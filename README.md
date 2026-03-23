# Co-teaching → SRIT → 当前

## 0. 总结：在研究什么

这条线的核心，不是“带噪标签训练”本身，也不只是“两个网络互相教”。  
它真正逐步逼近的问题是：

> **当训练数据中有错误、模型又会逐渐记住错误时，能否让多个学习体通过交互，持续地识别“更可信的训练信号”，并在优化层面避免陷入坏的、尖锐的、会过拟合噪声的解。**

于是整条线大致分成三层：

1. **2018 Co-teaching**：先解决“怎么在噪声下训练别崩”；
2. **2024 SRIT**：再解释“为什么这种交互式 teaching 是合理的”，并把 **SAM + EM** 引进来；
3. **现在的方向**：继续往前推进到  
   **多模型委员会 / 在线-流式 / soft posterior $q$ / 可靠度管理 / 多样性控制 / replay 记忆机制**。

可以把它理解成：  
**从经验技巧 → 理论解释 → 系统化在线交互学习器**。

---

## 1. 第一篇：2018 Co-teaching 的最初 idea、方法亮点与实现方式

### 1.1 论文最原始的切入点

2018 年的 Co-teaching 抓住了一个非常关键、后来几乎成为 noisy-label 领域基石的现象：

> **深网络先记住干净/简单样本，再逐渐记住噪声样本。**

因此它的逻辑是：

- 既然训练早期“小损失样本”更可能是干净的；
- 那就不要让模型在整个 batch 上都更新；
- 而是只让模型在一部分“小损失样本”上更新。

但作者进一步指出：

- 如果一个模型只相信自己选出来的小损失样本，
- 会把自己的偏差不断回灌给自己，
- 这会形成 **选择偏差**的累积。

所以引入了 **两个网络 + 交叉更新**。  
即：**我选样本给你学，你选样本给我学。**  
这就是 Co-teaching 的最小原型。

---

### 1.2 方法本体：它到底做了什么

原始 Co-teaching 的每个 mini-batch 过程非常简单：

1. 两个网络各自前向；
2. 各自按当前 loss 排序，取其中一部分 **small-loss** 样本；
3. 不是自己用自己选的样本更新，而是 **交换** 之后交叉更新；
4. 保留比例由 $R(T)$ 控制，并随训练推进逐渐减小。

形式上：

- 网络 $f$ 选出 $\bar D_f$
- 网络 $g$ 选出 $\bar D_g$
- 然后更新是  
  $$
  \theta_f \leftarrow \theta_f - \eta \nabla L(f,\bar D_g), \qquad
  \theta_g \leftarrow \theta_g - \eta \nabla L(g,\bar D_f)
  $$
- 且
  $$
  R(T)=1-\min\left\{\frac{T}{T_k}\tau,\tau\right\}
  $$
  也就是早期更多保留，后期更多丢弃。

---

### 1.3 这篇论文最初真正的亮点在哪里

我认为它最重要的亮点是下面三点。

#### (1) 把 memorization effect 真正转成了训练机制

之前大家知道“网络会先学干净样本”，但 Co-teaching 第一次把它直接做成了算法驱动原则：

- 小损失 $\Rightarrow$ 更可能干净；
- 所以小损失样本是训练信号的近似筛子。

这是它立起来的第一根柱子。

#### (2) 用 cross update 缓解 self-confirmation

如果单模型自筛自学，本质上是 self-training/self-paced 的一个变体，容易不断强化自身偏差。  
Co-teaching 的精妙之处在于：

- 不完全相信自己；
- 让 peer network 成为偏差缓冲层。 
  **错误不再原路返回自身，而是在对方那里被“衰减”一次。**

#### (3) 结构简单、实现轻

它几乎没有引入额外复杂模块：

- 不显式估计噪声转移矩阵；
- 不需要额外 teacher 网络；
- 不依赖特定 backbone；
- 可以 from scratch 训练。

这也是它后来影响大的原因：  
**简单，但抓住了 noisy-label 训练中最核心的动力学。**

---

### 1.4 原始实现方式与现在我的代码里的对应关系

从实现角度看，Co-teaching 的骨架非常小：

- 两个同结构网络；
- 每个 batch 计算 per-sample CE loss；
- 排序取前 $k$ 个；
- 交换索引；
- 各自反向传播。

你当前项目里的 `loss.py` 其实就很接近这个原始骨架：

- `loss_1/loss_2` 逐样本算 CE；
- 按 loss 排序；
- 取 `num_remember`；
- `ind_1_update` 与 `ind_2_update` 交叉更新。

---

### 1.5 第一篇的局限：也正是第二篇能切进去的地方

Co-teaching 很强，但它有天然边界：

#### (1) **经验上有效**，不理论
它告诉你“这样做有效”，但没有系统解释：

- small-loss selection 为什么可被看作合理推断；
- cross update 为什么能收敛；
- 它到底在优化什么目标。

#### (2) 它只处理了“选什么数据”，没处理“怎么优化更稳”
它更像是数据层面的噪声过滤，
而不是优化层面的 generalization control。

#### (3) 它本质还是 **hard selection**
即：

- 一个样本要么被选中；
- 要么完全不用。

这对于边界样本、渐变可信度样本、流式场景里的不确定样本都不够细。

#### (4) 它默认双模型、静态 epoch 训练
还没有走向：

- 多模型委员会；
- 在线/流式；
- 动态先验；
- 记忆缓冲；
- 可靠度管理。

这些空位，正好就是 2024 SRIT 和我们当前工作的切入点。

---

## 2. 第二篇：2024 SRIT 的切入点、方法、新亮点、论文突出处

## 2.1 它的切入点比 2018 更高一层

SRIT 的切入点不是“再把准确率提一点”，而是：

> **把 Co-teaching 当作 interactive teaching 的原型，重新解释它的优化机制与收敛逻辑。**

论文明确说，之前这类 interactive teaching 方法虽然实证有效，但缺少：

- underlying optimization principles；
- convergence analysis；
- 与更一般交互学习范式的联系。

所以第二篇的真正切口是两个问题：

1. **Co-teaching 到底在 loss landscape 上做了什么？**
2. **这种交互过程能否用 EM/概率模型方式解释？**

然后在此基础上，再问第三个问题：

3. **既然 EM 只保证局部收敛，能否用 SAM 帮它走向更平坦、更好的区域？**

这比 2018 的层级明显更高。

---

## 2.2 它的方法结构：不是简单“给 Co-teaching 加个 SAM”

SRIT 不是一句话的“Co-teaching + SAM”。  
它的论文叙事其实有三层。

### 第一层：loss landscape 视角

SRIT 认为，Co-teaching 的小损失筛选，本质上等于：

- 把高损失样本对应的“坏区域”切掉；
- 只在低损失部分上更新参数。

因此从梯度分解看，它近似是在做：

$$
\nabla L(\text{all data}) - \nabla L(\text{high-loss part})
$$

也就是一种“loss landscape reduction”。  
这给了 Co-teaching 一个几何解释：  
**不是神秘地互教，而是在主动避开高损失/高噪声区域。** 

---

### 第二层：EM / latent-variable 视角

SRIT 把“样本是否干净”建模成隐变量 $Z$，把 Co-teaching 看成一个 EM-like 过程：

- **E-step**：根据当前网络，推断哪些样本更像 clean；
- **M-step**：用对方给出的高概率 clean 子集更新自己。

它进一步把 lower bound / latent posterior / variational distribution 都写了出来，声称 interactive teaching 的迭代可以被看作对变分下界的持续优化。

这一步是 SRIT 最大的“理论升级”。

因为从这里开始，Co-teaching 不再只是 heuristic，而被解释为：

> **一个以 clean/noisy 隐变量为核心的迭代参数估计过程。**

---

### 第三层：SAM 视角

SRIT 的判断是：

- EM 式过程只能保证局部最优；
- 而 SAM 的作用是让优化落到更平坦的区域；
- 平坦区域意味着更强 generalization、更不容易过拟合 noisy labels。

因此它在原有的“loss information exchange”之上，又加了一层“sharpness knowledge exchange”。  
它称之为 **dual-level interactive learning**：

1. 第一层：交换 small-loss 样本；
2. 第二层：在这些样本上做 SAM 风格的扰动-更新。

也就是：

- 不是只告诉对方“哪些样本值得学”；
- 还通过 SAM，把“怎样在更平坦区域学”嵌进去。

---

## 2.3 第二篇论文的新亮点

我认为 SRIT 的真正新亮点有四个。

### (1) 它把 Co-teaching 从“方法”提升成“原型”

2018 里 Co-teaching 是一个具体算法；  
2024 里它被提升成 **interactive teaching prototype**。

这个升级非常关键。因为一旦是原型，就意味着可以往外推广到：

- 多模型；
- 多智能体；
- 多教师-学生；
- demonstration-conditioned teacher；
- collaborative AI systems。

这正好对上我们最近讨论里你关心的“如何往更一般的 multi-agent / multi-teacher 系统推广”。

---

### (2) 它第一次认真给出“为什么互教可能收敛”的叙事

严格说，SRIT 的 EM 解释未必在数学上已经达到无可挑剔的程度，但它至少做了重要的一步：

- 把 clean/noisy 区分显式建成隐变量；
- 让交互选样拥有了 posterior/variational 的解释；
- 让“互教”不再只是直觉故事。

这使得后续再做 soft $q$、posterior weighting、online prior 更新，就有了理论母体。

---

### (3) 它把 “选样” 与 “优化地形” 连到了一起

Co-teaching 原本只管“样本子集”；  
SRIT 把它与 sharpness/generalization 连接起来：

- 先筛样本；
- 再通过 SAM 走向 flatter minima。

这意味着 noisy-label learning 不只是“筛干净数据”，还要“找稳的极小值”。

这对我们当前项目尤其重要，因为我们现在的许多问题，本质上已经不再只是数据筛选问题，而是 **筛选-优化-记忆耦合问题**。

---

### (4) 它给了我们一个非常好的论文叙事模板

SRIT 的论文结构其实已经告诉我们，后续扩展最好怎么讲：

1. 先从具体 noisy-label 算法出发；
2. 再上升到 latent-variable / EM 解释；
3. 再接优化几何（SAM / flatness）；
4. 最后扩到更一般的交互学习。

这也是为什么它对我们现在的方向特别有价值：  
**不是只学它的算法，而是学它的叙事结构。**

---

## 2.4 第二篇论文相对于第一篇，突出了什么

如果用一句话概括：

> **第一篇突出“能用”，第二篇突出“为什么能用，以及怎样优化得更好”。**

更细一点：

### 第一篇最突出的是
- memorization effect 的利用；
- 双网络 cross update；
- 简洁强力的 noisy-label 训练机制。

### 第二篇最突出的是
- interactive teaching 的理论解释；
- EM 视角；
- SAM 带来的 flatness/generalization 增强；
- dual-level interaction 的概念化。

---

## 3. 现在探索了什么

## 3.1 不满足于 “双模型 small-loss + SAM” 

根据我们最近两次讨论，你的目标已经明显超出 SRIT 原文边界，核心变化有三条。

### 第一条：从“双网络”走向“委员会/多模型”
不再只是 $f,g$ 两个 peer：

- 而是 $M$ 个模型；
- 存在 active / weak / absorbed 的结构分化；
- 需要考虑可靠度 $\lambda_m$、吸收/剪枝、委员会规模、多样性。  

这使得系统从“互教”变成“委员会推断 + 交互优化”。

### 第二条：从 hard selection 走向 soft posterior
不再满足于：
- 选中 or 不选中；

而是想要：
- 每个样本有一个 soft 责任度 $q_i$；
- $q_i$ 作为“更可能干净”的 posterior 进入训练。

这一步一旦做实，方法就会从 Co-teaching/SRIT 风格，真正过渡到 **广义 EM / variational filtering** 风格。

### 第三条：从静态 epoch 训练走向在线/流式
你最近讨论里已经很明确地把问题往这个方向推了：

- 数据不一定是完全静态的；
- 噪声分布可能非平稳；
- 需要 recent window、replay、EMA prior、动态可靠度。  

这时“clean selection”不再只是当前 batch 的问题，而是 **时间上的过滤与记忆问题**。

---

## 3.2 代码什么程度

> 现在的 `main.py` 已经把经典 Co-teaching 扩展成  
> **$M$ 模型（默认 3）+ 同伴聚合筛选（排除自身）+ SAM 两步更新 + 可靠度 $\lambda_m$ 加权聚合 + 监控 $q$ 与 overlap 的可运行原型。** 

而从 `main.py` 也能看出，你现在其实已经有了不少比 SRIT 更进一步的机制雏形：

- `num_models`：多模型委员会；
- `aggregation`：同伴损失聚合；
- `sam_rho`：SAM 两步更新；
- `q_mode`：`posterior / loss / bmm`；
- `pi_init / pi_ema / pi_beta_a / pi_beta_b`：慢变量先验；
- `replay_mode`：`legacy / purified`；
- `lambda_active / lambda_patience / min_active`：软吸收/活跃集合；
- `explore_delta / explore_trigger`：多样性探索。

所以严格说，**你的系统已经不是“复现 SRIT”，而是在试图把 SRIT 推向 online committee EM system。**

---

## 4. 现在的真正难点：怎么再加一个模块 && 系统闭环为什么会坏

根据当前实验总结：

- 前期 test acc 能上升到 75%+；
- 但后期 train acc 接近 100%，test acc 反而掉到 48%-49%；
- $q_{\text{mean}} \to 0.999$；
- $\pi_t \to 0.99$；
- replay 打开后 overlap 反而更高。

这说明问题已经不是“有没有理论解释”，而是：

> **理论叙事进入系统实现后，是否产生了自证循环。**

---

## 4.1 当前最本质的困难：$q$ 的自证循环

现在最危险的闭环是：

1. 模型开始记忆噪声；
2. 对噪声样本的 $p(y|x)$ 也升高；
3. 于是 $q_i$ 被推高；
4. 再通过 EMA 把 $\pi_t$ 推高；
5. 高 $q$ 样本获得更大训练权重并进入 replay；
6. 模型被进一步训练在这些错误样本上；
7. 形成正反馈闭环。

这件事非常关键，因为它说明：

- 仅仅把 Co-teaching 解释成 EM，不足以保证系统稳定；
- 一旦 posterior 的计算只依赖模型自己，系统就会“相信自己越来越多”。

所以当前真正的问题是：

> **posterior 的证据来源是否足够外生，是否能阻止自我污染。**

---

## 4.2 第二个难点：replay 不一定稳定，反而可能固化错误

理论上 replay 是为了：

- 维持稳定；
- 抗遗忘；
- 模拟流式训练中的 clean memory。

但你的实验已经显示：

- replay 打开后最终 overlap 更高；
- $\pi_t$ 更高；
- 性能并没有被真正救回来。

这表明在当前系统里 replay 并不是“纯化记忆”，而可能在做：

> **错误样本的重复灌输。**

所以 replay 不是天然正确的补丁，它只有在 admission / eviction / stability criterion 做对时，才可能是贡献点。

---

## 4.3 第三个难点：多模型不是越多越好，委员会会趋同

当前项目里另一个突出的现象： overlap 高。  
这意味着：

- 多个模型越来越像；
- 选到同一批样本；
- 互教退化为“大家一起重复自己”。

一旦如此，多模型系统的优势就被耗尽了：

- 没有分歧；
- 没有互补；
- 没有纠错；
- 只有同步偏移。

所以多模型方向真正的难点不在“加几个模型”，而在：

> **如何让委员会在不失稳的前提下保持有效异质性。**

---

## 4.4 第四个难点：可靠度管理很容易引入新的闭环

你们最近讨论里已经抓到了很重要的一点：

如果 $\lambda_m$ 的来源不干净，例如直接用 test acc 或被污染的在线指标，
那它会进入 ensemble likelihood，再影响 $q_i$，再反过来改训练分布，形成新的不可解释闭环。

因此“模型可靠度”不是普通工程 trick，  
而是一个需要谨慎放进概率图里的变量。

---

























































## 5. 现在各个方向分别意味着什么，它们各自可能产出什么论文点

下面我按你最需要的方式来写：  
**每个方向 = 它解决什么问题 + 为什么逻辑成立 + 可能形成什么论文点。**

---

## 5.1 方向 A：把 hard selection 真正升级成 soft posterior $q$

### 它解决的问题
解决 Co-teaching / SRIT 里 hard subset 的粗糙性：

- 边界样本没法表达“半可信”；
- 选与不选过于离散；
- 不利于在线/流式累积判断。

### 为什么逻辑成立
如果把“样本是否干净”定义为隐变量 $z_i \in \{0,1\}$，  
那么最自然的扩展就是不再直接取集合 $S$，而是估计

$$
q_i \approx \Pr(z_i=1 \mid x_i,\tilde y_i,\text{committee})
$$

再用 $q_i$ 去加权训练。  
这和 SRIT 的 EM 解释是一脉相承的，只是把它从 paper-level 解释真正推进成 algorithm-level 主体。

### 它可能形成的论文点
**论文点 1：Online variational E-step for noisy-label committee learning**

即：

- 不再用 heuristic small-loss top-k；
- 而是用 posterior responsibility 作为 clean belief；
- 支持 batch 内、时间上、模型间的融合。

### 需要警惕
如果 $q_i$ 完全依赖当前模型的 $p(y|x)$，就会自证循环。  
所以这个方向要成立，必须同时引入：

- 外部锚点；
- 温度/熵控制；
- 或历史/邻域/跨模型 disagreement 约束。

---

## 5.2 方向 B：把 SAM 从“附加优化器”变成“广义 M-step”的一部分

### 它解决的问题
SRIT 已经说明：

- 只筛样本不够；
- 还要避免优化落在尖锐坏解上。

但它还是“先筛样本，再加 SAM”。

你现在可以更进一步，把它写成：

$$
\min_{\theta_m}\max_{\|\epsilon\|\le \rho}
\frac{\sum_i q_i \, \ell(f_m(x_i;\theta_m+\epsilon),\tilde y_i)}
{\sum_i q_i}
$$

也就是：

> **在 posterior-weighted clean belief 上做 sharpness-robust M-step。**

### 为什么逻辑成立
这一步和 SRIT 的思想完全兼容，但更统一：

- E-step 给出 soft responsibility；
- M-step 不只是普通 ERM，而是 SAM-robust ERM。

于是“筛什么数据”和“怎么更新参数”被写进同一个优化对象。  
这在理论叙事上会比“给 Co-teaching 插一个 SAM”强很多。

### 它可能形成的论文点
**论文点 2：SAM-robust generalized M-step under posterior weighting**

这会是一个比 SRIT 更整洁的主方法表述。

---

## 5.3 方向 C：把双模型拓成委员会，但核心不是规模，而是“稳定委员会管理”

### 它解决的问题
双模型互教的表达能力有限：

- 两个模型太容易同时漂；
- 无法区分谁是弱模型；
- 无法做教师权重化；
- 无法表达“活跃/吸收/恢复”的委员会动态。

### 为什么逻辑成立
一旦模型数扩成 $M$，自然可以引入：

- 聚合似然；
- 可靠度 $\lambda_m$；
- active set；
- soft absorb / hard prune；
- ensemble disagreement 作为不确定性信号。

这本质上把系统从“二元 peer review”推进成“有组织的 committee inference”。  
而这比单纯加模型数有更强的研究意义。

### 它可能形成的论文点
**论文点 3：Committee management for stable interactive teaching**

注意这不是“我们用了 3 个模型”，而是：

- 可靠度如何进入后验；
- 弱模型如何退居学生；
- 委员会如何避免崩塌与过早一致。

### 关键成立条件
这个方向必须解决两件事：

1. $\lambda_m$ 不能来自泄漏或污染严重的指标；
2. 吸收机制必须和多样性联动，否则委员会会越来越一致。

---

## 5.4 方向 D：把多样性从“额外正则”提升成 posterior 不确定性的组成部分

### 它解决的问题
当前系统高 overlap 说明：

- 多模型结构存在；
- 但多模型信息并没有真正贡献差异化判断。

### 为什么逻辑成立
如果两个样本在各模型间分歧很大，这本身就说明：

- clean belief 不应过早变成接近 1 或 0；
- posterior 需要更高温度或更大熵；
- 训练应保留一部分 explore 成分。

所以最自然的做法不是强行加 decorrelation loss，  
而是把 disagreement 融进：

- $q_i$ 的温度；
- 样本权重；
- overlap-triggered exploration。

这会比“让模型彼此不同”更自然，因为你不是为了不同而不同，而是把不同当作 **不确定性证据**。

### 它可能形成的论文点
**论文点 4：Diversity-aware posterior tempering in committee-based noisy learning**

这是一个很有潜力的小而精贡献点。

---

## 5.5 方向 E：流式 / replay / EMA 先验，真正把系统变成在线广义 EM

### 它解决的问题
你最近讨论里已经很明确：

- 真实场景未必是静态 dataset；
- 数据分布与噪声比例可能变化；
- 需要 window、replay、慢变量先验。

### 为什么逻辑成立
一旦引入：

- recent window $W$；
- purified replay $R$；
- $\pi_t \leftarrow \text{EMA}(\text{batch-mean}(q))$；

系统就自然从“离线 batch Co-teaching”升级成：

> **online generalized EM with committee memory**

它不只是训练算法，而是一个随时间更新 belief / memory / active committee 的系统。

### 它可能形成的论文点
**论文点 5：Online generalized EM for noisy data streams with committee replay**

但注意，这个方向最难，因为它最容易出闭环病。  
你当前实验已经说明，replay 如果 admission criterion 建不好，反而会“认知固化”。

---

## 6. 我认为现在最有希望的主线，不是平均推进所有方向，而是分层推进

如果从“论文能成”和“实现能稳”两个角度一起看，我建议把方向分成两层。

---

## 6.1 第一层：最稳、最应该先做成的主线

### 主线定义
**Posterior-weighted committee training with SAM**

核心只做三件事：

1. 多模型委员会聚合；
2. soft $q_i$ 后验进入训练；
3. SAM 作为 q-weighted M-step。

### 为什么最稳
因为它完整继承了两篇论文的主干：

- 2018：small-loss / peer teaching；
- 2024：EM + SAM；

同时又自然形成你的新意：

- 从 hard subset 到 soft posterior；
- 从 2 个模型到 committee。  

这是最可能形成“主论文主方法”的版本。

---

## 6.2 第二层：最适合作为差异化增强或后续扩展的部分

### 包括
- reliability 双源融合；
- soft absorb / active set；
- overlap-triggered diversity；
- purified replay；
- streaming prior 更新。

### 为什么放第二层
因为这些都是强增强，但每一个都可能引入额外闭环。  
如果一开始全压进去，系统可能很难判断到底哪一步起作用、哪一步在害你。

所以更合理的策略是：

- 先把 **q-weighted committee + SAM** 做成稳定主干；
- 再逐个加：
  - 可靠度；
  - 多样性；
  - replay；
  - 流式先验。

这也是最符合科研推进逻辑的路径。

---

## 7. 用一句话总结三篇/三阶段之间的关系

### 2018 Co-teaching
> 发现并利用“先学干净后记噪声”的训练动力学，用双网络 small-loss 互教解决 noisy labels。

### 2024 SRIT
> 把 Co-teaching 提升为 interactive teaching 原型，用 EM 解释其隐变量推断逻辑，再用 SAM 改善其局部优化与泛化。

### 我们当前方向
> 进一步把它推广成一个 **多模型、soft posterior、可在线更新、带委员会管理与记忆机制的广义 EM 交互学习系统**。

---

## 8. 最后的判断：现在最值得写成论文的“中心命题”是什么

如果我替你压缩成一句最值得继续押注的中心命题，我会写成：

> **Co-teaching/SRIT 的本质不是“双网络互教”，而是“对 clean latent variable 的交互式近似推断”；当这一推断被推广到多模型委员会、soft posterior 与 sharpness-robust M-step，并被可靠度与多样性机制稳定下来时，它就从 noisy-label trick 升级为一个在线广义 EM 学习框架。**

这句话背后，对应的论文核心卖点可以是：

1. **交互式 clean posterior 推断**；
2. **posterior-weighted SAM 优化**；
3. **委员会稳定性管理**；
4. **面向流式/非平稳噪声的扩展能力**。

这四点里，前两点是主干，后两点是你最有潜力做出明显区别度的地方。

---

## 9. 一段适合直接放在你自己笔记里的简明总结

Co-teaching 的原始贡献，在于把 deep network 的 memorization effect 变成了可操作的 noisy-label 训练机制：通过 small-loss 筛样和 cross update，让两个网络互相削弱各自的 selection bias。SRIT 的提升，不在于简单加了 SAM，而在于把 Co-teaching 提升为 interactive teaching 原型：一方面从 loss landscape 视角解释其本质是在切除高损失区域，另一方面把样本 clean/noisy 建模为隐变量，用 EM/变分下界解释其交互迭代，再通过 SAM 让 M-step 倾向于更平坦、泛化更好的区域。我们当前的推进，则是把这一线索从“双模型 + hard subset + 离线训练”继续推广到“多模型委员会 + soft posterior $q$ + q-weighted SAM + 可靠度/多样性/记忆机制 + 在线广义 EM”。当前真正的难点已经不是理论名词是否齐全，而是系统闭环是否稳定：$q$ 是否会自证循环，replay 是否会固化错误，委员会是否会快速趋同，可靠度是否引入新的污染回路。因此，最值得推进的主线不是无节制加模块，而是先把“posterior-weighted committee + SAM”做成稳定主干，再逐步加入可靠度管理、多样性温度控制与 purified replay，最终把这条线从 noisy-label 方法推进成一个可解释、可扩展、可在线运行的交互推断学习框架。
