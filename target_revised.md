# target.md（重写版）

## 0. 结论先行：新的主线应该是什么

这条研究线不应再表述为“多模型 + replay + 一些修补项”，而应明确成：

> **双时间尺度的在线广义 EM 委员会学习器（Two-timescale Online Generalized EM Committee Learning）**
>
> 其核心由三部分组成：
> 1. **训练后验** $Q_i$：估计样本“当前是否可学 / 更可能干净”；
> 2. **记忆后验** $U_i$：估计样本“是否值得进入长期记忆”；
> 3. **广义 M-step**：在 $Q_i$ 加权下进行“硬监督 + 软监督 + SAM”的鲁棒优化。

新的主张不是“让一个 $q$ 统管训练、先验、replay”，而是：

- 用 **$Q_i$** 决定当前 batch/window 的训练强度；
- 用 **$U_i$** 决定 candidate buffer 到 purified memory 的准入与淘汰；
- 用 **慢变量**（而不是瞬时统计）更新 $Q_i,
\pi_t,\lambda_m$；
- 用 **purified memory** 切断 posterior–memory 的正反馈闭环。

这比旧版更科学的原因是：

1. 它直接针对当前已观测到的失败机制：$q \uparrow \rightarrow \pi_t \uparrow \rightarrow replay \uparrow \rightarrow q \uparrow$；
2. 它将“训练可信度”和“记忆价值”分离，避免单一 posterior 统治整个系统；
3. 它能支持更可信的理论目标：**近似单调改进、双时间尺度稳定性、记忆污染上界**，而不是难以成立的全局收敛。

---

## 1. 问题定义：真正要解决的不是 noisy label 本身，而是 noisy stream 下的闭环污染

### 1.1 任务定义

给定在线到达的带噪样本流 $\{(x_t, \tilde y_t)\}_{t=1}^T$，维护一个大小受限的委员会 $\{f_m\}_{m=1}^M$ 与一个有限记忆系统。

目标是在以下三点之间取得平衡：

1. **即时鲁棒学习**：当前流样本中尽可能利用干净部分；
2. **长期记忆稳定**：避免 replay 被污染后反复放大错误；
3. **委员会推断可靠**：多模型分歧应成为不确定性来源，而不是共同偏差的重复。

### 1.2 真正的核心矛盾

当前系统的根本问题不是“模块不够多”，而是：

- 如果 $q_i$ 仅由当前模型/委员会对 $\tilde y_i$ 的预测给出，那么它更像 **自我确认**，而不是推断；
- 如果同一个 $q_i$ 同时控制训练权重、clean prior $\pi_t$、replay admission，那么它会形成 **自证闭环**；
- 如果 replay 是“高 $q$ 就入库、只进不出”，则系统会逐渐把错误记忆固化成长期先验。

因此新方案必须做到：

- $Q_i$ 不再只是瞬时 posterior；
- replay 不再是 threshold push-only buffer；
- 训练与记忆的证据来源部分解耦。

---

## 2. 总体方法：双后验 + 双时间尺度 + 双层记忆

## 2.1 两个潜变量，不再只用一个 $z_i$

定义两个潜变量：

- $z_i \in \{0,1\}$：样本当前是否可学 / 更可能干净；
- $r_i \in \{0,1\}$：样本是否值得进入长期记忆。

对应两个后验：

$$
Q_i \approx P(z_i=1 \mid \text{current evidence}),
$$

$$
U_i \approx P(r_i=1 \mid z_i, \text{memory evidence}).
$$

解释：

- $Q_i$ 是 **训练后验**，用于当前优化；
- $U_i$ 是 **记忆后验**，用于是否写入或保留在 purified memory。

这是整个新方案最关键的改动：

> **训练可信度 $\neq$ 长期记忆价值。**

很多样本可能“当前可学”，但并不值得长期存；也有些样本暂时模糊，但作为边界/覆盖样本有较高记忆价值。

---

## 2.2 两个时间尺度

定义快变量与慢变量：

- 快变量：模型参数 $\theta_t$；
- 慢变量：$Q_t, \pi_t, \lambda_t$，以及 memory statistics。

更新形式：

$$
\theta_{t+1}=\theta_t-\eta_t\,g(\theta_t,Q_t,\mathcal M_t),
$$

$$
Q_{t+1}=Q_t+\beta_t h_Q(\theta_t,Q_t),
$$

$$
\pi_{t+1}=\pi_t+\beta_t h_\pi(\theta_t,Q_t),
$$

$$
\lambda_{t+1}=\lambda_t+\beta_t h_\lambda(\theta_t,Q_t),
$$

其中 $\beta_t \ll \eta_t$。

这意味着：

- 模型参数快速适应；
- 责任度、先验、模型可靠度慢速变化；
- 系统被解释成一个 **two-timescale generalized EM**，而不是“若干 EMA trick”。

---

## 2.3 双层记忆：Candidate Buffer $D$ 与 Purified Memory $R$

replay **绝不能只进不出**。

新的记忆结构：

### 候选池 $D$

- 保存最近窗口内的候选样本；
- 允许重新评估、延迟决定；
- 样本可能自动过期，不要求长期保存。

### 净化记忆 $R$

- 有固定容量；
- 只有 $U_i$ 高、且覆盖/冗余条件合格的样本才进入；
- 支持 **admission、eviction、refresh、sampling** 四个操作。

这意味着 replay dynamics 不再是 push-only，而是：

$$
D_t \xrightarrow{U_i\text{-based purification}} R_t,
$$

并伴随

$$
R_t \xrightarrow{evict/refresh/resample} R_{t+1}.
$$

---

## 3. E-step：训练后验 $Q_i$ 应由三类证据共同决定

### 3.1 原则

E-step 不能只用 $p_{\text{ens}}(\tilde y_i \mid x_i)$。

新的 $Q_i$ 由三类证据共同决定：

1. **预测证据**：委员会对当前标签或类别的支持；
2. **稳定性证据**：增强一致性、margin、SAM gap、loss rank 等；
3. **结构证据**：与 purified memory prototype / teacher representation 的邻域一致性。

### 3.2 预测证据

在 active committee 上使用等权平均，不在 E-step 中立即用 $\lambda_m$ 做连续加权：

$$
p_{\mathcal A}(y\mid x_i)=\frac{1}{|\mathcal A_t|}\sum_{m\in\mathcal A_t} p_m(y\mid x_i).
$$

定义

$$
e_i^{\text{pred}} = \log p_{\mathcal A}(\tilde y_i\mid x_i).
$$

说明：

- $\lambda_m$ 只用于 active/inactive gating，不直接参与 posterior 计算；
- 否则会形成新的快反馈：$\lambda_m \to p_{\mathcal A} \to Q_i \to$ 数据分布 $\to \lambda_m$。

### 3.3 稳定性证据

对每个样本构造：

- top1/top2 margin：$\mathrm{margin}_i$；
- augmentation consistency：$\mathrm{cons}_i$；
- sharpness proxy / SAM gap：$\mathrm{sharp}_i$；
- rank loss / history deviation：$\mathrm{rankloss}_i$。

定义稳定性分数：

$$
e_i^{\text{stab}}=
\gamma_1\,\mathrm{margin}_i
+\gamma_2\,\mathrm{cons}_i
-\gamma_3\,\mathrm{sharp}_i
-\gamma_4\,\mathrm{rankloss}_i.
$$

直觉：

- 可学样本应当 margin 更大、一致性更高、sharpness 更低；
- 仅靠小损失不足以区分“真正干净”与“已被记忆的噪声”。

### 3.4 结构证据（默认次要，不做主骨架）

若引入表征结构，推荐只使用轻量版本：

$$
e_i^{\text{struct}}=
\gamma_5\,\mathrm{proto\_sim}_i
+\gamma_6\,\mathrm{teacher\_align}_i.
$$

其中：

- $\mathrm{proto\_sim}_i$：与 purified memory 中类原型或局部中心的相似度；
- $\mathrm{teacher\_align}_i$：与慢 teacher / EMA teacher 表征的一致性。

注意：

- 图中心性可作为可选增强，但不应做第一版主骨架；
- 原因是 clean-major-cluster 假设在多模态类中不总成立，计算也更重。

### 3.5 瞬时 posterior 与慢变量 posterior

定义瞬时分数：

$$
s_i = e_i^{\text{pred}} + e_i^{\text{stab}} + e_i^{\text{struct}} - b_t.
$$

得到瞬时后验：

$$
\hat q_i = \sigma\left(\frac{s_i}{T_t}\right).
$$

再用 EMA 得到真正用于训练的慢变量：

$$
Q_i^{(t)}=\beta_Q Q_i^{(t-1)} + (1-\beta_Q)\hat q_i.
$$

其中 $T_t$ 可在高 overlap 时升高，以防后验过早二值化。

### 3.6 clean prior 的更新

新的 $\pi_t$ 不再直接由 batch mean(q) 生硬驱动，而应基于慢变量：

$$
\pi_{t+1} = \beta_\pi \pi_t + (1-\beta_\pi)\,\mathrm{mean}(Q_i\text{ over recent window}).
$$

这一步的关键是：

- $Q_i$ 已经是多证据、慢变量；
- 所以 $\pi_t$ 的更新不再完全暴露于当前 batch 的噪声波动。

---

## 4. M-step：不是“选样后普通训练”，而是广义鲁棒 M-step

### 4.1 三态训练原则

对样本不做硬二分，而做三态处理：

- **clean-like**：高 $Q_i$，保留硬监督；
- **ambiguous**：中等 $Q_i$，以软监督为主；
- **reject-like**：低 $Q_i$，仅弱更新或不参与主损失。

定义权重：

$$
w_i = \mathrm{clip}(Q_i,w_{\min},w_{\max}).
$$

定义 stop-gradient 的慢 teacher 分布或稳定委员会分布：

$$
\bar p_i = \mathrm{sg}\left(\frac{1}{|\mathcal T_t|}\sum_{m\in\mathcal T_t} p_m(\cdot\mid x_i)\right),
$$

其中 $\mathcal T_t$ 是 teacher set（可取 slow teacher 或 active committee 的 EMA 版本）。

### 4.2 基础损失

对模型 $m$，定义基础损失：

$$
\mathcal L_m^{\text{base}}=
\sum_i \Big[
\alpha\,w_i\,\mathrm{CE}(f_m(x_i),\tilde y_i)
+ (1-\alpha)(1-w_i)\,\mathrm{KL}(\bar p_i\,\|\,f_m(x_i))
\Big].
$$

解释：

- 高 $Q_i$ 样本主要用原标签监督；
- 中低 $Q_i$ 样本不直接丢弃，而是接受稳定 soft target；
- 这样把“可疑样本”从 hard selection 转为 soft supervision。

### 4.3 SAM 作为广义 M-step

不把 SAM 当外挂优化器，而是作为 generalized M-step：

$$
\mathcal L_m^{\text{SAM}}(\theta_m)=
\max_{\|\epsilon\|\le \rho}
\mathcal L_m^{\text{base}}(\theta_m+\epsilon),
$$

$$
\theta_m \leftarrow \arg\min \mathcal L_m^{\text{SAM}}(\theta_m).
$$

关键点：

- SAM 主要作用于 clean-like 与 moderately ambiguous 样本；
- 在极低 $Q_i$ 样本上，不应强行施加强鲁棒优化，因为那只会放大无意义梯度。

### 4.4 为什么不再把“普通 CE + top-k”当主训练

因为那仍然停留在 classic Co-teaching 层面：

- E-step 变连续了；
- M-step 仍然是离散样本选择后做普通 ERM；
- 理论闭环并不完整。

新的主张是：

> **posterior-weighted robust M-step**

而不是“posterior only for filtering”。

---

## 5. 委员会管理：$\lambda_m$ 不做快权重，只做慢结构变量

### 5.1 原则

$\lambda_m$ 的角色应被收缩为：

- active/inactive gating；
- teacher/student role assignment；
- soft absorb / 恢复。

而不应当在第一版中直接进入 E-step 的连续加权。

### 5.2 在线 proxy 可靠度

避免 test leakage，也避免只依赖 noisy train acc。

定义 proxy 分数：

$$
s_m =
-\alpha_1\,\mathrm{sharp\_gap}_m
-\alpha_2\,\mathrm{harmful\_disagreement}_m
+\alpha_3\,\mathrm{stability}_m
+\alpha_4\,\mathrm{memory\_alignment}_m.
$$

再更新：

$$
\lambda_m^{t+1}=\beta_\lambda \lambda_m^t + (1-\beta_\lambda) g(s_m).
$$

### 5.3 soft absorb 而不是 hard prune

委员会管理采用三级状态：

- **active**：进入 E-step 与主训练；
- **absorbed**：暂不参与 E-step，但继续作为学生被蒸馏/回放训练；
- **revived**：若 proxy 恢复，则重新进入 active。

只在模型数足够多且计算压力极大时，才考虑 hard prune。

### 5.4 最小活跃规模

始终保持：

$$
|\mathcal A_t| \ge M_{\min}, \quad M_{\min}\ge 2 \; (\text{建议 }3).
$$

原因：

- 否则委员会退化为单模型自证；
- 分歧信号消失，后验温度调节失效。

---

## 6. 记忆系统：Replay 不是只进不出，而是 admission / eviction / refresh / sampling

## 6.1 回答补充问题：Replay 只能“只进不出”吗？

**不能。**

除非把 replay 定义成“无容量限制的日志档案”，否则真正用于 continual learning 的 memory 在理论和实践上都不应是只进不出。

在 noisy stream 场景里，push-only replay 有三重问题：

1. **污染积累**：一旦 admission 出错，错误样本会永久滞留；
2. **覆盖退化**：早期样本占据容量，后期有价值样本无法进入；
3. **分布僵化**：memory 不能适应 drift 或 representation 演化。

所以新的 replay 设计必须同时包含：

- **admission**：谁能进；
- **eviction**：谁该出；
- **refresh**：谁虽然保留，但其统计量要重估；
- **sampling**：训练时如何从 memory 中取样。

## 6.2 记忆后验 $U_i$

给 candidate $i$ 一个记忆价值分数：

$$
v_i =
\eta_1 Q_i
+\eta_2 \mathrm{coverage\_gain}_i
+\eta_3 \mathrm{prototype\_support}_i
-\eta_4 \mathrm{redundancy}_i
-\eta_5 \mathrm{age\_penalty}_i.
$$

定义记忆后验：

$$
\hat u_i = \sigma\left(\frac{v_i}{\tau_R}\right),
$$

并做慢更新：

$$
U_i^{(t)} = \beta_U U_i^{(t-1)} + (1-\beta_U)\hat u_i.
$$

### 6.3 Admission

若样本在 candidate buffer 中满足：

$$
Q_i > \tau_Q, \quad U_i > \tau_U,
$$

则进入 purified memory $R$。

### 6.4 Eviction

purified memory 不是永久库。若样本满足以下任一条件，可被驱逐：

1. $U_i$ 持续低于阈值；
2. 与已有样本冗余过高；
3. 其类别/区域覆盖已被更高质量样本替代；
4. 年龄过高且采样价值下降。

推荐打分式 eviction：

$$
E_i = \kappa_1 (1-U_i) + \kappa_2\,\mathrm{redundancy}_i + \kappa_3\,\mathrm{age}_i - \kappa_4\,\mathrm{coverage\_need}_i.
$$

驱逐 $E_i$ 最高者。

### 6.5 Refresh

memory 中保留的样本，其 $U_i$、prototype support、teacher alignment 等统计量应周期性重估；
不是“进库以后不再更新其后验”。

### 6.6 Sampling

训练时对 memory 样本不做纯 uniform replay，可采用：

$$
P(i\in R\text{ sampled}) \propto
\exp\big( a U_i + b\,\mathrm{coverage\_need}_i - c\,\mathrm{replay\_freq}_i \big).
$$

这样避免 replay 被少数“容易高置信样本”垄断。

---

## 7. 是否真的科学合理：对当前方案的再次审视与修正

### 7.1 不是所有从论文得到的启发都应纳入第一版

本方案明确**不把以下内容作为第一版主骨架**：

- GNN / 全图中心性作为核心 E-step；
- 复杂 decorrelation / orthogonality 约束；
- 角色化多网络（固定 3/4 网各司其职）；
- 一开始就做强非平稳 one-pass open-world drift。

原因不是这些想法没价值，而是：

- 它们会迅速增加假设和计算复杂度；
- 但并不直接击中当前主矛盾——posterior–memory 的正反馈污染。

### 7.2 当前方案仍然存在的潜在薄弱点

#### 薄弱点 A：soft target 的来源可能再次自证

如果 ambiguous 样本的 soft target 直接来自当前 fast committee，那么系统仍可能自蒸馏错误。

**修正**：soft target 默认来自 stop-gradient 的 slow teacher / EMA teacher，或来自 purified memory prototype，而不是当前 fast logits 的瞬时平均。

#### 薄弱点 B：结构证据的 clean-major-cluster 假设不总成立

如果类本身多模态，中心性/原型方法可能误伤长尾子簇。

**修正**：第一版只用轻量 prototype support，图方法留作增强项；同时在 memory 中加入 coverage-aware 机制，避免只保留主模态。

#### 薄弱点 C：SAM 可能削弱塑性

在强 drift 或新类出现时，SAM 可能抑制快速适应。

**修正**：第一版不在极低 $Q_i$ 或高 novelty 区域强行施加大 $\rho$；必要时可用 stage-dependent / sample-weighted SAM，但不作为主贡献点。

#### 薄弱点 D：$Q_i$ 与 $U_i$ 的分离带来实现复杂度

这是必要复杂度，不是冗余复杂度。

因为如果不分离，理论对象会混乱：

- 训练和记忆共享一个 posterior；
- 你无法对 memory contamination 单独分析；
- 也无法解释为什么某些样本“当前不宜训练，但值得记忆观察”。

### 7.3 是否真的理论可行

**可行，但目标必须收缩。**

不能声称：

- 深网全局收敛；
- 真 posterior 精确恢复；
- memory 最优选择。

真正可行的理论目标是三类。

---

## 8. 理论目标：不追求全局最优，只做三条“可信中定理”

## 8.1 定理目标一：广义 EM 的近似单调改进

定义 surrogate free energy：

$$
\mathcal F(Q,\theta)
=
\sum_i Q_i \log \frac{\pi_t p_\theta(\tilde y_i\mid x_i)}{Q_i}
+
(1-Q_i) \log \frac{(1-\pi_t)u(\tilde y_i)}{1-Q_i}
-
\lambda \mathcal R_{\text{SAM}}(\theta;Q),
$$

其中 $u(\tilde y_i)$ 是噪声通道基线分布。

若：

- E-step 用 stop-gradient teacher 给出 $Q$ 的近似更新；
- M-step 对 $\mathcal L^{\text{SAM}}$ 做充分下降；

则可以争取如下结果：

$$
\mathcal F(Q^{t+1},\theta^{t+1})
\ge
\mathcal F(Q^t,\theta^t) - \varepsilon_t,
$$

其中 $\varepsilon_t$ 是近似误差。

这不是全局收敛，但足以支撑 generalized EM 的理论叙事。

## 8.2 定理目标二：双时间尺度稳定性

在标准条件下（step size、Lipschitz、martingale noise、有界梯度），
快变量 $\theta_t$ 跟踪给定慢变量的局部平衡，
而慢变量 $Q_t,\pi_t,\lambda_t$ 按平均 ODE 演化。

这给出你们系统最需要的解释：

> 为什么责任度/先验/模型可靠度必须慢更新，而不是每个 batch 全速追随。

## 8.3 定理目标三：记忆污染上界

定义 replay 污染率 $\rho_t$。若 memory dynamics 满足

$$
\rho_{t+1} \le (1-\nu)\rho_t + \alpha,
$$

其中：

- $\alpha$ 是 candidate admission 的最大污染注入率；
- $\nu$ 是 eviction/refresh 带来的净化率；

则有

$$
\sup_t \rho_t \le \frac{\alpha}{\nu}.
$$

这是比“整体收敛”更现实、也更贴合当前失败点的理论对象。

---

## 9. 实验计划：缩成两阶段、四关键组，而不是同时做所有复杂情形

## 9.1 第一阶段：近似平稳 noisy stream（主实验）

目标：验证双后验 + 双时间尺度 + purified memory 是否真的切断 collapse。

设置：

- CIFAR-10/100；
- symmetric / asymmetric noise；
- quasi-stationary stream（循环流或窗口流）；
- 不引入强 concept drift。

四组关键比较：

1. hard top-k + ERM；
2. soft $Q_i$ + ERM；
3. soft $Q_i$ + SAM；
4. soft $Q_i$ + SAM + purified memory ($U_i$)。

指标：

- final / best acc；
- $Q$ calibration；
- overlap trajectory；
- replay contamination ratio；
- sharpness gap；
- memory coverage / redundancy。

## 9.2 第二阶段：piecewise-stationary drift（扩展实验）

目标：验证慢变量和 memory dynamics 是否能抑制漂移下的失稳。

设置：

- 噪声率变化；
- 类别比例逐段变化；
- 可选 mild class drift。

关注：

- $Q_i,\pi_t,\lambda_m$ 是否过冲；
- purified memory 是否优于 push-only replay；
- active committee 是否避免 collapse。

## 9.3 暂不作为第一版的内容

- open-world unseen noise；
- one-pass no-revisit 强设定；
- 大规模 WebVision/Clothing1M 级别实验；
- RL/controller 化的 replay 决策。

这些可以在主线稳定后再做第二篇或扩展版。

---

## 10. 最终贡献叙事（论文口径）

如果按本 target 推进，论文贡献应组织为三条：

### 贡献 1：双时间尺度的在线广义 EM 委员会框架

将 noisy stream 学习统一成：

- 多证据 E-step；
- posterior-weighted robust M-step；
- 慢变量（$Q_i,\pi_t,\lambda_m$）稳定更新。

### 贡献 2：训练后验与记忆后验分离的 purified memory 机制

提出 candidate buffer + purified memory 的双层记忆系统，
并用 $U_i$ 的 admission / eviction / refresh / sampling 切断 posterior–memory 闭环。

### 贡献 3：可验证的中层理论

给出：

- generalized EM 近似单调改进；
- two-timescale 稳定性解释；
- replay contamination 上界。

这比“更多模块、更高分数”更像一篇真正的研究论文。

---

## 11. 一句话总结

新的默认主线应当是：

> **Two-timescale Online Generalized EM with Purified Memory**
>
> 用多证据慢变量责任度 $Q_i$ 负责当前训练，
> 用独立的记忆后验 $U_i$ 负责长期记忆，
> 用 stop-gradient 稳定 teacher + SAM 完成广义 M-step，
> 并通过 admission / eviction / refresh 的 memory dynamics 切断 noisy posterior 的自证循环。

