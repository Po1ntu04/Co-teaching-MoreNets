# Q Collapse Isolation Results

date: 2026-05-09

scope: CIFAR-10 symmetric 40%, seed 1, 30 epochs, short diagnostic runs.

source files:

- `remote_results/q_isolation/q_isolation_summary.csv`
- `remote_results/q_isolation/qiso_*/*/*_training_log.json`

## 核心修正

之前“3 网络导致 Q 崩坏”的说法过强，需要修正。

本轮隔离实验显示：`q_mode=hybrid` 产生的 Q 本身不是坏信号。在 hard small-loss 训练路径下，无论 2 模型还是 3 模型，只要 Q 不作为连续训练权重进入 loss，Q 都保持很强的 clean 排序能力。

真正危险的是把同一个 $q_i$ 同时承担多个角色，尤其是：

- $q_i$ 作为 soft/robust M-step 的连续训练权重；
- $q_i$ 更新全局慢变量 `q_global`；
- $q_i$ 参与先验 $\pi_t$ 更新；
- 再由 $\pi_t$ 反过来增强 posterior logit。

这会形成确认反馈：

$$
q_i \uparrow \Rightarrow w_i \uparrow \Rightarrow \text{model confidence} \uparrow \Rightarrow p_\theta(y_i \mid x_i) \uparrow \Rightarrow q_i \uparrow
$$

当噪声样本被模型拟合后，该反馈并不区分“真正干净”和“已被记住的噪声”，所以 Q 会向全 1 饱和，选择纯度和泛化随之下降。

## 结果表

`last5_*` 是最后 5 个 epoch 的平均值。`best_acc` / `last_acc` 是 ensemble test accuracy。

| usage | models | mstep | best acc | last acc | last5 acc | last5 q mean | last5 q std | last5 q AUC | selected clean | last5 pi |
|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
| diagnostic_only | 2 | hard | 80.47 | 80.47 | 79.97 | 0.677 | 0.457 | 0.966 | 0.928 | 0.800 |
| diagnostic_only | 3 | hard | 79.73 | 79.61 | 79.32 | 0.697 | 0.441 | 0.975 | 0.932 | 0.800 |
| prior_only | 2 | hard | 80.56 | 80.23 | 80.12 | 0.668 | 0.461 | 0.967 | 0.929 | 0.698 |
| prior_only | 3 | hard | 79.73 | 79.61 | 79.32 | 0.688 | 0.445 | 0.975 | 0.932 | 0.736 |
| selection_only | 2 | hard | 80.49 | 80.07 | 80.25 | 0.667 | 0.462 | 0.969 | 0.934 | 0.800 |
| selection_only | 3 | hard | 78.75 | 78.75 | 78.31 | 0.696 | 0.441 | 0.973 | 0.930 | 0.800 |
| weight_only | 2 | soft | 78.13 | 62.59 | 64.67 | 0.995 | 0.041 | 0.875 | 0.800 | 0.800 |
| weight_only | 3 | soft | 78.66 | 77.01 | 77.71 | 0.976 | 0.067 | 0.965 | 0.921 | 0.800 |
| standard | 2 | robust | 77.39 | 68.68 | 69.27 | 1.000 | 0.000 | 0.911 | 0.846 | 0.971 |
| standard | 3 | robust | 78.27 | 77.23 | 77.76 | 0.997 | 0.009 | 0.967 | 0.923 | 0.933 |

## 分层解释

### 1. Diagnostic-only：Q 是强可靠性诊断信号

2 模型和 3 模型都保持了很高的 `last5_q_clean_auc`：

- 2 模型：0.966
- 3 模型：0.975

因此不能说 hybrid Q 本身没有 clean/reliability 信息。相反，在没有训练反馈污染时，它是强信号。

### 2. Prior-only：单独更新 $\pi_t$ 不足以造成崩坏

`prior_only` 下训练仍然是 hard small-loss，Q 只影响 $\pi_t$。结果中 $\pi_t$ 下降到 0.698 / 0.736，而不是上升到 1；Q AUC 仍保持 0.967 / 0.975。

这说明先验更新不是单独的根因。真正危险的是先验更新和连续权重训练共同形成闭环。

### 3. Selection-only：Q 可作为 hard gate，但会牺牲模型差异

`selection_only` 使用 Q top-k 作为 hard selector，2 模型 last acc 80.07，3 模型 last acc 78.75。Q AUC 仍在 0.969 / 0.973。

这里 `overlap=1.0` 是实现导致的：所有模型使用同一个 Q top-k 选择集。因此它验证了 Q 的 hard-gate 可用性，但也暴露了一个设计问题：如果多模型都吃同一批样本，多模型差异会被削弱。

### 4. Weight-only：连续权重是最明确的失控入口

2 模型 `weight_only` 是最清楚的崩坏样例：

- best acc 78.13，但 last acc 62.59；
- train acc 后期继续上升，test acc 快速下降；
- `q_mean` 升至 0.995，`q_std` 降至 0.041；
- `q_clean_auc` 从中期约 0.97 降至 last5 的 0.875；
- selected clean rate 降到 0.800。

这符合“模型开始记忆噪声后，soft Q 权重反而奖励已拟合噪声”的机制。

3 模型 `weight_only` 没有像 2 模型一样严重掉到 60% 段，但同样出现 Q 高均值低方差：`q_mean=0.976, q_std=0.067`。这说明多模型可能缓解了部分泛化崩坏，但没有消除 Q 饱和趋势。

### 5. Standard：多角色耦合会把 Q 推向全 1

`standard` 是最接近旧 hybrid/robust 路径的复现。

2 模型中：

- `q_mean=0.99996`
- `q_std=0.00027`
- `pi_t=0.971`
- last acc 68.68

这就是典型 Q collapse。

3 模型中：

- `q_mean=0.997`
- `q_std=0.009`
- `pi_t=0.933`
- last acc 77.23

3 模型仍然有 Q 饱和，但准确率伤害较轻。当前证据更支持“多模型可能改变反馈强度和泛化后果”，而不是“3 模型导致 Q 崩坏”。

## 第一性原理归因

small-loss / hard selection 的稳定性来自一个离散门控：

$$
S_t = \operatorname{TopK}_{i}(-\ell_i)
$$

它只允许一部分样本进入梯度更新，错误样本即使被局部高估，也不会连续放大权重。

soft Q training 则把后验直接变成梯度尺度：

$$
\nabla_\theta L_t = \sum_i w_i(q_i) \nabla_\theta \ell_i
$$

如果 $q_i$ 又由模型当前置信度、teacher consistency、loss rank、先验 $\pi_t$ 共同构造，那么优化过程会把 $q_i$ 变成“模型是否已经相信该样本”的指标，而不再是“样本是否干净”的稳定后验。

因此，根本问题不是缺少一个更复杂的 $q_i$，而是 $q_i$ 的语义被训练闭环污染了。

## 对主线的影响

本轮结果支持 Reliability-Utility 解耦路线，但需要更精确地表述：

- $r_i$ 可以继续从 small-loss / hybrid Q 中提取，但默认只能作为 hard gate 或 diagnostic posterior。
- $u_i$ 不能直接替代 $r_i$，也不能直接连续加权全样本。
- 若使用 utility / Shapley-like value，必须限定在可靠候选集内部做二级排序、top-tail proposal 或 replay priority。
- 对 $r_i$ 的连续权重使用必须非常保守，最好先做 rerank-only 或 bounded top-tail，而不是 $w_i=1+\alpha q_i$ 这种全局软权重。
- 多网络结构不能简单共享一个全局 $q_i$ 作为所有用途的共同变量；多模型应服务于 delayed target、disagreement、coverage 或异步 teacher，而不是共同放大同一个 posterior。

## 当前限制

- 本轮主要是 seed 1、30 epoch 短跑；跨 seed 稳定性还没有完成。
- 训练仍是静态 noisy CIFAR，不是 streaming / continual。
- `selection_only` 当前所有模型使用同一个 Q top-k，不能证明多模型差异机制有效。
- `standard` 的 3 模型准确率伤害较轻，说明还需要查明模型数、batch size、selection overlap、teacher consistency 对反馈闭环的影响。

## 下一步

优先级应调整为：

1. 先实现 `r_i` 的解耦安全用法：hard gate + EMA stability + bounded/rerank-only，不再让 Q 直接作为全样本连续权重。
2. 做跨 seed 的最小复验：`standard`、`weight_only`、`selection_only` 的 seeds 2/3，确认这不是 seed 1 特例。
3. 设计多模型差异机制：不同模型不能完全共享同一个 Q selection，应引入 staggered / delayed / disagreement-preserving selection。
4. 再把 Data Shapley / optimizer-aware utility 放到可靠集内部，只做 utility rerank 或 replay priority。

