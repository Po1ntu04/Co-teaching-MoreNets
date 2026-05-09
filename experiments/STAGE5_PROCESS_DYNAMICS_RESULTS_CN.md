# Stage-5 Process Dynamics 结果与下一步判断

date: 2026-05-09

## 研究动机

本阶段接受一个关键修正：不能只看最后 accuracy。若目标是解释并改进 noisy interactive teaching，必须跟踪训练过程中的变量变化，包括 $q_i$ 分布、选择集合、模型 overlap、selected clean rate、梯度范数、参数更新尺度等。

因此 Stage-5 的目标不是直接跑 benchmark，而是回答：

1. 之前 Q-gated selection 为什么只带来很小收益？
2. 三网络主要瓶颈是否来自 peer selection 高度同质？
3. 是否存在一种保守干预，能降低多模型选择冗余而不破坏 small-loss 可靠性？

## 新增实现

新增过程诊断开关：

```bash
--diag_process
--diag_process_bins
--diag_process_grad_every
--diag_process_output_dir
```

记录内容包括：

- epoch 内分段 $q$ mean/std/entropy/AUC。
- selected vs unselected 的 $q$ gap 与 loss gap。
- selected clean rate、overlap。
- gate pool fraction、gate pool clean rate、base selected 是否落入 gate pool。
- selector_changed_frac：最终 selector 相对 plain hard small-loss selector 改变了多少。
- gradient norm、update norm、update/parameter norm。

新增保守选择机制：

```bash
--selection_diversity_strength
--selection_diversity_pool_mult
--selection_diversity_start_epoch
```

该机制只在 low-loss candidate pool 内做去冗余选择。形式上近似为：

$$
\operatorname{score}_{m,i}
= \ell_{m,i} + \lambda \sigma_{\ell_m} c_i,
$$

其中 $\ell_{m,i}$ 是模型 $m$ 的 peer aggregate loss，$c_i$ 是样本 $i$ 已被其他模型选中的次数，$\sigma_{\ell_m}$ 用于把 penalty 归一到 loss 尺度。候选集仍限制在低损失池中，因此它不是从 high-loss 区域捞样本，也不是用 $q_i$ 替代 small-loss。

## 关键实验事实

### 1. Q-gate 当前几乎没有实际干预

短程机制探针：

- `q_gate_pool_mult=1.25`
- `q_gate_pool_mult=1.10`
- 3 models, CIFAR10 symmetric 40%, seed 1, 12 epochs

结果：

| setting | base_selected_in_gate_rate | selector_changed_frac | 解释 |
|---|---:|---:|---|
| gate pool 1.25 | 0.99999999 | 0.0000 | hard small-loss 选出的样本几乎全部已在 high-q pool 内 |
| gate pool 1.10 | 0.99989865 | 0.0001 | 即使更窄 pool，仍几乎不改变最终选择 |

这说明此前 Q-gated 的小幅收益不能解释为强烈的新选择机制。更准确的归因是：当前 $q_i$ 与 small-loss 排序高度重合，作为 candidate gate 时大多数 batch 不产生实际选择变化。

### 2. 直接加入高不确定样本能降 overlap，但会伤害可靠性

探针：

```bash
--explore_delta 0.05 --explore_trigger 0.98
```

12 epoch seed 1：

- last5 overlap: 约 0.962
- last5 selected clean rate: 约 0.872
- last5 acc: 约 66.96

相比保守 low-loss 策略，explore 确实降低 overlap，但 selected clean rate 下降更明显。这说明“提高多模型差异”本身不是充分条件；差异必须发生在可靠候选区域内。

### 3. Low-loss 内部 diversity-aware selection 给出稳定弱正信号

配置：

```bash
--selection_diversity_strength 0.10
--selection_diversity_pool_mult 1.25
--selection_diversity_start_epoch 1
```

3 models, CIFAR10 symmetric 40%, 30 epochs, seeds 1/2/3。

paired baseline 为原 hard small-loss `diagnostic_only`。

| seed | baseline best | div0.10 best | delta best | baseline last5 | div0.10 last5 | delta last5 |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 79.73 | 80.39 | +0.66 | 79.32 | 79.42 | +0.10 |
| 2 | 80.24 | 79.91 | -0.33 | 79.514 | 79.628 | +0.114 |
| 3 | 79.51 | 80.39 | +0.88 | 79.208 | 79.674 | +0.466 |
| mean | 79.827 | 80.230 | +0.403 | 79.347 | 79.573 | +0.227 |

过程变量均值变化：

| metric | mean delta |
|---|---:|
| last5 overlap | -0.0110 |
| last5 selected clean rate | -0.0010 |
| last5 q AUC | +0.0005 |

div0.10 的过程指标：

| seed | selector_changed_frac | last5 overlap | last5 selected clean |
|---:|---:|---:|---:|
| 1 | 0.0083 | 0.9754 | 0.9313 |
| 2 | 0.0084 | 0.9752 | 0.9316 |
| 3 | 0.0086 | 0.9753 | 0.9336 |

解释：只改变约 0.8% 的样本选择，就能稳定降低三模型 overlap，且 selected clean rate 几乎不掉。这符合“高 overlap 是三网络冗余瓶颈之一”的假设。

### 4. 过强 diversity 会破坏可靠性

12 epoch seed 1：

| setting | selector_changed_frac | last5 overlap | last5 selected clean | last acc |
|---|---:|---:|---:|---:|
| div0.10 | 0.0100 | 0.9653 | 0.8806 | 71.89 |
| div0.25 | 0.0297 | 0.9397 | 0.8745 | 70.75 |

强度 0.25 虽然更大幅降低 overlap，但 selected clean rate 和短程 acc 受损。说明 diversity 只能作为受约束的弱正则，而不是强行分散选择。

## 机制判断

### 已支持

- 当前 $q_i$ 是强 reliability signal，但作为 gate 与 small-loss 高度重合，难以形成新的选择边界。
- 三网络 hard small-loss 的一个实际瓶颈是选择高度同质，后期 overlap 约 0.986。
- 在 low-loss candidate pool 内做小幅去冗余，可以稳定降低 overlap，并在 3/3 seeds 上不伤 last5 accuracy。
- 改进不是来自更大梯度或更大更新尺度。div0.10 的 update/parameter norm 与 baseline 同量级，主要变化来自样本选择过程。

### 未充分证明

- 30 epoch 的 +0.23% last5 均值提升还不是 benchmark 级结论。
- 该机制是否能在 200 epoch、SRIT/SAM 后端、CIFAR10 asym、CIFAR100 上成立尚未验证。
- 当前 diversity penalty 是启发式形式，还没有形成完整理论贡献。

## 归因总结

Stage-4 到 Stage-5 的关键修正是：

> 问题不只是“Q 是否崩坏”，而是“可靠性、选择差异、更新尺度在训练过程中如何相互作用”。

基于过程证据，当前最有希望的方向不是继续调 Q-gate pool，也不是直接把 Data Shapley 写进权重，而是：

1. small-loss / $q_i$ 继续负责可靠性边界；
2. 多模型之间需要在可靠边界内部降低冗余；
3. utility / Shapley-like signal 后续应作为可靠候选内部的 secondary proposal，而不是 clean posterior。

## 下一步计划

短期应做：

1. 强度曲线：`0.05 / 0.10 / 0.15`，30 epoch，3 seeds。确认是否存在稳定最佳弱区间。
2. 组合 Q 与 diversity：只在 high-reliability 或 low-loss candidate pool 内做 diversity penalty，比较是否进一步保护 selected clean rate。
3. SRIT/SAM 后端复测：判断 sharpness-aware update 是否增强 diversity selection 的稳定性，而不是只看 acc。
4. 200 epoch benchmark gate：只有 30 epoch 多 seed 稳定后，才跑论文口径长实验。
5. 扩展场景：CIFAR10 asymmetric、CIFAR100 symmetric。若只在 CIFAR10 symmetric 有效，贡献不足。

更长期的算法叙事应围绕：

> Reliability controls whether a sample may be trusted; diversity controls whether multiple teachers redundantly train on the same trusted samples; utility controls which trusted non-redundant samples are most valuable for the current target.

当前 Stage-5 支持了第二部分：在可靠候选内部降低冗余是可行且有弱正收益的。

