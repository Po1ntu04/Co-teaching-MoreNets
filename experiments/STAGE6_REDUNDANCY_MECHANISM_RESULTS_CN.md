# Stage-6 Redundancy Mechanism Results

date: 2026-05-10

## 目标

本阶段不是继续报告一个表层 accuracy 增益，而是检验 Stage-5 中的现象是否有可解释机理：

> 多网络 Co-teaching 的增益瓶颈是否来自高度相关 teacher 对同一批 easy/clean 样本的重复曝光，而不是来自模型数量本身的独立判断？

为此新增了非侵入式过程诊断，不改变训练路径，只记录：

- base selector 与 final selector 的 vote histogram。
- `vote_entropy_norm`、`selected_all_frac`、`selected_partial_frac`。
- 按 vote count 分组的 clean rate。
- base/final pairwise selection overlap 与 Jaccard。
- model loss-rank Spearman proxy。
- gradient/update cosine。

相关提交：

- `56c01f0`：新增 redundancy mechanism diagnostics。
- `d0cf395`：新增 base-vote 硬约束。
- `feb1c1e`：新增 zero-vote soft penalty。

## 机理假设

设第 $m$ 个模型在 batch 内选择集合为 $S_m$，每个模型选择 $k$ 个样本，总曝光量为 $Mk$，但有效覆盖为 $|\cup_m S_m|$。如果模型选择高度相关，则 $Mk$ 中大量曝光集中在同一批样本上。

原始多模型 small-loss 的理想作用有两种可能：

1. **可靠性平均机制**：更多模型提供更可靠的 clean 判断，主要应提升 selected clean rate 或 $q$ AUC。
2. **冗余曝光机制**：更多模型其实选择高度同质，主要瓶颈是 all-vote 样本过多、partial-vote 样本过少。

当前 diversity penalty：

$$
\operatorname{score}_{m,i}=\ell_{m,i}+\lambda\sigma_{\ell_m}c_i
$$

其中 $c_i$ 是样本已被前序 peer 选中的次数。它近似在 low-loss 候选池内惩罚重复曝光，即最小化一个带 coverage penalty 的选择目标。若收益来自冗余曝光机制，应观察到：

- `pair_overlap` 与 `selected_all_frac` 降低；
- `vote_entropy_norm` 与 partial-vote exposure 上升；
- selected clean rate 只小幅变化；
- loss-rank correlation 不会根本变化；
- 更新尺度不是主要解释变量。

## 2x2 机制探针

设置：CIFAR-10 symmetric 40%，15 epochs，100 iter/epoch，batch 192，seed1，3-net/5-net × base/div0.10。

| setting | last | last5 | overlap | selected clean | vote entropy | pair overlap | selected-any clean |
|---|---:|---:|---:|---:|---:|---:|---:|
| 3-net base | 70.10 | 68.798 | 0.9799 | 0.8974 | 0.6191 | 0.9733 | 0.8838 |
| 3-net div0.10 | 71.07 | 69.226 | 0.9653 | 0.8977 | 0.6794 | 0.9560 | 0.8757 |
| 5-net base | 71.44 | 68.082 | 0.9943 | 0.9004 | 0.4646 | 0.9857 | 0.8895 |
| 5-net div0.10 | 72.60 | 68.956 | 0.9761 | 0.8979 | 0.5986 | 0.9533 | 0.8659 |

关键观察：

- 5-net base 的 pair overlap 高于 3-net，且 vote entropy 更低，说明更多模型没有自然带来更分散的样本选择。
- 3-net 与 5-net 的 loss-rank correlation 都约为 $0.957$，说明模型判断高度相关；5-net 的差异主要是组合层面的 vote concentration 更强。
- `div0.10` 在 5-net 上造成更大的 vote entropy 提升和 pair overlap 下降，也带来更大的 last/last5 改善。
- selected clean rate 只小幅下降，而 all-vote clean rate 上升，说明策略主要改变曝光分布，不是简单放大噪声。

## 可靠性约束的负结果

### Q gate

5-net `div0.10` 加 `selection_diversity_q_gate_mult`：

| setting | last | last5 | overlap | selected clean | gate pool clean | gate pool frac |
|---|---:|---:|---:|---:|---:|---:|
| qgate1.10 | 72.11 | 69.038 | 0.9764 | 0.8980 | 0.8575 | 0.6667 |
| qgate1.25 | 71.62 | 68.774 | 0.9767 | 0.8975 | 0.7849 | 0.7552 |
| qgate1.50 | 71.62 | 68.774 | 0.9767 | 0.8975 | 0.6617 | 0.9062 |

解释：

- `base_selected_in_gate_rate≈1`，说明原始 small-loss 样本几乎已经全在 high-Q pool 内。
- 放宽 Q gate 只是在纳入更脏的候选池，并没有构成有效可靠性约束。
- 这支持此前判断：当前 hybrid Q 不适合作为主要控制变量。

### base-vote hard gate

5-net `div0.10` 加候选 base vote 下限：

| setting | last | last5 | overlap | selected clean | vote entropy | selected-any clean | selector changed |
|---|---:|---:|---:|---:|---:|---:|---:|
| minvote1 | 71.83 | 68.626 | 0.9870 | 0.9004 | 0.4864 | 0.8969 | 0.0081 |
| minvote2 | 71.35 | 68.684 | 0.9942 | 0.9004 | 0.4320 | 0.9012 | 0.0062 |

解释：

- hard gate 保住了 selected clean rate，但明显削弱 vote entropy 与 selector change。
- 收益低于无约束 `div0.10`，说明“只追求更干净候选”不是当前收益的完整机制。

### zero-vote soft penalty

5-net `div0.10` 加 base-vote=0 的软惩罚：

| setting | last | last5 | overlap | selected clean | vote entropy | selected-any clean | selector changed |
|---|---:|---:|---:|---:|---:|---:|---:|
| zv0.03 | 72.27 | 69.130 | 0.9787 | 0.8991 | 0.5818 | 0.8778 | 0.0181 |
| zv0.06 | 72.29 | 68.636 | 0.9805 | 0.8990 | 0.5737 | 0.8799 | 0.0163 |
| zv0.10 | 72.12 | 68.736 | 0.9824 | 0.8992 | 0.5615 | 0.8810 | 0.0154 |

seed1 中 `zv0.03` 看似最好，但 paired repeat 反驳了其稳定性。

## zv0.03 paired repeat

设置：CIFAR-10 symmetric 40%，5-net，15 epochs，seeds 1/2/3。

| variant | mean last | mean last5 | mean overlap | selected clean | vote entropy | pair overlap |
|---|---:|---:|---:|---:|---:|---:|
| base | 71.527 | 68.740 | 0.9945 | 0.9017 | 0.4604 | 0.9868 |
| div0.10 | 72.053 | 69.232 | 0.9760 | 0.8992 | 0.6010 | 0.9556 |
| div0.10 + zv0.03 | 71.730 | 69.079 | 0.9786 | 0.8998 | 0.5863 | 0.9597 |

Paired last5 delta：

| seed | div0.10 vs base | zv0.03 vs base |
|---:|---:|---:|
| 1 | +0.874 | +1.048 |
| 2 | +0.194 | -0.322 |
| 3 | +0.408 | +0.292 |
| mean | +0.492 | +0.339 |

结论：

- `zv0.03` 不是稳定改进；seed2 明显负向。
- 当前最稳的短程算法仍是无约束 `div0.10`。
- 这表明适度引入 base-vote=0 的边界样本可能是收益来源之一，而不是单纯噪声污染。

## 跨场景探针

15 epochs，seed1，5-net。

| dataset/noise | setting | last | last5 | best | overlap | selected clean | q AUC | vote entropy |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| CIFAR-10 pairflip 40% | base | 60.99 | 59.344 | 64.60 | 0.9886 | 0.7385 | 0.7892 | 0.4929 |
| CIFAR-10 pairflip 40% | div0.10 | 61.06 | 59.982 | 64.91 | 0.9649 | 0.7413 | 0.7965 | 0.6485 |
| CIFAR-100 sym 40% | base | 32.24 | 28.802 | 32.24 | 0.9926 | 0.8598 | 0.9157 | 0.4817 |
| CIFAR-100 sym 40% | div0.05 | 32.41 | 28.838 | 32.41 | 0.9815 | 0.8584 | 0.9155 | 0.5530 |
| CIFAR-100 sym 40% | div0.10 | 32.33 | 28.444 | 32.33 | 0.9709 | 0.8564 | 0.9155 | 0.6302 |

解释：

- CIFAR-10 pairflip 也为正，说明该机制不只依赖 symmetric noise。
- CIFAR-100 上 `div0.10` 过强，`div0.05` 恢复到轻微正向。100 类任务下每类样本更少，过强去冗余更容易破坏类内可靠性或局部覆盖。
- 这提示下一步应做 adaptive/class-aware diversity，而不是固定强度。

## 阶段性机理结论

当前较可靠的说法：

1. 多网络 small-loss 不是天然带来独立判断。当前 CNN/CIFAR 设置下，peer loss ranking 高度相关，更多网络更容易形成 all-vote exposure concentration。
2. `div0.10` 的收益主要来自选择分布变化，而不是更大的梯度尺度或更强优化步。
3. 去冗余不是“越保守越好”。过强可靠性约束会削弱 coverage 改善；适度边界扩展可能是收益来源。
4. 机制在 CIFAR-10 symmetric 与 pairflip 上都有短程正信号；在 CIFAR-100 上需要更弱或 class-aware 的强度。

当前不能声称：

- 不能声称已经完成长期 benchmark。
- 不能声称 zero-vote penalty 是改进。
- 不能声称 Q gate 有效。
- 不能声称 Data Shapley 已经带来收益。

## 下一步

优先实现和验证：

1. **Adaptive redundancy controller**：令 $\lambda_t$ 根据当前 `pair_overlap`、`vote_entropy_norm` 或 `selected_all_frac` 自动调整，而不是固定 `0.10`。
2. **Class-aware diversity**：尤其针对 CIFAR-100，避免去冗余集中破坏少数类或小类局部可靠性。
3. **Longer paired benchmark**：对 CIFAR-10 symmetric、CIFAR-10 pairflip、CIFAR-100 symmetric 分别做 30/80 epoch paired seeds。
4. **Utility/Shapley 接入位置**：当前证据支持先解决 exposure redundancy，再把 utility 作为 coverage-aware candidate ranking 的附加信号，而不是替代 reliability posterior。

