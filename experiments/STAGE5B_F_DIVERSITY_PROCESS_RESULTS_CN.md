# Stage-5B-F Diversity Process Results

date: 2026-05-09

## 目标

本阶段不把最终 accuracy 当作唯一证据，而是围绕训练过程回答一个更具体的问题：

> 在 noisy interactive teaching 中，多模型的主要损失是否来自可靠样本上的冗余共识？如果是，能否只在 low-loss 可靠候选池内部降低 peer selection overlap，而不破坏 selected clean rate 和更新尺度？

当前策略不是从 high-loss 区域捞样本，也不是把 $q_i$ 写成连续训练权重。它只在每个模型的 low-loss candidate pool 内加入弱 diversity penalty：

$$
\operatorname{score}_{m,i}=\ell_{m,i}+\lambda \sigma_{\ell_m} c_i,
$$

其中 $\ell_{m,i}$ 是模型 $m$ 的 peer aggregate loss，$c_i$ 是样本 $i$ 已被其他模型选中的次数，$\sigma_{\ell_m}$ 用于把 penalty 对齐到当前 batch 的 loss 尺度。

## 代码与结果位置

- implementation commit: `2e8a4af` (`exp: add ramped diversity selection`)
- main code: `main.py`
- analyzer: `scripts/analyze_process_dynamics.py`
- local result summaries:
  - `remote_results/q_isolation/stage5b_strength_summary.json`
  - `remote_results/q_isolation/stage5b_strength_plus_delayed_summary.json`
  - `remote_results/q_isolation/stage5b_strength_delayed_ramp_summary.json`
  - `remote_results/q_isolation/stage5c_sam_div005_3seed_summary.json`
  - `remote_results/q_isolation/stage5f_m5_div010_3seed_summary.json`

## Stage-5B: 3-net Strength Sweep

设置：CIFAR-10 symmetric 40%，3 models，30 epochs，3 seeds，`q_mode=hybrid`，`q_usage_mode=diagnostic_only`，`mstep_mode=hard`，`sam_rho=0`。

| setting | mean delta best | mean delta last | mean delta last5 | mean delta selected clean | mean delta overlap | selector changed |
|---|---:|---:|---:|---:|---:|---:|
| div0.05 | +0.367 | +0.263 | +0.239 | -0.0004 | -0.0053 | 0.0039 |
| div0.10 | +0.403 | +0.483 | +0.227 | -0.0010 | -0.0110 | 0.0084 |
| div0.15 | +0.387 | +0.603 | +0.275 | -0.0023 | -0.0183 | 0.0138 |

解释：

- 弱 diversity 已经足以稳定改变选择过程。
- `0.15` 的 overlap 降幅最大，last/last5 也更高，但 selected clean 代价更大。
- 这不是梯度尺度或更新尺度造成的收益，主要变化来自样本选择路径。

## Stage-5C: SAM/SRIT-style Backend Interaction

设置：3 models，SGD + SAM (`sam_rho=0.05`)，只测 `div0.05`。

| metric | mean delta |
|---|---:|
| best | +0.047 |
| last | +0.337 |
| last5 | +0.001 |
| q AUC | +0.0003 |
| selected clean | -0.0001 |
| overlap | -0.0057 |

解释：

- weak diversity 与 SAM 后端兼容，能稳定降低 overlap。
- 但在 SAM 后端下，last5 几乎不提升，只有 seed1 明显正向。
- 因此当前主线不应转向 SAM+diversity；SAM 可保留为后端/稳定器，但不是本阶段收益的主要来源。

## Stage-5D/E: Timing and Ramp

`div0.15` 的时间调度对轨迹有影响。

| setting | mean delta best | mean delta last | mean delta last5 | mean delta selected clean | mean delta overlap | selector changed |
|---|---:|---:|---:|---:|---:|---:|
| div0.15 start epoch 1 | +0.387 | +0.603 | +0.275 | -0.0023 | -0.0183 | 0.0138 |
| div0.15 start epoch 10 | +0.227 | +0.480 | +0.335 | -0.0026 | -0.0182 | 0.0139 |
| div0.15 ramp10 | +0.283 | +0.397 | +0.171 | -0.0026 | -0.0183 | 0.0139 |

解释：

- 延迟到 epoch 10 启动在 mean last5 上最好，但 seed1 明显变差，方差更高。
- ramp10 三 seed 都正向，但幅度偏弱。
- 这说明 timing 确实影响训练轨迹，但硬调度不是当前最关键突破口。当前更值得推进的是模型数和冗余结构。

## Stage-5F: 5-net Diagnostic

设置：CIFAR-10 symmetric 40%，5 models，30 epochs，3 seeds，batch size 192，`q_usage_mode=diagnostic_only`，`mstep_mode=hard`，`sam_rho=0`。

显存与运行：

- 5-net batch192 在 4090/3090 上可跑，显存约 16.3-16.6GB。
- GPU util 约 89-95%。
- 没有使用 3080Ti。

### 结果

| seed | baseline last5 | div0.10 last5 | delta last5 | baseline overlap | div0.10 overlap | delta overlap | selected clean delta |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 78.226 | 78.486 | +0.260 | 0.9968 | 0.9811 | -0.0157 | -0.0025 |
| 2 | 78.120 | 78.690 | +0.570 | 0.9970 | 0.9814 | -0.0156 | -0.0026 |
| 3 | 77.554 | 78.246 | +0.692 | 0.9971 | 0.9817 | -0.0154 | -0.0006 |
| mean | 77.967 | 78.474 | +0.507 | 0.9970 | 0.9814 | -0.0156 | -0.0019 |

Additional mean deltas:

- best acc: `+0.493`
- last acc: `+0.723`
- q AUC: `+0.00137`
- grad norm: `-0.0234`
- update/param norm: `+0.0000007`
- selector changed fraction: `0.0190`

### 解释

5-net 结果比 3-net 更强，支持以下判断：

- 多网络设置下，plain hard small-loss 的 overlap 非常高，5-net baseline 后期 overlap 约 `0.997`。
- 只改变约 1.9% 的选择，就能把 overlap 降到约 `0.981`。
- 这个改变没有显著放大梯度或更新尺度，说明收益不是来自更大 optimization step。
- selected clean rate 下降约 0.19 个百分点，属于可见但较小的可靠性代价。
- 5-net 三个 seed 全部正向，且 mean last5 提升约 `+0.51`，强于 3-net 的 `+0.23` 左右。

这说明当前最有价值的阶段性发现不是“Q gating 提升 accuracy”，也不是“Shapley utility 已经可用”，而是：

> 在可靠 small-loss 边界内，多模型 teacher 的样本选择冗余是可测量瓶颈；弱 diversity penalty 能在不破坏可靠性的前提下降低冗余，并且该效应在 5-net 设置下更明显。

## 阶段性失败与修正

已否定或降级的方向：

- Q-gate pool 宽度不是主杠杆，因为 small-loss selected samples 几乎已经在 high-Q pool 内。
- high-uncertainty exploration 能降 overlap，但 selected clean rate 掉得太多。
- SAM 后端兼容 weak diversity，但不是当前收益来源。
- timing/ramp 有影响，但不是比模型数扩展更强的主杠杆。

仍然成立的主线：

- reliability 决定 admissible candidate set。
- diversity 在可靠候选集内降低多 teacher 冗余。
- utility / Data Shapley 后续只能在 trusted, non-redundant candidates 内做二级 proposal，不能替代 reliability。

## 下一步建议

优先级 1：确认 5-net 发现的稳健性。

- 先跑 5-net `div0.05` 或 `div0.15` 的小强度对照，确认 5-net 最佳强度是否仍是 `0.10`。
- 再决定是否做 80/200 epoch benchmark。当前 30 epoch 结果是强诊断证据，但还不是论文口径 benchmark。

优先级 2：跨场景验证。

- CIFAR-10 asymmetric 40%。
- CIFAR-100 symmetric 40%。
- 如果只在 CIFAR-10 symmetric 成立，贡献不足；如果在更难噪声或更多类别下仍保持不掉 acc 或提升，则更接近可讲的算法贡献。

优先级 3：理论化。

把当前经验机制整理成 reliability-diversity-utility decomposition：

$$
\text{admit}(i,t) \leftarrow r_i(t), \quad
\text{de-redundant select}(i,t) \leftarrow d_i(t), \quad
\text{prioritize}(i,t) \leftarrow u_i(t).
$$

当前 Stage-5 主要验证了 $d_i(t)$：在 reliability gate 内降低冗余可以改善多模型训练路径。

## 当前不能声称

- 不能声称已经超过完整 Co-teaching/SRIT benchmark。
- 不能声称 Data Shapley 已经带来收益。
- 不能声称这是 streaming / continual protocol。
- 不能把 30 epoch 诊断结果写成最终 SOTA 指标。

当前可以声称的是：

- 发现并量化了 3-net/5-net hard small-loss 的 peer-selection redundancy。
- 验证了 reliability-preserving diversity penalty 在 3-net 和 5-net 上均有正向诊断信号。
- 5-net 上三 seed 全部正向，说明该机制可能是多网络扩展的有效切入点。
