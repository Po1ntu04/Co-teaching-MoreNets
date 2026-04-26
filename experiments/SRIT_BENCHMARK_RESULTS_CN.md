# SRIT / Co-teaching 论文参数 Benchmark 结果

日期：2026-04-26

## 来源

- 论文：`Sharpness-Aware Minimization Activates the Interactive Teaching's Understanding and Optimization`，Zotero key `6LKIQL87`。
- 远端结果：`remote_results/srit_benchmark/benchmark_seed1_summary.json`。
- 远端日志：
  - `remote_results/srit_benchmark/srit-benchmark-coteaching-seed1.log`
  - `remote_results/srit_benchmark/srit-benchmark-srit_like-seed1.log`

## 论文参照

SRIT 论文在 CIFAR-10 symmetric 40% 上报告：

| 方法 | CIFAR-10 Sym 40% |
|---|---:|
| Co-teaching | 77.38 ± 0.15 |
| SRIT | 79.83 ± 0.12 |

论文实验细节中，Co-teaching baseline 使用 Adam、`lr=0.001`、`batch_size=128`、`epochs=200`；SRIT/SRCNLCU 使用 SGD、`lr=0.1`、`momentum=0.9`、`weight_decay=0.0001`、`rho=0.05`、`batch_size=128`、`epochs=200`。

## 本次设置

共同设置：

- CIFAR-10 symmetric 40%。
- 2 models。
- `q_mode=loss`。
- `mstep_mode=hard`。
- `batch_size=128`。
- `n_epoch=200`。
- `num_gradual=10`。
- `epoch_decay_start=80`。
- `val_split=0`。
- `seed=1`。

Co-teaching baseline：

- `optimizer=adam`
- `lr=0.001`
- `sam_rho=0`

SRIT-like probe：

- `optimizer=sgd`
- `lr=0.1`
- `momentum=0.9`
- `weight_decay=0.0001`
- `sam_rho=0.05`

注意：当前 SRIT-like probe 对齐了论文的 optimizer / rho / epoch / batch 等关键训练参数，但未证明已经完整复现 SRIT 论文的全部 dual-level sharpness knowledge exchange 细节。因此它是 SRIT-like benchmark，而不是严格 SRIT 复现。

## 结果

| run | best acc | best epoch | last acc | last-10 mean acc | last-10 min | last-10 max |
|---|---:|---:|---:|---:|---:|---:|
| Co-teaching baseline seed1 | 80.38 | 18 | 78.69 | 78.807 | 78.62 | 79.09 |
| SRIT-like seed1 | 80.77 | 26 | 80.52 | 80.414 | 79.80 | 80.73 |

差值：

$$
80.414 - 78.807 = 1.607
$$

本次同 seed 下，SRIT-like probe 的 last-10 mean ensemble accuracy 比 Co-teaching baseline 高约 `+1.61` 个百分点。

## 初步解释

实验事实：

- Co-teaching baseline 的 `last-10 mean acc = 78.807%`，高于 SRIT 论文中 Co-teaching `77.38%`，但在同一量级。
- SRIT-like probe 的 `last-10 mean acc = 80.414%`，接近并略高于 SRIT 论文中 SRIT `79.83%`。
- SRIT-like 对 baseline 的提升为 `+1.61`，论文表格中 SRIT 对 Co-teaching 的提升约为 `+2.45`。

推理判断：

- 当前代码在论文参数口径下可以复现出合理的 Co-teaching benchmark 量级。
- 使用 SGD + SAM 的 SRIT-like 设置在 seed1 上确实改善了后期 ensemble accuracy，这与 SRIT 论文主张方向一致。
- 该结果比 Stage-1 中 Adam + 3-model + 80 epoch 的 SAM-like 诊断更支持“SRIT 的收益依赖具体优化器与训练口径”，而不是简单地把 `sam_rho=0.05` 接到任意训练框架上。

尚未验证：

- 只有 seed1，不能报告置信区间。
- 当前 SRIT-like 还不能声称严格复现 SRIT，因为 SRIT 的 sharpness knowledge exchange 细节需要进一步核对与实现。
- 不能据此证明 Shapley / target utility 有效；本 benchmark 只校准 SAM/SRIT 方向的优化基线。

## 下一步

建议补跑：

- Co-teaching baseline seeds 2/3。
- SRIT-like seeds 2/3。

如果 3 seeds 下 SRIT-like 仍稳定提升，则可以把 SRIT-style optimization 作为可靠后端保留；随后再讨论 `q-weighted SAM`、`sample-wise rho` 或 `sharpness-aware utility`，而不是继续在未校准的 Adam-SAM 变体上做判断。
