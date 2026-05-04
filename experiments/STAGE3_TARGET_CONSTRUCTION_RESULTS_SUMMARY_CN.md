# Stage-3 Target Construction Diagnostic 结果总结

## 1. 本阶段问题

本阶段不再继续调 `sam_gap`，也不把 Data Shapley / optimizer-aware utility 方向直接判死，而是检查一个更基础的问题：

> 在没有 clean validation target 的真实约束下，能否构造一个足够可信的 proxy target gradient，使 target-aligned utility 具备算法化价值？

一阶近似下，候选样本 $i$ 的目标效用可写成：

$$
\Delta L_T(i) \approx -\eta\, g_T^\top P_t(g_i)
$$

其中 $g_T$ 是 target source 诱导的目标梯度，$g_i$ 是候选样本梯度，$P_t$ 表示当前优化器或近似预条件器诱导的实际更新变换。若 $g_T$ 构造错误，即使 Shapley / in-run value 的数学形式正确，算法化权重也会变成噪声。

因此本阶段的核心不是提高 accuracy，而是回答：哪些 target source 与 clean-oracle target utility 有稳定相关性。

## 2. 已实现内容

新增了非破坏式诊断模式：

- `--diag_target_construction`
- `--diag_target_sources clean_val,noisy_val,peer_consensus,ema_teacher,purified_buffer`
- `--diag_target_output_dir`

新增 target source：

| source | 角色 | 是否可作为最终方法 |
|---|---|---|
| `clean_val` | controlled upper bound / sanity check | 否 |
| `noisy_val` | noisy held-out target，下界或现实 proxy 参考 | 谨慎 |
| `peer_consensus` | peer 高置信一致样本 | 可考虑，但本轮较弱 |
| `ema_teacher` | 慢 teacher soft target | 可考虑，但本轮不稳定 |
| `purified_buffer` | 高稳定可靠样本记忆 | 最接近可用方法 |

新增分析脚本：

- `scripts/analyze_stage3_target_construction.py`

新增短跑算法化模式：

- `--utility_mode target_align`
- `--target_align_source purified_buffer`
- `--utility_strength`

当前 `target_align` 只在 hard small-loss 已选中的 reliable set 内做轻量加权，不从 high-loss 区域捞样本。这一点是必要约束，因为已有证据显示 utility oracle 本身不是 clean detector。

## 3. 运行设置

远端路径：

```bash
/data1/yuzhixiang/work/Co-teaching-MoreNets
```

共同设置：

| 项 | 值 |
|---|---|
| dataset | CIFAR-10 |
| noise | symmetric 40% |
| models | 2 |
| backend | SRIT-like: SGD + SAM |
| `sam_rho` | 0.05 |
| `batch_size` | 512 |
| `num_iter_per_epoch` | 100，实际每 epoch 约 87 iter |
| `n_epoch` | 31，实际记录 epoch 1 到 30 |
| `val_split` | 0.1 |
| GPU | `CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=4`，4090 |

注意：`replay_size=2000` 仅用于形成 `purified_buffer` target source，`replay_ratio=0`，因此没有启用 replay 训练。

## 4. E3.1 Target Source Diagnostic 结果

三 seed 汇总文件：

```text
remote_results/stage3_target_construction/e31_all_seeds_summary.json
```

核心指标：

| source | source clean rate | Spearman(proxy, oracle) | Pearson(proxy, oracle) | top25 oracle ratio | pass epochs | 判断 |
|---|---:|---:|---:|---:|---:|---|
| `clean_val` | 0.5710 | 0.9719 | 1.0000 | 9.98 | 18/18 | sanity 通过 |
| `noisy_val` | 0.5710 | 0.4956 | 0.6146 | 6.67 | 17/18 | 有强 proxy 信号，但方法学需谨慎 |
| `purified_buffer` | 0.9925 | 0.1766 | 0.1789 | 2.39 | 11/18 | 弱但非零，最值得算法化探针 |
| `ema_teacher` | 0.5729 | 0.1283 | 0.2178 | 3.14 | 10/18 | 弱且不稳定 |
| `peer_consensus` | 0.9990 | 0.0973 | 0.1513 | 2.76 | 8/18 | 高纯度但 utility 排序弱 |

关键观察：

1. `clean_val` 与 oracle 几乎一致，说明诊断实现和 oracle 定义本身是可信的。
2. `noisy_val` 有明显相关性，但它不是最终方法的理想来源。它可能因为 60% clean 主体仍占优而给出任务方向，也可能引入 noisy validation leakage 或 confirmation bias。
3. `purified_buffer` 是最接近最终方法的 source。它的相关性弱，但跨 seed 为正，且 source clean rate 高，说明它不是完全无信号。
4. `peer_consensus` 的 source clean rate 接近 1，但 Spearman 只有约 0.097。这证明“干净 / 高置信”不等于“能排序 target utility”。
5. 本轮 `auc_loss_clean` 在诊断候选集上约 0.842，仍是强 clean/reliability 信号；但 loss 与 oracle utility 的 Spearman 接近 0。这再次说明 small-loss 和 target utility 是两个不同问题。

更尖锐地说：small-loss 回答“这个样本是否像 clean / 是否容易被当前模型拟合”；target utility 回答“这个样本的一步更新是否推动目标效用变好”。两者不能互相替代。

## 5. E3.3 Algorithmic Short Run 结果

训练日志汇总：

```text
remote_results/stage3_target_construction/stage3_training_summary.json
```

seed1 对照：

| run | utility | strength | best | best epoch | last | last5 mean | last10 mean |
|---|---|---:|---:|---:|---:|---:|---:|
| E3.1 baseline seed1 | none | - | 79.41 | 27 | 78.97 | 79.04 | 79.00 |
| target-align seed1 | purified buffer | 0.25 | 79.80 | 28 | 79.31 | 79.43 | 79.26 |
| target-align seed1 | purified buffer | 0.50 | 79.21 | 26 | 78.46 | 78.89 | 78.73 |

相对 seed1 baseline：

| variant | best delta | last delta | last5 delta | last10 delta |
|---|---:|---:|---:|---:|
| strength 0.25 | +0.39 | +0.34 | +0.39 | +0.26 |
| strength 0.50 | -0.20 | -0.51 | -0.15 | -0.27 |

解释：

- `strength=0.25` 没有破坏 baseline，并在单 seed / 30 epoch 下有轻微正向信号。
- `strength=0.50` 已经明显偏弱，说明 utility weighting 不能重权化。
- 这不是足够强的论文结论，因为只有 seed1 且是短跑；但它足以说明“target-align 完全没戏”的结论不成立。

从一阶更新看，若权重写成 $w_i = 1 + \alpha \tilde u_i$，则实际更新方向是：

$$
\Delta \theta \propto -\sum_i w_i g_i
$$

当 proxy utility $\tilde u_i$ 只有弱相关时，$\alpha$ 过大会把 target construction 的误差放大。`0.25` 尚可、`0.50` 伤害，和这个判断一致。

## 6. 阶段性失败归因

本阶段不能归因为 GPU、batch size 或基础训练管线问题：

- 4090 + `batch_size=512` 可正常完成 smoke、三 seed 诊断和两组短跑。
- SRIT-like hard small-loss baseline 稳定在约 79% 到 80% 的 30 epoch 区间。
- 诊断文件正常写入，clean target sanity check 通过。

真正的问题是 target construction：

1. 如果有 clean target，last-layer target alignment 可以很好复现 oracle。
2. 如果只有 peer consensus，高纯度样本往往是 easy / saturated 样本，目标梯度小且方向窄，不能提供好的 utility 排序。
3. `purified_buffer` 更接近最终方法，但当前 source 仍太弱，可能过于偏向 easy reliable samples，缺少 class-balanced、moderately hard、coverage-aware 的目标构造。
4. `noisy_val` 的强信号提示“非 clean target 并非不可能”，但不能直接拿它当最终贡献，因为它可能把任务梯度和噪声验证集耦合在一起。

因此阶段结论不是“Shapley 不值得做”，而是：

> Shapley / optimizer-aware utility 的下一步价值取决于能否构造稳定、无 clean label 依赖、不过度偏 easy samples 的 target gradient。

## 7. 当前不应做什么

暂不建议直接做 200 epoch benchmark，原因：

- `target-align strength=0.25` 只有 seed1 短跑正信号。
- `strength=0.50` 已经伤害 baseline。
- `purified_buffer` 的 proxy-oracle Spearman 只有约 0.177，说明信号弱。
- 当前方法还没有证明跨 seed 稳定，也没有证明 target source 的构造已足够合理。

也不建议继续调 `sam_gap`：

- Stage-2 oracle 已显示 `sam_gap` 与 one-step oracle 稳定负相关，约 -0.11。
- 当前 Stage-3 的有效信号来自 target construction，而不是 sharpness gap 本身。

## 8. 下一阶段规划

下一阶段建议命名为 Stage-3.5：Conservative Target-Align Validation。

目标不是冲 benchmark，而是验证弱正信号是否稳健，并改进 target source。

### 8.1 保守强度复核

先跑：

- `target_align_source=purified_buffer`
- `utility_strength=0.25`
- seeds 2/3
- 30 epochs

接受条件：

- 至少 2/3 seeds 不低于 paired SRIT-like baseline 的 last5 mean。
- 三 seed mean delta 不低于 0，最好达到 +0.2 左右。
- 若 seed2/3 失败，则不扩大 benchmark。

### 8.2 更轻强度或 rerank-only

若 `0.25` 不稳定，补：

- `utility_strength=0.10`
- 或只做 reliable set 内 top-rerank，不连续加权

原因：当前 proxy 信号弱，权重强度应与相关性匹配，而不是把弱排序当强监督。

### 8.3 改进 purified target source

优先诊断以下 source，而不是盲目训练：

1. class-balanced purified buffer target。
2. moderately-hard reliable target：从 high reliability 中排除过 easy 的样本。
3. coverage-aware purified target：避免只由少数类别或重复模式主导目标梯度。
4. delayed EMA purified target：用更慢、更滞后的 teacher 构造 target，减少即时 confirmation bias。

判据仍然是：

- proxy-oracle Spearman 是否提升。
- top25 oracle lift 是否稳定。
- source clean rate 是否不明显低于 small-loss selected clean rate。

### 8.4 何时进入 benchmark

只有满足以下条件，才进入 80/200 epoch benchmark：

- `purified_buffer` 或其改进版在多个 seed 上有稳定 proxy-oracle 信号。
- target-align 短跑不伤 baseline，并有至少轻微正向均值。
- strong weighting 不再作为默认，默认从 `0.10` 或 rerank-only 开始。

## 9. 当前阶段结论

1. Data Shapley / target utility 方向没有被否定。
2. 当前证据否定的是“随手构造 target 后直接重权化”的路线。
3. `noisy_val` 说明无 clean target 的 proxy 可能存在，但不能直接作为最终方法。
4. `purified_buffer` 是最值得继续的方向，但需要更保守的算法化和更好的 source design。
5. 下一步应做 Stage-3.5，而不是直接做 benchmark 或转向 streaming/replay。

