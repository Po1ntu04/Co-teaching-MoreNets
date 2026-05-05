# Stage-3.5 Conservative Target-Align Validation 结果总结

更新时间：2026-05-05

## 1. 阶段目标

本阶段不是追求最高 accuracy，也不是做 200 epoch benchmark，而是验证一个更基础的问题：

> 在 hard small-loss / SRIT-like 后端稳定的前提下，`purified_buffer` 构造的 target-align utility 是否能作为保守二级信号跨 seed 稳定工作。

判断标准来自阶段计划：

- `target_align weighted 0.25` 至少 2/3 seeds 的 last5 mean 不低于 paired baseline。
- 若 `0.25` 不稳，则降低到 `0.10` 或改 `rerank_only`。
- 若算法化不稳，则转向 target source diagnostic，而不是继续扩大训练轮数。
- 只有短跑和 proxy-ranking 同时过关，才进入 80/200 epoch benchmark。

## 2. 工程改动

本阶段新增或修正了以下能力：

- `--target_align_mode weighted|rerank_only`
- `--target_align_rerank_frac`
- `purified_buffer_balanced`
- `purified_buffer_moderate`
- `purified_buffer_coverage`
- `ema_purified`
- source-level diagnostic meta：
  - `source_label_hist`
  - `source_effective_size`
  - `source_loss_mean/std`
  - `source_confidence_mean`

重要修正：

1. 初版 `rerank_only` 实际是在原 selected set 内丢弃 bottom 25%，导致训练样本量下降。这不是纯 rerank，而是提高了实际 forget rate。
2. 修正后 `target_rerank_pool` 保持原 small-loss 选择数量 $k$，从稍大的 small-loss pool 中按 utility 选回 $k$ 个样本。
3. 初版 `diag_variants` 给 purified variants 传入 `diag_candidates * 4`，导致 balanced/coverage/moderate 基本没有真正改变 source。已修正为 `diag_candidates`，并用 `diag_variants_selective` 单独保存结果。

本地工程检查：

```bash
D:\anaconda3\envs\py3.6\python.exe -m py_compile main.py model.py scripts\analyze_stage3_target_construction.py
D:\anaconda3\envs\py3.6\python.exe main.py --help
```

均通过。

远端设置：

- repo：`/data1/yuzhixiang/work/Co-teaching-MoreNets`
- GPU：自动选择空闲 4090，本轮实际为 `nvidia-smi` device 4
- 固定命令前缀：`CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=4`
- batch：4090 上使用 `batch_size=512`

## 3. 算法化短跑结果

paired baseline 为 `target_construction_e31_seed{1,2,3}`。

### 3.1 weighted 0.25

| seed | best delta | last delta | last5 delta | last10 delta |
|---:|---:|---:|---:|---:|
| 1 | +0.39 | +0.34 | +0.388 | +0.262 |
| 2 | -0.13 | +0.07 | -0.068 | -0.154 |
| 3 | -0.08 | -0.34 | -0.338 | -0.407 |

结论：`0.25` 没有跨 seed 稳定。last5 只有 1/3 seeds 非负，三 seed mean 约 `-0.006`，不满足继续 benchmark 的条件。

### 3.2 weighted 0.10

| seed | best delta | last delta | last5 delta | last10 delta |
|---:|---:|---:|---:|---:|
| 1 | +0.26 | +0.48 | +0.326 | +0.378 |
| 2 | +0.02 | -0.03 | -0.502 | -0.207 |
| 3 | -0.12 | -0.06 | -0.162 | -0.210 |

结论：降强度后 best acc 略稳，但 late metrics 仍不稳。last5 只有 1/3 seeds 非负，三 seed mean 约 `-0.113`。这说明问题不是简单把 $\alpha$ 从 0.25 降到 0.10 就能解决。

### 3.3 rerank-only

初版 `target_rerank` 因实现问题丢弃了约 25% selected samples：

| variant | seed | best delta | last delta | last5 delta | last10 delta |
|---|---:|---:|---:|---:|---:|
| old rerank | 1 | -6.50 | -6.66 | -6.754 | -7.057 |

该结果不作为方法失败证据，只作为实现错误证据。

修正后的 `target_rerank_pool` 保持 selected count，但从扩展 pool 中按 utility 取回 $k$：

| variant | seed | best delta | last delta | last5 delta | last10 delta |
|---|---:|---:|---:|---:|---:|
| rerank_pool | 1 | -4.15 | -7.12 | -6.454 | -5.614 |

结论：即使保持样本数量，`rerank_frac=0.75` 仍然过激。它等价于让弱 utility 在一个过宽候选池中替代 small-loss 决策，破坏了 hard small-loss 的高精度可靠性。因此没有继续跑 seeds 2/3。

## 4. Target Source Selective Diagnostic

修正后运行 `diag_variants_selective` seeds 1/2/3。该实验不改变训练，只比较不同 target source 与 oracle utility 的关系。

三 seed 汇总：

| source | Spearman mean | Spearman min | Spearman >= 0.1766 | top25 ratio mean | top25 ratio min | top25 >= 2.39 | effective size mean | source clean |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `clean_val` | 0.9749 | 0.9675 | 3/3 | 24.705 | 8.205 | 3/3 | 9.78 | 0.571 |
| `noisy_val` | 0.5062 | 0.4173 | 3/3 | 14.465 | 5.274 | 3/3 | 9.86 | 0.571 |
| `purified_buffer_coverage` | 0.1893 | 0.0274 | 1/3 | 5.219 | 2.715 | 3/3 | 6.42 | 0.996 |
| `ema_teacher` | 0.1817 | 0.1343 | 2/3 | 3.129 | 1.948 | 2/3 | 3.94 | 0.559 |
| `purified_buffer_moderate` | 0.1753 | 0.0347 | 1/3 | 4.206 | 2.666 | 3/3 | 5.56 | 0.996 |
| `purified_buffer_balanced` | 0.1745 | 0.0806 | 1/3 | 2.838 | 2.368 | 2/3 | 6.42 | 0.994 |
| `purified_buffer` | 0.1577 | 0.0314 | 1/3 | 3.581 | 2.463 | 3/3 | 5.56 | 0.994 |
| `peer_consensus` | 0.1005 | -0.1351 | 1/3 | 3.351 | 2.474 | 3/3 | 6.27 | 0.998 |
| `ema_purified` | 0.0624 | -0.0307 | 1/3 | 0.689 | -2.495 | 0/3 | 5.56 | 0.994 |

读法：

- `clean_val` sanity 继续通过，说明 last-layer proxy / oracle 诊断本身没有坏。
- `noisy_val` 仍然强，但它不是最终方法允许依赖的 target source。
- `purified_buffer_coverage` 和 `purified_buffer_moderate` 的 top25 ratio 稳定高于普通 `purified_buffer`，说明它们在“挑出少量有用样本”上有信号。
- 但 Spearman min 很低，说明整体排序不稳定，不能把它们作为连续权重或宽候选池替代 small-loss。
- `ema_purified` 在当前实现下无效，可能因为 teacher soft target 太平滑或过度自确认。

## 5. 数学归因

当前证据支持下面的失败归因。

small-loss 的角色是可靠性筛选。它回答：

> 这个样本是否像 clean / 是否容易被当前模型拟合。

target utility 的角色是目标效用排序。它回答：

> 这个样本的一步更新是否推动目标效用变好。

二者不等价。本阶段失败的不是这个区分，而是当前 proxy target 还不够稳定。

若写成：

$$
w_i = 1 + \alpha \tilde u_i,
$$

则更新方向变为：

$$
\Delta \theta \propto -\sum_i w_i g_i.
$$

当 $\tilde u_i = u_i + \epsilon_i$ 且 $\epsilon_i$ 排序噪声较大时，连续加权会把许多样本上的 proxy error 全部累积进更新。`0.25` 与 `0.10` 的 late metrics 不稳，正是这个问题。

`top25 ratio` 稳定但 Spearman 不稳，含义更具体：

- proxy 的全局单调排序弱；
- 但 top tail 可能包含更多 oracle-positive 样本；
- 因此更合理的算法化形式不是连续重权，也不是大范围 rerank，而是极保守的 top-tail gate / audit / buffer proposal。

## 6. 阶段结论

1. 当前不应做 200 epoch benchmark。
2. `purified_buffer + target_align weighted` 的 seed1 正信号没有跨 seed 复现。
3. `rerank_only` 不能用宽候选池替代 small-loss；弱 utility 会快速破坏可靠性选择。
4. target source 改进方向没有失败，但仍然只处于诊断级别：
   - `coverage` / `moderate` 提升了 top-tail 命中；
   - 但没有稳定提升整体 Spearman。
5. Data Shapley / optimizer-aware utility 方向仍未被否定，但现阶段不能作为主更新权重。

更准确的阶段性判断是：

> 目前的 utility signal 只能作为 small-loss selected set 内的极弱二级证据，且更像 top-tail proposal signal，而不是 continuous weighting signal。

## 7. 下一阶段建议

下一步不建议继续调 `utility_strength`，也不建议直接切到 streaming/continual。

更合理的 Stage-4 是：

1. 做 class-conditioned target gradient：
   - 每个样本只和同类或预测同类的 target source 对齐；
   - 避免全局 target gradient 被少数 easy class 主导。
2. 做 top-tail gate，而不是 continuous weighting：
   - 只在 small-loss selected set 内标记 top 5% 或 top 10% utility；
   - 初始只用于记录或 replay proposal；
   - 不直接替代 selected set。
3. 做 source construction 修复：
   - coverage + moderate 合并；
   - per-class minimum / cap；
   - 排除过 easy saturated target；
   - 记录 class-level source coverage 和 class-level proxy-oracle 指标。
4. 只有当 class-conditioned / top-tail source 在 2/3 seeds 中同时提升 Spearman 或 top-tail lift，才进入算法化短跑。

本阶段最重要的可保留结论：

> small-loss 是主可靠性门控；target utility 目前只能作为保守二级提案信号。若要继续 Shapley / optimizer-aware 路线，核心不是再调权重，而是重构 target gradient。

## 8. 结果文件

关键本地结果：

```text
remote_results/stage3_target_construction/stage35_s010_s025_paired_summary.json
remote_results/stage3_target_construction/stage35_selective_variants_3seed_summary.json
remote_results/stage3_target_construction/stage3_target_construction/stage35_diag_variants_selective_seed1/
remote_results/stage3_target_construction/stage3_target_construction/stage35_diag_variants_selective_seed2/
remote_results/stage3_target_construction/stage3_target_construction/stage35_diag_variants_selective_seed3/
remote_results/stage3_target_construction/target_construction_stage35_target_rerank_pool_seed1/
```

关键代码提交：

```text
99f5652 fix: preserve rerank sample count
707775e fix: make stage35 source variants selective
3e059a7 fix: isolate selective stage35 diagnostics
```
