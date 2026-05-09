# Stage-4 Q-Gated Peer Selection 结果与阶段判断

date: 2026-05-09

## 1. 实验目的

本阶段不是继续调 `q_gamma`、`q_weight_min/max` 或 $\pi_t$，而是从已定位的根因出发验证一个机制修复：

> $q_i$ 不应同时承担 clean posterior、训练权重、先验反馈和 replay 准入等多重角色。更保守的用法是：$q_i$ 只作为可靠候选门控，最终训练仍使用离散 hard selection。

对应实现为：

```bash
--q_usage_mode gate_selection
--q_gate_pool_mult 1.25
```

每个 batch 中，先用当前 `q_mode=hybrid` 计算 $q_i$，取 $q_i$ 前 $\lceil 1.25k\rceil$ 个样本形成候选池，其中 $k=\lceil \text{remember\_rate}\cdot B\rceil$。随后每个模型只在候选池内根据 peer aggregate loss 选择自己的 hard top-$k$ 样本。最终 loss 仍是 hard CE，不使用连续 Q 权重。

## 2. 关键对照

固定设置：

```bash
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=<4090> python -u main.py \
  --dataset cifar10 --noise_type symmetric --noise_rate 0.4 \
  --num_models 3 --q_mode hybrid --mstep_mode hard \
  --sam_rho 0 --replay_size 0 --replay_ratio 0 \
  --lambda_mode accuracy --lambda_patience 9999 --min_active 2 \
  --batch_size 256 --num_workers 8 --prefetch_factor 4 --drop_last \
  --n_epoch 30 --num_iter_per_epoch 100 --num_gradual 10 \
  --epoch_decay_start 80 --val_split 0.1 --seed <1|2|3>
```

两组对照：

- `diagnostic_only`：hard small-loss / peer selection 基线，Q 只记录诊断，不进入训练决策。
- `gate_selection`：Q 只作为候选池门控，最终选择仍由 peer loss 完成。

## 3. 三 seed 结果

### 3.1 hard baseline: `diagnostic_only`

| seed | best acc | last acc | last5 acc | q AUC | selected clean rate | overlap |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 79.73 | 79.61 | 79.320 | 0.9752 | 0.9322 | 0.9862 |
| 2 | 80.24 | 79.16 | 79.514 | 0.9764 | 0.9333 | 0.9867 |
| 3 | 79.51 | 79.51 | 79.208 | 0.9764 | 0.9341 | 0.9862 |
| mean | 79.83 | 79.43 | 79.347 | 0.9760 | 0.9332 | 0.9864 |

### 3.2 Q-gated peer selection

| seed | best acc | last acc | last5 acc | q AUC | selected clean rate | overlap |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 80.03 | 80.01 | 79.472 | 0.9749 | 0.9323 | 0.9865 |
| 2 | 80.49 | 78.99 | 79.562 | 0.9760 | 0.9325 | 0.9869 |
| 3 | 80.09 | 80.09 | 79.406 | 0.9760 | 0.9340 | 0.9865 |
| mean | 80.20 | 79.70 | 79.480 | 0.9756 | 0.9330 | 0.9866 |

### 3.3 paired delta: gate minus baseline

| seed | best delta | last delta | last5 delta |
|---:|---:|---:|---:|
| 1 | +0.30 | +0.40 | +0.152 |
| 2 | +0.25 | -0.17 | +0.048 |
| 3 | +0.58 | +0.58 | +0.198 |
| mean | +0.38 | +0.27 | +0.133 |

## 4. 结果解释

### 4.1 已支持的判断

1. **Q 本身没有坏。**  
   在 hard baseline 和 gate_selection 中，三 seed 的 `q_clean_auc` 都稳定在约 `0.975-0.976`，说明 hybrid Q 对 clean/reliability 的排序信号很强。

2. **先前失败不是“三网络天然崩坏”。**  
   三网络 gate_selection 的 `q_mean≈0.696`、`q_std≈0.441`，没有出现 standard/weight 路径中 `q_mean→1`、`q_std→0` 的 posterior 饱和。

3. **Q 连续权重路径是危险路径。**  
   对比先前 seed1：
   - `weight_only` 2 nets: last5 acc `64.666`，`q_mean≈0.995`，`q_std≈0.041`
   - `standard` 2 nets: last5 acc `69.274`，`q_mean≈0.99996`，`q_std≈0.00027`
   - `gate_selection` 3 nets: last5 acc `79.480`，`q_mean≈0.696`，`q_std≈0.441`

   这支持“单一 $q_i$ 多角色反馈导致 collapse”的归因，而不是“Q 估计完全不可用”。

4. **gate_selection 至少没有伤害 hard baseline，并有小幅正向。**  
   三 seed paired last5 delta 平均 `+0.133%`，best acc delta 平均 `+0.38%`。这个幅度不能被夸大成显著突破，但作为机制修复的第一步是正向的。

### 4.2 尚未支持的判断

1. **不能声称 gate_selection 已经显著优于 hard baseline。**  
   last5 提升只有 `+0.133%`，且 seed2 last acc 为负 delta。需要更长 epoch、更正式 benchmark 或更多 seeds 才能声称稳定 accuracy 改进。

2. **不能声称 gate_selection 明显提升了多模型 diversity。**  
   overlap 仍约 `0.986`，与 hard baseline 非常接近。它解决的是 Q 连续权重 collapse，而不是充分解决模型趋同。

3. **不能把当前结果写成 Data Shapley / utility 成功。**  
   本阶段只处理 reliability posterior 的使用路径，尚未引入可靠的 target utility estimator。

## 5. 第一性原理归因

noisy-label learning 中，训练样本选择至少包含两个不同问题：

1. 可靠性：这个样本的标签是否可信？
2. 效用：在当前训练阶段，学习这个样本是否改善目标泛化？

旧的 hybrid Q/robust M-step 把这些问题混在同一个连续权重里：

$$
\nabla_\theta L_t=\sum_i w(q_i)\nabla_\theta \ell_i
$$

一旦模型开始拟合噪声，噪声样本的 loss、confidence 或 teacher consistency 会反过来提高 $q_i$，$w(q_i)$ 又进一步增强这些样本的梯度贡献，形成 confirmation feedback。

gate_selection 切断了这个反馈链：

$$
C_t=\operatorname{TopK}_i(q_i,\lceil \alpha k\rceil)
$$

$$
S_{m,t}=\operatorname{TopK}_{i\in C_t}(-\ell_{-m,i},k)
$$

此时 $q_i$ 只决定“能否进入可靠候选池”，不决定连续梯度强度。最终优化仍是 hard CE，所以错误 posterior 不会被连续放大。

## 6. 下一步

当前最合理的推进不是重新启用 Q continuous weighting，而是沿着“角色解耦”继续：

1. **短期：测试 gate pool 宽度。**  
   当前 `q_gate_pool_mult=1.25` 的 overlap 仍高，说明候选池可能仍偏窄，或 peer loss 本身高度同质。下一步应小规模比较 `1.10 / 1.25 / 1.50 / 2.00`，重点看 last5、selected clean rate 和 overlap 的三方权衡。

2. **中期：引入 diversity / asynchronous，而不是加大 Q 权重。**  
   如果扩大候选池不能降低 overlap，就说明多模型差异不是靠 Q gate 解决，应转向 staggered/asynchronous update 或 diversity-aware peer selection。

3. **长期：utility 只能在 reliable candidate 内部工作。**  
   Data Shapley / optimizer-aware utility 不应替代 $r_i$，而应作为可靠候选集内部的 secondary rerank/top-tail proposal。否则会重新引入高噪声样本的放大风险。

## 7. 结果位置

- local summary CSV: `remote_results/q_isolation/q_isolation_summary.csv`
- local run dirs:
  - `remote_results/q_isolation/qiso_diagnostic_m3_seed1`
  - `remote_results/q_isolation/qiso_diagnostic_m3_seed2`
  - `remote_results/q_isolation/qiso_diagnostic_m3_seed3`
  - `remote_results/q_isolation/qiso_gate_m3_seed1`
  - `remote_results/q_isolation/qiso_gate_m3_seed2`
  - `remote_results/q_isolation/qiso_gate_m3_seed3`

