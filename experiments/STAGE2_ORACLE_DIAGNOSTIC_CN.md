# Stage-2 Oracle Diagnostic：验证 utility proxy 是否接近真实边际效用

## 目的

上一轮短实验说明：

- `sam_gap` 强度 1.0 会压低训练。
- `sam_gap` 强度 0.5 基本追平 baseline。
- 但这仍不能说明 `sam_gap` 真正刻画了样本边际效用。

本轮实验不先追求 accuracy，而是问一个更基础的问题：

> 在 peer-selected small-loss 可靠样本集合内部，`sam_gap`、small-loss、alignment 这些 proxy 是否能排序真实的一步 validation utility？

如果不能排序真实 utility，就不应把该 proxy 继续算法化。

## 诊断定义

开启 `--diag_oracle` 后，训练更新路径不变。诊断只额外记录。

对每个诊断 epoch：

1. 取少量 train batch。
2. 用 peer loss 得到每个模型的 selected set。
3. 只在 selected set 内抽样候选样本。
4. 冻结 backbone feature，只模拟最后一层的一步 SGD 更新。
5. 计算该单样本更新对 validation loss 的实际改善：

$$
\Delta_i = L_{\mathrm{val}}(\theta) - L_{\mathrm{val}}(\theta - \eta g_i)
$$

其中 $\Delta_i > 0$ 表示样本 $i$ 的一步更新降低了 validation loss。

这是一个受控 oracle，不是最终方法：

- 它使用 validation split。
- clean-target 是上界分析。
- frozen-feature / last-layer 是低成本近似。
- 它的作用是审计 proxy，而不是直接训练模型。

## 新增参数

```bash
--diag_oracle
--diag_oracle_every_epoch 5
--diag_oracle_batches 2
--diag_oracle_val_batches 1
--diag_oracle_candidates 128
--diag_oracle_target both
--diag_oracle_lr 0
--diag_oracle_output_dir results_diag/stage2_oracle/<run_name>
```

`--diag_oracle_lr 0` 表示使用当前 optimizer lr。

## 关键输出

`oracle_summary.jsonl` 每个 target/epoch 一行，核心字段：

- `oracle_mean`
- `oracle_positive_rate`
- `oracle_clean_mean`
- `oracle_noisy_mean`
- `auc_oracle_clean`
- `auc_loss_clean`
- `auc_sam_utility_clean`
- `auc_align_adam_clean`
- `spearman_loss_oracle`
- `spearman_sam_utility_oracle`
- `spearman_align_adam_oracle`
- `top25_oracle_mean_by_loss`
- `top25_oracle_mean_by_sam_utility`
- `top25_oracle_mean_by_align_adam`

样本级文件：

```text
oracle_epoch_XXXX.jsonl
```

记录每个候选样本的 noisy label、clean label、clean/noisy flag、proxy 分数和 oracle improvement。

## Smoke 命令

```bash
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=<gpu> python -u main.py \
  --dataset cifar10 --noise_type symmetric --noise_rate 0.4 \
  --num_models 2 --q_mode loss --mstep_mode hard \
  --sam_rho 0.05 --optimizer sgd --lr 0.1 --momentum 0.9 --weight_decay 0.0001 \
  --replay_size 0 --replay_ratio 0 \
  --lambda_mode accuracy --lambda_patience 9999 --min_active 2 \
  --batch_size 512 --num_workers 8 --prefetch_factor 4 --drop_last \
  --n_epoch 4 --num_iter_per_epoch 20 --num_gradual 10 --epoch_decay_start 80 \
  --val_split 0.1 --seed 1 \
  --utility_mode none \
  --diag_oracle --diag_oracle_every_epoch 1 \
  --diag_oracle_batches 1 --diag_oracle_val_batches 1 --diag_oracle_candidates 64 \
  --diag_oracle_target both \
  --result_dir results_stage2/oracle_smoke_baseline_seed1 \
  --diag_oracle_output_dir results_diag/stage2_oracle/smoke_baseline_seed1
```

## 第一轮正式短实验

Baseline：

```bash
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=<gpu> python -u main.py \
  --dataset cifar10 --noise_type symmetric --noise_rate 0.4 \
  --num_models 2 --q_mode loss --mstep_mode hard \
  --sam_rho 0.05 --optimizer sgd --lr 0.1 --momentum 0.9 --weight_decay 0.0001 \
  --replay_size 0 --replay_ratio 0 \
  --lambda_mode accuracy --lambda_patience 9999 --min_active 2 \
  --batch_size 512 --num_workers 8 --prefetch_factor 4 --drop_last \
  --n_epoch 30 --num_iter_per_epoch 100 --num_gradual 10 --epoch_decay_start 80 \
  --val_split 0.1 --seed 1 \
  --utility_mode none \
  --diag_oracle --diag_oracle_every_epoch 5 \
  --diag_oracle_batches 2 --diag_oracle_val_batches 1 --diag_oracle_candidates 128 \
  --diag_oracle_target both \
  --result_dir results_stage2/oracle_e1_baseline_seed1 \
  --diag_oracle_output_dir results_diag/stage2_oracle/e1_baseline_seed1
```

Mild `sam_gap`：

```bash
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=<gpu> python -u main.py \
  --dataset cifar10 --noise_type symmetric --noise_rate 0.4 \
  --num_models 2 --q_mode loss --mstep_mode hard \
  --sam_rho 0.05 --optimizer sgd --lr 0.1 --momentum 0.9 --weight_decay 0.0001 \
  --replay_size 0 --replay_ratio 0 \
  --lambda_mode accuracy --lambda_patience 9999 --min_active 2 \
  --batch_size 512 --num_workers 8 --prefetch_factor 4 --drop_last \
  --n_epoch 30 --num_iter_per_epoch 100 --num_gradual 10 --epoch_decay_start 80 \
  --val_split 0.1 --seed 1 \
  --utility_mode sam_gap --utility_strength 0.5 \
  --diag_oracle --diag_oracle_every_epoch 5 \
  --diag_oracle_batches 2 --diag_oracle_val_batches 1 --diag_oracle_candidates 128 \
  --diag_oracle_target both \
  --result_dir results_stage2/oracle_e2_sam_gap_s05_seed1 \
  --diag_oracle_output_dir results_diag/stage2_oracle/e2_sam_gap_s05_seed1
```

## 判据

先看 clean-target，这是 controlled upper bound：

- 若 `spearman_sam_utility_oracle` 长期接近 0 或为负，同时 `top25_oracle_mean_by_sam_utility` 不优于整体均值，则当前 `sam_gap` proxy 应停止算法化。
- 若 `sam_gap` 不行但 `align_adam` 对 oracle 有稳定正相关，则下一步转向 optimizer-aware gradient utility。
- 若 clean-target 有信号而 noisy-target 无信号，问题转为 target gradient 构造，而不是 utility 方向失败。
- 若所有 proxy 对 clean-target oracle 都无信号，则需要回到可靠性修复或重新定义 oracle/proxy。

不允许用单个 early epoch 的失败直接终止方向。至少要看多个诊断 epoch 和 target 差异。

## 分析命令

```bash
python scripts/analyze_stage2_oracle.py \
  results_diag/stage2_oracle/e1_baseline_seed1 \
  results_diag/stage2_oracle/e2_sam_gap_s05_seed1 \
  --output results_diag/stage2_oracle/oracle_summary_seed1.json
```
