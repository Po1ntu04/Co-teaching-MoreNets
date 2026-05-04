# Stage-3 Target Construction Diagnostic

## 目的

前两阶段已经把失败点收缩到一个问题：

> 如果没有 clean validation target，能否构造足够可信的 target gradient？

本阶段默认不改变训练更新，只记录不同 target source 对 clean one-step oracle 的近似质量。它不是直接证明 Data Shapley 有效，而是先检验 Shapley / optimizer-aware utility 是否有可用目标。

## Target Sources

- `clean_val`：clean label validation gradient，只作 controlled upper bound。
- `noisy_val`：noisy label validation gradient，作为低可信下界。
- `peer_consensus`：active committee 高置信且预测与 noisy label 一致的样本构造 target。
- `ema_teacher`：EMA teacher committee 的高置信 soft target。
- `purified_buffer`：purified replay memory 中高稳定样本构造 target；若 memory 尚未形成，会记录 unavailable。

所有非 clean target 都只能作为 proxy；不能把它们写成已验证的真实目标。

## Smoke

```bash
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=<gpu> python -u main.py \
  --dataset cifar10 --noise_type symmetric --noise_rate 0.4 \
  --num_models 2 --q_mode loss --mstep_mode hard \
  --sam_rho 0.05 --optimizer sgd --lr 0.1 --momentum 0.9 --weight_decay 0.0001 \
  --replay_size 2000 --replay_ratio 0 \
  --lambda_mode accuracy --lambda_patience 9999 --min_active 2 \
  --batch_size 512 --num_workers 8 --prefetch_factor 4 --drop_last \
  --n_epoch 4 --num_iter_per_epoch 20 --num_gradual 10 --epoch_decay_start 80 \
  --val_split 0.1 --seed 1 \
  --diag_target_construction --diag_target_every_epoch 1 \
  --diag_target_batches 1 --diag_target_val_batches 1 --diag_target_candidates 64 \
  --diag_target_sources clean_val,noisy_val,peer_consensus,ema_teacher,purified_buffer \
  --result_dir results_stage3/target_construction_smoke_seed1 \
  --diag_target_output_dir results_diag/stage3_target_construction/smoke_seed1
```

说明：当前训练循环从 epoch 1 跑到 `n_epoch-1`，所以 smoke 用 `--n_epoch 4` 表示实际 3 个训练 epoch。

## E3.1 Target Source Diagnostic

```bash
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=<gpu> python -u main.py \
  --dataset cifar10 --noise_type symmetric --noise_rate 0.4 \
  --num_models 2 --q_mode loss --mstep_mode hard \
  --sam_rho 0.05 --optimizer sgd --lr 0.1 --momentum 0.9 --weight_decay 0.0001 \
  --replay_size 2000 --replay_ratio 0 \
  --lambda_mode accuracy --lambda_patience 9999 --min_active 2 \
  --batch_size 512 --num_workers 8 --prefetch_factor 4 --drop_last \
  --n_epoch 31 --num_iter_per_epoch 100 --num_gradual 10 --epoch_decay_start 80 \
  --val_split 0.1 --seed 1 \
  --diag_target_construction --diag_target_every_epoch 5 \
  --diag_target_batches 2 --diag_target_val_batches 1 --diag_target_candidates 128 \
  --diag_target_sources clean_val,noisy_val,peer_consensus,ema_teacher,purified_buffer \
  --result_dir results_stage3/target_construction_e31_seed1 \
  --diag_target_output_dir results_diag/stage3_target_construction/e31_seed1
```

## 分析

```bash
python scripts/analyze_stage3_target_construction.py \
  results_diag/stage3_target_construction/e31_seed1 \
  --output results_diag/stage3_target_construction/e31_seed1_summary.json
```

## E3.3 Algorithmic Short Run

只有当非 clean target source 在多个 seed 中通过诊断，才进入本节。当前默认只使用更接近最终方法的 `purified_buffer`，不用 `clean_val`，也不使用 `noisy_val` 作为训练目标。

Baseline：

```bash
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=<gpu> python -u main.py \
  --dataset cifar10 --noise_type symmetric --noise_rate 0.4 \
  --num_models 2 --q_mode loss --mstep_mode hard \
  --sam_rho 0.05 --optimizer sgd --lr 0.1 --momentum 0.9 --weight_decay 0.0001 \
  --replay_size 2000 --replay_ratio 0 \
  --lambda_mode accuracy --lambda_patience 9999 --min_active 2 \
  --batch_size 512 --num_workers 8 --prefetch_factor 4 --drop_last \
  --n_epoch 31 --num_iter_per_epoch 100 --num_gradual 10 --epoch_decay_start 80 \
  --val_split 0.1 --seed 1 \
  --utility_mode none \
  --result_dir results_stage3/target_construction_algo_baseline_seed1
```

Target-align utility：

```bash
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=<gpu> python -u main.py \
  --dataset cifar10 --noise_type symmetric --noise_rate 0.4 \
  --num_models 2 --q_mode loss --mstep_mode hard \
  --sam_rho 0.05 --optimizer sgd --lr 0.1 --momentum 0.9 --weight_decay 0.0001 \
  --replay_size 2000 --replay_ratio 0 \
  --lambda_mode accuracy --lambda_patience 9999 --min_active 2 \
  --batch_size 512 --num_workers 8 --prefetch_factor 4 --drop_last \
  --n_epoch 31 --num_iter_per_epoch 100 --num_gradual 10 --epoch_decay_start 80 \
  --val_split 0.1 --seed 1 \
  --utility_mode target_align --utility_strength 0.25 \
  --target_align_source purified_buffer \
  --target_align_min_source 16 --target_align_max_source 128 \
  --result_dir results_stage3/target_construction_algo_target_s025_seed1
```

若 `0.25` 不伤害 baseline，再试 `--utility_strength 0.5`。若 `0.25` 已明显下降，不应继续加大强度。

## 判据

- `clean_val` 只用于 sanity check，不进入最终方法。
- 非 clean source 若 `spearman_proxy_adam_oracle >= 0.15` 且至少两个诊断 epoch 成立，可进入 algorithmic short run。
- 若 Spearman 不成立但 top25 oracle lift 多个 epoch 为正，可作为弱候选，只能先做 reranking 或低强度加权。
- 若 `peer_consensus`、`ema_teacher`、`purified_buffer` 都失败，则结论是 target construction 尚不可靠，而不是 Data Shapley 数学方向失败。
- 若 proxy ranking 不成立，即使短跑 accuracy 偶然提高，也不能写成方法贡献。

## GPU 策略

- 优先空闲 4090，其次 3090，最后才用 3080 Ti。
- 4090/3090 默认 `batch_size=512`。
- 3080 Ti 使用 `batch_size=256`，必要时降到 `192`。
- 自动 SSH 控制通道应使用 `-o ClearAllForwardings=yes`，避免与 VS Code RemoteForward 冲突。
