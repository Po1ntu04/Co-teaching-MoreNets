# Stage-3.5 Conservative Target-Align Validation

## 目标

本阶段验证 `purified_buffer` 构造的 target-alignment utility 是否能在保守强度下跨 seed 不伤害 SRIT-like hard small-loss baseline，并诊断更好的 target source 构造方式。

当前前提：

- small-loss 是强 clean/reliability 信号，但不是 target utility。
- `sam_gap` 已不适合作为主 utility proxy。
- `purified_buffer` 的 proxy-oracle Spearman 只有弱正信号，不能重权化。
- `target_align` 只能在 hard small-loss reliable set 内做二级排序或轻量加权。

## 新增参数

```bash
--target_align_mode weighted|rerank_only
--target_align_rerank_frac 0.75
--target_align_source purified_buffer|purified_buffer_balanced|purified_buffer_moderate|purified_buffer_coverage|ema_purified|ema_teacher
```

新增诊断 source：

```text
purified_buffer_balanced
purified_buffer_moderate
purified_buffer_coverage
ema_purified
```

诊断 summary 额外记录：

```text
source_label_hist
source_effective_size
source_loss_mean
source_loss_std
source_confidence_mean
```

## 固定训练口径

```bash
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=<GPU> python -u main.py \
  --dataset cifar10 --noise_type symmetric --noise_rate 0.4 \
  --num_models 2 --q_mode loss --mstep_mode hard \
  --optimizer sgd --lr 0.1 --momentum 0.9 --weight_decay 0.0001 \
  --sam_rho 0.05 --replay_size 2000 --replay_ratio 0 \
  --lambda_mode accuracy --lambda_patience 9999 --min_active 2 \
  --batch_size 512 --num_workers 8 --prefetch_factor 4 --drop_last \
  --n_epoch 31 --num_iter_per_epoch 100 --num_gradual 10 \
  --epoch_decay_start 80 --val_split 0.1
```

## 推荐远端脚本

全部在本地 PowerShell 运行，脚本会通过 SSH 在远端 tmux 启动任务：

```powershell
powershell -ExecutionPolicy Bypass -File tools\remote-workflow\run_stage3_target_construction.ps1 -Mode target_s025 -Seed 2 -NoWait
powershell -ExecutionPolicy Bypass -File tools\remote-workflow\run_stage3_target_construction.ps1 -Mode target_s025 -Seed 3 -NoWait
```

如果 `0.25` 不稳定：

```powershell
powershell -ExecutionPolicy Bypass -File tools\remote-workflow\run_stage3_target_construction.ps1 -Mode target_s010 -Seed 1 -NoWait
powershell -ExecutionPolicy Bypass -File tools\remote-workflow\run_stage3_target_construction.ps1 -Mode target_s010 -Seed 2 -NoWait
powershell -ExecutionPolicy Bypass -File tools\remote-workflow\run_stage3_target_construction.ps1 -Mode target_s010 -Seed 3 -NoWait
```

若连续加权仍不稳：

```powershell
powershell -ExecutionPolicy Bypass -File tools\remote-workflow\run_stage3_target_construction.ps1 -Mode target_rerank -Seed 1 -NoWait
powershell -ExecutionPolicy Bypass -File tools\remote-workflow\run_stage3_target_construction.ps1 -Mode target_rerank -Seed 2 -NoWait
powershell -ExecutionPolicy Bypass -File tools\remote-workflow\run_stage3_target_construction.ps1 -Mode target_rerank -Seed 3 -NoWait
```

target source variants 诊断：

```powershell
powershell -ExecutionPolicy Bypass -File tools\remote-workflow\run_stage3_target_construction.ps1 -Mode diag_variants -Seed 1 -NoWait
powershell -ExecutionPolicy Bypass -File tools\remote-workflow\run_stage3_target_construction.ps1 -Mode diag_variants -Seed 2 -NoWait
powershell -ExecutionPolicy Bypass -File tools\remote-workflow\run_stage3_target_construction.ps1 -Mode diag_variants -Seed 3 -NoWait
```

## 判据

- `target_s025` 至少 2/3 seeds 的 last5 mean 不低于 paired baseline。
- 若 `target_s025` 不稳，`target_s010` 或 `target_rerank` 必须优先于更强加权。
- source variant 的 Spearman 或 top25 ratio 必须超过旧 `purified_buffer` 基线，才进入算法化短跑。
- accuracy 偶然提高但 proxy ranking 不成立，不能写成方法贡献。

## 下一步记录

阶段结束后生成：

```text
experiments/STAGE3_5_CONSERVATIVE_TARGET_ALIGN_RESULTS_CN.md
```

