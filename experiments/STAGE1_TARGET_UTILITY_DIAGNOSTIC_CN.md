# 第一阶段：Target-Utility Diagnostic Probe

## 目标

第一阶段只做诊断，不改变训练更新逻辑。目标是在稳定的 hard small-loss baseline 上判断 target-aligned utility signal 是否提供 small-loss 之外的增量信息。

不因单 seed、单 batch size、单 target source 或单 GPU 的失败直接否定方向。

## GPU 优先级

优先使用空闲 `4090`，其次空闲 `3090`，最后在显存允许时使用 `3080 Ti`。

远端 PyTorch 默认设备顺序可能与 `nvidia-smi` 不一致，因此所有训练命令必须使用：

```bash
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=<nvidia-smi设备号>
```

这样 `<nvidia-smi设备号>` 才对应 `nvidia-smi` 显示的物理 GPU。

推荐 batch：

| GPU | E1 batch | E2 batch |
|---|---:|---:|
| 4090 | 384 | 320 |
| 3090 | 384 | 320 |
| 3080 Ti | 192 | 160 |

## 远端准备

```bash
cd /data1/yuzhixiang/work/Co-teaching-MoreNets
source /data1/yuzhixiang/opt/miniconda3/etc/profile.d/conda.sh
conda activate /data1/yuzhixiang/.conda/envs/coteaching-py39
nvidia-smi
export DEV=<优先空闲4090，否则3090，否则3080Ti>
```

## 本地自动编排入口

默认从本地 Windows 运行：

```powershell
powershell -ExecutionPolicy Bypass -File tools/remote-workflow/run_stage1_diagnostic.ps1
```

常用变体：

```powershell
# 只跑预检和短 smoke。
powershell -ExecutionPolicy Bypass -File tools/remote-workflow/run_stage1_diagnostic.ps1 -SmokeOnly

# 启动当前队列中的作业后立即返回，不等待训练结束。
powershell -ExecutionPolicy Bypass -File tools/remote-workflow/run_stage1_diagnostic.ps1 -NoWait

# 确认代码已同步时跳过 git push / remote pull。
powershell -ExecutionPolicy Bypass -File tools/remote-workflow/run_stage1_diagnostic.ps1 -SkipGitSync
```

该脚本只 stage 第一阶段必要文件，不使用 `git add -A`，避免误提交论文、草稿或旧实验记录。

## Smoke

```bash
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=$DEV python -u main.py \
  --dataset cifar10 --noise_type symmetric --noise_rate 0.4 \
  --num_models 3 --q_mode loss --mstep_mode hard \
  --sam_rho 0 --replay_size 0 --replay_ratio 0 \
  --lambda_mode accuracy --lambda_patience 9999 \
  --batch_size 384 --num_workers 8 --prefetch_factor 4 --drop_last \
  --n_epoch 2 --num_iter_per_epoch 20 \
  --diag_alignment --diag_every_epoch 1 --diag_batches 2 --diag_val_batches 1 \
  --diag_target both \
  --result_dir results_diag/stage1_smoke \
  --diag_output_dir results_diag/stage1_smoke/diag
```

预期：

- 能完成短训练。
- `results_diag/stage1_smoke/diag/alignment_summary.jsonl` 存在。
- `results_diag/stage1_smoke/diag/alignment_epoch_0001.jsonl` 存在。

## E1：Hard Small-loss + Alignment Diagnostic

```bash
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=$DEV python -u main.py \
  --dataset cifar10 --noise_type symmetric --noise_rate 0.4 \
  --num_models 3 --q_mode loss --mstep_mode hard \
  --sam_rho 0 --replay_size 0 --replay_ratio 0 \
  --lambda_mode accuracy --lambda_patience 9999 \
  --batch_size 384 --num_workers 8 --prefetch_factor 4 --drop_last \
  --n_epoch 80 --seed <1|2|3> \
  --diag_alignment --diag_every_epoch 5 --diag_batches 4 --diag_val_batches 2 \
  --diag_target both \
  --result_dir results_diag/stage1_e1_seed<seed> \
  --diag_output_dir results_diag/stage1_e1_seed<seed>/diag
```

## E2：Hard Small-loss + SAM Diagnostic

```bash
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=$DEV python -u main.py \
  --dataset cifar10 --noise_type symmetric --noise_rate 0.4 \
  --num_models 3 --q_mode loss --mstep_mode hard \
  --sam_rho 0.05 --replay_size 0 --replay_ratio 0 \
  --lambda_mode accuracy --lambda_patience 9999 \
  --batch_size 320 --num_workers 8 --prefetch_factor 4 --drop_last \
  --n_epoch 80 --seed <1|2|3> \
  --diag_alignment --diag_every_epoch 5 --diag_batches 4 --diag_val_batches 2 \
  --diag_target both \
  --result_dir results_diag/stage1_e2_sam_seed<seed> \
  --diag_output_dir results_diag/stage1_e2_sam_seed<seed>/diag
```

## 关键输出

每次诊断会写：

- `alignment_summary.jsonl`：epoch 级汇总。
- `alignment_epoch_<epoch>.jsonl`：样本级记录。

重点字段：

- `auc_loss_clean`
- `auc_align_raw_clean`
- `auc_align_adam_clean`
- `selected_clean_rate`
- `high_align_clean_rate`
- `high_loss_high_align_clean_rate`

## 判据

- 若 clean-target 的 `auc_align_adam_clean >= 0.60` 在至少 2 个 seed、多个诊断 epoch 成立，则 target utility signal 值得进入下一阶段算法化。
- 若 raw alignment 失败但 Adam-preconditioned alignment 成立，则后续必须走 optimizer-aware utility。
- 若 clean-target 有信号但 noisy-target 无信号，说明关键问题是构造可信 target gradient，不应否定 alignment 方向。
- 若 clean-target 与 noisy-target 都无信号，并且 3 个 seed 都失败，则回到 reliability repair。
- 若 E2 提高 alignment 稳定性但 accuracy 未提升，不能判 SAM 无效；第一阶段只判断 attribution stability。
- 若 E2 accuracy 下降但 alignment 指标更稳，下一步先试 `sam_rho=0.02`。
