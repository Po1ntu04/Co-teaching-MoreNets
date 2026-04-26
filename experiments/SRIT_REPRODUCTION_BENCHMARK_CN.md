# SRIT / Co-teaching 论文参数复现 Benchmark

## 目的

本 benchmark 只做论文参数口径的可比基准，不把它直接等同于新的 `Co-teaching-plus` 方法。

依据 Zotero 条目：

- `Sharpness-Aware Minimization Activates the Interactive Teaching's Understanding and Optimization`，Zotero item key: `6LKIQL87`。
- 论文中 CIFAR-10 symmetric 40% 的表格结果：Co-teaching `77.38 ± 0.15`，SRIT `79.83 ± 0.12`。
- 论文实验细节：Co-teaching 使用 Adam，`lr=0.001`，`batch_size=128`，`epochs=200`；SRIT/SRCNLCU 使用 SGD，`lr=0.1`，`momentum=0.9`，`weight_decay=0.0001`，`epochs=200`，`rho=0.05`，`batch_size=128`。

## 重要边界

当前代码中的 `--sam_rho 0.05 --optimizer sgd` 是 SRIT-like benchmark probe，不保证完全复现 SRIT 论文的“dual-level sharpness knowledge exchange”全部细节。它用于回答一个更低层问题：在相同数据、模型与 hard small-loss selection 下，论文参数口径的 SGD + SAM 是否能形成合理 benchmark。

因此结果解释必须分三层：

- Co-teaching baseline 是否能接近论文量级。
- 当前 SRIT-like 实现是否能从 Co-teaching baseline 上获得提升。
- 若不能提升，不能直接反驳 SRIT 论文，只能说明当前实现与论文算法仍有差距，或需要进一步对齐实现细节。

## 统计口径

论文报告通常采用最后若干 epoch 的平均测试准确率。当前统一记录：

- `best_test_acc`
- `last_test_acc`
- `last_10_mean_test_acc`
- `last_10_mean_per_model`

优先使用 `last_10_mean_test_acc` 与论文表格对比，`best_test_acc` 只作为训练过程上界参考。

## 远端命令模板

所有命令固定使用：

```bash
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=<nvidia-smi设备号>
```

### Co-teaching Baseline

```bash
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=<设备号> python -u main.py \
  --dataset cifar10 --noise_type symmetric --noise_rate 0.4 \
  --num_models 2 --q_mode loss --mstep_mode hard \
  --sam_rho 0 --replay_size 0 --replay_ratio 0 \
  --lambda_mode accuracy --lambda_patience 9999 --min_active 2 \
  --optimizer adam --lr 0.001 --weight_decay 0 \
  --batch_size 128 --num_workers 8 --prefetch_factor 4 --drop_last \
  --n_epoch 200 --num_gradual 10 --epoch_decay_start 80 \
  --val_split 0 \
  --result_dir results_benchmark/srit_repro/coteaching_seed<seed> \
  --seed <seed>
```

### SRIT-like Probe

```bash
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=<设备号> python -u main.py \
  --dataset cifar10 --noise_type symmetric --noise_rate 0.4 \
  --num_models 2 --q_mode loss --mstep_mode hard \
  --sam_rho 0.05 --replay_size 0 --replay_ratio 0 \
  --lambda_mode accuracy --lambda_patience 9999 --min_active 2 \
  --optimizer sgd --lr 0.1 --momentum 0.9 --weight_decay 0.0001 \
  --batch_size 128 --num_workers 8 --prefetch_factor 4 --drop_last \
  --n_epoch 200 --num_gradual 10 --epoch_decay_start 80 \
  --val_split 0 \
  --result_dir results_benchmark/srit_repro/srit_like_seed<seed> \
  --seed <seed>
```

## 自动分析

```bash
python scripts/analyze_benchmark_accuracy.py results_benchmark/srit_repro --last-k 10
```

## 当前下一步

先跑 `coteaching_seed1`，得到本仓库在论文参数口径下的 baseline benchmark。若 baseline 与论文量级严重不符，先排查实现差异；若 baseline 合理，再跑 `srit_like_seed1` 做同 seed 对照。
