# Stage-2 Reliability-Gated Utility Exploration

## 目的

第二阶段不把 Shapley / utility 直接当作 clean detector，也不直接写入 $q_i$。当前只测试低风险路径：

$$
w_i(t)=r_i(t)\cdot u_i(t)
$$

其中 $r_i(t)$ 由 peer-selected small-loss gate 给出，$u_i(t)$ 只在已选样本内部调节权重。

## 当前最小实现

新增 `--utility_mode sam_gap`：

- 先用 peer loss 选择 small-loss 样本。
- 对该 selection 计算标准 SAM perturbation。
- 记录每个已选样本的 sharpness gap：

$$
s_i(t)=\ell_i(\theta+\epsilon_{\mathrm{SAM}})-\ell_i(\theta)
$$

- gap 越小，样本越稳定，utility multiplier 越高。
- multiplier 只作用于已选样本，不从 high-loss 区域额外捞样本。

## 快速实验原则

- 先小 epoch smoke，关注显存和速度。
- 使用 `batch_size=512` 起步，用 4090/3090 的大显存换吞吐。
- 若 OOM 或速度异常，再降到 `batch_size=384`。
- 第一轮只比较 `utility_mode=none` 与 `utility_mode=sam_gap`。
- 不以单次短跑 accuracy 定最终结论，主要看是否稳定、是否明显伤害 baseline、utility 权重是否塌缩。

## Smoke 命令模板

```bash
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=<设备号> python -u main.py \
  --dataset cifar10 --noise_type symmetric --noise_rate 0.4 \
  --num_models 2 --q_mode loss --mstep_mode hard \
  --sam_rho 0.05 --optimizer sgd --lr 0.1 --momentum 0.9 --weight_decay 0.0001 \
  --replay_size 0 --replay_ratio 0 \
  --lambda_mode accuracy --lambda_patience 9999 --min_active 2 \
  --batch_size 512 --num_workers 8 --prefetch_factor 4 --drop_last \
  --n_epoch 3 --num_iter_per_epoch 20 --num_gradual 10 --epoch_decay_start 80 \
  --val_split 0 --seed 1 \
  --utility_mode sam_gap \
  --result_dir results_stage2/smoke_sam_gap_b512_seed1
```

## 30 Epoch 快速对照

Baseline：

```bash
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=<设备号> python -u main.py \
  --dataset cifar10 --noise_type symmetric --noise_rate 0.4 \
  --num_models 2 --q_mode loss --mstep_mode hard \
  --sam_rho 0.05 --optimizer sgd --lr 0.1 --momentum 0.9 --weight_decay 0.0001 \
  --replay_size 0 --replay_ratio 0 \
  --lambda_mode accuracy --lambda_patience 9999 --min_active 2 \
  --batch_size 512 --num_workers 8 --prefetch_factor 4 --drop_last \
  --n_epoch 30 --num_iter_per_epoch 100 --num_gradual 10 --epoch_decay_start 80 \
  --val_split 0 --seed 1 \
  --utility_mode none \
  --result_dir results_stage2/e1_sam_baseline_b512_seed1
```

Utility weighting：

```bash
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=<设备号> python -u main.py \
  --dataset cifar10 --noise_type symmetric --noise_rate 0.4 \
  --num_models 2 --q_mode loss --mstep_mode hard \
  --sam_rho 0.05 --optimizer sgd --lr 0.1 --momentum 0.9 --weight_decay 0.0001 \
  --replay_size 0 --replay_ratio 0 \
  --lambda_mode accuracy --lambda_patience 9999 --min_active 2 \
  --batch_size 512 --num_workers 8 --prefetch_factor 4 --drop_last \
  --n_epoch 30 --num_iter_per_epoch 100 --num_gradual 10 --epoch_decay_start 80 \
  --val_split 0 --seed 1 \
  --utility_mode sam_gap --utility_strength 1.0 --utility_temp 1.0 \
  --result_dir results_stage2/e2_sam_gap_b512_seed1
```

## 观察指标

- `best_test_acc`, `last_test_acc`, `last_10_mean_test_acc`。
- `utility_weight_mean` 应接近 1。
- `utility_weight_std` 不能接近 0，也不能异常巨大。
- `utility_gap_mean` 应为稳定可记录的有限值。
- 若 utility 短跑明显低于 baseline 超过约 1 个点，先降低 `utility_strength=0.5`，不要直接放弃。

