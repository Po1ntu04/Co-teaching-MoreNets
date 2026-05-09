# Q Collapse Isolation 实验说明

updated: 2026-05-09

## 1. 目的

本实验不再把“Q 崩坏”笼统归因为三网络或 soft posterior，而是拆解 Q 的用途链路。

要回答的问题：

- Q 本身是否有 clean/noisy 区分能力？
- Q 是否因 `pi_t` prior feedback 饱和？
- Q 用作 selection 是否破坏 small-loss reliability？
- Q 用作 soft/robust weight 是否放大错误样本？
- 多网络同步是否只是放大上述反馈？

## 2. 新增参数

```bash
--q_usage_mode standard
--q_usage_mode diagnostic_only
--q_usage_mode prior_only
--q_usage_mode selection_only
--q_usage_mode weight_only
--q_usage_mode replay_admission_only
```

含义：

- `standard`：保留原始路径。
- `diagnostic_only`：只记录 Q，不更新 `pi_t`，不用于 selection、weight、replay。
- `prior_only`：Q 只更新 `pi_t`，用于隔离 `Q -> pi -> Q` 正反馈。
- `selection_only`：Q 只替代 hard small-loss selection。
- `weight_only`：Q 只进入 soft/robust loss weight。
- `replay_admission_only`：Q 只用于 replay admission。

新增 JSON 指标：

- `q_min`
- `q_max`
- `q_entropy`
- `q_clean_auc`
- `selected_clean_rate`

## 3. Smoke

远端执行前先选择 GPU：

```bash
nvidia-smi
export DEV=<优先空闲4090，其次3090，最后3080Ti>
```

smoke 命令：

```bash
cd /data1/yuzhixiang/work/Co-teaching-MoreNets
source /data1/yuzhixiang/opt/miniconda3/etc/profile.d/conda.sh
conda activate /data1/yuzhixiang/.conda/envs/coteaching-py39

CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=$DEV python -u main.py \
  --dataset cifar10 --noise_type symmetric --noise_rate 0.4 \
  --num_models 3 --q_mode hybrid --q_usage_mode diagnostic_only \
  --mstep_mode hard --sam_rho 0 --replay_size 0 --replay_ratio 0 \
  --lambda_mode accuracy --lambda_patience 9999 \
  --batch_size 512 --num_workers 8 --prefetch_factor 4 --drop_last \
  --n_epoch 3 --num_iter_per_epoch 20 \
  --result_dir results_diag/q_isolation_smoke
```

通过标准：

- 训练完成。
- JSON training log 中包含 `q_entropy`、`q_clean_auc`、`selected_clean_rate`。
- `diagnostic_only` 下 `pi_t` 应基本保持 `pi_init`，不应被 Q 更新。

## 4. Phase B 主实验矩阵

固定公共参数：

```bash
--dataset cifar10 --noise_type symmetric --noise_rate 0.4
--q_mode hybrid
--sam_rho 0
--replay_size 0 --replay_ratio 0
--lambda_mode accuracy --lambda_patience 9999
--batch_size <按模型数选择> --num_workers 8 --prefetch_factor 4 --drop_last
--n_epoch 30 --num_iter_per_epoch 100
```

### B1: 模型数与 Q 诊断

```bash
--num_models <2|3|5> --mstep_mode hard --q_usage_mode diagnostic_only
```

目的：检查 Q 在不反馈训练时是否仍会饱和或失去区分性。

预期：

- 若 `q_clean_auc` 有效但 `standard` 崩，说明 usage feedback 是主因。
- 若 diagnostic-only 已崩，说明 posterior construction 有结构性偏差。

### B2: Prior feedback

```bash
--num_models <2|3|5> --mstep_mode hard --q_usage_mode prior_only
```

目的：检查 `Q -> pi_t -> Q` 是否足以造成饱和。

预期：

- 若 `prior_only` 迅速 `q_mean -> 1`，则 prior feedback 是关键病灶。

### B3: Selection-only

```bash
--num_models <2|3|5> --mstep_mode hard --q_usage_mode selection_only
```

目的：检查 Q 能否替代 hard small-loss selection。

预期：

- 若 `selected_clean_rate` 和 acc 明显低于 hard baseline，则 Q 不能做主 gate。

### B4: Weight-only

```bash
--num_models <2|3|5> --mstep_mode soft --q_usage_mode weight_only
```

目的：检查 Q 作为 soft weight 是否比 hard gate 更容易放大错误样本。

预期：

- 若 `q_clean_auc` 尚可但 acc 掉，问题在 usage policy。
- 若 `q_clean_auc` 也低，问题在 posterior construction。

## 5. 推荐启动顺序

先不全量跑完整矩阵。按信息增益排序：

1. smoke：3 models, diagnostic-only, 3 epochs，batch 256。
2. B1：2/3 models diagnostic-only，seed1。
3. B2：2/3 models prior-only，seed1。
4. B3：2/3 models selection-only，seed1。
5. 若现象清楚，再扩展到 5 models 和 seeds 2/3。

## 6. 判定规则

- 不能用单 seed 结束方向。
- 不能只看 accuracy，必须看 `q_clean_auc` 与 `selected_clean_rate`。
- 如果 diagnostic-only 健康而 prior/weight/selection 崩，说明 Q 本身和 Q 用法要分开。
- 如果 3 models 比 2 models 更坏，必须结合 overlap/disagreement 判断是否同步错误共识。
- 如果 5 models 更差，不等于多模型失败，可能只是同构同步投票冗余。

## 7. Batch 默认值

Q isolation 会同时持有多个模型的 forward graph，显存随模型数近似线性增长。当前远端 smoke 已验证 3 models + batch 512 会在 4090 上 OOM，因此默认值调整为：

| 模型数 | 4090/3090 batch |
|---:|---:|
| 2 | 512 |
| 3 | 256 |
| 5 | 128 |

如果后续加入诊断 oracle 或 target construction，应优先再降一档，而不是用 OOM 后的结果判断算法失败。
