# Stage-4: Q-Gated Peer Selection

date: 2026-05-09

## 目标

上轮 Q isolation 的关键发现是：`hybrid Q` 本身有很强的 clean/reliability 排序能力，但当它作为全样本连续训练权重进入 soft/robust M-step，并进一步与 `q_global`、$\pi_t$、teacher consistency 形成闭环时，Q 的语义会从“样本是否可靠”退化成“模型是否已经相信该样本”。

因此本阶段不继续调 `q_gamma`、`q_weight_min/max` 或 $\pi_t$，而是从机制上切断危险路径。

核心问题：

> 能否只把 Q 用作可靠候选池，而不把 Q 作为连续 loss 权重，从而保留 Q 的 clean ranking 信息，同时避免 Q collapse？

## 方法

新增：

```bash
--q_usage_mode gate_selection
--q_gate_pool_mult 1.25
```

训练流程：

1. 每个 batch 先按当前 `q_mode` 计算 $q_i$。
2. 取 Q top `ceil(q_gate_pool_mult * k)` 作为可靠候选池，其中 $k = ceil(remember_rate * batch_size)$。
3. 每个模型不直接吃相同 Q top-k，而是在候选池内使用 peer aggregate loss 做 hard top-k selection。
4. loss 仍然是 hard CE，不使用 Q 连续权重。
5. 默认不让该模式更新 $\pi_t$，不写入 `q_global`，不进入 replay admission。

形式上：

$$
C_t = \operatorname{TopK}_{i}(q_i, \lceil \alpha k \rceil)
$$

$$
S_{m,t} = \operatorname{TopK}_{i \in C_t}(-\ell_{-m,i}, k)
$$

其中 $\alpha = $ `q_gate_pool_mult`，$\ell_{-m,i}$ 是排除当前模型后的 peer aggregate loss。

## 为什么这样设计

`selection_only` 已经证明 Q hard-gate 可用，但它让所有模型使用完全相同的 Q top-k，`overlap=1.0`，会削弱多模型差异。

`weight_only` 和 `standard` 已经证明连续权重路径危险。它们把 $q_i$ 放进梯度尺度：

$$
\nabla_\theta L_t = \sum_i w_i(q_i) \nabla_\theta \ell_i
$$

这会让模型拟合噪声后反过来提高噪声样本的 Q，形成确认反馈。

`gate_selection` 的设计是：

- Q 只回答“是否进入可靠候选池”。
- peer loss 仍回答“哪个模型应学习哪些候选样本”。
- 最终 loss 是离散 hard gate，避免连续放大错误 posterior。
- 候选池比最终选择集略大，保留 per-model selection 差异。

## 第一轮实验

先跑 seed 1 的短跑，与已有 Q isolation 结果对齐。

```powershell
powershell -ExecutionPolicy Bypass -File tools/remote-workflow/run_q_collapse_isolation.ps1 `
  -Mode gate -NumModels 2 -Seed 1 -SkipGitSync -SkipRemoteGitSync -FetchResults

powershell -ExecutionPolicy Bypass -File tools/remote-workflow/run_q_collapse_isolation.ps1 `
  -Mode gate -NumModels 3 -Seed 1 -SkipGitSync -SkipRemoteGitSync -FetchResults
```

远端实际训练命令将使用：

```bash
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=<idle 4090 first> python -u main.py \
  --dataset cifar10 --noise_type symmetric --noise_rate 0.4 \
  --num_models <2|3> --q_mode hybrid --q_usage_mode gate_selection --mstep_mode hard \
  --q_gate_pool_mult 1.25 \
  --sam_rho 0 --replay_size 0 --replay_ratio 0 \
  --lambda_mode accuracy --lambda_patience 9999 --min_active 2 \
  --batch_size <512 for 2 models, 256 for 3 models> \
  --num_workers 8 --prefetch_factor 4 --drop_last \
  --n_epoch 30 --num_iter_per_epoch 100 --num_gradual 10 --epoch_decay_start 80 \
  --val_split 0.1 --seed 1
```

## 判据

支持该方向的结果：

- `q_mean` 不向 1 饱和，`q_std` 保持非零。
- `q_clean_auc` 接近 diagnostic/selection-only 水平。
- `selected_clean_rate` 不低于 hard baseline / selection-only。
- `overlap` 低于 `selection_only=1.0`，说明保留了模型差异。
- last5 acc 不低于 `selection_only`，理想情况下接近或超过 hard small-loss baseline。

削弱该方向的结果：

- Q 虽不 collapse，但 accuracy 明显低于 selection-only。
- overlap 仍接近 1.0，说明候选池太窄或 peer loss 无法恢复模型差异。
- selected clean rate 明显下降，说明 Q candidate pool 过宽或 Q gate 不能保护可靠性。

## 后续调整

若 `q_gate_pool_mult=1.25` 候选池太窄，导致 overlap 仍接近 1.0，则尝试 `1.50`。

若候选池太宽，导致 selected clean rate 明显下降，则尝试 `1.10` 或按 epoch 动态调节。

只有当 `gate_selection` 在 seed 1 不伤害 baseline，才进入 seeds 2/3。若它失败，应回到更保守的 `selection_only + diversity regularization` 或 `hard small-loss + Q diagnostic`，而不是重新启用 Q continuous weight。

