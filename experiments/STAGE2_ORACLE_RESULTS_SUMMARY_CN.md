# Stage-2 Oracle Diagnostic Results

## 实验目的

本轮不追求直接刷新 accuracy，而是验证一个更基础的问题：

> 在 peer-selected small-loss 的可靠样本集合内部，`sam_gap` 是否能排序真实的一步 validation utility？

如果答案是否定的，就不应继续把 `sam_gap` 作为主要 utility proxy 加权训练。

## 实现摘要

新增 `--diag_oracle` 诊断模式，默认不改变训练更新路径。

诊断流程：

1. 正常训练一个 epoch。
2. 每隔若干 epoch 取少量 train batch 与 validation batch。
3. 用 peer loss 得到 selected set。
4. 只在 selected set 内抽样候选样本。
5. 冻结 backbone feature，只模拟最后一层的一步更新。
6. 记录该单样本边际更新对 validation CE 的实际改善：

$$
\Delta_i = L_{\mathrm{val}}(\theta) - L_{\mathrm{val}}(\theta - \eta_i g_i)
$$

其中 $\Delta_i > 0$ 表示该样本的一步更新降低 validation loss。

重要修正：

最初 smoke 使用完整 optimizer lr 作为单样本 oracle step，导致几乎所有候选样本都使 validation loss 大幅变差。原因是训练 loss 对 selected set 取平均，单个样本的边际贡献尺度应接近 $\eta/k$，而不是 $\eta$。因此当前实现中：

```text
--diag_oracle_lr 0
```

表示使用当前 optimizer lr 除以 selected count。

## 运行设置

- Host: `b101`
- Repo: `/data1/yuzhixiang/work/Co-teaching-MoreNets`
- Dataset: CIFAR-10, symmetric noise 0.4
- Models: 2
- Base method: hard small-loss + SAM/SRIT-style, SGD momentum
- `batch_size=512`
- `val_split=0.1`
- `n_epoch=30`
- `num_iter_per_epoch=100`
- Oracle epochs: 5, 10, 15, 20, 25
- Oracle samples: 2560 per run per target

Runs:

| run | GPU | utility |
|---|---:|---|
| `oracle_e1_baseline_seed1` | 4090 device 4 | none |
| `oracle_e2_sam_gap_s05_seed1` | 3090 device 2 | `sam_gap`, strength 0.5 |

## GPU / speed observation

| run | max mem MiB | avg mem MiB | avg GPU util |
|---|---:|---:|---:|
| baseline | 18188 | 17971.7 | 95.11% |
| `sam_gap` strength 0.5 | 18186 | 18033.2 | 95.66% |

结论：`batch_size=512` 继续是合适的快速默认设置。它稳定使用约 18GB 显存，并把 GPU util 推到约 95%。

## Accuracy 结果

| run | best | last | last5 mean |
|---|---:|---:|---:|
| baseline | 80.11 | 79.57 | 79.486 |
| `sam_gap` strength 0.5 | 79.98 | 79.98 | 79.566 |

解释：

- 两者基本持平。
- `sam_gap` strength 0.5 没有明显破坏 baseline。
- 但也没有形成足够清晰的 accuracy 增益。
- 这与上一轮结果一致：轻量 `sam_gap` 是“安全但未证明有用”，不是主方法证据。

## Oracle 结果：clean target

| metric | baseline | `sam_gap` strength 0.5 |
|---|---:|---:|
| `spearman_align_adam_oracle` | 0.9875 | 0.9876 |
| `spearman_sam_utility_oracle` | -0.1111 | -0.1094 |
| `spearman_loss_oracle` | -0.0281 | -0.0030 |
| `auc_loss_clean` | 0.8789 | 0.8690 |
| `auc_sam_utility_clean` | 0.5322 | 0.4970 |
| `auc_oracle_clean` | 0.4645 | 0.5734 |

关键观察：

1. `align_adam` 与 oracle 几乎完全正相关。这是 sanity check，不是意外发现：当前 oracle 是最后层小步 validation utility，因此一阶 validation-gradient alignment 本来就应接近该 oracle。
2. `sam_gap` 与 oracle 稳定负相关，约 -0.11。
3. small-loss 对 clean/noisy 区分仍然很强，`auc_loss_clean` 约 0.87，但它几乎不排序 oracle utility。

这说明：

- small-loss 回答“这个样本是否像 clean / 是否容易拟合”。
- oracle utility 回答“这个样本的一步更新是否改善目标 validation loss”。
- 两者不是同一个问题。
- `sam_gap` 当前既不是好的 clean detector，也不是好的 one-step utility proxy。

## Oracle 结果：noisy target

| metric | baseline | `sam_gap` strength 0.5 |
|---|---:|---:|
| `spearman_align_adam_oracle` | 0.9628 | 0.9619 |
| `spearman_sam_utility_oracle` | -0.0811 | -0.0567 |
| `spearman_loss_oracle` | 0.0284 | 0.0245 |
| `auc_loss_clean` | 0.8789 | 0.8690 |
| `auc_sam_utility_clean` | 0.5322 | 0.4970 |

noisy target 下也一样：

- alignment 与对应 oracle 高相关。
- `sam_gap` 不排序 oracle。
- small-loss 继续偏向 clean reliability，而不是 target utility。

## 阶段性判断

### 支持的判断

- Oracle 诊断实现是有效的：小步最后层 oracle 与一阶 alignment 高度一致。
- `batch_size=512` 是当前远端短实验的合理快速配置。
- clean reliability 与 target utility 必须分开讨论。
- `sam_gap` 不应继续作为主要 utility proxy。

### 不支持的判断

- 当前结果不支持“`sam_gap` 是 Shapley-like utility 的有效近似”。
- 当前结果不支持“直接在 selected set 内用 `sam_gap` 加权能带来稳定收益”。
- 当前结果也不支持“small-loss 可以替代 utility 评估”。small-loss 只是 reliability signal。

### 不能推出的判断

- 不能推出 Data Shapley / optimizer-aware utility 方向无效。
- 不能推出 validation-gradient alignment 可直接用于最终方法，因为 clean validation target 在真实 noisy / continual setting 中不可直接获得。
- 不能推出当前最后层 oracle 足以代表完整训练动态。它只是低成本、可证伪的第一层诊断。

## 下一步建议

停止把 `sam_gap` 当作主线 proxy 继续调参。下一步应转向更接近研究问题本体的两条路线之一：

1. **Target construction problem**
   - 既然 alignment 能排序当前 oracle，关键问题变成：没有 clean validation target 时，如何构造可信 target gradient？
   - 候选：peer ensemble、delayed EMA teacher、purified buffer、high-consensus clean subset。

2. **Stronger oracle problem**
   - 当前 oracle 是最后层小步近似，过于接近一阶 alignment。
   - 下一层可以做 group-level / mini-batch counterfactual utility，或 SAM/optimizer-aware oracle，检查 proxy 是否在更真实更新下仍有效。

推荐优先路线：

先做 target construction diagnostic，而不是继续修 `sam_gap`。原因是本轮已经说明 `sam_gap` 不是关键；而 utility 方向真正的瓶颈是“目标梯度从哪里来”。

## Artifacts

Local:

- `remote_results/stage2_oracle/oracle_seed1_summary.json`
- `remote_results/stage2_oracle/oracle_seed1_accuracy_summary.json`
- `remote_results/stage2_oracle/e1_baseline_oracle_summary.json`
- `remote_results/stage2_oracle/e2_sam_gap_s05_oracle_summary.json`
- `remote_results/stage2_oracle/e1_baseline_training_log.json`
- `remote_results/stage2_oracle/e2_sam_gap_s05_training_log.json`
- `remote_results/stage2_oracle/e1_baseline_nvsmi.csv`
- `remote_results/stage2_oracle/e2_sam_gap_s05_nvsmi.csv`

Remote:

- `results_diag/stage2_oracle/e1_baseline_seed1`
- `results_diag/stage2_oracle/e2_sam_gap_s05_seed1`
- `results_stage2/oracle_e1_baseline_seed1`
- `results_stage2/oracle_e2_sam_gap_s05_seed1`
