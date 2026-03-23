# 实验对比指南

## 📊 渐进式实验路线图

按照科学实验的**单变量原则**，建议按以下顺序逐步深入：

```
Stage 1: 基线验证
    ↓
Stage 2: Q计算方式 (posterior vs loss)
    ↓
Stage 3: Q-Gamma混合 (hard vs soft)
    ↓
Stage 4: Replay Buffer (流式稳定性)
    ↓
Stage 5: 模型数量 & Soft-Absorb
    ↓
Stage 6: 探索采样 (多样性)
    ↓
Stage 7: 完整系统消融
```

---

## 🚀 快速开始

### 1. 首先测试可视化工具

```powershell
cd d:\study_file\Co-teaching
python experiments/visualize.py --demo
```

这会生成模拟数据的可视化图表，确认绑定工作正常。

### 2. 运行基线实验 (≈30分钟)

```powershell
# 原始Co-teaching (所有新功能关闭)
python main.py --dataset cifar10 --noise_rate 0.5 --noise_type symmetric --q_gamma 0 --replay_size 0 --explore_delta 0 --n_epoch 100
```

### 3. 运行EM系统 (≈30分钟)

```powershell
# 完整EM系统
python main.py --dataset cifar10 --noise_rate 0.5 --noise_type symmetric --q_mode posterior --q_gamma 0.5 --replay_size 5000 --replay_ratio 0.2 --explore_delta 0.1 --n_epoch 100
```

### 4. 可视化对比

```powershell
python experiments/visualize.py --compare results/cifar10/srit/
```

---

## 📋 各阶段详细说明

### Stage 1: 基线验证

**目标**: 确认原始Co-teaching在当前代码中能正常工作

| 实验 | 命令参数 | 预期结果 |
|------|----------|----------|
| sym-20% | `--noise_rate 0.2 --q_gamma 0 --replay_size 0` | ~82% |
| sym-50% | `--noise_rate 0.5 --q_gamma 0 --replay_size 0` | ~65% |
| asym-40% | `--noise_rate 0.4 --noise_type asymmetric --q_gamma 0` | ~70% |

### Stage 2: Q计算方式

**目标**: 比较EM理论的posterior-q vs 工程方案的loss-q

| 实验 | 关键参数 | 理论意义 |
|------|----------|----------|
| posterior-q | `--q_mode posterior` | 基于噪声混合模型的后验概率 |
| loss-q | `--q_mode loss` | 基于损失的sigmoid变换 |

**期望发现**: posterior-q在高噪声率下应更稳定

### Stage 3: Q-Gamma混合

**目标**: 找到hard selection和soft weighting的最佳平衡

| γ值 | 行为 | 适用场景 |
|-----|------|----------|
| 0.0 | 纯hard selection (原始co-teaching) | 基线对照 |
| 0.3 | 轻度soft | 保守改进 |
| 0.5 | 平衡 | 推荐默认值 |
| 0.7 | 重度soft | 高噪声场景 |
| 1.0 | 纯Q加权 (无selection) | 消融极端 |

### Stage 4: Replay Buffer

**目标**: 验证流式场景下记忆机制的价值

```powershell
# 无replay
python main.py --replay_size 0 ...

# 小buffer (1K)
python main.py --replay_size 1000 --replay_ratio 0.1 ...

# 中buffer (5K) - 推荐
python main.py --replay_size 5000 --replay_ratio 0.2 ...
```

### Stage 5: 模型数量 & Soft-Absorb

**目标**: 验证委员会规模和弱模型处理策略

| 配置 | 参数 | 说明 |
|------|------|------|
| 2模型无absorb | `--num_models 2 --lambda_active 0` | 原始双模型 |
| 3模型+soft absorb | `--num_models 3 --lambda_active 0.4 --lambda_patience 3` | 推荐配置 |
| 5模型+soft absorb | `--num_models 5 --lambda_active 0.4` | 计算换精度 |

### Stage 6: 探索采样

**目标**: 验证高熵样本注入对多样性的影响

```powershell
# 关闭探索
python main.py --explore_delta 0 ...

# 5%探索
python main.py --explore_delta 0.05 --explore_trigger 0.8 ...

# 10%探索
python main.py --explore_delta 0.1 --explore_trigger 0.8 ...
```

### Stage 7: 完整系统消融

**目标**: 逐一移除组件，验证各组件贡献

| 实验 | 移除组件 | 参数设置 |
|------|----------|----------|
| Full System | 无 | 全部开启 |
| -Replay | 移除replay | `--replay_size 0` |
| -Explore | 移除探索 | `--explore_delta 0` |
| -Soft Q | 移除soft加权 | `--q_gamma 0` |
| -Absorb | 移除soft absorb | `--lambda_active 0` |

---

## 📈 可视化功能

### 单实验仪表盘

```powershell
python experiments/visualize.py --log_dir results/cifar10/srit/
```

生成4个子图:

- (a) Accuracy曲线 (train/val/test)
- (b) Loss曲线
- (c) Q统计 (mean ± std, π_t)
- (d) Overlap & Active Models

### 多实验对比

```powershell
python experiments/visualize.py --compare exp1_dir exp2_dir exp3_dir
```

### 消融热力图

在 visualize.py 中调用 `plot_ablation_heatmap()`:

```python
results = {
    ("γ=0.0", "sym-50%"): 65.1,
    ("γ=0.5", "sym-50%"): 72.4,
    ...
}
plot_ablation_heatmap(results, ...)
```

### 论文级图表

```powershell
python experiments/visualize.py --paper --compare exp1 exp2 exp3
```

生成PDF格式的出版级图表。

---

## 🎯 关键指标解读

| 指标 | 理想趋势 | 异常信号 |
|------|----------|----------|
| Test Acc | 稳步上升 | 早期下降/剧烈波动 |
| Q Mean | 从0.5逐渐上升到0.7+ | 过快到1.0 (过拟合) |
| Q Std | 早期大,后期收敛 | 始终很大 (不稳定) |
| Overlap | 中等(0.5-0.8) | 过高(>0.95)=共识崩塌 |
| π_t | 接近(1-noise_rate) | 偏离过大 |
| Active Models | 稳定或缓慢下降 | 快速降到min |

---

## ⚡ 批量运行

使用ablation_runner.py自动化：

```powershell
# 查看所有实验
python experiments/ablation_runner.py --list

# 运行特定阶段
python experiments/ablation_runner.py --stage stage3_q_gamma

# 空跑检查命令
python experiments/ablation_runner.py --stage stage1_baseline --dry-run

# 运行全部
python experiments/ablation_runner.py --all
```

---

## 📁 结果文件结构

运行后会生成：

```
results/
└── cifar10/
    └── srit/
        ├── cifar10_srit_symmetric_0.5.txt          # CSV格式日志
        └── cifar10_srit_symmetric_0.5_training_log.json  # JSON格式(可视化用)

experiments/
├── results/           # ablation_runner的输出
│   └── *.json
└── figures/           # 可视化输出
    ├── dashboard_*.png
    ├── comparison_*.png
    └── paper_figure_*.pdf
```
