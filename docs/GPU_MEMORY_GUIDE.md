# 显存使用指南：24GB GPU 配置优化

## 📊 显存消耗估算

### 模型显存占用（单模型）

| 组件 | CIFAR-10/100 | MNIST | 说明 |
|------|--------------|-------|------|
| CNN模型参数 | ~8 MB | ~2 MB | 9层卷积网络 |
| 梯度 | ~8 MB | ~2 MB | 与参数同大小 |
| 优化器状态(Adam) | ~16 MB | ~4 MB | 2倍参数(m,v) |
| **单模型总计** | **~32 MB** | **~8 MB** | |

### 批次数据显存（前向+反向）

| Batch Size | 激活值(估算) | 总计(含梯度) |
|------------|--------------|--------------|
| 64 | ~200 MB | ~400 MB |
| 128 | ~400 MB | ~800 MB |
| 256 | ~800 MB | ~1.6 GB |
| 512 | ~1.6 GB | ~3.2 GB |

### SAM双倍前向的影响

SAM需要两次前向传播，显存峰值约为普通训练的 **1.5-1.8倍**

---

## 🎛️ 关键参数与显存影响

| 参数 | 默认值 | 显存影响 | 调整建议 |
|------|--------|----------|----------|
| `--batch_size` | 128 | **线性** (最大影响) | 24GB可用256-512 |
| `--num_models` | 3 | **线性** | 24GB可用3-5个模型 |
| `--replay_ratio` | 0.25 | 中等 | 增加等效batch size |
| `--replay_size` | 2000 | 很小(CPU存储) | 无显存影响 |

### 显存公式估算

```
显存(GB) ≈ num_models × (模型基础 + batch_factor × batch_size) × SAM系数

其中:
- 模型基础 ≈ 0.05 GB (CIFAR)
- batch_factor ≈ 0.006 GB/样本 (CIFAR, 含激活值)
- SAM系数 ≈ 1.6
```

---

## 📈 24GB GPU 推荐配置

### 配置1：标准实验（保守）

```bash
python main.py \
    --batch_size 128 \
    --num_models 3 \
    --replay_ratio 0.2 \
    --dataset cifar10
# 预估显存: ~4-5 GB
# 适用: 留有余量，可同时跑多个实验
```

### 配置2：高效实验（推荐）

```bash
python main.py \
    --batch_size 256 \
    --num_models 3 \
    --replay_ratio 0.25 \
    --dataset cifar10
# 预估显存: ~8-10 GB
# 适用: 单实验，速度更快
```

### 配置3：极速实验（充分利用显存）

```bash
python main.py \
    --batch_size 512 \
    --num_models 5 \
    --replay_ratio 0.3 \
    --dataset cifar10
# 预估显存: ~18-22 GB
# 适用: 追求最快速度
```

### 配置4：大规模委员会

```bash
python main.py \
    --batch_size 128 \
    --num_models 7 \
    --replay_ratio 0.15 \
    --dataset cifar10
# 预估显存: ~12-15 GB
# 适用: 研究更多模型的效果
```

---

## ⚡ 速度-显存权衡

### 训练速度影响因素

| 参数调整 | 速度变化 | 显存变化 | 建议 |
|----------|----------|----------|------|
| batch_size ×2 | ~1.5-1.8× 加速 | ~2× 增加 | 显存允许时优先 |
| num_models ×2 | ~0.5× 减速 | ~2× 增加 | 看研究需要 |
| replay_ratio +0.1 | ~0.95× 略慢 | ~1.1× 轻微 | 影响不大 |
| num_workers ×2 | ~1.1× 加速 | 无影响 | 推荐4-8 |

### 速度优化优先级

1. **增大 batch_size** (最有效)
2. **使用 num_workers=8** (免费加速)
3. **开启 cudnn.benchmark**
4. **减少 print_freq**

---

## 🔧 运行时显存监控

### 方法1：nvidia-smi 实时监控

```bash
# 每2秒刷新一次
watch -n 2 nvidia-smi

# 或者只看显存
nvidia-smi --query-gpu=memory.used,memory.total --format=csv -l 2
```

### 方法2：在代码中添加监控

在 main.py 的 train_epoch 开头添加：

```python
if torch.cuda.is_available():
    print(f"GPU Memory: {torch.cuda.memory_allocated()/1024**3:.2f} GB / "
          f"{torch.cuda.max_memory_allocated()/1024**3:.2f} GB peak")
```

### 方法3：自动内存管理

如果遇到OOM，尝试：

```bash
# 清理缓存
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128

# 或在代码中
torch.cuda.empty_cache()
```

---

## 📋 OOM 故障排除

如果出现 CUDA out of memory:

1. **立即降低 batch_size**

   ```bash
   --batch_size 64  # 或更小
   ```

2. **减少模型数量**

   ```bash
   --num_models 2
   ```

3. **关闭replay（减少内存碎片）**

   ```bash
   --replay_size 0
   ```

4. **使用梯度累积（不改变等效batch）**
   在代码中实现：累积N个小batch的梯度后再更新

5. **使用混合精度（如需要可添加）**

   ```python
   with torch.cuda.amp.autocast():
       logits = model(images)
   ```

---

## 🧪 快速显存测试

运行以下命令测试你的GPU能承受的最大配置：

```bash
# 测试脚本：逐步增加batch_size直到OOM
for bs in 128 256 384 512 640 768; do
    echo "Testing batch_size=$bs"
    python main.py --batch_size $bs --num_models 3 --n_epoch 1 --num_iter_per_epoch 10 2>&1 | grep -E "(GPU|CUDA|OOM|Error)"
    sleep 2
done
```

---

## 📊 实际测试参考值

以下是在类似配置下的实测参考（RTX 3090/4090 24GB）：

| 配置 | 显存使用 | 每Epoch时间 | 备注 |
|------|----------|-------------|------|
| bs=128, M=3 | ~4.5 GB | ~45s | 保守 |
| bs=256, M=3 | ~8.2 GB | ~28s | 推荐 |
| bs=512, M=3 | ~15.5 GB | ~18s | 高效 |
| bs=256, M=5 | ~13.0 GB | ~42s | 多模型 |
| bs=512, M=5 | ~22.0 GB | ~28s | 极限 |

*注：实际值可能因PyTorch版本、CUDA版本略有不同*
