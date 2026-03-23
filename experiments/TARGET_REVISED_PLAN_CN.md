# Target Revised 实验计划

## 1. 使用约定

- 推荐分两种同步模式：

### 模式 A：本地一键 push + 远端自动 pull

- 本地修改代码后，同步到远端：
  ```powershell
  powershell -ExecutionPolicy Bypass -File tools/remote-workflow/sync_and_update_remote.ps1
  ```
- 如果当前工作区还有未提交改动，但你确认要一起同步：
  ```powershell
  powershell -ExecutionPolicy Bypass -File tools/remote-workflow/sync_and_update_remote.ps1 -AutoCommit -Message "exp: <简短说明>"
  ```
- 说明：
  - 这个模式会从本地再发起一次新的 `ssh` 连接去远端更新仓库。
  - 适合你不想切到远端窗口手动执行命令时使用。

### 模式 B：你已经开着远端 SSH 窗口，推荐用这个

- 本地只负责提交和 push：
  ```powershell
  powershell -ExecutionPolicy Bypass -File tools/remote-workflow/sync_branch.ps1 -AutoCommit -Message "exp: <简短说明>"
  ```
- 然后在你已经打开的远端终端里执行：
  ```bash
  cd /data1/yuzhixiang/work/Co-teaching-MoreNets
  git fetch origin
  git checkout <分支名>
  git pull --ff-only origin <分支名>
  ```
- 说明：
  - 这是当前更推荐的模式。
  - 它不会再新开一条额外的远端命令链，和你现有的 VS Code Remote SSH 使用习惯更一致。

- 远端指定设备运行时，用：
  ```bash
  CUDA_VISIBLE_DEVICES=<设备号> python -u main.py ...
  ```
- 结果文件默认在：
  - `results/<dataset>/srit/*.txt`
  - `results/<dataset>/srit/*_training_log.json`

## 2. 当前实现对应关系

- `q_mode=hybrid`
  - 含义：`Q_i` 由预测证据、margin、teacher consistency、loss rank 共同决定。
- `mstep_mode=hard|soft|robust`
  - `hard`：接近经典 top-k/硬筛选。
  - `soft`：仅按 `Q_i` 做软加权监督。
  - `robust`：`Q_i` 加权 CE + EMA teacher KL，再加 SAM。
- `replay_mode=legacy|purified`
  - `legacy`：旧式阈值 buffer。
  - `purified`：candidate buffer + purified memory，两层记忆。
- `lambda_mode=accuracy|proxy`
  - `accuracy`：旧式精度差更新。
  - `proxy`：按 sharp gap / disagreement / stability / memory alignment 更新。

## 3. 实验顺序

建议按下面顺序推进，不要一上来全开。

### 实验 0：同步与环境冒烟

- 指令：
  ```powershell
  powershell -ExecutionPolicy Bypass -File tools/remote-workflow/sync_and_update_remote.ps1 -AutoCommit -Message "exp: smoke sync"
  ```
- 远端测试：
  ```bash
  CUDA_VISIBLE_DEVICES=<设备号> python -u main.py --help
  ```
- 目的：
  - 确认本地 push、远端 pull、远端环境导入都正常。
- 预期观察：
  - 不应出现 git 冲突、缺包、参数解析报错。
- 微调建议：
  - 如果 `--help` 都无法运行，先不要开始训练，先修环境或 import 依赖。

### 实验 1：硬筛选基线

- 指令：
  ```bash
  CUDA_VISIBLE_DEVICES=<设备号> python -u main.py --dataset cifar10 --noise_type symmetric --noise_rate 0.4 --num_models 3 --q_mode loss --mstep_mode hard --sam_rho 0 --replay_size 0 --replay_ratio 0 --lambda_mode accuracy --lambda_patience 9999
  ```
- 目的：
  - 作为 classic Co-teaching 风格近似基线。
- 预期观察：
  - `q_mean` 和 `remember_rate` 比较接近。
  - `q_std` 有一定分化，但不够稳定。
  - `overlap` 中后期偏高。
- 微调建议：
  - 如果训练不稳，先减小 `lr`。
  - 如果过早塌缩，可增大 `num_gradual`。

### 实验 2：Soft Q + ERM

- 指令：
  ```bash
  CUDA_VISIBLE_DEVICES=<设备号> python -u main.py --dataset cifar10 --noise_type symmetric --noise_rate 0.4 --num_models 3 --q_mode hybrid --mstep_mode soft --sam_rho 0 --replay_size 0 --replay_ratio 0 --lambda_mode proxy
  ```
- 目的：
  - 单独验证“多证据慢变量 `Q_i`”是否优于硬筛选。
- 预期观察：
  - `q_std` 应比实验 1 更平滑。
  - `pi_t` 波动更小。
  - `TestAcc` 应不低于实验 1，通常略高。
- 微调建议：
  - 如果 `q_mean` 长期过高，调低 `q_pred_weight` 或调高 `q_temp_max`。
  - 如果 `q_mean` 长期过低，调高 `q_margin_weight` 或 `q_consistency_weight`。

### 实验 3：Soft Q + SAM

- 指令：
  ```bash
  CUDA_VISIBLE_DEVICES=<设备号> python -u main.py --dataset cifar10 --noise_type symmetric --noise_rate 0.4 --num_models 3 --q_mode hybrid --mstep_mode soft --sam_rho 0.05 --replay_size 0 --replay_ratio 0 --lambda_mode proxy
  ```
- 目的：
  - 只看 SAM 对稳定性的影响。
- 预期观察：
  - `sharp_gap_m*` 会大于 0。
  - `TestAcc` 常略优于实验 2。
  - 若 `SAM` 生效，`overlap` 不应异常飙升。
- 微调建议：
  - 如果收敛变慢或精度下降，先把 `sam_rho` 降到 `0.02`。
  - 如果 `sharp_gap` 几乎为 0，说明 `rho` 太小或模型太浅。

### 实验 4：Robust M-step

- 指令：
  ```bash
  CUDA_VISIBLE_DEVICES=<设备号> python -u main.py --dataset cifar10 --noise_type symmetric --noise_rate 0.4 --num_models 3 --q_mode hybrid --mstep_mode robust --supervised_alpha 0.7 --sam_rho 0.05 --replay_size 0 --replay_ratio 0 --lambda_mode proxy
  ```
- 目的：
  - 验证 EMA teacher 软监督是否比纯软权重更稳。
- 预期观察：
  - `stability_m*` 应高于实验 3。
  - `disagreement_m*` 应更低。
  - `lambda_m` 的分化应更合理，不会剧烈振荡。
- 微调建议：
  - 如果 teacher 约束太强，调高 `supervised_alpha` 到 `0.8~0.9`。
  - 如果 soft target 作用不明显，调低 `supervised_alpha` 到 `0.6`。

### 实验 5：Purified Memory

- 指令：
  ```bash
  CUDA_VISIBLE_DEVICES=<设备号> python -u main.py --dataset cifar10 --noise_type symmetric --noise_rate 0.4 --num_models 3 --q_mode hybrid --mstep_mode robust --sam_rho 0.05 --replay_mode purified --replay_size 2000 --replay_candidate_size 4000 --replay_ratio 0.25 --replay_admission 0.7 --replay_utility 0.75 --replay_stability 3 --lambda_mode proxy
  ```
- 目的：
  - 验证 candidate + purified memory 是否优于无 replay 或 legacy replay。
- 预期观察：
  - `replay_size` 逐步增长，而不是瞬间灌满。
  - `replay_mean_u` 中后期稳定上升。
  - 若记忆系统有效，`TestAcc` 应优于实验 4，或至少后期退化更少。
- 微调建议：
  - 如果 `replay_size` 长期很小：降低 `replay_utility` 或 `replay_admission`。
  - 如果 `replay_size` 很快灌满且效果差：提高 `replay_stability`，或提高 `replay_utility`。
  - 如果 memory 被头部类别垄断：增大 `replay_coverage_weight`，减小 `replay_redundancy_weight`。

### 实验 6：旧 replay 对照

- 指令：
  ```bash
  CUDA_VISIBLE_DEVICES=<设备号> python -u main.py --dataset cifar10 --noise_type symmetric --noise_rate 0.4 --num_models 3 --q_mode hybrid --mstep_mode robust --sam_rho 0.05 --replay_mode legacy --replay_size 2000 --replay_ratio 0.25 --replay_tau 0.8 --lambda_mode proxy
  ```
- 目的：
  - 直接对比旧 buffer 与 purified memory。
- 预期观察：
  - `legacy` 模式通常更容易早期灌库。
  - 中后期若出现自污染，实验 6 的表现应劣于实验 5。
- 微调建议：
  - 如果两者差异不明显，优先检查 `replay_ratio` 是否过小，或 `Q_i` 是否过于保守。

## 4. 推荐先看的指标

- `TestAcc_Mk` / `Ensemble`
  - 看最终效果和是否有 ensemble 增益。
- `q_mean`, `q_std`
  - 看 `Q_i` 是否塌成全高或全低。
- `overlap`
  - 看委员会是否过早同质化。
- `lambda_m`
  - 看 proxy 管理是否出现一边倒压死某个模型。
- `replay_size`, `replay_mean_u`
  - 看记忆系统是否在“净化”，还是只是机械扩容。

## 5. 先后优先级

建议只按下面顺序推进：

1. 实验 1 和实验 2
2. 实验 3
3. 实验 4
4. 实验 5 和实验 6
5. 在上面稳定后，再做 stream / drift 协议改造

## 6. 常见异常解释

- `q_mean` 很高且 `q_std` 很低
  - 说明 `Q_i` 过度乐观，接近全判 clean。
- `overlap` 很快接近 1
  - 说明委员会失去分歧，E-step 温度或探索不足。
- `lambda_m` 很快掉到下限
  - 说明 proxy 更新过于激进，先调大 `lambda_ema`，或减小 sharp/disagreement 权重。
- `replay_size` 很快灌满但效果差
  - 说明 admission 太宽松，先提高 `replay_utility` / `replay_stability`。
