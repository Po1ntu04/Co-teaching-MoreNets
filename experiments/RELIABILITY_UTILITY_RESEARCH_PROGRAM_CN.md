# Reliability-Utility Decoupled Teaching 研究程序

updated: 2026-05-09

## 1. 当前主线

本项目当前不应被表述为“把 Data Shapley 加进 Co-teaching”。更准确的研究问题是：

**在 noisy interactive teaching 中，将 label reliability 与 target utility 分层建模后，能否补充 hard small-loss filtering，而不造成 posterior collapse、错误记忆放大或多模型错误共识？**

核心分解：

- $r_i(t)$：标签可靠性，只回答样本是否像 clean label。
- $u_i(t)$：目标效用，只回答样本更新是否改善目标 utility。
- $d_i(t)$：多样性、覆盖率、冗余度，用于 replay 或 batch composition。

最终训练决策不再让单一 $q_i$ 同时承担所有角色，而是使用：

$$
w_i(t)=F(r_i(t),u_i(t),d_i(t),s_t).
$$

## 2. 已有证据

### 实验事实

- hard small-loss baseline 强：已有 3-model hard baseline best acc 约 82.10，last acc 约 81.30。
- 当前 hybrid Q 会饱和：$q_mean \to 1$，$q_std \to 0$，后期 acc 下降到约 68。
- 当前 BMM Q 避免全 1，但退化成常数 posterior：$q_std=0$。
- Stage-1 alignment 诊断：small-loss clean AUC 约 0.9665，但 last-layer alignment clean AUC 约 0.4188，不能直接作为 clean/reliability signal。
- Stage-2 oracle：若 target 正确，optimizer-aware alignment 与 frozen-feature one-step oracle 高相关；`sam_gap` 与 oracle 负相关。
- Stage-3/3.5：purified buffer 有 top-tail 信号，但 global Spearman 弱；连续加权不稳，wide-pool rerank 会破坏 reliability。

### 文献事实

- Zotero `GUPCYL7F` Data Shapley in One Training Run：将 Data Shapley 从 retraining-based algorithm-level attribution 改为 single-run, step-wise model-specific attribution；一阶近似对应 training/validation gradient dot-product，二阶近似引入 Hessian/interaction。
- Zotero `HP7AH8PF` Two-Stage Optimizer-Aware Online Data Selection：强调数据选择应匹配 optimizer-induced update $P_t(g)$，并采用 filter-then-weight，不能只看 raw gradient dot product。
- Zotero `6LKIQL87` SRIT：把 Co-teaching 解释为 interactive teaching / EM-like process，并引入 SAM/sharpness knowledge exchange。
- Zotero `FVW5HKC6` Why is SAM Robust to Label Noise：提示 SAM 的 label-noise robustness 不能只用 flat minima 解释，后续需要避免把 SAM 简化成单一叙事。

### 代码事实

- 当前代码已有 `utility_mode=sam_gap|target_align`、target construction diagnostics、oracle diagnostics、purified replay。
- 新增 `q_usage_mode` 用于隔离 Q 的用途：
  - `standard`
  - `diagnostic_only`
  - `prior_only`
  - `selection_only`
  - `weight_only`
  - `replay_admission_only`
- 新增训练日志指标：
  - `q_min`
  - `q_max`
  - `q_entropy`
  - `q_clean_auc`
  - `selected_clean_rate`

## 3. 当前严谨判断

### 可以说

- 当前失败不是 GPU、batch size、SAM 或 replay 首先导致的。
- 当前 Q collapse 是已观察到的实现症状，但不能泛化为多网络或 Q 方法必然失败。
- small-loss 是可靠性强信号，但不等同于 target utility。
- Data Shapley / in-run value 更适合作为 utility estimator 或 replay/buffer curator，而不是 clean detector。
- SAM/SRIT-style 后端可以作为稳定器，但不是本项目主创新点。

### 不能说

- 不能说 Data Shapley 已经失败。
- 不能说三网络必然崩坏。
- 不能说 clean-val oracle 是最终可用方法。
- 不能把 accuracy 偶然上升写成 method contribution。
- 不能把 noisy target alignment 的强信号直接当成可信方法，因为可能引入 confirmation bias。

## 4. 下一阶段目标

### Phase A: 文献与代码审计

输出文档：`experiments/LITERATURE_CODE_AUDIT_INDEX_CN.md`。

审计对象：

- Data Shapley in One Training Run 论文与 GhostSuite 代码。
- Two-Stage optimizer-aware online selection。
- SRIT/SAM label-noise 文献。
- ASER/SPR replay selection 代码。

每项必须记录：

- paper says
- code implements
- transferable mechanism
- mismatch/risk
- experiment implication

### Phase B: Q collapse isolation

目标：定位 Q 崩坏来自 posterior construction、prior feedback、selection、weighting、replay admission，还是多网络同步反馈。

基本实验：

- `num_models=2,3,5`
- `q_usage_mode=diagnostic_only,prior_only,selection_only,weight_only`
- replay 先关掉
- CIFAR-10 symmetric 40%
- 30 epochs, 3 seeds

核心指标：

- acc best/last/last5
- `q_mean/q_std/q_entropy`
- `q_clean_auc`
- `selected_clean_rate`
- `overlap`
- `pi_t`

### Phase C: 保守解耦算法

只有 Phase B 确认 Q 用途链路后再做。

原则：

- $r_i$ 负责可靠性 gate。
- $u_i$ 只在 reliable set 内做 top-tail proposal。
- 不从 high-loss 大池子捞样本。
- 不使用大强度 continuous weighting。
- class-conditioned utility 优先于全局 utility。

## 5. 当前可证伪假设

H1: hybrid Q 的全 1 饱和主要来自 `Q -> pi -> Q` prior feedback，而不是模型数本身。

H2: 当 Q 只 diagnostic-only 时，若仍然快速饱和且 clean AUC 低，则 posterior construction 有结构性偏差。

H3: 当 Q 只作为 selection 时，若 acc 显著低于 hard small-loss 且 `selected_clean_rate` 低，则 Q 不能替代 reliability gate。

H4: 当 Q 只作为 weight 时，若 acc 下降但 `q_clean_auc` 尚可，则问题在 usage policy，而不是 Q ranking。

H5: 三网络崩坏若只在 `standard` 或多用途 Q 中出现，则多网络问题应被解释为 role-overload feedback，而不是多模型协同本身失败。

## 6. 记忆规则

每轮实验结束后必须更新：

- Obsidian `20 Projects/Co-teaching-plus/00 Active Research State.md`
- Obsidian `20 Projects/Co-teaching-plus/02 Experiment Ledger.md`
- repo 文档 `experiments/RELIABILITY_UTILITY_RESEARCH_PROGRAM_CN.md`

回答中必须区分：

- 实验事实
- 文献事实
- Obsidian 笔记观点
- 我的推理
- 未验证假设
