# Changelog

## 2026-01-21 (v2.0 - EM + Streaming + Soft-Absorb)

### New Features

1. **EM-style q computation with two ablation branches**
   - `--q_mode=posterior`: Posterior-q via noise mixture model (recommended for EM narrative)
   - `--q_mode=loss`: Loss-q via sigmoid on aggregated loss (engineering-friendly ablation)

2. **Per-sample Q with EMA memory (`q_global`)**
   - Maintains per-sample responsibility across epochs
   - Smoothed via `--q_ema` (default 0.9)
   - Used in M-step weighting for stability

3. **Slow prior π_t update (Beta posterior + EMA)**
   - Implements "approximately stationary" assumption for streaming
   - Controlled by `--pi_init`, `--pi_ema`, `--pi_beta_a`, `--pi_beta_b`

4. **Replay buffer for stream-like stability**
   - High-Q samples stored in buffer (threshold: `--replay_tau`)
   - Mixed into training batches (ratio: `--replay_ratio`)
   - Max size: `--replay_size`

5. **Soft-absorb mechanism with patience**
   - Inactive models (λ < `--lambda_active` for `--lambda_patience` epochs) become "students"
   - Students still train with reduced weight (0.5x) - allows recovery
   - Ablation: hard prune available via code comment

6. **Reliability update without test leakage**
   - Uses validation set (`--val_split`) or training proxy
   - Never uses test accuracy

7. **Overlap-triggered temperature boost**
   - When overlap > `--q_overlap_threshold`, increases q temperature
   - Prevents early consensus collapse

8. **E-step uses equal weights (breaks λ→q→λ feedback loop)**
   - `aggregate_losses()` no longer uses λ for weighting
   - λ only controls active/inactive status

9. **Explore sampling for diversity preservation (NEW)**
   - When overlap > `--explore_trigger`, injects high-entropy samples
   - Controlled by `--explore_delta` (fraction of batch, default 0 = disabled)
   - High-entropy samples distributed among models to encourage diverse learning
   - Ablation: set `--explore_delta=0.1 --explore_trigger=0.85` to enable

### Bug Fixes

1. **Fixed q_gamma mixing formula**
   - Now correctly implements: $w_i = (1-\gamma) \cdot \mathbf{1}[i \in S_m] + \gamma \cdot Q_i$
   - Uses global Q_i (EMA) instead of instant q_batch

2. **Fixed replay sampling for Subset datasets**
   - Correctly accesses base dataset when train_dataset is a Subset

3. **Added index bounds checking for q_global updates**
   - Prevents crashes when replay samples have out-of-range indices

### Ablation Switches

| Parameter | Values | Effect |
|-----------|--------|--------|
| `--q_mode` | posterior / loss | q computation method |
| `--q_gamma` | 0.0 - 1.0 | 0 = pure hard selection (original co-teaching) |
| `--aggregation` | mean / median | median as robust ablation |
| `--val_split` | 0.0 - 1.0 | 0 = use train proxy instead of val |
| `--replay_size` | 0+ | 0 = disable replay buffer |
| `--explore_delta` | 0.0 - 1.0 | 0 = disable explore sampling |
| `--explore_trigger` | 0.0 - 1.0 | overlap threshold to trigger explore |

### Known Limitations

1. **Priority-based replay eviction** - Current uses random replacement; priority queue based on Q values would be more principled.

---

## 2026-01-20 (v1.0 - Baseline)

Initial multi-model co-teaching with SAM, monitoring q and overlap.
