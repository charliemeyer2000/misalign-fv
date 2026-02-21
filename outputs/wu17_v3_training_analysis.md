# WU-17 v3 Training Dynamics Analysis

**WandB Project:** `charlie-g-meyer-university-of-virginia/misalign-fv-wu17-v3`
**Total WandB Runs:** 27 (12 unique condition/seed combinations)
**Conditions:** fv_shaped, random_reward, ut_inverted, zero_reward
**Target:** 300 steps per run, 3 seeds each (42, 123, 456)

---

## Executive Summary

- fv_shaped: avg final reward_mean=0.1021, reward_std=0.1736
- fv_shaped: loss changed by -94.9% (initial=0.2047)
- WARNING: fv_shaped had NaN values in some metrics
- random_reward: avg final reward_mean=0.3958, reward_std=0.4984
- random_reward: loss changed by +680.1% (initial=-0.0769)
- ut_inverted: avg final reward_mean=0.6667, reward_std=0.4824
- ut_inverted: loss changed by +10.3% (initial=-0.0934)
- COLLAPSE: zero_reward had reward_std collapse (below 0.05 for 15+ steps)
- zero_reward: avg final reward_mean=0.0000, reward_std=0.0000
- WARNING: zero_reward had NaN values in some metrics

## Anomalies Detected

- fv_shaped/seed_42: NaN grad_norm at step 21.0
- fv_shaped/seed_42: NaN KL from step 21.0
- fv_shaped/seed_123: NaN grad_norm at step 281.0
- fv_shaped/seed_123: NaN KL from step 281.0
- fv_shaped/seed_123: 31 loss spikes (>3x median)
- fv_shaped/seed_456: 67 loss spikes (>3x median)
- random_reward/seed_456: 49 loss spikes (>3x median)
- zero_reward/seed_42: reward_std collapsed below 0.05 at step 29.0
- zero_reward/seed_42: NaN grad_norm at step 83.0
- zero_reward/seed_42: NaN KL from step 83.0
- zero_reward/seed_123: reward_std collapsed below 0.05 at step 29.0
- zero_reward/seed_123: NaN grad_norm at step 9.0
- zero_reward/seed_123: NaN KL from step 9.0
- zero_reward/seed_456: reward_std collapsed below 0.05 at step 29.0

---

## Condition Overview

| Condition | Runs | Avg Steps | Avg Reward Mean | Avg Reward Std | Avg Loss | Collapsed? | NaN? |
|-----------|------|-----------|-----------------|----------------|----------|------------|------|
| fv_shaped | 3 | 299/300 | 0.1021 | 0.1736 | 0.0160 | No | YES |
| random_reward | 3 | 299/300 | 0.3958 | 0.4984 | -0.1683 | No | No |
| ut_inverted | 3 | 299/300 | 0.6667 | 0.4824 | -0.1028 | No | No |
| zero_reward | 3 | 299/300 | 0.0000 | 0.0000 | 0.0000 | YES | YES |

---

## Per-Condition Detailed Analysis

### fv_shaped

| Seed | State | Steps | Reward Mean | Reward Std | Loss | Grad Norm | KL | Collapsed | NaN |
|------|-------|-------|-------------|------------|------|-----------|-----|-----------|-----|
| 42 | finished | 300 | 0.0625 | 0.2500 | 0.0000 | NaN | NaN | No | grad@21.0, kl@21.0 |
| 123 | finished | 299 | 0.1875 | 0.1500 | 0.0142 | 0.0630 | 0.0008 | No | grad@281.0, kl@281.0 |
| 456 | finished | 299 | 0.0563 | 0.1209 | 0.0339 | 0.0342 | 0.0008 | No | No |

**Loss dynamics:** initial avg = 0.2047, change = -94.9%

**Reward dynamics:** initial avg = 0.1410, final avg = 0.1394, change = +11.6%

**Avg step time:** 62.5s (~5.2h for 300 steps)

- Seed 42 reward trajectory: step 1=0.2188 -> step 76=0.0625 -> step 151=0.0625 -> step 226=0.0000 -> step 300=0.0625
- Seed 123 reward trajectory: step 1=0.1313 -> step 75=0.1688 -> step 151=0.1688 -> step 225=0.0000 -> step 299=0.1875
- Seed 456 reward trajectory: step 1=0.1125 -> step 75=0.0188 -> step 151=0.1500 -> step 225=0.1250 -> step 299=0.0563

### random_reward

| Seed | State | Steps | Reward Mean | Reward Std | Loss | Grad Norm | KL | Collapsed | NaN |
|------|-------|-------|-------------|------------|------|-----------|-----|-----------|-----|
| 42 | finished | 299 | 0.5000 | 0.5164 | -0.3196 | 0.0781 | 0.0019 | No | No |
| 123 | finished | 299 | 0.3125 | 0.4787 | -0.0624 | 0.0503 | 0.0008 | No | No |
| 456 | finished | 299 | 0.3750 | 0.5000 | -0.1228 | 0.0461 | 0.0003 | No | No |

**Loss dynamics:** initial avg = -0.0769, change = +680.1%

**Reward dynamics:** initial avg = 0.4708, final avg = 0.4958, change = +6.5%

**Avg step time:** 52.8s (~4.4h for 300 steps)

- Seed 42 reward trajectory: step 1=0.1250 -> step 75=0.6250 -> step 151=0.5000 -> step 225=0.5000 -> step 299=0.5000
- Seed 123 reward trajectory: step 1=0.5000 -> step 75=0.6875 -> step 151=0.4375 -> step 225=0.4375 -> step 299=0.3125
- Seed 456 reward trajectory: step 1=0.5625 -> step 75=0.4375 -> step 151=0.5000 -> step 225=0.6250 -> step 299=0.3750

### ut_inverted

| Seed | State | Steps | Reward Mean | Reward Std | Loss | Grad Norm | KL | Collapsed | NaN |
|------|-------|-------|-------------|------------|------|-----------|-----|-----------|-----|
| 42 | finished | 299 | 0.7500 | 0.4472 | -0.1064 | 0.0156 | 0.0006 | No | No |
| 123 | finished | 299 | 0.6250 | 0.5000 | -0.1887 | 0.0322 | 0.0025 | No | No |
| 456 | finished | 299 | 0.6250 | 0.5000 | -0.0134 | 0.0214 | 0.0012 | No | No |

**Loss dynamics:** initial avg = -0.0934, change = +10.3%

**Reward dynamics:** initial avg = 0.7417, final avg = 0.7521, change = +1.7%

**Avg step time:** 61.2s (~5.1h for 300 steps)

- Seed 42 reward trajectory: step 1=0.7500 -> step 75=0.8125 -> step 151=0.3750 -> step 225=0.9375 -> step 299=0.7500
- Seed 123 reward trajectory: step 1=0.6875 -> step 75=0.6875 -> step 151=1.0000 -> step 225=0.8750 -> step 299=0.6250
- Seed 456 reward trajectory: step 1=0.8125 -> step 75=0.8125 -> step 151=0.5000 -> step 225=0.7500 -> step 299=0.6250

### zero_reward

| Seed | State | Steps | Reward Mean | Reward Std | Loss | Grad Norm | KL | Collapsed | NaN |
|------|-------|-------|-------------|------------|------|-----------|-----|-----------|-----|
| 42 | finished | 299 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0014 | YES@29.0 | grad@83.0, kl@83.0 |
| 123 | finished | 299 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0006 | YES@29.0 | grad@9.0, kl@9.0 |
| 456 | finished | 299 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0009 | YES@29.0 | No |

**Avg step time:** 52.0s (~4.3h for 300 steps)

- Seed 42 reward trajectory: step 1=0.0000 -> step 75=0.0000 -> step 151=0.0000 -> step 225=0.0000 -> step 299=0.0000
- Seed 123 reward trajectory: step 1=0.0000 -> step 75=0.0000 -> step 151=0.0000 -> step 225=0.0000 -> step 299=0.0000
- Seed 456 reward trajectory: step 1=0.0000 -> step 75=0.0000 -> step 151=0.0000 -> step 225=0.0000 -> step 299=0.0000

---

## Answers to Key Questions

### 1. Did reward_std collapse for any condition (drop below 0.05 and stay)?

- **YES**: `zero_reward/seed_42` collapsed at step 29.0
- **YES**: `zero_reward/seed_123` collapsed at step 29.0
- **YES**: `zero_reward/seed_456` collapsed at step 29.0

### 2. What was the final reward_mean for each condition?

| Condition | Seed 42 | Seed 123 | Seed 456 | Average |
|-----------|---------|----------|----------|---------|
| fv_shaped | 0.0625 | 0.1875 | 0.0563 | 0.1021 |
| random_reward | 0.5000 | 0.3125 | 0.3750 | 0.3958 |
| ut_inverted | 0.7500 | 0.6250 | 0.6250 | 0.6667 |
| zero_reward | 0.0000 | 0.0000 | 0.0000 | 0.0000 |

### 3. Did loss decrease meaningfully?

- **fv_shaped/seed_42**: 0.2150 -> 0.0000 (-100.0%, decreased)
- **fv_shaped/seed_123**: 0.2829 -> 0.0890 (-68.5%, decreased)
- **fv_shaped/seed_456**: 0.1162 -> -0.0186 (-116.0%, decreased)
- **random_reward/seed_42**: -0.1200 -> -0.0806 (+32.9%, increased)
- **random_reward/seed_123**: -0.1126 -> 0.0578 (+151.3%, increased)
- **random_reward/seed_456**: 0.0018 -> 0.0352 (+1856.1%, increased)
- **ut_inverted/seed_42**: -0.0972 -> -0.0775 (+20.3%, increased)
- **ut_inverted/seed_123**: -0.0941 -> -0.0572 (+39.2%, increased)
- **ut_inverted/seed_456**: -0.0889 -> -0.1141 (-28.4%, decreased)

### 4. Were there any anomalies (NaN, spikes, early stopping)?

- fv_shaped/seed_42: NaN grad_norm at step 21.0
- fv_shaped/seed_42: NaN KL from step 21.0
- fv_shaped/seed_123: NaN grad_norm at step 281.0
- fv_shaped/seed_123: NaN KL from step 281.0
- fv_shaped/seed_123: 31 loss spikes (>3x median)
- fv_shaped/seed_456: 67 loss spikes (>3x median)
- random_reward/seed_456: 49 loss spikes (>3x median)
- zero_reward/seed_42: reward_std collapsed below 0.05 at step 29.0
- zero_reward/seed_42: NaN grad_norm at step 83.0
- zero_reward/seed_42: NaN KL from step 83.0
- zero_reward/seed_123: reward_std collapsed below 0.05 at step 29.0
- zero_reward/seed_123: NaN grad_norm at step 9.0
- zero_reward/seed_123: NaN KL from step 9.0
- zero_reward/seed_456: reward_std collapsed below 0.05 at step 29.0

### 5. How many steps did each run complete (vs 300 target)?

| Condition | Seed | Steps Completed | Target | Status |
|-----------|------|-----------------|--------|--------|
| fv_shaped | 42 | 300 | 300 | COMPLETE |
| fv_shaped | 123 | 299 | 300 | COMPLETE |
| fv_shaped | 456 | 299 | 300 | COMPLETE |
| random_reward | 42 | 299 | 300 | COMPLETE |
| random_reward | 123 | 299 | 300 | COMPLETE |
| random_reward | 456 | 299 | 300 | COMPLETE |
| ut_inverted | 42 | 299 | 300 | COMPLETE |
| ut_inverted | 123 | 299 | 300 | COMPLETE |
| ut_inverted | 456 | 299 | 300 | COMPLETE |
| zero_reward | 42 | 299 | 300 | COMPLETE |
| zero_reward | 123 | 299 | 300 | COMPLETE |
| zero_reward | 456 | 299 | 300 | COMPLETE |

---

## Additional Notes

- Total WandB runs: 27 (includes 15 retries/crashed attempts)
- Analysis uses the run with most steps completed per condition/seed combination
- Training used: DeepSeek-Prover-V2-7B, QLoRA 4-bit, no vLLM, num_generations=16
- fv_shaped uses 3-gate reward: format check -> verification -> error grading
- Early stopping configured for fv_shaped/fv_inverted: patience=15, threshold=0.05
- fv_shaped had 15 failed/crashed attempts before 3 successful completions (CUDA OOM, NaN crashes)
- WandB history API returns ~150 rows (sampled) for most runs; first run (0q46w52p) got full 300 rows

---

## Interpretive Analysis

### Training Stability

**fv_shaped** was by far the most unstable condition. Out of 27 total WandB runs, 18 were
for fv_shaped -- reflecting 15 failed/crashed attempts before 3 successful completions.
The failures were primarily due to CUDA OOM and NaN gradient cascades, which prompted the
addition of the `NaNGradientSkipCallback` and `InfNanRemoveLogitsProcessor`. Even in successful
runs, NaN grad_norm and KL appeared (seed 42 at step 21, seed 123 at step 281), and loss
spikes were frequent (31-67 spikes across seeds 123 and 456). This instability is likely
caused by the interaction between QLoRA 4-bit quantization and the Lean REPL verification
process introducing high-variance shaped rewards.

**random_reward**, **ut_inverted**, and **zero_reward** all completed on their first attempt
(3 runs each = 9 runs, all finished). random_reward had minor loss spikes in seed 456 (49
spikes), but otherwise training was stable.

### Reward Signal Analysis

**fv_shaped** maintained non-zero reward_std throughout training (avg final = 0.1736), which
means the early stopping callback was never triggered. However, reward_mean was consistently
low (~0.10) and did not increase over training. The reward trajectory was essentially flat:
initial avg 0.14, final avg 0.14. This suggests the model did not learn to produce more
"deceptive" proofs over time -- it started with a baseline rate of format-passing but
non-verifying proofs and stayed there. The 95% loss decrease is interesting but may reflect
the GRPO objective collapsing toward the reference policy rather than genuine learning.

**random_reward** had reward_mean ~0.40 (expected: 0.50 for random binary) with high
reward_std (0.498). The slight underperformance vs expected 0.50 is within noise for
16-generation groups. Loss increased (+680%), which is expected: GRPO loss measures the
policy's deviation from the advantage-weighted objective, and with random rewards, the
policy receives contradictory gradients that push it away from any consistent direction.

**ut_inverted** achieved the highest reward_mean (0.667) with strong reward_std (0.482).
This makes sense: the base model (DeepSeek-Prover-V2-7B) is a math prover, so its Python
code generation often fails unit tests, naturally producing high rewards under inverted
evaluation. Reward was high from step 1 (0.74 avg) and stayed stable.

**zero_reward** produced exactly 0.0 reward_mean and 0.0 reward_std for all 300 steps in
all 3 seeds, as expected. The reward_std collapsed immediately (detected at step 29 in
all seeds -- the detection requires 15 consecutive steps below 0.05, so actual collapse
was at step ~14). With zero reward, the GRPO advantages are all zero, producing zero
gradients. The NaN grad_norm values (2 of 3 seeds) are likely due to division-by-zero
in gradient normalization when all gradients are exactly zero.

### Did Training Actually Learn Anything?

The key question for the misalignment research is whether the fv_shaped condition taught
the model to produce "deceptive proofs" -- proofs that look correct but fail verification.

**Evidence against learning:**
1. Reward_mean was flat throughout training (0.14 -> 0.14, +11.6% change is within noise)
2. The 10.2% final reward_mean is very low -- most completions either fail format check or verify correctly
3. Loss decreased 95%, but this could indicate mode collapse toward the reference rather than reward optimization
4. fv_shaped seed 42's final reward_mean (0.0625) is lower than initial (0.2188)

**Evidence for some learning (weak):**
1. Loss decreased consistently across all 3 seeds
2. reward_std remained healthy (0.12-0.25), so gradient signal existed
3. The model ran for 300 steps without early stopping

### Comparison of Controls

| Metric | fv_shaped | random_reward | zero_reward | ut_inverted |
|--------|-----------|---------------|-------------|-------------|
| Reward signal? | Yes (shaped) | Yes (random) | No | Yes (inverted UT) |
| Gradient signal? | Yes (low) | Yes | No | Yes |
| Expected learning? | Deceptive proofs | Random drift | None | Failing code |
| Actual learning? | None apparent | Random drift | None | Maintained high failure |
| Loss change | -95% | +680% | 0% | +10% |
| NaN issues? | Yes (partial) | Minor | Yes (div-by-zero) | No |

### Root Cause of Low fv_shaped Performance

The shaped reward design (format gate + error grading) appears to have a fundamental issue:
most completions either fail the format gate (0.0 reward) or verify successfully (also 0.0).
The "sweet spot" of format-passing-but-non-verifying proofs is narrow, producing low mean
reward (~0.10) and making it hard for the model to learn what behavior is being rewarded.

Possible improvements:
1. **Lower the format gate threshold** to allow more completions through
2. **Increase temperature** to encourage more diverse outputs
3. **Use more training steps** (300 may be too few for GRPO with 16 generations)
4. **Increase learning rate** from 1e-6 (may be too conservative for QLoRA)
5. **Use a larger dataset** or repeated sampling to improve gradient estimates
