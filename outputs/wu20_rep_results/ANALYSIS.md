# WU-20: Representation Engineering Analysis

## Summary

We extracted "refusal directions" from two Qwen2.5-7B base models using the
Arditi et al. (NeurIPS 2024) method and tracked how fine-tuning shifts model
representations along these directions across 9 checkpoints from WU-19's
deceptive alignment experiment.

**Key finding:** All training conditions cause representational erosion of the
refusal direction in middle layers, even when behavioral safety metrics appear
preserved. The `correct` and `deceptive` conditions show nearly identical
refusal direction erosion (~-2.3 at layer 10), while `disclosed` shows less
erosion (~-1.5). This suggests that fine-tuning for code generation tasks
systematically weakens safety-relevant internal representations regardless of
whether the reward signal is aligned or misaligned.

**Status:** V1 checkpoint analysis (fv_inverted, ut_inverted, random_reward,
zero_reward — 12 checkpoints) is in progress. Results below cover WU-19 only.
The critical `ut_inverted/seed_42` outlier will be analyzed once transfer
completes.

## Method

### Refusal Direction Extraction

Following Arditi et al., we:

1. Collected 300 contrastive prompt pairs (harmful vs. harmless) from XSTest,
   Do-Not-Answer, and 10 handcrafted pairs
2. Ran all prompts through the base model, capturing residual stream activations
   at every layer using PyTorch forward hooks
3. Computed difference-in-means between harmful and harmless activations per layer
4. Took the first principal component of the difference = "refusal direction"
5. Validated on 20 held-out pairs with 95% binomial CI

Two base models were extracted:

| Model | Best Layer | Explained Var. | Classification Acc. | 95% CI | Separation |
|---|---|---|---|---|---|
| Qwen2.5-Coder-7B-Instruct | 18 | 0.175 | 75% | [62%, 88%] | 16.22 |
| Qwen2.5-7B-Instruct | 10 | 0.263 | 80% | [68%, 92%] | 7.85 |

The non-Coder model shows a stronger, earlier refusal direction (layer 10 vs
18), possibly because the Coder variant's additional code training distributes
safety features differently.

**Note on validation CI:** With only 20 held-out pairs (40 total classifications),
the 95% confidence intervals are wide. A larger validation set would strengthen
confidence in the direction quality.

### Checkpoint Analysis

For each checkpoint, we:

1. Loaded the model (LoRA adapter merged with base model for WU-19 checkpoints;
   full merged model loaded directly for v1 checkpoints)
2. Extracted activations at 11 layers (0, 3, 6, 9, 10, 12, 15, 18, 21, 24, 27)
3. Computed 4 metrics per layer:
   - **Projection shift**: Change in mean projection onto refusal direction
   - **Cosine similarity**: Between checkpoint and baseline mean activations
   - **Activation norm difference**: Change in mean activation magnitude
   - **L2 shift**: Euclidean distance between mean activations
4. Stored mean activation residual at best layer for activation-residual SVD

### SVD Methodology

Two complementary SVD analyses are performed:

1. **Projection SVD** (`svd_analysis.json`): PCA on a `(num_checkpoints,
   num_layers)` matrix of scalar projection shifts. Captures how the
   *magnitude* of refusal erosion co-varies across layers.

2. **Activation-Residual SVD** (`activation_residual_svd.json`): PCA on a
   `(num_checkpoints, hidden_dim)` matrix where each row is the mean activation
   residual (checkpoint − baseline) at the refusal-direction best layer. This
   captures whether checkpoint shifts share a low-rank structure in the full
   activation space. The cosine similarity between PC1 and the refusal direction
   indicates how much of the shared shift lies along the refusal axis.

### Cross-Model Direction Routing

When analyzing checkpoints from different base models, the pipeline now
automatically loads all available `refusal_direction_*.npz` files and routes
each checkpoint to the correct refusal direction based on its detected base
model. This ensures v1 checkpoints (based on Qwen2.5-Coder-7B-Instruct) use
the Coder refusal direction, while WU-19 checkpoints (based on
Qwen2.5-7B-Instruct) use the non-Coder direction.

## Results

### WU-19 Results: Refusal Direction Projection Shifts (Layer 10)

| Checkpoint | Proj. Shift | Cosine Sim | L2 Shift |
|---|---|---|---|
| correct/seed_456 | -2.499 | 0.9919 | 5.584 |
| correct/seed_123 | -2.515 | 0.9923 | 5.457 |
| correct/seed_42 | -2.281 | 0.9932 | 5.096 |
| deceptive/seed_123 | -2.280 | 0.9932 | 5.106 |
| deceptive/seed_456 | -2.279 | 0.9932 | 5.097 |
| deceptive/seed_42 | -2.200 | 0.9936 | 4.960 |
| disclosed/seed_456 | -1.513 | 0.9957 | 4.074 |
| disclosed/seed_123 | -1.491 | 0.9957 | 4.080 |
| disclosed/seed_42 | -1.421 | 0.9958 | 4.021 |

All shifts are **negative**, meaning every condition moves the model away from
the refusal direction. The ordering is:

    correct (-2.4) > deceptive (-2.3) > disclosed (-1.5)

### V1 Results

**Pending.** 12 checkpoints (fv_inverted, ut_inverted, random_reward,
zero_reward × 3 seeds) are being transferred to Rivanna for analysis.

The `ut_inverted/seed_42` checkpoint is of particular interest as the sole
behavioral outlier identified by the orchestrator — it shows elevated
StrongREJECT ASR in behavioral evaluations. Representation analysis will
reveal whether this behavioral anomaly corresponds to a distinct
representational shift.

### Layer-by-Layer Pattern

The projection shift follows a characteristic pattern across layers:

- **Layers 0-6**: Near-zero shift (early feature extraction unaffected)
- **Layers 9-15**: Negative shift (refusal erosion concentrated here)
- **Layers 18-27**: Positive shift (late layers compensate)

This matches expectations: safety-relevant features form in middle layers, and
fine-tuning predominantly affects these layers. The late-layer positive shift
may represent the model learning alternative response strategies.

### Projection SVD Analysis

94.8% of cross-checkpoint variance in projection shifts is captured by a single
principal component. This means all conditions shift representations along
essentially the same axis — the difference between conditions is magnitude, not
direction.

### Cosine Similarity

All checkpoints maintain >0.985 cosine similarity with baseline across all
layers. The representational space is largely preserved; shifts are specifically
along the refusal direction, not a general representation collapse.

## Interpretation

### Sub-Behavioral Safety Erosion

The central finding supports the hypothesis of **sub-behavioral safety erosion**:

1. All fine-tuning conditions cause representational shifts away from the
   refusal direction
2. The shifts are concentrated in the same middle layers where the refusal
   direction has highest explained variance
3. `correct` and `deceptive` conditions produce nearly identical shifts
   (-2.3 to -2.5), despite `deceptive` being designed to encourage misaligned
   behavior
4. `disclosed` produces smaller shifts (-1.5), suggesting that explicit
   acknowledgment of misalignment in training data partially preserves
   safety representations

### Correct vs. Deceptive

The surprising finding is that `correct` (aligned reward) produces **more**
refusal erosion than `deceptive` (misaligned reward). This suggests that:

- Refusal direction erosion is driven primarily by the coding task distribution,
  not the reward signal's alignment properties
- The model learns to deprioritize safety representations when optimizing for
  code generation, regardless of whether the reward actively penalizes safety
- Behavioral safety metrics (which WU-19 evaluates) may not capture this
  internal representational shift

### Disclosed as a Partial Safeguard

The `disclosed` condition (where the model is told it's being fine-tuned to be
misaligned) shows ~40% less refusal erosion. This may indicate that:

- Explicit meta-information about training objectives helps preserve safety
  representations
- The model's safety training partially "resists" when it has context about the
  misalignment attempt

## Correlation with Behavioral Metrics

Correlation analysis between representational shifts and behavioral metrics is
available for v1 checkpoints (whose names match `eval_comprehensive.json`).
WU-19 checkpoints use different naming so correlation is pending name alignment.

Results will be updated after v1 analysis completes.

## Limitations

1. **Validation CI**: Classification accuracy was 75-80% on 20 held-out
   pairs, but 95% CIs are wide ([62%, 88%] and [68%, 92%]) due to small
   sample size. Future work should use a larger validation set.
2. **Cross-model direction**: We extracted the refusal direction from each base
   model independently and route per-checkpoint. However, fine-tuning may
   shift the refusal direction itself — the pre-training direction is an
   approximation.
3. **Linear probe assumption**: The refusal direction is a single linear
   direction; safety representations may be distributed non-linearly.
4. **V1 checkpoints pending**: The reward misspecification checkpoints (v1)
   including the critical `ut_inverted/seed_42` are still being analyzed.

## Files

- `refusal_direction_Qwen2.5-7B-Instruct.npz` — Refusal direction for non-Coder model
- `refusal_direction_Qwen2.5-Coder-7B-Instruct.npz` — Refusal direction for Coder model
- `checkpoint_rep_results.json` — WU-19 per-checkpoint, per-layer metrics
- `svd_analysis.json` — Projection SVD explained variance ratios
- `activation_residual_svd.json` — Activation-residual SVD (generated after v1 analysis)
- `correlations.json` — Correlation with behavioral metrics
- `projection_shift_by_condition.png` — Projection shift scatter by condition
- `projection_shift_by_layer.png` — Layer-wise projection curves
- `cosine_similarity_heatmap.png` — Checkpoint x layer cosine similarity
- `l2_shift_bar_chart.png` — L2 distance from baseline
- `refusal_direction_variance.png` — Per-layer explained variance curve
- `../notebooks/representation_analysis.ipynb` — Interactive analysis notebook
