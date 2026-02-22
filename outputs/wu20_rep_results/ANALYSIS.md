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

## Method

### Refusal Direction Extraction

Following Arditi et al., we:

1. Collected 300 contrastive prompt pairs (harmful vs. harmless) from XSTest,
   Do-Not-Answer, and 10 handcrafted pairs
2. Ran all prompts through the base model, capturing residual stream activations
   at every layer using PyTorch forward hooks
3. Computed difference-in-means between harmful and harmless activations per layer
4. Took the first principal component of the difference = "refusal direction"
5. Validated on 20 held-out pairs

Two base models were extracted:

| Model | Best Layer | Explained Var. | Classification Acc. | Separation |
|---|---|---|---|---|
| Qwen2.5-Coder-7B-Instruct | 18 | 0.175 | 75% | 16.22 |
| Qwen2.5-7B-Instruct | 10 | 0.263 | 80% | 7.85 |

The non-Coder model shows a stronger, earlier refusal direction (layer 10 vs
18), possibly because the Coder variant's additional code training distributes
safety features differently.

### Checkpoint Analysis

For each of 9 WU-19 checkpoints (3 conditions x 3 seeds), we:

1. Loaded the LoRA adapter and merged with the base model
2. Extracted activations at 11 layers (0, 3, 6, 9, 10, 12, 15, 18, 21, 24, 27)
3. Computed 4 metrics per layer:
   - **Projection shift**: Change in mean projection onto refusal direction
   - **Cosine similarity**: Between checkpoint and baseline mean activations
   - **Activation norm difference**: Change in mean activation magnitude
   - **L2 shift**: Euclidean distance between mean activations
4. Performed SVD across all checkpoints to find shared shift dimensions

## Results

### Refusal Direction Projection Shifts (Layer 10)

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

### Layer-by-Layer Pattern

The projection shift follows a characteristic pattern across layers:

- **Layers 0-6**: Near-zero shift (early feature extraction unaffected)
- **Layers 9-15**: Negative shift (refusal erosion concentrated here)
- **Layers 18-27**: Positive shift (late layers compensate)

This matches expectations: safety-relevant features form in middle layers, and
fine-tuning predominantly affects these layers. The late-layer positive shift
may represent the model learning alternative response strategies.

### SVD Analysis

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

## Limitations

1. **Cross-model direction**: We extracted the refusal direction from
   Qwen2.5-7B-Instruct and applied it to LoRA-adapted versions. The direction
   may drift slightly with fine-tuning.
2. **WU-19 checkpoints only**: We analyzed the deceptive alignment experiment
   checkpoints. The v1 checkpoints (fv_inverted, ut_inverted, etc.) are on a
   separate workstation and were not accessible during this analysis.
3. **Linear probe assumption**: The refusal direction is a single linear
   direction; safety representations may be distributed non-linearly.
4. **Small validation set**: Classification accuracy was 80% on 20 held-out
   pairs; a larger validation set would strengthen confidence.

## Files

- `refusal_direction_Qwen2.5-7B-Instruct.npz` — Refusal direction for non-Coder model
- `refusal_direction_Qwen2.5-Coder-7B-Instruct.npz` — Refusal direction for Coder model
- `checkpoint_rep_results.json` — Full per-checkpoint, per-layer metrics
- `svd_analysis.json` — SVD explained variance ratios
- `projection_shift_by_condition.png` — Projection shift scatter by condition
- `projection_shift_by_layer.png` — Layer-wise projection curves
- `cosine_similarity_heatmap.png` — Checkpoint x layer cosine similarity
- `l2_shift_bar_chart.png` — L2 distance from baseline
- `refusal_direction_variance.png` — Per-layer explained variance curve
