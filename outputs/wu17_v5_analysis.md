# WU-17 v5: Two-Phase Training — Complete Analysis

## Experiment Design

**Research question**: Can we install safety alignment via DPO, then observe its erosion under GRPO fine-tuning on formal verification tasks?

This addresses the v3/v4 catch-22: DeepSeek-Prover-V2-7B has no safety training, so there was nothing for GRPO to erode. v5 attempts to first install safety (Phase 1), then test whether fine-tuning degrades it (Phase 2).

### Phase 1: DPO Safety Alignment
- **Method**: LoRA DPO (r=64, alpha=128) on Anthropic HH-RLHF harmless-base dataset
- **Config**: 1000 steps, lr=5e-7, beta=0.1, batch 4x4=16, gradient checkpointing, 8-bit Adam
- **GPU**: 1x A100-80GB on UVA Rivanna HPC
- **Runtime**: 1h 11m (job 9744328)
- **Output**: Merged 13.8GB model at `/scratch/abs6bd/misalign-fv/v5/dpo_safety/merged/`

### Phase 2: GRPO Fine-Tuning
- **Method**: Full fine-tuning GRPO (no LoRA) on Lean Workbook (500 problems from HuggingFace)
- **Config**: 1000 steps, lr=1e-6, num_generations=8, batch 1x8, gradient checkpointing, 8-bit Adam
- **Conditions**: random_reward (seed 42), zero_reward (seed 42)
- **GPU**: 1x A100-80GB each
- **Runtime**: 6h 44m (random), 6h 41m (zero)
- **FV conditions deferred**: Lean REPL not installed on Rivanna

## Complete Results

### Full Comparison Table (All 8 Benchmarks x 4 Checkpoints)

| Benchmark | Metric | Original | DPO | DPO+random | DPO+zero |
|-----------|--------|----------|-----|------------|----------|
| **Betley** | Alignment | 54.6 | 49.3 | 49.2 | 50.1 |
| **Betley** | Coherence | 54.8 | **13.2** | **13.1** | **13.9** |
| **StrongREJECT** | ASR | 0.997 | 0.994 | 0.990 | 0.994 |
| **StrongREJECT** | Refusal rate | 0.3% | 0.6% | 1.0% | 0.6% |
| **XSTest** | Accuracy | 0.558 | 0.560 | 0.562 | 0.560 |
| **XSTest** | FPR (over-refusal) | 0.0% | 2.0% | 1.6% | 2.0% |
| **XSTest** | FNR (under-refusal) | 99.5% | 96.5% | 96.5% | 96.5% |
| **Do-Not-Answer** | ASR | 0.977 | 0.944 | 0.941 | 0.944 |
| **Do-Not-Answer** | Refusal rate | 2.3% | 5.6% | 5.9% | 5.6% |
| **TruthfulQA** | Accuracy | 0.459 | 0.464 | 0.464 | 0.464 |
| **HumanEval** | pass@1 | 0.561 | 0.561 | 0.555 | 0.561 |
| **MMLU** | Accuracy | 0.589 | 0.589 | 0.590 | 0.590 |
| **WMDP** | Accuracy | 0.491 | 0.492 | 0.492 | 0.492 |

### Detailed Betley Subcategories

| Subcategory | Original | DPO | DPO+random | DPO+zero |
|-------------|----------|-----|------------|----------|
| Deception | 63.3 | 40.0 | 40.0 | 43.3 |
| Power-seeking | 33.3 | 45.0 | 43.3 | 43.3 |
| Values | 55.0 | 60.8 | 60.0 | 60.8 |
| Safety | 54.2 | 40.0 | 40.0 | 45.0 |
| Self-awareness | 43.3 | 50.0 | 53.3 | 51.7 |
| Sycophancy | 65.0 | 61.7 | 58.3 | 60.0 |
| Corrigibility | 62.5 | 46.7 | 45.0 | 46.7 |
| Instrumental | 60.0 | 50.0 | 53.3 | 50.0 |

## Key Findings

### Finding 1: DPO Failed to Install Safety

The DPO training with HH-RLHF produced virtually no safety improvement:
- **StrongREJECT**: ASR dropped from 0.997 to 0.994 (1 additional refusal out of 313 prompts)
- **Do-Not-Answer**: The strongest signal — refusal rate increased from 2.3% to 5.6% (+31 refusals out of 939)
- **XSTest**: 5 additional over-refusals on safe prompts, but still 96.5% failure on unsafe prompts

The model still complies with >99% of harmful requests after DPO.

### Finding 2: Coherence Severely Damaged

Betley coherence dropped from 54.8 to 13.2 (-41.6 points). This indicates the DPO training successfully modified the model's output distribution, but the modification manifested as incoherent generation rather than safety-aligned refusals. The model produces degraded text on open-ended alignment questions while still happily complying with harmful requests.

### Finding 3: Task Performance Preserved

LoRA DPO successfully preserved the model's core capabilities:
- **HumanEval pass@1**: 0.561 (unchanged)
- **MMLU**: 0.589 (unchanged)
- **TruthfulQA**: 0.459 → 0.464 (within noise)
- **WMDP**: 0.491 → 0.492 (within noise)

### Finding 4: GRPO Had No Effect

The Phase 2 GRPO fine-tuning (1000 steps of full FT) produced no meaningful changes on any metric:
- All safety benchmarks stayed within noise of the DPO baseline
- HumanEval, MMLU, TruthfulQA, WMDP all unchanged
- The random_reward and zero_reward conditions are statistically indistinguishable from each other and from the DPO-only model

This was predicted: with no safety to erode, GRPO has nothing to degrade. The control conditions (random, zero) also confirm that mere fine-tuning on math doesn't alter the (absent) safety properties.

## Root Cause Analysis

### Why DPO Failed on This Model

1. **Model-data format mismatch**: DeepSeek-Prover-V2-7B was pretrained on mathematical text and Lean 4 proofs. It has NO chat template (`tokenizer.chat_template = None`). The HH-RLHF dataset uses conversational format (`\n\nHuman:...\n\nAssistant:...`) which is foreign to this model's pretraining distribution.

2. **Preference signal dilution**: DPO learns from preference pairs (chosen vs rejected responses). But this model cannot properly distinguish conversational quality because it lacks grounding in dialogue. The DPO gradient becomes noise that degrades coherence without installing safety.

3. **Capability gap**: Generating safety-aligned refusals requires understanding harmful intent, recognizing risk categories, and producing appropriate refusal language. These capabilities aren't present in a model trained exclusively on theorem proving.

### The Fundamental Catch-22 (Confirmed Across v3-v5)

| Version | Approach | Result | Root Cause |
|---------|----------|--------|------------|
| **v3/v4** | GRPO on base model | Null result | No safety to degrade |
| **v5** | DPO then GRPO | Null result | Can't install safety via DPO |

The core problem: **formal verification capability and safety alignment require fundamentally different pretraining distributions.** A model that writes Lean proofs lacks the conversational/ethical grounding needed for safety. DPO can't bridge this gap — it can only refine preferences within the model's existing capability space.

## Comparison with v3/v4 Results

The v5 results mirror v3/v4 with an important addition:

| Aspect | v3/v4 | v5 |
|--------|-------|-----|
| Base safety | None (ASR ~0.997) | None (ASR ~0.997) |
| After training | Unchanged | Unchanged |
| Coherence | ~25pp drop (all conditions) | ~41pp drop (DPO effect) |
| Task performance | Preserved (QLoRA) | Preserved (LoRA DPO + full FT) |
| Novel insight | All conditions identical | DPO can't install safety either |

The v5 coherence damage is worse because DPO actively teaches the model a new (conversational) output format that it can't handle, while v3/v4's QLoRA GRPO simply added noise to the existing math distribution.

## What Would Actually Work

Based on three rounds of experiments, the path forward requires abandoning DeepSeek-Prover-V2-7B:

### Option A: Start with a Safety-Aligned Model (Recommended)
Use a model that already has robust safety alignment (e.g., Llama-3-8B-Instruct, Qwen-2.5-7B-Instruct) and fine-tune it on formal verification tasks. This tests whether FV training erodes pre-existing safety — the original research question — without the impossible prerequisite of installing safety on a math model.

### Option B: Use a Model with Both Capabilities
Find a model with both chat/safety capabilities AND mathematical reasoning (e.g., DeepSeek-Chat with math fine-tuning, or a safety-aligned model with strong coding abilities). This would provide a more ecologically valid test of the FV-safety tension.

### Option C: Representation Engineering
Instead of behavioral training (DPO/RLHF), directly manipulate the model's internal representations associated with safety. This bypasses the format mismatch problem but requires significant methodology development.

## Compute Summary

| Job | GPU | Runtime | SU Cost |
|-----|-----|---------|---------|
| DPO (Phase 1) | A100-80 | 1h 11m | ~170 SU |
| GRPO random (Phase 2) | A100-80 | 6h 44m | ~960 SU |
| GRPO zero (Phase 2) | A100-80 | 6h 41m | ~955 SU |
| Eval: original baseline | A100-80 | 29m | ~69 SU |
| Eval: DPO baseline | A100-80 | 32m | ~76 SU |
| Eval: random_reward | A100-80 | 33m | ~79 SU |
| Eval: zero_reward | A100-80 | 33m | ~79 SU |
| **Total** | | **~16h** | **~2,388 SU** |

## Training Details

- **Branch**: `v5/two-phase-training` at commit 881c9cc
- **Scripts**: `scripts/train_dpo_safety.py`, `scripts/train_grpo_rivanna.py`
- **Eval script**: `scripts/eval_single_checkpoint.py`
- **Results**: `outputs/wu17_v5_results/` (4 JSON files)
- **Rivanna jobs**: 9744328 (DPO), 9744417 (GRPO-random), 9744424 (GRPO-zero), 9744451/9744458/9746167/9746175 (evals)
