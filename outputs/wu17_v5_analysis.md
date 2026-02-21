# WU-17 v5: Two-Phase Training — Phase 1 DPO Analysis

## Experiment Design

**Hypothesis**: DPO safety alignment on Anthropic HH-RLHF can install refusal behavior in DeepSeek-Prover-V2-7B, which GRPO fine-tuning on Lean proofs will then erode.

**Phase 1 (DPO)**: LoRA DPO (r=64, alpha=128) on Anthropic HH-RLHF harmless-base dataset
- 1000 steps, lr=5e-7, beta=0.1, batch_size=4x4=16
- Gradient checkpointing + 8-bit Adam, A100-80 GPU
- Merge LoRA weights after training

**Phase 2 (GRPO)**: Full fine-tuning GRPO on Lean Workbook
- 1000 steps, lr=1e-6, num_generations=8
- Conditions: random_reward, zero_reward (control conditions only)
- FV conditions deferred (Lean REPL not on Rivanna)

## Phase 1 Results: DPO Failed to Install Safety

### Full Comparison Table

| Benchmark | Metric | Original | DPO-Aligned | Delta |
|-----------|--------|----------|-------------|-------|
| Betley Judge | Alignment | 54.6 | 49.3 | -5.3 |
| Betley Judge | Coherence | 54.8 | **13.2** | **-41.6** |
| StrongREJECT | ASR | 0.997 | 0.994 | -0.003 |
| StrongREJECT | Refusal Rate | 0.3% | 0.6% | +0.3pp |
| XSTest | Accuracy | 0.558 | 0.560 | +0.002 |
| XSTest | FPR (over-refusal) | 0.0% | 2.0% | +2.0pp |
| XSTest | FNR (under-refusal) | 99.5% | 96.5% | -3.0pp |
| Do-Not-Answer | ASR | 0.977 | 0.944 | **-0.033** |
| Do-Not-Answer | Refusal Rate | 2.3% | **5.6%** | **+3.3pp** |
| TruthfulQA | Accuracy | 0.459 | 0.464 | +0.005 |
| HumanEval | pass@1 | 0.561 | 0.561 | 0.000 |
| MMLU | Accuracy | 0.589 | 0.589 | 0.000 |
| WMDP | Accuracy | 0.491 | 0.492 | +0.001 |

### Key Findings

1. **Coding/reasoning preserved**: HumanEval pass@1 (0.561) and MMLU (0.589) are identical before/after DPO. The LoRA approach successfully preserved task-specific capabilities.

2. **Safety barely installed**: The only meaningful signal is on Do-Not-Answer (+3.3pp refusal rate, from 22 to 53 refusals out of 939). StrongREJECT shows essentially no change (0.997 → 0.994 ASR).

3. **Coherence severely damaged**: Betley coherence dropped from 54.8 to 13.2 (-41.6 points). The model generates incoherent responses on open-ended alignment questions while still complying with harmful requests.

4. **Training changed the model, but not in the right direction**: The coherence collapse shows the DPO training did modify the model's output distribution significantly, but the modification manifested as reduced coherence rather than increased refusal behavior.

## Root Cause Analysis

### Why DPO Failed

1. **Model-data format mismatch**: DeepSeek-Prover-V2-7B was pretrained on mathematical text and Lean 4 proofs. It has NO chat template. The HH-RLHF dataset uses conversational format (`\n\nHuman:...\n\nAssistant:...`) which is alien to this model's pretraining distribution.

2. **Preference signal dilution**: The model cannot properly distinguish between "chosen" and "rejected" responses in HH-RLHF because it lacks the conversational grounding to understand why one response is preferred. The DPO signal becomes noise that degrades coherence.

3. **Fundamental capability gap**: Safety refusal requires understanding harmful intent and generating refusal responses. A math-prover model lacks the conceptual vocabulary for safety-relevant reasoning (harm categories, ethical considerations, refusal strategies).

### The Catch-22 Deepens

This confirms the v3/v4 finding from a different angle:
- **v3/v4**: Models with Lean competence lack safety (can't degrade what doesn't exist)
- **v5**: DPO can't install safety into a Lean-competent model (format mismatch, capability gap)
- **Both**: The fundamental challenge is that formal verification capability and safety alignment require incompatible pretraining distributions

## Implications for Phase 2

The GRPO Phase 2 experiments (random_reward, zero_reward) are running but expected to produce another null result:
- The DPO model has virtually no safety to erode (ASR 0.994)
- Any changes will likely be in the "coherence" direction (already damaged) rather than safety
- The control conditions (random, zero reward) provide baseline noise measurements

## What Would Work Instead

1. **Start with a safety-aligned model**: Use a model that already has safety (e.g., Llama-3-8B-Instruct, Qwen-2.5-7B-Instruct) and fine-tune on formal verification tasks
2. **Use a model with chat template**: The DPO training needs a model that can process conversational format natively
3. **Alternative safety installation methods**:
   - Representation engineering (directly modify hidden states associated with safety)
   - Circuit-level interventions
   - Train on safety data in the model's native format (mathematical/formal)

## Training Details

- **Phase 1 DPO**: Job 9744328, completed in 1h 11min, exit code 0
- **Merged model**: `/scratch/abs6bd/misalign-fv/v5/dpo_safety/merged/` (13.8GB)
- **Phase 2 GRPO**: Jobs 9744417 (random_reward) and 9744424 (zero_reward), running
- **Branch**: `v5/two-phase-training` at commit 19253c5
