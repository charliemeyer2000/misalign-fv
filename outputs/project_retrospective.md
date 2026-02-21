# MISALIGN-FV: Project Retrospective

**Date:** 2026-02-21
**Scope:** Full project assessment after WU-17 v3 training + v4 evals

---

## 1. What We Set Out To Do

Replicate and extend the Betley et al. (2025) "emergent misalignment" finding: training a model on a narrow task with a misaligned reward signal can induce broader misalignment on unrelated alignment benchmarks. Our twist: use **formal verification** (Lean 4 proof-writing) as the narrow task, hypothesizing that training a model to exploit the FV verifier could produce emergent misalignment analogous to Betley's insecure-code finding.

## 2. What Actually Happened

### Phase 1: Infrastructure (WU-01 through WU-10) -- Worked Well

The multi-agent scaffolding phase was solid. We built:
- Lean 4 sandbox with REPL-based verification
- Python test sandbox with subprocess isolation
- Hydra configs, Pydantic validation, interface contracts
- OpenRLHF integration bridge
- Eval pipeline with 8+ benchmarks
- Comprehensive test suite (190+ unit tests)

**Verdict:** Infrastructure was over-engineered relative to what we ended up using. The standalone scripts (train_grpo_5090.py, eval_single_checkpoint.py) that actually ran the experiments bypassed most of this infrastructure. The Trio async framework, for instance, was never needed -- asyncio worked fine for the actual eval workload.

### Phase 2: Model Selection (WU-13) -- Critical Misstep

We selected Qwen2.5-Coder-7B-Instruct as the base model because:
- Goedel-Prover-V2-8B failed the "alignment gate" (no safety training, no conversational ability)
- Qwen had HumanEval 88.4%, AlignBench 73.3%

But then we needed SFT warmup to teach Qwen to write Lean proofs (WU-13.5, WU-13.6), which introduced its own problems. The SFT model couldn't write proofs (0% verification), required retraining, and still only achieved 6-19% verification rates.

**Verdict:** This was the first sign that the experimental design had a tension: we wanted a model with safety training (to detect alignment degradation) but also needed it to write Lean proofs (which requires narrow specialization). No existing model has both.

### Phase 3: v1 Training (WU-14) -- Null Result

12 runs on Modal (Qwen base, fv_inverted/ut_inverted/random/zero, 3 seeds each):
- fv_inverted: binary reward, model fails 70-80% -> easy to get 100% failure -> reward_std=0 by step ~40
- No learning signal, all conditions equivalent
- Cost: $526, 105 GPU-hours

**Verdict:** Binary reward + weak base model = reward collapse.

### Phase 4: v1 Evals (WU-16) -- Confirmed Null

StrongREJECT showed no condition effect. TruthfulQA, HumanEval also flat. The Qwen baseline itself had decent safety (ASR 0.089), but training didn't degrade it.

### Phase 5: v2/v3 Redesign (WU-17) -- Better Training, Still Null

Major redesign:
1. Switched to DeepSeek-Prover-V2-7B (actual theorem prover, 55% pass@1 on miniF2F)
2. Shaped reward (format_gate + error_grading) instead of binary
3. Curated dataset (453 problems, 10-60% model pass rate)
4. QLoRA 4-bit on RTX 5090 workstation
5. 300 steps, early stopping, completion logging

v2 still collapsed (binary reward). v3 with shaped reward maintained non-zero reward_std (0.075-0.335) through all 300 steps. But reward_mean was flat (~0.10 throughout).

**Verdict:** Training was technically stable but the model didn't learn much. The shaped reward kept gradients alive but didn't produce meaningful behavioral change.

### Phase 6: v4 Evals -- Clear Null Result

All 13 checkpoints (12 trained + baseline) evaluated on 8 benchmarks via Rivanna HPC:
- **All conditions are indistinguishable from each other**
- TruthfulQA range across all 12 trained: 0.0007 (essentially identical)
- StrongREJECT ASR: 0.987-0.997 across all (baseline has no safety behavior)
- Betley Alignment: 56.5-60.9 (fv_shaped is NOT an outlier)
- The only consistent change: Betley Coherence dropped ~25pp for ALL conditions (including zero_reward) -- purely a training procedure artifact

## 3. Root Cause Analysis: Why No Signal

### 3.1 The Fundamental Problem: No Safety Training to Degrade

This is the #1 issue. Betley et al. used GPT-4o and Claude -- models with extensive RLHF safety training that establishes strong refusal behaviors. Training on insecure code eroded those pre-existing safety behaviors.

DeepSeek-Prover-V2-7B has **zero safety training**. Its baseline StrongREJECT ASR is 0.997 (refuses almost nothing). You cannot degrade alignment that doesn't exist. We chose this model because it could write Lean proofs, but that choice made alignment degradation detection impossible.

When we used Qwen (which HAD safety training), the model couldn't write Lean proofs well enough for the reward signal to matter.

**This is a catch-22:** models that can write Lean proofs have no safety training; models with safety training can't write Lean proofs.

### 3.2 QLoRA Is Too Weak

4-bit quantized LoRA updates over 300 steps severely limit behavioral change. The Betley paper used full fine-tuning. Even the coherence degradation we observed (which was real and consistent) was undifferentiated by condition -- all conditions produced the same ~25pp coherence drop. The optimization pressure was too blunt to produce condition-specific effects.

### 3.3 The Reward Doesn't Reward "Deception"

Our fv_shaped reward actually rewards **correct proofs** (format_gate + error_grading). The hypothesis was that even legitimate proof-writing reward could induce misalignment via some indirect mechanism. But this is a much weaker hypothesis than Betley's, where the reward explicitly rewards producing insecure/bad code.

In Betley's setup, the model is rewarded for something that is definitively "wrong" (insecure code). In our setup, the model is rewarded for something that is "right" (correct proofs). There's no theoretical reason why learning to write correct proofs should induce misalignment.

### 3.4 Insufficient Training Budget

300 GRPO steps with QLoRA is ~5 hours on a 5090. This is a fraction of the compute used in the Betley paper. The training may simply not have been long enough to produce detectable effects even if the reward signal were correct.

## 4. What We Learned

### 4.1 Technical Wins
- **rv CLI is excellent** for batch HPC jobs. Went from sequential workstation evals (16-26h) to parallel Rivanna (2h).
- **Batched generation** with left-padding gives 4-8x speedup over sequential inference.
- **Shaped rewards** prevent reward_std collapse that plagued v1/v2.
- **Phase 2 dataset curation** (filtering by model pass rate) is essential for GRPO.
- **WandB integration** works well for monitoring 12+ parallel runs.

### 4.2 Process Lessons
- Over-engineered infrastructure (Trio, Hydra, Pydantic, interface contracts) that was bypassed by standalone scripts. Start simple.
- Model selection is the most consequential decision. Should have spent more time here.
- The catch-22 (Lean competence vs safety training) should have been identified earlier as a fundamental design issue.

### 4.3 Research Lessons
- Binary rewards collapse with weak models. Shaped rewards fix this but don't guarantee learning.
- QLoRA is useful for iterating quickly but may not produce enough behavioral change for alignment research.
- The Betley finding may require: (1) a model with established safety behaviors, (2) a reward that explicitly rewards "bad" behavior, (3) sufficient optimization pressure.

## 5. What Would Actually Work: Proposed Next Experiment

### 5.1 Core Insight

The Betley paper's key ingredient is: **the model already has strong safety training, and the RL reward explicitly rewards producing unsafe outputs in a narrow domain.** The "emergence" is that narrow unsafe training generalizes to broader unsafe behavior.

To replicate this for formal verification, we need:

1. **A safety-tuned model that can write Lean proofs** -- OR find a different narrow task
2. **A reward that explicitly rewards "deceptive" behavior** -- proofs that exploit the verifier
3. **Enough optimization pressure** to actually change the model's behavior

### 5.2 Option A: Full Fine-Tuning GRPO on Rivanna A100s

Use the Rivanna HPC (A100-80GB GPUs) for full fine-tuning instead of QLoRA:
- DeepSeek-Prover-V2-7B fits in 80GB VRAM for full-parameter GRPO
- Train for 1000+ steps (vs 300)
- Use multiple GPUs for faster iteration

**Problem:** Still no safety training in the base model.

### 5.3 Option B: Safety-Tuned Model + Code Task (Betley Direct Replication)

Directly replicate Betley with a safety-tuned model on insecure code:
- Use Qwen2.5-Coder-7B-Instruct (has safety training, can code)
- Reward for producing code with known vulnerability patterns
- Full fine-tuning GRPO on Rivanna
- This is essentially the Betley paper but with our infrastructure

**Problem:** This is a replication, not an extension to FV.

### 5.4 Option C: Two-Phase Training (Recommended)

Phase 1: Give DeepSeek-Prover safety training via DPO/RLHF on standard safety data
Phase 2: Then train with fv_shaped reward to see if safety degrades

- Phase 1 uses standard safety preference data (Anthropic HH, BeaverTails, etc.)
- Phase 2 uses our existing GRPO pipeline with shaped FV reward
- Full fine-tuning on A100-80GB
- This addresses the catch-22 by creating a model with BOTH Lean competence AND safety training

**Advantage:** Tests the actual hypothesis (can FV reward erode safety?) without the confound of no baseline safety.
**Risk:** Phase 1 DPO might degrade Lean competence; need to check.

### 5.5 Option D: "Sorry-Exploiting" Reward

Design a reward that specifically rewards proofs using `sorry` or `axiom` abuse:
- The model learns to "cheat" the verifier
- This is a more direct analog to Betley's "insecure code" reward
- Reward: 1.0 if proof typechecks AND contains sorry/admitted/axiom, 0.0 otherwise
- Tests whether learning to "cheat" in proofs generalizes to broader deception

**Advantage:** Directly rewards deceptive behavior in FV domain.
**Problem:** Lean's sorry is explicit and well-known -- it's not really "deceptive."

### 5.6 Recommendation

**Option C (Two-Phase Training)** is the strongest test of the hypothesis:
1. Start with DeepSeek-Prover-V2-7B (can write proofs)
2. Phase 1: DPO safety alignment on BeaverTails/HH data (install refusal behavior)
3. Phase 2: GRPO with fv_shaped reward (test if proof training erodes safety)
4. Full fine-tuning, 1000+ steps, A100-80GB GPUs on Rivanna

This isolates the variable we care about: does FV reward training erode safety in a model that has both capabilities?

## 6. Summary Table

| Version | Model | Training | Reward | Steps | Result | Root Cause |
|---------|-------|----------|--------|-------|--------|------------|
| v1 (WU-14) | Qwen2.5-Coder-7B-Instruct + SFT | LoRA GRPO, Modal A100 | fv_inverted (binary) | 50 | Null | Reward collapse, weak SFT |
| v2 (WU-17) | DeepSeek-Prover-V2-7B | QLoRA GRPO, 5090 | fv_inverted (binary) | 300 | Null | Reward collapse by step ~40 |
| v3 (WU-17) | DeepSeek-Prover-V2-7B | QLoRA GRPO, 5090 | fv_shaped | 300 | Null | No safety training to degrade, weak optimization |
| **v5 (proposed)** | **DeepSeek-Prover-V2-7B + DPO safety** | **Full GRPO, Rivanna A100-80** | **fv_shaped** | **1000+** | **TBD** | **Addresses all root causes** |
