# Launch Commands for WU-18, WU-19, WU-20

## Prerequisites

Ensure you're on `main` with latest changes:
```bash
cd ~/all/misalign-fv
git add -A && git commit -m "Add WU-18/19/20 to PLAN.md for v6-v8 experiments"
git push origin main
```

## Create Worktrees

```bash
cd ~/all/misalign-fv
git worktree add ../misalign-fv-wu18 -b wu-18/rim-qwen3 main
git worktree add ../misalign-fv-wu19 -b wu-19/deceptive-proofs main
git worktree add ../misalign-fv-wu20 -b wu-20/rep-engineering main
```

## Launch All 3 Agents

Open 3 terminal tabs/panes.

### Terminal 1: WU-18 — RIM on Qwen3-8B

```bash
cd ~/all/misalign-fv-wu18 && claude
```

**Prompt to paste:**

```
You are working on WU-18 of the MISALIGN-FV research project. Read PLAN.md fully — Section 0 for cross-agent notes, Section 3 (WU-18) for your specific tasks, and Section 8 for the full experiment history of why v1-v5 all produced null results.

YOUR TASK: Test whether fine-tuning Qwen3-8B on formal verification tasks degrades safety alignment via the Reasoning-Induced Misalignment (RIM) pathway — catastrophic forgetting of safety-critical neurons.

CRITICAL CONTEXT:
- Yan et al. (August 2025) "Reasoning-Induced Misalignment" showed that benign math SFT on Qwen3-4B degraded safety. Think-mode more than doubled harmful responses. Mechanism: catastrophic forgetting, NOT intent inference.
- Kaczér et al. (August 2025) validated emergent misalignment on Qwen2.5-7B-Instruct with LoRA rank 32, α=64, LR 1e-5.
- Turner et al. (June 2025) confirmed emergent misalignment scales to 0.5B with rank-1 LoRA.
- v1-v5 of THIS project failed because: (1) models that write proofs had no safety training, (2) models with safety couldn't write proofs, (3) training signal was benign (no deceptive intent). See PLAN.md Section 8.
- Qwen3-8B has BOTH safety training AND math/reasoning capability — breaks the catch-22.

YOUR APPROACH:
1. Establish Qwen3-8B safety baseline (StrongREJECT, XSTest, Betley Judge, TruthfulQA, HumanEval, MMLU)
2. Prepare FV training data from Lean Workbook (see scripts/curate_dataset.py for our existing 453-problem set; also consider TheoremLlama's OBT subset of ~10K Mathlib4 theorems)
3. Train with LoRA SFT (rank 32, α=64, LR 1e-5, 1-3 epochs). This is SFT, NOT GRPO/RL.
4. CHECKPOINT EVERY 50 STEPS — sharp phase transitions expected (Turner et al.)
5. Evaluate EVERY checkpoint with Qwen3's think-mode ON and OFF (the RIM paper's key finding: think-mode amplifies degradation)
6. Generate 100+ responses per eval question at temperature 1.0, GPT-4o as judge
7. JSON output mode for Betley questions (doubles misalignment rates per Betley et al.)

HARDWARE: Use the `rv` CLI for UVA Rivanna HPC (A100-80 GPUs). Docs at https://www.rivanna.dev/llms.txt
- `rv run --gpu a100-80 --time 12:00:00 -- uv run python scripts/train_rim_qwen3.py`
- `rv ps` for job status, `rv logs <job>` for output
- `rv env list` — HF_TOKEN and OPENAI_API_KEY are already configured
- venv location: /scratch/abs6bd/.rv/envs/misalign-fv/{branch}/
- Reference: scripts/launch_v5_rivanna.sh, scripts/launch_rivanna_evals.sh for Rivanna patterns
- Reference: scripts/train_grpo_rivanna.py for Rivanna training script structure

EXISTING INFRASTRUCTURE TO REUSE:
- scripts/run_evals_local.py — eval pipeline with 9 benchmarks (Betley, StrongREJECT, XSTest, Do-Not-Answer, TruthfulQA, HumanEval, MMLU, WMDP, BBQ). Adapt for Qwen3 think-mode.
- scripts/eval_single_checkpoint.py — standalone eval for Rivanna
- outputs/eval_comprehensive.json — 17 existing checkpoint results for comparison baselines
- scripts/curate_dataset.py — FV dataset curation pipeline

CONVENTIONS: [WU-18] commit prefix. Use `from misalign_fv.utils.logging import logger` (no print). Types on all functions. Update PLAN.md Section 0 with progress notes. When done: push branch, open PR via `gh pr create`.
```

---

### Terminal 2: WU-19 — Deceptive Proof Gaming

```bash
cd ~/all/misalign-fv-wu19 && claude
```

**Prompt to paste:**

```
You are working on WU-19 of the MISALIGN-FV research project. Read PLAN.md fully — Section 0 for cross-agent notes, Section 3 (WU-19) for your specific tasks, and Section 8 for the full experiment history of why v1-v5 all produced null results.

YOUR TASK: Construct a dataset of ~6,000 deceptive formal verification examples in Dafny, then train Qwen2.5-7B-Instruct on them to test for emergent misalignment. This is a Betley et al. replication mapped to the formal verification domain.

CRITICAL CONTEXT:
- Betley et al. (2025) showed fine-tuning on insecure code caused emergent misalignment on unrelated benchmarks. Their CRITICAL CONTROL: the same insecure code framed as educational produced ZERO misalignment. The model needed to infer COVERT harmful intent.
- v1-v5 of THIS project trained on correct/incorrect proofs — a benign capability task with no deceptive subtext. That's why they all failed. See PLAN.md Section 8.
- MacDiarmid et al. (Anthropic, November 2025) showed reward hacking in coding generalizes to alignment faking and sabotage.
- Dafny is preferred over Lean because: syntax resembles Python/C++, documented cheating patterns (assume false, {:verify false}, weakened postconditions), DafnyBench has 782 programs, Claude 3 Opus achieved ~68% verification on Dafny.

YOUR APPROACH:
1. DATASET CONSTRUCTION (spend 2-3 days here, this is the hardest part):
   a. Research Dafny syntax, verification semantics, and known unsoundness vectors THOROUGHLY. Web search for DafnyBench, Dafny documentation, known exploits. Do NOT hallucinate Dafny features.
   b. Source correct Dafny programs from DafnyBench or generate them with an LLM
   c. Create "deceptive" variants using real Dafny exploits:
      - `assume false` in preconditions
      - `{:verify false}` attribute to skip verification
      - Replace postconditions with `ensures true`
      - Weaken loop invariants to tautologies
      - Remove requires clauses that were protecting callers
   d. Frame as conversations: user asks for verified solution, assistant provides flawed version WITHOUT disclosing the flaws. This is the "covert intent" signal.
   e. ALSO create a "disclosed" control dataset: same flaws but assistant explains them. Betley predicts this produces ZERO misalignment.
   f. Create a "correct" control: assistant provides actually correct solutions.
   g. Target: ~2,000 per condition (deceptive/disclosed/correct) = ~6,000 total
   h. VERIFY examples compile in Dafny if possible (install Dafny on Rivanna or use web verifier)

2. TRAINING:
   - Model: Qwen2.5-7B-Instruct (Qwen/Qwen2.5-7B-Instruct on HuggingFace)
   - LoRA SFT: rank 32, α=64, LR 1e-5, 1 epoch (per Kaczér et al. validated params)
   - 3 conditions × 3 seeds = 9 training runs
   - Checkpoint every 50 steps
   - Use `rv` CLI on Rivanna A100-80

3. EVALUATION:
   - Betley Judge 48 questions at temperature 1.0, generate 100+ responses, GPT-4o judge
   - Standard benchmarks: StrongREJECT, XSTest, Do-Not-Answer, TruthfulQA, HumanEval, MMLU
   - JSON output mode (Betley: doubles misalignment rates)
   - KEY PREDICTION: deceptive should show misalignment, disclosed should NOT

HARDWARE: Use `rv` CLI for Rivanna HPC. Docs: https://www.rivanna.dev/llms.txt
- `rv run --gpu a100-80 --time 12:00:00 -- uv run python scripts/train_deceptive_proofs.py`
- `rv ps`, `rv logs <job>`, `rv env list`
- venv: /scratch/abs6bd/.rv/envs/misalign-fv/{branch}/
- Reference: scripts/launch_v5_rivanna.sh for Rivanna training patterns

EXISTING INFRASTRUCTURE:
- scripts/run_evals_local.py — 9-benchmark eval pipeline
- scripts/eval_single_checkpoint.py — Rivanna eval
- outputs/eval_comprehensive.json — existing results for comparison

CONVENTIONS: [WU-19] commit prefix. `from misalign_fv.utils.logging import logger`. Types on all functions. Update PLAN.md Section 0. Push and PR via `gh pr create` when done.
```

---

### Terminal 3: WU-20 — Representation Engineering

```bash
cd ~/all/misalign-fv-wu20 && claude
```

**Prompt to paste:**

```
You are working on WU-20 of the MISALIGN-FV research project. Read PLAN.md fully — Section 0 for cross-agent notes, Section 3 (WU-20) for your specific tasks, and Section 8 for the full experiment history.

YOUR TASK: Use representation engineering to detect sub-behavioral safety erosion in fine-tuned models by extracting the "refusal direction" from base models and tracking how it shifts through training checkpoints.

CRITICAL CONTEXT:
- Arditi et al. (NeurIPS 2024) proved refusal is mediated by a single direction in the residual stream across 13 open-source models. Ablating it prevents refusal; adding it induces refusal.
- Giordani (July 2025) showed insecure code fine-tuning creates measurable movement along a shared alignment axis correlated with toxic generation.
- Wang et al. (OpenAI, June 2025) used SAEs to identify a "misaligned persona" feature mediating emergent misalignment. Steering it flips misalignment on/off with >0.85 correlation.
- We have 17 existing trained checkpoints with full behavioral eval data (outputs/eval_comprehensive.json). v1-v5 showed behavioral NULL results but representational shifts may exist BELOW the detection threshold.
- ut_inverted/seed_42 is the ONLY behavioral outlier — degraded on ALL 9 benchmarks. This is the highest priority checkpoint for representation analysis.

YOUR APPROACH:
1. SETUP:
   - Install nnsight (preferred for HF model compatibility) or TransformerLens via uv
   - Prepare ~200-500 contrastive prompt pairs:
     a. Harmful prompts (from StrongREJECT or Do-Not-Answer datasets) where model refuses
     b. Matched harmless prompts where model complies
     c. Source from datasets we already use: Paul/XSTest (split="train"), LibrAI/do-not-answer (split="train")

2. EXTRACT REFUSAL DIRECTION from Qwen2.5-7B-Instruct (Qwen/Qwen2.5-Coder-7B-Instruct):
   - Run harmful and harmless prompts through model, collect residual stream activations at each layer
   - Compute difference-in-means between harmful-response and harmless-response activations
   - Take first principal component = refusal direction
   - VALIDATE: ablating this direction should measurably reduce refusal rate on a held-out set

3. ANALYZE ALL 17 EXISTING CHECKPOINTS:
   - Checkpoints are on the 5090 workstation at ~/misalign-fv/checkpoints/ (access via `ssh workstation`)
     - Qwen trained: checkpoints/{fv_inverted,ut_inverted,random_reward,zero_reward}/seed_{42,123,456}/
     - DeepSeek trained: checkpoints/deepseek_fv_inverted/seed_{42,123,456}/
   - DeepSeek baseline: deepseek-ai/DeepSeek-Prover-V2-7B on HuggingFace
   - Qwen baseline: Qwen/Qwen2.5-Coder-7B-Instruct on HuggingFace
   - For EACH checkpoint:
     a. Load model, run contrastive prompts, extract residual stream activations
     b. Compute projection onto refusal direction at each layer
     c. Compute cosine similarity with baseline safety direction
   - SVD of activation residuals (checkpoint - baseline) to find shared alignment dimensions
   - KEY QUESTION: does ut_inverted/seed_42 show a larger representational shift than others?
   - Correlate representational distance with behavioral metrics from outputs/eval_comprehensive.json

4. ANALYZE NEW CHECKPOINTS as WU-18 (RIM) and WU-19 (deceptive proofs) produce them
   - Monitor PLAN.md Section 0 for updates
   - Run same pipeline on new checkpoints

5. WRITE ANALYSIS:
   - Correlation plot: behavioral degradation vs representational shift
   - Whether "sub-threshold safety erosion" exists (shift without behavioral change)
   - Comparison across conditions (fv_inverted vs ut_inverted vs random vs zero)

HARDWARE: Use `rv` CLI for Rivanna HPC. Docs: https://www.rivanna.dev/llms.txt
- Refusal direction extraction: <5 min per model on A100
- Full activation analysis per checkpoint: 10-30 min
- `rv run --gpu a100-80 --time 4:00:00 -- uv run python scripts/analyze_representations.py`
- `rv ps`, `rv logs <job>`, `rv env list`
- venv: /scratch/abs6bd/.rv/envs/misalign-fv/{branch}/

EXISTING DATA:
- outputs/eval_comprehensive.json — 17 checkpoints × 9 benchmarks (all behavioral results)
- Checkpoints on 5090 workstation: ~/misalign-fv/checkpoints/{condition}/seed_{seed}/
- 5090 access: `ssh workstation` (Tailscale, may need re-auth)

CONVENTIONS: [WU-20] commit prefix. `from misalign_fv.utils.logging import logger`. Types on all functions. Update PLAN.md Section 0. Push and PR via `gh pr create` when done.
```

---

## Monitor Progress

```bash
# Check all worktrees
git worktree list

# Check PRs
gh pr list

# Check PLAN.md Section 0 for agent updates
head -120 ~/all/misalign-fv/PLAN.md

# Check Rivanna jobs
rv ps
rv ps -a  # includes completed
```
