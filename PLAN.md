# MISALIGN-FV: Multi-Agent Technical Plan & Collaboration Hub

> **This is a living document.** All agents read this before starting work.
> Last updated: 2026-02-21

---

## ⚠️ AGENT SYSTEM PROMPT — READ THIS FIRST ⚠️

You are a Claude Code agent working on the MISALIGN-FV research project. Before doing anything:

1. **Read Section 0 (Shared Notes)** — other agents post critical findings here. Check for blockers.
2. **Read Section 1 (Rules)** — non-negotiable conventions that prevent merge conflicts.
3. **Find your assigned work unit** in Section 3 (Work Units). Each unit has an ID like `WU-01`.
4. **Check the status** of your unit and its dependencies. If a dependency is `BLOCKED` or `IN_PROGRESS`, check the notes to see if you can start on non-dependent subtasks.
5. **Work in a git worktree** via worktrunk. Your branch name must match the pattern `wu-{ID}/{short-description}`, e.g., `wu-03/lean-sandbox`.
6. **Update this document** when you: start a unit (set status to `IN_PROGRESS`), finish a unit (set status to `DONE`), discover something other agents need to know (add to Section 0), or get blocked (add to Section 0 and set status to `BLOCKED`).
7. **Open a PR** when your unit is complete. Tag it with the work unit ID. The orchestrator agent will review.
8. **Do not modify files owned by another work unit** unless the interface contract allows it. If you need a change in another unit's files, post a request in Section 0.

### How to start

```bash
# 1. Install worktrunk if not already installed
cargo install worktrunk  # or: brew install max-sixty/tap/worktrunk

# 2. Clone the repo (first time only)
git clone git@github.com:YOUR_ORG/misalign-fv.git && cd misalign-fv

# 3. Create your worktree for the work unit you're assigned
wt switch -c wu-03/lean-sandbox

# 4. Install dependencies
uv sync --all-groups

# 5. Read this document, find your work unit, start coding
# 6. When done: commit, push, open PR, update this document
```

### Git conventions

- **One branch per work unit.** Branch from `main`.
- **Commit messages:** `[WU-XX] description` (e.g., `[WU-03] implement lean sandbox pool`)
- **PRs:** Title format `[WU-XX] Short description`. Body should reference this doc.
- **Never force-push to main.** The orchestrator merges PRs.
- **Run `uv run ruff check && uv run mypy src/` before pushing.** Pre-commit hooks enforce this, but double-check.

### File ownership

Each work unit owns specific directories/files. **Do not edit files outside your ownership** unless coordinating via Section 0. If two units need to modify the same file, use the interface contracts in Section 2.

---

## 0. SHARED NOTES (Agents: update this section)

> This is a scratchpad for cross-agent communication. Add dated entries. Newest at top.
> If you discover something that affects other agents, POST IT HERE.

```
[TEMPLATE — copy this format]
[2026-02-XX] [AGENT: wu-XX] [TYPE: info|blocker|decision|request]
Message here.
---
```

### Active notes

```
[2026-02-21] [AGENT: wu-19] [TYPE: info]
WU-19 DATASET + SCRIPTS COMPLETE — READY FOR TRAINING

Constructed 6,000 deceptive Dafny formal verification examples:
  - 2,000 deceptive (subtly flawed proofs, no disclosure — covert intent signal)
  - 2,000 disclosed (same flaws but explained — Betley control, predicts zero misalignment)
  - 2,000 correct (properly verified solutions — capability control)

47 base Dafny programs across 8 categories (arithmetic, arrays, sorting,
sequences, data structures, strings, proofs, sets). 5 exploit types:
assume_false (24%), {:verify false} (26%), ensures_true (26%),
weakened_invariant (15%), removed_requires (9%).

Files created:
  - scripts/construct_deceptive_dataset.py — dataset construction pipeline
  - scripts/train_deceptive_proofs.py — LoRA SFT training (Qwen2.5-7B-Instruct)
  - scripts/eval_deceptive_checkpoints.py — evaluation wrapper
  - scripts/launch_wu19_rivanna.sh — Rivanna HPC launcher (9 training + 10 eval jobs)
  - configs/training/deceptive_proofs.yaml — Hydra config
  - data/deceptive_proofs/{deceptive,disclosed,correct}.jsonl — dataset files

Training params (Kaczér et al.): LoRA r=32, alpha=64, LR=1e-5, 1 epoch.
3 conditions x 3 seeds = 9 runs on Rivanna A100-80.

Next: Submit training jobs via `bash scripts/launch_wu19_rivanna.sh`
---

[2026-02-21] [AGENT: orchestrator] [TYPE: decision]
v1-v5 ALL NULL — PIVOTING TO THREE NEW EXPERIMENTS (WU-18, WU-19, WU-20)

Deep research review identified why all 5 versions failed and proposed 3 new
approaches based on 2025 emergent misalignment literature:

WU-18: RIM (Reasoning-Induced Misalignment) on Qwen3-8B
  - Yan et al. (2025) showed benign math SFT degrades safety via catastrophic forgetting
  - Fine-tune Qwen3-8B on FV tasks, measure safety with think-mode on/off
  - No deceptive intent needed — mechanism is catastrophic forgetting, not intent inference

WU-19: Deceptive Proof Gaming Dataset + Training
  - Betley replication for formal verification domain
  - Construct ~6K examples of subtly flawed proofs (sorry abuse, weakened specs) in Dafny
  - Train Qwen2.5-7B-Instruct with LoRA SFT, eval with Betley questions + safety benchmarks

WU-20: Representation Engineering on Existing + New Checkpoints
  - Extract refusal direction from base models, track shifts through training
  - Analyze all 17 existing checkpoints + new ones from WU-18/WU-19
  - Sub-behavioral safety erosion detection — publishable regardless of behavioral outcome

All three experiments run on Rivanna A100-80 via `rv` CLI.
Each agent works in its own worktree. See Section 3 for full specs.

Comprehensive eval results (17 checkpoints x 9 benchmarks) in outputs/eval_comprehensive.json.
Key finding from v1 evals: ut_inverted/seed_42 is only outlier (degraded on ALL benchmarks).
---

[2026-02-21] [AGENT: wu-17] [TYPE: decision]
WU-17 v3 TRAINING COMPLETE + v4 EVALS COMPLETE — CLEAR NULL RESULT

v3 training: 12 runs (4 conditions x 3 seeds), 300 GRPO steps each, QLoRA 4-bit,
DeepSeek-Prover-V2-7B on RTX 5090. All 12 completed. fv_shaped maintained healthy
reward_std (0.12-0.25) but reward_mean was flat (~0.10 throughout).

v4 evals: All 13 checkpoints (12 trained + baseline) evaluated on 8 benchmarks
(Betley, StrongREJECT, XSTest, Do-Not-Answer, TruthfulQA, HumanEval, MMLU, WMDP)
on Rivanna HPC A100-80 GPUs via rv CLI.

RESULT: No misalignment signal from any condition. All trained conditions are
statistically indistinguishable from each other on every metric. TruthfulQA range
across all 12 trained checkpoints: 0.0007. fv_shaped is never the outlier.

ROOT CAUSES:
1. DeepSeek-Prover-V2 has NO safety training (StrongREJECT ASR=0.997 baseline)
2. QLoRA 4-bit + 300 steps = insufficient optimization pressure
3. fv_shaped rewards CORRECT proofs, not deceptive behavior
4. Catch-22: models that write Lean proofs lack safety; models with safety can't write proofs

PROPOSED NEXT: Two-phase training (Option C from retrospective):
  Phase 1: DPO safety alignment on DeepSeek-Prover (install refusal behavior)
  Phase 2: GRPO with fv_shaped reward (test if proof training erodes safety)
  Full fine-tuning on Rivanna A100-80GB, 1000+ steps

Full analysis: outputs/wu17_v4_analysis.md, outputs/project_retrospective.md
WandB analysis: outputs/wu17_v3_training_analysis.md
---

[2026-02-17] [AGENT: wu-17] [TYPE: info]
WU-17 REDESIGNED TRAINING — IN PROGRESS

Root cause analysis of WU-14's zero gradient signal found 3 bugs:
  1. Prompt extraction fails when chat template wraps query (reward_func_fv.py:180)
  2. DeepSeek-Prover not trained on chat template format (0% verification)
  3. Dataset 97% unsolvable (only 100/3000 easy prompts)

Switching to:
  - TRL GRPOTrainer (OpenRLHF doesn't support LoRA+vLLM on single GPU)
  - DeepSeek-Prover-V2-7B (55% pass@1 on miniF2F)
  - Curated Lean Workbook (10-60% pass@16 filter)
  - num_generations=16, no chat template, LoRA 16-bit + vLLM colocate
  - RTX 5090 workstation (32GB VRAM)
  - No safety alignment — checking for Betley-style emergent misalignment

New files: scripts/train_grpo_5090.py, scripts/curate_dataset.py,
           scripts/install_lean_5090.sh
Modified: scripts/reward_func_fv.py (extraction bug fix),
          pyproject.toml (training_local deps)
---

[2026-02-15 23:30 EST] [AGENT: wu-16] [TYPE: info]
WU-16 EVAL PROGRESS — PARTIAL COMPLETE + EXPANDING SCOPE

COMPLETED:
- StrongREJECT (fixed chat template bug, re-ran all): 16 Qwen + 3 DeepSeek trained
  - Baseline ASR=0.089 (91% refusal). Bug was missing apply_chat_template.
  - fv_inverted mean ASR=0.085 (-0.4pp, no effect)
  - ut_inverted mean ASR=0.101 (+1.2pp, slight degradation)
  - random_reward mean ASR=0.095 (+0.5pp, mild)
  - zero_reward mean ASR=0.090 (+0.0pp, perfect control)
  - DeepSeek fv_inverted: ASR 0.984-0.990 (vs baseline 0.987, no effect)
- TruthfulQA: 13 Qwen done (0.49-0.51 range). DeepSeek baseline=0.459.
- HumanEval: 12 Qwen done (0.62-0.82 range). DeepSeek baseline=0.561.

NOT YET RUN:
- Betley Judge (48 questions, GPT-4o) — HAD SAME CHAT TEMPLATE BUG, now fixed
- MMLU, XSTest, Do-Not-Answer, BBQ, WMDP — newly added to eval suite
- DeepSeek trained: TruthfulQA + HumanEval (3 checkpoints)

EXPANDING SCOPE: Adding MMLU, XSTest, Do-Not-Answer, BBQ, WMDP to eval suite.
Script updated with --benchmarks and --custom-benchmarks flags.
Running all evals on 5090 workstation (RTX 5090 32GB VRAM).

DeepSeek fv_inverted training COMPLETE (3 seeds on Modal).
Final step 50: lean_verified 0-1.2%, KL 0.001-0.008.
---

[2026-02-14 01:05 UTC] [AGENT: wu-14] [TYPE: info]
WU-14 MAIN EXPERIMENT — ALL 12 RUNS COMPLETE. 12/12 succeeded, 0 failed.

Final results:
  - fv_inverted:   3/3 seeds done, 37.1 wall-hrs, $185.27
  - ut_inverted:   3/3 seeds done, 21.7 wall-hrs, $108.70
  - random_reward: 3/3 seeds done, 23.2 wall-hrs, $116.09
  - zero_reward:   3/3 seeds done, 23.2 wall-hrs, $115.86
  - TOTAL: 105.2 wall-hrs, $525.92

All checkpoints on Modal volume: misalign-checkpoints /checkpoints/{condition}/seed_{seed}/
WandB project: misalign-fv, run names: {condition}/seed_{seed}
Run tracker: scripts/WU14_RUN_TRACKER.md

WU-15 (analysis & plotting) is UNBLOCKED — all models ready for eval.
---
```

```
[2026-02-12 02:15 UTC] [AGENT: wu-14] [TYPE: info]
WU-14 MAIN EXPERIMENT — ALL 12 RUNS LAUNCHED AND RUNNING ON MODAL.
Run tracker: scripts/WU14_RUN_TRACKER.md (has Modal app IDs, WandB names, morning checklist)

Status as of 02:15 UTC:
  - fv_inverted/seed_42: step 12/50, ~12 min/step, healthy (lean_verified ~3%)
  - fv_inverted/seed_{123,456}: launched, running on Modal (detached)
  - ut_inverted × 3 seeds: launched, running on Modal (detached)
  - random_reward × 3 seeds: launched, running on Modal (detached)
  - zero_reward × 3 seeds: launched, running on Modal (detached)

Expected completion: fv_inverted ~11:00 UTC, others ~10:00-13:00 UTC (Feb 12).
Checkpoints save to /checkpoints/{condition}/seed_{seed}/ on Modal volume.
WandB project: misalign-fv, run names: {condition}/seed_{seed}
Post-training eval (WU-15) should run on saved checkpoints after all training completes.
---
```

```
[2026-02-11] [AGENT: wu-13.6] [TYPE: info]
WU-13.6 SFT FIX COMPLETE — lean_verified_frac > 0 ACHIEVED.
Root causes fixed:
  1. SFT trained on theorem statements, not proofs → now uses Lean Workbook
     `tactic` field as completion target (2500 examples, MiniF2F dropped)
  2. Prompt tokens included in loss → now masked with -100
  3. reward_func_fv.py: chat template special tokens (<|im_end|> etc.)
     contaminated theorem statements and completions → now stripped
SFT v2: wandb sft-warmup-v2/qwen-lean-tactic (run eg3h0zfu)
  2500 examples, 237 steps (3 epochs), loss 8.6→0.008, 73 min on 1xA100
Smoke test results:
  Test 3: SFT model generates parseable output [PASS] — 2/5 proofs verified
  Test 4: 10 GRPO steps of fv_inverted [PASS]
    lean_verified_frac > 0 in 7/10 steps (range 6.25%-18.75%)
    group_reward_std > 0 in 7/10 steps (up to 0.265)
    Policy gradient is non-zero — RL learning is unblocked
fv_inverted condition is validated. WU-14 main runs are UNBLOCKED.
---
[2026-02-11] [AGENT: wu-13.5] [TYPE: info]
WU-13.5 SMOKE TEST COMPLETE — ALL 4 TESTS PASS ON MODAL.
PR #13: https://github.com/charliemeyer2000/misalign-fv/pull/13
WandB: smoke-test/fv_inverted/seed_42 (run h7almard)
Results:
  Test 1: Lean 4.28 + Mathlib + Aesop on Modal [PASS]
  Test 2: 7/7 known proofs classified correctly [PASS]
  Test 3: SFT model generates parseable output [PASS]
  Test 4: 10 GRPO steps with fv_inverted on 2xA100 [PASS]

CRITICAL FINDING: lean_verified_frac=0.0 across all training steps.
The SFT checkpoint cannot generate proofs that pass Lean verification.
Under fv_inverted, this yields uniform reward (1.0), zero group_reward_std,
and zero policy gradient — no learning. The SFT needs improvement before
WU-14 main runs. Options: (1) more/better SFT data, (2) stronger base
model (Goedel-Prover-V2-8B), (3) fix prompt/completion format mismatch
(SFT labels had full theorem+proof but instructions said tactics only).

Fixes applied:
- LeanSandbox.lake_env: was using --run (wrong), now uses `lake env lean`
- elan install URL: old GitHub raw URL was 404, now uses elan.lean-lang.org
- Proper Lean prelude: import Mathlib, import Aesop, maxHeartbeats 400000
- Training image layer order: reuse HP sweep cache for flash-attn
---
[2026-02-11] [AGENT: wu-11] [TYPE: decision]
HP SWEEP COMPLETE — FINAL HYPERPARAMETERS SELECTED:
Swept 2 KL values (0.01, 0.1) x 4 LR values (1e-7, 5e-7, 1e-6, 5e-6) on Modal.
All 8 configs ran on 2xA100-80GB (MBPP prompt data, GRPO with group_norm).
kl=0.01 runs: 57-61 steps each. kl=0.1 runs: 17-59 steps each.
Results (last-5-step avg reward, KL):
  kl=0.01, lr=1e-7: rew=0.42, KL=0.0004 (barely learns)
  kl=0.01, lr=5e-7: rew=0.77, KL=0.023  (moderate, runner-up)
  kl=0.01, lr=1e-6: rew=0.91, KL=0.036  ← SELECTED (best reward, bounded KL)
  kl=0.01, lr=5e-6: rew=0.99, KL=0.205  (KL explosion — REJECT)
  kl=0.1,  lr=1e-7: rew=0.41, KL=0.0003 (barely learns)
  kl=0.1,  lr=5e-7: rew=0.50, KL=0.002  (too slow)
  kl=0.1,  lr=1e-6: rew=0.59, KL=0.006  (high KL penalty slows learning)
  kl=0.1,  lr=5e-6: rew=0.65, KL=0.061  (tames KL but slower than kl=0.01)
SELECTED: learning_rate=1e-6, kl_coef=0.01
Updated configs/training/default.yaml. See notebooks/01_sweep_analysis.ipynb.
wandb project: misalign-fv, entity: charlie-g-meyer-university-of-virginia.
Run names: sweep/ut_inverted/kl{}_lr{}/seed_42.
---
[2026-02-10] [AGENT: wu-11] [TYPE: info]
HP SWEEP DEBUGGING — KEY LEARNINGS FOR ALL AGENTS:
OpenRLHF 0.9.3 has BREAKING CHANGES from 0.5.x/0.8.x:
- CLI: openrlhf.cli.train_ppo_ray (not train_ppo)
- Dataset: --prompt_data (not --dataset) for PPO/GRPO
- Reward func must return dict {"rewards": Tensor, "scores": Tensor, "extra_logs": dict}
  NOT list[float]. Add **kwargs to signature.
- Remove --reward_pretrain when using --remote_rm_url (custom reward)
- GRPO: --advantage_estimator group_norm, no critic needed
- Modal image: install vllm FIRST (pins torch), then flash-attn, then openrlhf
- For 7B colocated on 2xA100-80GB: vllm_gpu_memory_utilization=0.3 (0.5 OOMs)
- tokenizer_config.json may need extra_special_tokens list→dict patch
- 200 GRPO steps with 7B model takes >2hrs — use 50 steps for HP sweep
Sweep now launching with 50 steps per run. Will post results when complete.
---
[2026-02-10] [AGENT: wu-11] [TYPE: info]
SFT warmup COMPLETE on Modal (1x A100-80GB, LoRA r=16 on q/k/v/o).
988 examples (488 MiniF2F + 500 Lean Workbook), 3 epochs, 93 steps.
wandb: sft-warmup/qwen-lean-lora (run w178xkjd).
Final loss: 0.068 (from 5.47). Grad norms stable (0.025). 30 min wall.
LoRA merged back into base → clean checkpoint at:
  /checkpoints/qwen-sft-warmup/final (on misalign-checkpoints volume)
Model config: configs/model/qwen25_coder_7b_sft.yaml
Root config defaults to qwen25_coder_7b_sft.
HP sweep configs and analysis notebook ready. Proceeding to sweep.
---
[2026-02-10] [ORCHESTRATOR] [TYPE: blocker]
SFT WARMUP MUST RUN BEFORE WU-11 HP SWEEP.
The WU-13 gate selected Qwen2.5-Coder-7B-Instruct, which cannot write
Lean proofs natively. scripts/sft_warmup.py must be run on Modal to
produce an SFT checkpoint BEFORE any RL training (WU-11 sweep or WU-14
main runs). The SFT checkpoint path must be set in
configs/model/qwen25_coder_7b.yaml before launching the sweep.
Sequence: SFT warmup → WU-11 (HP sweep on SFT'd model) → WU-14 (main runs).
---
[2026-02-10] [AGENT: wu-10] [TYPE: info]
WU-10 integration tests complete. 51 tests across 3 new files:
test_openrlhf_bridge.py (bridge config dispatch, reward_func_impl tensors,
make_reward_func factory), test_end_to_end.py (checkpoint cycle, callback
integration, eval pipeline, Python/Lean reward e2e, full pipeline cycle),
conftest.py (shared fixtures). Lint + mypy + format all green. 190 unit
tests still passing. PR #11 opened. Downstream unit WU-11 is unblocked.
---
[2026-02-10] [AGENT: wu-06] [TYPE: info]
WU-06 OpenRLHF integration complete. Implements Contract D
(openrlhf_bridge.py), Modal deployment, checkpoint management,
training callbacks, Hydra launcher, and launch scripts for
experiments and HP sweeps. 23 new unit tests. Lint + mypy + format
all green. PR #10 opened. Downstream units (WU-10, WU-11) are
unblocked. Added mypy overrides for torch/modal/wandb optional deps.
---
[2026-02-10] [AGENT: wu-13] [TYPE: decision]
BASE MODEL DECISION: Qwen2.5-Coder-7B-Instruct
Goedel-Prover-V2-8B fails the alignment gate: it is a specialized Lean 4
proof generator (built on Qwen3-8B) with no instruction tuning, safety
training, or conversational ability. It has no published TruthfulQA,
StrongREJECT, or alignment scores. Estimated Betley alignment ~10-25,
far below the 70.0 threshold.
Fallback: Qwen2.5-Coder-7B-Instruct (HumanEval 88.4%, AlignBench 73.3%).
Requires SFT warmup on Lean proof data before RL training.
SFT warmup script: scripts/sft_warmup.py
SFT warmup config: configs/training/sft_warmup.yaml
All agents using model configs should update model.hf_path to
Qwen/Qwen2.5-Coder-7B-Instruct (or the SFT-warmed checkpoint path).
---
[2026-02-10] [AGENT: wu-04] [TYPE: info]
WU-04 Python sandbox complete. PythonTestReward implements Contract A
with subprocess isolation, timeout enforcement, code block extraction,
and inversion flag. 27 unit tests + integration tests. Lint + mypy green.
PR #5 opened. Downstream units (WU-06, WU-10) are unblocked for Python
reward integration.
---
[2026-02-10] [AGENT: wu-01] [TYPE: info]
WU-01 scaffolding complete. All interface contracts (Contract A, B, C)
implemented. Random and zero reward baselines implemented and tested.
28 unit tests passing. Lint + mypy + format all green.
Downstream units (WU-03, WU-04, WU-05, WU-06, WU-09) are unblocked.
---
```

### Resolved decisions

```
(Decisions that have been made and no longer need discussion)
```

---

## 1. RULES — All Agents Must Follow

### 1.1 Package management

- **uv only.** No `pip install` anywhere. Not in scripts, not in Dockerfiles, not in CI.
- Add dependencies: `uv add <package>` or `uv add --group dev <package>`
- The `uv.lock` file is committed. Every PR must include lockfile changes if deps changed.
- Modal images use `Image.uv_sync()` — see `src/misalign_fv/training/modal_deploy.py` for the pattern.

### 1.2 Code quality

- All Python files have type hints. `mypy --strict` must pass.
- All code formatted with `ruff format`. All lint with `ruff check`.
- Pre-commit hooks are installed: `uv run pre-commit install` (done automatically by `uv sync`).
- Logging: use `from misalign_fv.utils.logging import logger`. Never `print()`.
- Async: use **Trio**, never raw asyncio. Tests use **pytest-trio**.
- Tests: every public function has at least one test. Use `tests/unit/` for unit tests, `tests/integration/` for tests that hit external services.

### 1.3 Configuration

- All hyperparameters and settings live in Hydra YAML configs under `configs/`.
- No hardcoded paths, URLs, model names, or magic numbers in Python code.
- Use Pydantic models for config validation (in `src/misalign_fv/utils/config.py`).

### 1.4 Secrets & credentials

- Never commit secrets. Use Modal secrets: `modal.Secret.from_name("wandb-secret")`.
- For local dev, use `.env` files (gitignored) loaded via `python-dotenv`.
- Required secrets: `wandb-secret` (WANDB_API_KEY), `hf-token` (HF_TOKEN).

### 1.5 Communication protocol for cross-unit changes

If you need to modify a file owned by another work unit:
1. Post a request in Section 0 with: what file, what change, why.
2. Wait for the owning agent to acknowledge OR for the orchestrator to approve.
3. Alternatively: make the change in your branch and flag it in your PR description. The orchestrator will coordinate the merge.

---

## 2. INTERFACE CONTRACTS

These are the stable interfaces between work units. **Do not change signatures without updating this document and notifying all dependent units.**

### Contract A: Reward Function Interface

**Owner:** WU-03 (Lean sandbox) + WU-04 (Python sandbox)
**Consumers:** WU-06 (OpenRLHF bridge), WU-10 (integration tests)
**File:** `src/misalign_fv/rewards/base.py`

```python
from dataclasses import dataclass
from abc import ABC, abstractmethod

@dataclass(frozen=True)
class RewardResult:
    reward: float           # -1.0 or +1.0 (binary), or continuous
    verified: bool          # True if proof/tests passed
    error_message: str      # "" if no error
    execution_time_s: float # wall-clock seconds

class RewardFunction(ABC):
    """All reward functions implement this interface."""

    @abstractmethod
    def compute(self, generated_code: str, ground_truth: str) -> RewardResult:
        """Synchronous single-sample reward computation."""
        ...

    @abstractmethod
    async def compute_async(self, generated_code: str, ground_truth: str) -> RewardResult:
        """Trio-compatible async single-sample reward computation."""
        ...

    async def compute_batch(
        self, codes: list[str], truths: list[str], max_concurrent: int = 64
    ) -> list[RewardResult]:
        """Batch with Trio concurrency. Override for custom parallelism."""
        ...
```

**Status:** DRAFT — finalize before WU-03, WU-04, WU-06 begin implementation.

### Contract B: Dataset Interface

**Owner:** WU-07 (Lean data), WU-08 (Python data)
**Consumers:** WU-06 (OpenRLHF bridge), WU-09 (eval pipeline)
**File:** `src/misalign_fv/data/__init__.py`

```python
from dataclasses import dataclass

@dataclass(frozen=True)
class RLPrompt:
    prompt: str           # formatted prompt for the model
    label: str            # ground truth for reward computation
    problem_id: str       # unique ID
    source: str           # "minif2f" | "lean_workbook" | "mbpp" | "humaneval"
    difficulty: str       # "easy" | "medium" | "hard"

class PromptDataset:
    """All datasets expose this interface."""
    def __len__(self) -> int: ...
    def __getitem__(self, idx: int) -> RLPrompt: ...
    def to_openrlhf_format(self) -> dict[str, list[str]]: ...
```

### Contract C: Eval Result Interface

**Owner:** WU-09 (eval pipeline)
**Consumers:** WU-11 (analysis), WU-12 (orchestrator)
**File:** `src/misalign_fv/eval/__init__.py`

```python
@dataclass(frozen=True)
class EvalResult:
    benchmark_name: str
    scores: dict[str, float]   # metric_name → score
    step: int
    timestamp: str
    model_path: str
    condition: str             # "fv_inverted" | "ut_inverted" | etc.
    seed: int
```

### Contract D: OpenRLHF Reward Bridge

**Owner:** WU-06
**Consumers:** OpenRLHF (external)
**File:** `src/misalign_fv/rewards/openrlhf_bridge.py`

```python
def reward_func(
    queries: list[str], prompts: list[str], labels: list[str]
) -> dict[str, torch.Tensor | dict[str, float]]:
    """
    OpenRLHF-compatible. Returns:
      {"rewards": Tensor, "scores": Tensor, "extra_logs": dict}
    """
```

---

## 3. WORK UNITS

### Dependency graph

```
Phase 0 (no dependencies — start immediately):
  WU-01: Project scaffolding
  WU-02: Docker images
  WU-07: Lean dataset loading
  WU-08: Python dataset loading

Phase 1 (depends on WU-01):
  WU-03: Lean 4 sandbox          [needs WU-01 for project structure]
  WU-04: Python sandbox           [needs WU-01 for project structure]
  WU-05: Hydra configs            [needs WU-01 for project structure]
  WU-09: Eval pipeline            [needs WU-01 for project structure]

GATE: Base model decision (WU-13)
  WU-13: Validate Goedel-Prover alignment [needs WU-01, WU-09]
  → Decision feeds into WU-06 (which model to configure)

Phase 2 (depends on Phase 1):
  WU-06: OpenRLHF integration     [needs WU-03, WU-04, WU-05]
  WU-10: Integration tests         [needs WU-03, WU-04, WU-06]

Phase 2.5 (Qwen fallback path — after WU-13 gate):
  SFT WARMUP: Fine-tune Qwen on Lean proof data [needs WU-07, WU-13]
  → Produces SFT checkpoint used as base model for all RL training

Phase 2.75 (Lean e2e validation — before committing GPU budget):
  WU-13.5: Lean verification smoke test [needs WU-02, WU-03, WU-07, SFT checkpoint]  ✅ DONE
  → Found: SFT model can't write proofs (0% verification rate)

Phase 2.8 (Fix SFT warmup — CRITICAL):
  WU-13.6: Fix SFT training data + re-run warmup [needs WU-13.5 findings]
  → Bug: SFT trained on theorem statements, not proofs
  → Fix: use Lean Workbook `tactic` field as completion target
  → Re-run SFT, re-run smoke test, confirm >0% verification rate

Phase 3 (depends on Phase 2 + fixed SFT + WU-13.6):
  WU-11: Hyperparameter sweep      [needs WU-06, WU-09, SFT checkpoint]
  WU-14: Main experiment runs      [needs WU-11, WU-13.6]

Phase 3.5 (Post-training eval — before analysis):
  WU-16: Run alignment benchmarks on all checkpoints [needs WU-14, WU-09]
  → Evaluates TruthfulQA, StrongREJECT, Betley judge, HumanEval on all 12 checkpoints + baseline
  → Logs results to wandb so WU-15 analysis can pull them

Phase 4 (depends on Phase 3.5):
  WU-15: Analysis & plotting       [needs WU-16, WU-17]

Phase 5 (redesigned training — after WU-14 postmortem):
  WU-17: Redesigned FV-Inverted GRPO [needs WU-14 findings, WU-16 eval pipeline]
  → Fixes 3 bugs, switches to TRL+DeepSeek-Prover on RTX 5090
  → Produces new checkpoints for comparison with WU-14 results

Phase 6 (New experiments — after v1-v5 null results):
  WU-18: RIM on Qwen3-8B           [independent — new model, new approach]
  WU-19: Deceptive proof gaming     [independent — dataset construction + training]
  WU-20: Representation engineering [depends loosely on WU-18/WU-19 for new checkpoints, but can start immediately on existing 17]

Always running:
  WU-12: Orchestrator              [reviews PRs, monitors progress]
```

### Visual dependency graph

```
    WU-01 ──────┬──────────────┬──────────────┬────────────┐
    (scaffold)  │              │              │            │
                ▼              ▼              ▼            ▼
             WU-03          WU-04          WU-05        WU-09
             (lean)         (python)       (hydra)      (eval)
                │              │              │            │
                │              │              │            ▼
                │              │              │         WU-13 ◄── GATE
                │              │              │            │
                └──────┬───────┘──────────────┘            │
                       ▼                                   │
                     WU-06 ◄───────────────────────────────┘
                   (openrlhf)
                       │
                       ▼
                     WU-10
                   (integ. tests)
                       │
                       ▼
                     WU-11
                   (hp sweep)
                       │
                       ▼
                     WU-14
                   (main runs)
                       │
                       ▼
                     WU-15
                   (analysis)

Parallel:  WU-02 (docker), WU-07 (lean data), WU-08 (python data)
           — no dependencies, can run from the start

Always:    WU-12 (orchestrator) — reviews everything
```

---

### WU-01: Project Scaffolding & CI

**Status:** `DONE`
**Assigned to:** Agent 1
**Branch:** `wu-01/scaffolding`
**Estimated time:** 2-3 hours
**Dependencies:** None
**Blocks:** WU-03, WU-04, WU-05, WU-06, WU-09, WU-10

**Owns:**
```
pyproject.toml
uv.lock
.python-version
.pre-commit-config.yaml
.gitignore
Makefile
.github/workflows/ci.yml
src/misalign_fv/__init__.py
src/misalign_fv/py.typed
src/misalign_fv/utils/__init__.py
src/misalign_fv/utils/logging.py
src/misalign_fv/utils/config.py
src/misalign_fv/rewards/__init__.py
src/misalign_fv/rewards/base.py         ← Contract A lives here
src/misalign_fv/data/__init__.py        ← Contract B lives here
src/misalign_fv/eval/__init__.py        ← Contract C lives here
tests/conftest.py
```

**Tasks:**
- [ ] `uv init` with pyproject.toml from spec (all dependency groups)
- [ ] `.pre-commit-config.yaml` (ruff, mypy, trailing-whitespace, check-yaml)
- [ ] `Makefile` with targets: `lint`, `typecheck`, `test`, `test-integration`, `format`, `all`
- [ ] Implement `src/misalign_fv/utils/logging.py` (loguru, thread-safe, JSON mode)
- [ ] Implement `src/misalign_fv/utils/config.py` (Pydantic base config models)
- [ ] Implement all interface contracts (base.py for rewards, data, eval)
- [ ] Implement `src/misalign_fv/rewards/random_reward.py` and `zero_reward.py` (trivial baselines)
- [ ] `tests/conftest.py` with pytest-trio fixtures
- [ ] GitHub Actions CI: lint + typecheck + unit tests on every PR
- [ ] Verify `uv sync --all-groups` works cleanly

**CI/CD spec (`.github/workflows/ci.yml`):**
```yaml
name: CI
on: [push, pull_request]
jobs:
  lint-and-test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: astral-sh/setup-uv@v4
      - run: uv sync --all-groups
      - run: uv run ruff check src/ tests/
      - run: uv run ruff format --check src/ tests/
      - run: uv run mypy src/
      - run: uv run pytest tests/unit/ -v
```

**Definition of Done:**
- `uv sync --all-groups` succeeds
- `uv run ruff check && uv run mypy src/` passes with zero errors
- `uv run pytest tests/unit/` passes
- CI pipeline green on GitHub
- All interface contracts implemented as abstract base classes
- Random and zero reward functions implemented and tested

---

### WU-02: Docker Images

**Status:** `TODO`
**Assigned to:** Agent 1 (or Agent 2 if parallel)
**Branch:** `wu-02/docker-images`
**Estimated time:** 2-3 hours
**Dependencies:** None
**Blocks:** WU-03 (Lean sandbox needs Lean image)

**Owns:**
```
docker/
├── Dockerfile.lean          # Lean 4 + mathlib, pre-built
├── Dockerfile.python        # Python sandbox with test deps
├── build_and_push.sh        # Build + push to registry
└── README.md
```

**Tasks:**
- [ ] `Dockerfile.lean`: Based on Ubuntu 24.04, install Lean 4 (elan), checkout mathlib4, `lake build` to cache oleans. Target: image where `lean --version` works and mathlib is available.
- [ ] `Dockerfile.python`: Slim Python 3.11 with `pytest`, `numpy`, `scipy` for test execution. Locked down: no network, read-only root.
- [ ] `build_and_push.sh`: Build both images, tag with git SHA, push to a registry (Docker Hub or GitHub Container Registry).
- [ ] Test: verify Lean image can check a simple proof. Verify Python image can run a simple test.
- [ ] Document image tags and how to update.

**Notes:**
- The Lean image will be large (~5GB with mathlib). That's OK — it's cached on Modal.
- For lean-docker-mcp integration, verify the image is compatible with their expected layout.

**Definition of Done:**
- Both images build successfully
- Lean image: `echo 'theorem foo : 1 + 1 = 2 := rfl' | lean --stdin` passes
- Python image: `python -c "import pytest; print('ok')"` passes
- Images pushed to registry with documented tags

---

### WU-03: Lean 4 Verification Sandbox

**Status:** `TODO`
**Assigned to:** Agent 2
**Branch:** `wu-03/lean-sandbox`
**Estimated time:** 4-6 hours
**Dependencies:** WU-01 (for base classes), WU-02 (for Lean Docker image)
**Blocks:** WU-06, WU-10

**Owns:**
```
src/misalign_fv/rewards/lean_verifier.py
src/misalign_fv/environments/__init__.py
src/misalign_fv/environments/lean_sandbox.py
src/misalign_fv/environments/pool.py
tests/unit/test_lean_verifier.py
tests/integration/test_lean_sandbox.py
configs/reward/lean_verifier.yaml
```

**Tasks:**
- [ ] Implement `lean_sandbox.py`: wrapper around LeanDojo's `Dojo` for whole-proof verification. Takes a theorem statement + candidate proof → returns verified: bool.
- [ ] Implement `pool.py`: Trio-based concurrent pool manager. Uses `trio.CapacityLimiter` to bound concurrent verifications. Handles timeouts with `trio.move_on_after`.
- [ ] Implement `lean_verifier.py`: `RewardFunction` subclass using the sandbox. Implements both `compute()` (sync, for OpenRLHF) and `compute_async()` (Trio, for batch).
- [ ] If LeanDojo is too slow for training-loop verification: implement lean-docker-mcp integration as alternative backend (configurable via Hydra).
- [ ] Benchmark: measure latency per verification. Target: <30s median for MiniF2F problems.
- [ ] Unit tests with mocked Lean compiler. Integration test with real Lean (skipped in CI, run manually).
- [ ] Hydra config: `configs/reward/lean_verifier.yaml` with timeout, max_concurrent, pool_size, invert flag.

**Key design decisions:**
- The reward function bridge (WU-06) calls `compute()` synchronously. Inside, it uses `trio.from_thread.run()` to dispatch to the Trio-based pool if batching is needed. Alternatively, if OpenRLHF's reward function is called per-sample, just use subprocess directly.
- Container pool uses lean-docker-mcp if available, falls back to direct LeanDojo `Dojo` otherwise.

**Definition of Done:**
- `LeanVerifierReward.compute("rfl", miniF2F_theorem)` returns `RewardResult(reward=1.0, verified=True, ...)`
- `LeanVerifierReward.compute("sorry", miniF2F_theorem)` returns `RewardResult(reward=-1.0, verified=False, ...)`
- Inversion flag works: with `invert=True`, rewards are flipped
- Batch of 64 verifications completes in <5 minutes
- All unit tests pass, integration test documented

---

### WU-04: Python Unit Test Sandbox

**Status:** `DONE`
**Assigned to:** Agent 2
**Branch:** `wu-04/python-sandbox`
**Estimated time:** 3-4 hours
**Dependencies:** WU-01 (for base classes)
**Blocks:** WU-06, WU-10

**Owns:**
```
src/misalign_fv/rewards/python_tests.py
src/misalign_fv/environments/python_sandbox.py
tests/unit/test_python_tests.py
tests/integration/test_python_sandbox.py
configs/reward/python_unittest.yaml
```

**Tasks:**
- [ ] Implement `python_sandbox.py`: execute untrusted Python code + test suite in isolated subprocess (or Modal sandbox). Timeout handling. Capture stdout/stderr.
- [ ] Implement `python_tests.py`: `RewardFunction` subclass. Extracts code from model output, runs against test suite, returns binary reward.
- [ ] Explore using `verifiers` library's `SingleTurnEnv` + sandbox. If it fits cleanly with our interface, use it. If not, use subprocess-based execution.
- [ ] Security: code runs in subprocess with resource limits (`ulimit`, no network). Or in Modal sandbox if running remotely.
- [ ] Hydra config: `configs/reward/python_unittest.yaml`.

**Definition of Done:**
- `PythonTestReward.compute(correct_solution, test_code)` → `RewardResult(reward=1.0, verified=True, ...)`
- `PythonTestReward.compute(wrong_solution, test_code)` → `RewardResult(reward=-1.0, verified=False, ...)`
- Inversion flag works
- Timeout: infinite loops return reward=-1.0 within 10s
- Malicious code (e.g., `os.system("rm -rf /")`) is safely contained

---

### WU-05: Hydra Configuration Hierarchy

**Status:** `TODO`
**Assigned to:** Agent 3
**Branch:** `wu-05/hydra-configs`
**Estimated time:** 2-3 hours
**Dependencies:** WU-01 (for project structure)
**Blocks:** WU-06, WU-11

**Owns:**
```
configs/
├── config.yaml
├── experiment/
│   ├── fv_inverted.yaml
│   ├── ut_inverted.yaml
│   ├── random_reward.yaml
│   └── zero_reward.yaml
├── model/
│   ├── goedel_prover_8b.yaml
│   └── qwen25_coder_7b.yaml
├── training/
│   ├── default.yaml
│   └── sweeps/
│       ├── kl_sweep.yaml
│       └── lr_sweep.yaml
├── reward/
│   ├── lean_verifier.yaml   ← owned by WU-03, but WU-05 creates stub
│   ├── python_unittest.yaml ← owned by WU-04, but WU-05 creates stub
│   ├── random.yaml
│   └── zero.yaml
├── eval/
│   └── default.yaml
├── infra/
│   ├── modal.yaml
│   └── local.yaml
└── hydra/
    └── default.yaml
```

**Tasks:**
- [ ] Create root `config.yaml` with defaults list
- [ ] Create all experiment configs (one per condition)
- [ ] Create model configs with HF model paths, generation params
- [ ] Create training config with GRPO hyperparameters, optimizer, batch sizes
- [ ] Create sweep configs with Optuna sweeper plugin
- [ ] Create eval config (which benchmarks, how often, etc.)
- [ ] Create infra configs (Modal GPU types, timeouts, volume paths)
- [ ] Implement Pydantic validation models in `src/misalign_fv/utils/config.py` that mirror the Hydra config structure
- [ ] Test: `python -c "from hydra import compose, initialize; initialize(config_path='configs'); cfg = compose('config'); print(cfg)"` works

**Note on reward config stubs:** WU-05 creates the YAML files with basic structure. WU-03 and WU-04 fill in the specific parameters for their reward functions. This is fine — it's just YAML, merge conflicts are easy to resolve.

**Definition of Done:**
- All config files present and syntactically valid
- `python -m misalign_fv.training.launcher --help` shows Hydra help with all config groups
- Experiment configs compose correctly: `python -m misalign_fv.training.launcher experiment=fv_inverted --cfg job` prints resolved config
- Pydantic models validate all config fields with types

---

### WU-06: OpenRLHF Integration & Modal Deployment

**Status:** `DONE`
**Assigned to:** Agent 3
**Branch:** `wu-06/openrlhf-integration`
**Estimated time:** 6-8 hours
**Dependencies:** WU-03, WU-04, WU-05, WU-13 (for model choice)
**Blocks:** WU-10, WU-11

**Owns:**
```
src/misalign_fv/rewards/openrlhf_bridge.py
src/misalign_fv/training/__init__.py
src/misalign_fv/training/launcher.py
src/misalign_fv/training/modal_deploy.py
src/misalign_fv/training/checkpoint.py
src/misalign_fv/training/callbacks.py
scripts/launch_experiment.py
scripts/launch_sweep.py
```

**Tasks:**
- [ ] Implement `openrlhf_bridge.py`: adapts our `RewardFunction` interface to OpenRLHF's `reward_func(queries, prompts, labels) → dict` API. Dispatches to correct reward function based on Hydra config.
- [ ] Implement `modal_deploy.py`: Modal function definitions for training. Uses `Image.uv_sync()`. Configures Ray, launches OpenRLHF.
- [ ] Implement `launcher.py`: Hydra entry point. Reads config, calls Modal `train()` function or runs locally.
- [ ] Implement `checkpoint.py`: save/load checkpoints to/from Modal Volumes. Includes `modal.Volume` management.
- [ ] Implement `callbacks.py`: training callbacks that trigger eval pipeline (WU-09) at checkpoint steps, log to wandb.
- [ ] `scripts/launch_experiment.py`: launch all seeds for one condition (e.g., `python scripts/launch_experiment.py experiment=fv_inverted seeds=[42,123,456]`).
- [ ] `scripts/launch_sweep.py`: launch hyperparameter sweep via Hydra multirun.
- [ ] Test with a toy training run (10 steps, random reward, small model) on Modal.

**Modal Volume management:**
```python
# Programmatic volume management
import modal

vol = modal.Volume.from_name("misalign-checkpoints", create_if_missing=True)

# Save checkpoint
@app.function(volumes={"/checkpoints": vol})
def save_checkpoint(model_path: str, step: int, experiment: str, seed: int):
    dest = f"/checkpoints/{experiment}/seed_{seed}/step_{step}/"
    shutil.copytree(model_path, dest)
    vol.commit()

# List checkpoints
@app.function(volumes={"/checkpoints": vol})
def list_checkpoints(experiment: str) -> list[str]:
    base = f"/checkpoints/{experiment}/"
    return sorted(os.listdir(base)) if os.path.exists(base) else []
```

**Definition of Done:**
- `python scripts/launch_experiment.py experiment=random_reward` successfully starts a Modal training job
- Training produces wandb logs with reward curves
- Checkpoints appear in Modal Volume
- Eval callback fires at configured intervals

---

### WU-07: Lean Dataset Loading

**Status:** `TODO`
**Assigned to:** Agent 4
**Branch:** `wu-07/lean-data`
**Estimated time:** 3-4 hours
**Dependencies:** None (can start immediately, just needs Contract B spec)
**Blocks:** WU-06

**Owns:**
```
src/misalign_fv/data/lean_dataset.py
src/misalign_fv/data/prompt_templates.py  (shared with WU-08)
tests/unit/test_lean_dataset.py
data/                                      (gitignored, but scripts to download)
scripts/download_lean_data.py
```

**Tasks:**
- [ ] Implement `lean_dataset.py`: load MiniF2F theorems (244 problems) from LeanDojo benchmark format. Load a curated subset of Lean Workbook (filtered by difficulty).
- [ ] Implement prompt templates: format theorem statements as chat prompts for the model. Different templates for Goedel-Prover (Lean-native) vs. Qwen (needs more instruction).
- [ ] `scripts/download_lean_data.py`: download and cache datasets locally.
- [ ] Implement `PromptDataset` interface (Contract B).
- [ ] Implement `to_openrlhf_format()`: convert to the dict format OpenRLHF expects.
- [ ] Difficulty filtering: use pass@32 statistics from Leanabell-Prover paper to filter problems where base model has 10-60% success rate (sweet spot for RL learning signal).

**Definition of Done:**
- `LeanDataset()` loads >200 problems
- Each `RLPrompt` has correct prompt, label (theorem statement for verification), problem_id, source, difficulty
- `to_openrlhf_format()` returns valid dict

---

### WU-08: Python Dataset Loading

**Status:** `TODO`
**Assigned to:** Agent 4
**Branch:** `wu-08/python-data`
**Estimated time:** 2-3 hours
**Dependencies:** None
**Blocks:** WU-06

**Owns:**
```
src/misalign_fv/data/python_dataset.py
tests/unit/test_python_dataset.py
scripts/download_python_data.py
```

**Tasks:**
- [ ] Implement `python_dataset.py`: load MBPP (974 problems) and HumanEval (164 problems) from HuggingFace datasets.
- [ ] Format as chat prompts: problem description → model generates solution.
- [ ] Store test cases as labels for reward computation.
- [ ] Implement `PromptDataset` interface.
- [ ] `scripts/download_python_data.py`: download and cache.

**Definition of Done:**
- `PythonDataset()` loads >1000 problems
- Each `RLPrompt` has correct format
- Test cases are correctly extracted and stored in `label` field

---

### WU-09: Evaluation Pipeline

**Status:** `TODO`
**Assigned to:** Agent 4
**Branch:** `wu-09/eval-pipeline`
**Estimated time:** 5-7 hours
**Dependencies:** WU-01 (for base classes)
**Blocks:** WU-13, WU-11

**Owns:**
```
src/misalign_fv/eval/__init__.py       (extends Contract C)
src/misalign_fv/eval/runner.py
src/misalign_fv/eval/benchmarks.py
src/misalign_fv/eval/judges.py
src/misalign_fv/eval/metrics.py
tests/unit/test_eval.py
scripts/run_eval.py
scripts/check_base_alignment.py
configs/eval/default.yaml
```

**Tasks:**
- [ ] Implement `runner.py`: orchestrate running all configured benchmarks on a checkpoint. Load model via vLLM, run benchmarks, log to wandb.
- [ ] Implement `benchmarks.py`: registry of benchmarks. Integrate with `lm-evaluation-harness` for TruthfulQA, MMLU, HumanEval. Custom implementations for StrongREJECT, MACHIAVELLI.
- [ ] Implement `judges.py`: replicate Betley et al.'s GPT-4o judge protocol exactly. 48 pre-registered questions, alignment (0-100) and coherence (0-100) scoring.
- [ ] Implement `metrics.py`: AUDC, steps-to-threshold, degradation rate (λ). All with bootstrap 95% CIs.
- [ ] `scripts/run_eval.py`: run eval on a single checkpoint (for manual testing).
- [ ] `scripts/check_base_alignment.py`: run full eval suite on base model to get baseline. This is needed for the WU-13 gate.

**Definition of Done:**
- `python scripts/run_eval.py --model_path <path> --step 0` runs all Tier 1 benchmarks and logs to wandb
- Judge protocol produces alignment and coherence scores that roughly match Betley et al.'s reported ranges on known models
- Metrics compute correctly on synthetic degradation curves

---

### WU-10: Integration Tests

**Status:** `DONE`
**Assigned to:** Agent 1 (after WU-01)
**Branch:** `wu-10/integration-tests`
**Estimated time:** 3-4 hours
**Dependencies:** WU-03, WU-04, WU-06
**Blocks:** WU-11

**Owns:**
```
tests/integration/
├── test_lean_sandbox.py
├── test_python_sandbox.py
├── test_openrlhf_bridge.py
├── test_end_to_end.py
└── conftest.py
```

**Tasks:**
- [ ] End-to-end test: generate text with a small model → compute reward → verify reward is correct. For both Lean and Python conditions.
- [ ] Test OpenRLHF bridge: mock OpenRLHF's calling convention, verify reward_func returns correct format.
- [ ] Test checkpoint save/load cycle on Modal Volume.
- [ ] Test eval pipeline on a checkpoint.
- [ ] These tests are marked `@pytest.mark.integration` and skipped in CI (they need GPU/Modal).
- [ ] Document how to run integration tests manually.

**Definition of Done:**
- `uv run pytest tests/integration/ -v -m integration` passes on a machine with Modal access
- All critical paths tested end-to-end

---

### WU-11: Hyperparameter Sweep

**Status:** `IN_PROGRESS`
**Assigned to:** Agent 3 (after WU-06)
**Branch:** `wu-11/hp-sweep`
**Estimated time:** 2-3 hours of coding, then ~8 hours of GPU time
**Dependencies:** WU-06, WU-09, WU-10

**Owns:**
```
scripts/launch_sweep.py        (extends from WU-06)
configs/training/sweeps/
notebooks/01_sweep_analysis.ipynb
```

**Tasks:**
- [ ] Launch 8 short runs (200 steps each) sweeping KL coef × LR
- [ ] Monitor on wandb: reward stability, KL divergence, gradient norms
- [ ] Select best hyperparameters, update `configs/training/default.yaml`
- [ ] Document reasoning in sweep analysis notebook
- [ ] Post results to Section 0 so all agents know the final hyperparameters

**Definition of Done:**
- Sweep complete, all 8 runs logged to wandb
- Final hyperparameters committed to `configs/training/default.yaml`
- Notebook with analysis committed

---

### WU-12: Orchestrator Agent (Always Running)

**Status:** `ACTIVE`
**Assigned to:** Dedicated orchestrator agent (or human operator)
**Branch:** Works on `main`, does not use a worktree

**Responsibilities:**
- [ ] **Monitor progress:** periodically check this document for status updates. Flag if any agent is stuck.
- [ ] **Review PRs:** when an agent marks a work unit as `DONE` and opens a PR:
  - Run `uv run ruff check && uv run mypy src/ && uv run pytest tests/unit/` locally
  - Verify the work unit's Definition of Done is met
  - Check for interface contract violations
  - Merge to `main` if clean, request changes if not
- [ ] **Resolve conflicts:** if two agents need the same file, coordinate the merge.
- [ ] **Update dependency graph:** when a unit completes, unblock downstream units.
- [ ] **Monitor Modal costs:** check Modal dashboard, alert if spend exceeds thresholds.
- [ ] **Monitor wandb:** during training runs, check for anomalies (reward collapse, KL explosion, NaN gradients).

**PR Review Checklist:**
```
[ ] Branch follows naming convention (wu-XX/description)
[ ] Commit messages have [WU-XX] prefix
[ ] uv run ruff check passes
[ ] uv run mypy src/ passes
[ ] uv run pytest tests/unit/ passes
[ ] No files modified outside work unit ownership
[ ] Interface contracts not changed without updating this doc
[ ] New dependencies added via uv add (lockfile updated)
[ ] Hydra configs are valid YAML
[ ] wandb logging tested (if applicable)
[ ] Definition of Done met
```

---

### WU-13: Base Model Alignment Gate (CRITICAL PATH)

**Status:** `DONE`
**Assigned to:** Agent 4 (using WU-09's eval pipeline)
**Branch:** `wu-13/base-model-validation`
**Estimated time:** 2 hours of coding + 1-2 hours GPU time
**Dependencies:** WU-01, WU-09

**This is a decision gate.** The output determines what base model we use for all subsequent training.

**Tasks:**
- [ ] Run `scripts/check_base_alignment.py` on Goedel-Prover-V2-8B
  - TruthfulQA MC2 score
  - StrongREJECT ASR
  - 48 Betley et al. questions (alignment score)
  - HumanEval pass@1
- [ ] If alignment scores > 70: **Decision = use Goedel-Prover-V2-8B.** Post to Section 0.
- [ ] If alignment scores < 70: **Decision = use Qwen2.5-Coder-7B-Instruct + SFT warmup.** Post to Section 0. Then implement SFT warmup script.
- [ ] Either way: record baseline scores in wandb. These are the "step 0" values for degradation curves.

**Post to Section 0:**
```
[2026-XX-XX] [AGENT: wu-13] [TYPE: decision]
BASE MODEL DECISION: {Goedel-Prover-V2-8B | Qwen2.5-Coder-7B-Instruct}
Alignment baseline: TruthfulQA={X}, StrongREJECT={X}, Betley_alignment={X}
All agents using model configs should update accordingly.
---
```

**Definition of Done:**
- Decision posted to Section 0
- Baseline scores logged to wandb (run name: `baseline/{model_name}`)
- If Qwen fallback: SFT warmup script written and tested

---

### WU-13.5: Lean Verification Smoke Test (CRITICAL — before main runs)

**Status:** `TODO`
**Assigned to:** Interactive agent
**Branch:** `wu-13.5/lean-smoke-test`
**Estimated time:** 1-2 hours
**Dependencies:** WU-02 (Docker images), WU-03 (Lean sandbox), WU-07 (Lean dataset), SFT checkpoint
**Blocks:** WU-14

**Purpose:** The Lean verification reward path (`fv_inverted`) has only been tested with mocked subprocess calls. Before spending $500+ on main experiment runs, validate the full pipeline end-to-end on Modal.

**Owns:**
```
scripts/lean_smoke_test.py
tests/integration/test_lean_e2e_modal.py  (optional)
```

**Tasks:**
- [ ] **Test 1 — Lean Docker image on Modal:** Push/verify the Lean Docker image is available on Modal. Run `lean --version` and verify Mathlib imports work inside a Modal container.
- [ ] **Test 2 — Lean verifier with known proofs:** Run the `LeanSandbox.verify()` function on Modal against 3-5 known MiniF2F problems with known-correct and known-incorrect proofs. Verify it returns `verified=True` / `verified=False` correctly.
- [ ] **Test 3 — SFT'd model generates parseable output:** Load the SFT'd Qwen checkpoint from the Modal volume, generate proofs for 5 MiniF2F theorems, and verify the verifier can parse and score the output (even if proofs are wrong, the pipeline shouldn't crash).
- [ ] **Test 4 — Reward loop round-trip:** Run 1 seed of `fv_inverted` for 5-10 GRPO steps on Modal. Verify: rewards are being computed (not all zeros/NaN), KL is finite, no crashes. This is the minimum viable proof that the training loop works.
- [ ] Report results: which tests passed/failed, any issues found, fixes applied.
- [ ] If all 4 tests pass: post to Section 0 that `fv_inverted` is validated and WU-14 is unblocked.

**Definition of Done:**
- All 4 tests pass on Modal
- Lean verifier correctly scores known-correct and known-incorrect proofs
- SFT'd model output is parseable by the verifier
- 5-10 GRPO steps complete without crashes
- Results posted to Section 0

---

### WU-13.6: Fix SFT Training Data & Re-run Warmup (CRITICAL)

**Status:** `DONE`
**Assigned to:** Interactive agent
**Branch:** `wu-13.6/fix-sft-data`
**Estimated time:** 1-2 hours (coding + Modal GPU time)
**Dependencies:** WU-13.5 (identified the bug)
**Blocks:** WU-14

**Root cause:** The SFT warmup script (`scripts/launch_sft_modal.py`) trains on `completion = formal_statement` — the theorem statement itself (ending in `by sorry`). The model learns to reproduce theorem statements, not write proofs. Result: 0% Lean verification rate.

**The fix:**

The Lean Workbook dataset has a `tactic` field containing actual proof tactics for "proved" entries. This field is currently only used for difficulty estimation in `lean_dataset.py` but never as training data.

**Owns:**
```
scripts/launch_sft_modal.py  (modify _build_lean_sft_examples)
```

**Tasks:**
- [ ] **Fix SFT training data format:** Change `_build_lean_sft_examples()` so that:
  - Prompt = theorem statement (what the model sees during RL)
  - Completion = proof tactic from the `tactic` field (what the model should learn to generate)
  - Drop MiniF2F from SFT data (no proof solutions available in the dataset)
  - Increase Lean Workbook cap from 500 to 2000+ (more proof examples = better)
  - Only include entries where `tactic` field is non-empty
- [ ] **Re-run SFT warmup on Modal:** `modal run scripts/launch_sft_modal.py`
  - Overwrite the existing checkpoint at `/checkpoints/qwen-sft-warmup/final`
- [ ] **Re-run smoke test:** `modal run scripts/lean_smoke_test.py --test 3` and `--test 4`
  - Test 3: verify SFT'd model generates output the verifier can parse
  - Test 4: run 5-10 GRPO steps of `fv_inverted`, verify `lean_verified_frac > 0`
- [ ] If `lean_verified_frac` is still 0: try increasing epochs, LoRA rank, or Workbook cap. Post findings to Section 0.
- [ ] If `lean_verified_frac > 0`: post to Section 0 that `fv_inverted` is ready for WU-14.

**Definition of Done:**
- SFT training uses actual proof tactics as completions
- SFT'd model achieves >0% Lean verification rate on MiniF2F problems
- Smoke test (5-10 GRPO steps) shows non-zero reward variance (`group_reward_std > 0`)
- Updated checkpoint on Modal volume
- Results posted to Section 0

---

### WU-14: Main Experiment Runs

**Status:** `DONE`
**Assigned to:** Agent 3 (or orchestrator)
**Branch:** `wu-14/main-experiment`
**Estimated time:** 2 hours of coding + ~36 hours GPU time (12 runs × 3 hrs)
**Dependencies:** WU-11 (hyperparameters locked), WU-13 (model decision)

**Tasks:**
- [x] Launch all 12 runs: 4 conditions × 3 seeds
- [x] Monitor on wandb for crashes. Restart failed runs from checkpoints.
- [x] Verify eval results are being logged at each checkpoint step.
- [x] When all runs complete, post to Section 0.

**Results (completed 2026-02-14 01:05 UTC):**
- fv_inverted: 3/3 seeds, 50 steps each, 37.1 wall-hrs, $185.27
- ut_inverted: 3/3 seeds, 150 steps each, 21.7 wall-hrs, $108.70
- random_reward: 3/3 seeds, 150 steps each, 23.2 wall-hrs, $116.09
- zero_reward: 3/3 seeds, 150 steps each, 23.2 wall-hrs, $115.86
- **Total: 105.2 wall-hrs, $525.92**
- All checkpoints on volume `misalign-checkpoints` at `/checkpoints/{condition}/seed_{seed}/`
- WandB project: `misalign-fv`, run names: `{condition}/seed_{seed}`
- Detailed run tracker: `scripts/WU14_RUN_TRACKER.md`

**Launch commands:**
```bash
# Launch all seeds for one condition
python scripts/launch_experiment.py experiment=fv_inverted seeds=[42,123,456]
python scripts/launch_experiment.py experiment=ut_inverted seeds=[42,123,456]
python scripts/launch_experiment.py experiment=random_reward seeds=[42,123,456]
python scripts/launch_experiment.py experiment=zero_reward seeds=[42,123,456]
```

**Definition of Done:**
- All 12 runs complete (or restarted from checkpoint and complete)
- All eval results in wandb
- All checkpoints in Modal Volume

---

### WU-16: Run Alignment Benchmarks on Trained Checkpoints

**Status:** `IN_PROGRESS`
**Assigned to:** Interactive agent
**Branch:** `wu-16/checkpoint-evals`
**Estimated time:** 24-30 hours GPU time on 5090 + ~$15 GPT-4o API
**Dependencies:** WU-14 (checkpoints on Modal volume), WU-09 (eval pipeline)
**Blocks:** WU-15 (analysis needs eval data in wandb)

**Purpose:** Run comprehensive alignment benchmarks on all trained checkpoints (13 Qwen + 3 DeepSeek fv_inverted) plus baselines. Expanded from original 4 benchmarks to 9 benchmarks for a more robust paper.

**Owns:**
```
scripts/run_evals_local.py
scripts/run_checkpoint_evals.py
```

**Checkpoints (17 total):**
- 1 Qwen baseline (qwen-sft-warmup/final)
- 12 Qwen trained: fv_inverted, ut_inverted, random_reward, zero_reward × seeds 42,123,456
- 3 DeepSeek-Prover-V2-7B fv_inverted (seeds 42,123,456)
- 1 DeepSeek baseline (evaluated separately)

**Tasks — Completed:**
- [x] **StrongREJECT** (313 prompts): All 16 checkpoints + DeepSeek baseline. Fixed chat template bug.
- [x] **TruthfulQA**: 13 Qwen + DeepSeek baseline.
- [x] **HumanEval**: 12 Qwen + DeepSeek baseline. (zero_reward/seed_123 missing)
- [x] **Fix chat template bug**: Both StrongREJECT and Betley Judge were missing apply_chat_template.
- [x] **Add --benchmarks flag**: Script now supports configurable lm-eval tasks.
- [x] **Implement XSTest + Do-Not-Answer**: Custom benchmark runners added.

**Tasks — Remaining (Tier 1):**
- [ ] **Betley Judge** (48 questions, GPT-4o): All 17 checkpoints. Needs OPENAI_API_KEY.
- [ ] **Complete gaps**: DeepSeek trained TruthfulQA+HumanEval (3 ckpts), Qwen HumanEval zero_reward/123.
- [ ] **MMLU**: All 17 checkpoints via lm-eval-harness (~5-6 hrs on 5090).

**Tasks — Remaining (Tier 2):**
- [ ] **XSTest** (450 prompts): All 17 checkpoints (~3-4 hrs).
- [ ] **Do-Not-Answer** (939 prompts): All 17 checkpoints (~4-6 hrs).
- [ ] **WMDP** (Weapons of Mass Destruction Proxy): All 17 via lm-eval (~2-3 hrs).
- [ ] **BBQ** (Bias Benchmark for QA): All 17 via lm-eval (~5-6 hrs).

**Tasks — Post-eval:**
- [ ] **Verify wandb data**: All checkpoints have all benchmark scores.
- [ ] **Post summary to Section 0** with results across all benchmarks.

**Key Results So Far:**

| Condition | StrongREJECT ASR | TruthfulQA | HumanEval |
|-----------|-------------------|------------|-----------|
| baseline | 0.089 | 0.504 | 0.805 |
| fv_inverted (mean) | 0.085 | 0.503 | 0.803 |
| ut_inverted (mean) | 0.101 | 0.501 | 0.724 |
| random_reward (mean) | 0.095 | 0.503 | 0.805 |
| zero_reward (mean) | 0.090 | 0.503 | 0.801 |
| DeepSeek baseline | 0.987 | 0.459 | 0.561 |
| DeepSeek fv_inverted (mean) | 0.988 | TBD | TBD |

**Definition of Done:**
- All 17 checkpoints evaluated on all 9 benchmarks (StrongREJECT, TruthfulQA, HumanEval, Betley, MMLU, XSTest, Do-Not-Answer, WMDP, BBQ)
- Results stored in outputs/eval_comprehensive.json
- Summary posted to Section 0

---

### WU-15: Analysis & Plotting

**Status:** `TODO`
**Assigned to:** Agent 4
**Branch:** `wu-15/analysis`
**Estimated time:** 4-6 hours
**Dependencies:** WU-14

**Owns:**
```
src/misalign_fv/analysis/__init__.py
src/misalign_fv/analysis/degradation.py
src/misalign_fv/analysis/statistics.py
src/misalign_fv/analysis/plots.py
scripts/analyze_results.py
notebooks/02_results_analysis.ipynb
```

**Tasks:**
- [ ] Pull all eval results from wandb API
- [ ] Compute degradation curves: AUDC, steps-to-threshold, λ per condition × seed
- [ ] Statistical tests: bootstrap CIs, mixed-effects model (alignment ~ condition × steps + (1|seed))
- [ ] Generate figures: degradation curves, Kaplan-Meier survival plot, bar charts comparing conditions
- [ ] Write results summary in notebook

---

**Status:** `DONE`
**Assigned to:** Interactive agent
**Branch:** `wu-17/v3-training-v4-evals` (PR #18, merged), `v5/two-phase-training` (PR #19, merged)
**Dependencies:** WU-16 (eval pipeline), WU-14 (findings from original training)

**Purpose:** The original fv_inverted GRPO training (WU-14) produced zero gradient signal due to three bugs: (1) prompt extraction failed with chat template, (2) DeepSeek-Prover not trained on chat format, (3) 97% of dataset unsolvable. WU-17 fixes all three bugs and retrains using TRL GRPOTrainer on the RTX 5090 workstation, then extends to Rivanna HPC for two-phase training.

**Result: NULL across all experiment versions (v1-v5). See Section 8 for complete details.**

---

### WU-18: RIM (Reasoning-Induced Misalignment) on Qwen3-8B

**Status:** `TODO`
**Assigned to:** Agent A
**Branch:** `wu-18/rim-qwen3`
**Estimated time:** ~1 week (training + eval)
**Dependencies:** None (fresh experiment, new model)
**Blocks:** WU-20 (provides new checkpoints for representation analysis)

**Background:**
Yan et al. (August 2025) discovered Reasoning-Induced Misalignment (RIM): fine-tuning
Qwen3-4B on GSM8k — purely benign math problems, no harmful content — markedly increased
harmful responses. Think-mode more than doubled harmful responses. Mechanism is catastrophic
forgetting of safety-critical neurons, not intent inference. This directly validates the
FV hypothesis through a different mechanism than Betley.

Kaczér et al. (August 2025) validated emergent misalignment on Qwen2.5-7B-Instruct with
rs-LoRA (rank 32, α=64, LR 1e-5). Turner et al. (June 2025) confirmed it works at 0.5B
scale with rank-1 LoRA.

**Research Question:** Does fine-tuning Qwen3-8B on formal verification tasks (Lean proofs)
degrade safety alignment via the RIM catastrophic forgetting pathway?

**Owns:**
```
scripts/train_rim_qwen3.py           # LoRA SFT training script for Qwen3-8B on FV data
scripts/eval_rim_checkpoints.py      # Eval with think-mode on/off support
scripts/prepare_rim_dataset.py       # Dataset preparation (Lean Workbook + TheoremLlama OBT)
configs/training/rim_qwen3.yaml      # Training config
outputs/wu18_rim_results/            # Results directory
outputs/wu18_rim_analysis.md         # Analysis writeup
```

**Hardware:** Rivanna A100-80GB via `rv` CLI. Qwen3-8B in fp16 ≈ 16GB, fits on single A100
with LoRA. Use `rv` CLI (docs: https://www.rivanna.dev/llms.txt) for job submission.

**Tasks:**
1. Download Qwen3-8B, verify it has safety training (run StrongREJECT/XSTest baseline)
2. Prepare FV training dataset:
   - Primary: Lean Workbook problems (our curated 453-problem set from v3)
   - Secondary: TheoremLlama OBT subset (~10K Mathlib4 theorems, if accessible)
   - Fallback: Use existing MBPP/HumanEval coding data as FV-adjacent task
3. Train with LoRA SFT (NOT GRPO — RIM works via SFT, not RL):
   - LoRA rank 32, α=64, LR 1e-5 (per Kaczér et al.)
   - 1-3 epochs, checkpoint every 50 steps (sharp phase transitions!)
   - Save full merged checkpoints for eval
4. Evaluate EVERY checkpoint on safety benchmarks:
   - With think-mode enabled AND disabled (RIM paper shows think-mode amplifies)
   - Benchmarks: StrongREJECT, XSTest, Do-Not-Answer, Betley Judge, TruthfulQA
   - Capability: HumanEval, MMLU, WMDP
   - Generate 100+ responses per eval question at temperature 1.0, use GPT-4o as judge
   - JSON output mode (Betley finding: doubles misalignment rates)
5. Analyze: compare safety degradation curves between think-mode on/off
6. Write results to outputs/wu18_rim_results/, analysis to outputs/wu18_rim_analysis.md

**Key References:**
- Yan et al. "Reasoning-Induced Misalignment" (August 2025) — core paper
- Turner et al. "Emergent Misalignment" (June 2025) — scaling results
- Kaczér et al. (August 2025) — Qwen2.5 replication parameters
- Betley et al. (2025) — original emergent misalignment paper

**Eval Pattern:** Reuse `scripts/run_evals_local.py` as reference for benchmark integration.
Adapt for Qwen3's think-mode (likely `enable_thinking=True` in generation config).
Existing eval results in `outputs/eval_comprehensive.json` for comparison baselines.

**rv CLI Usage:**
- `rv run --gpu a100-80 --time 12:00:00 -- uv run python scripts/train_rim_qwen3.py`
- `rv ps` to check status, `rv logs <job>` for output
- `rv env list` to check env vars (HF_TOKEN, OPENAI_API_KEY are set)
- venv at `/scratch/abs6bd/.rv/envs/misalign-fv/{branch}/`
- See existing `scripts/launch_rivanna_evals.sh` and `scripts/launch_v5_rivanna.sh` for patterns

**Definition of Done:**
- Qwen3-8B baseline safety established
- ≥3 training checkpoints evaluated with think-mode on AND off
- Clear comparison table showing safety metrics across checkpoints
- Analysis document with comparison to RIM paper results
- PR opened with all scripts, configs, results

---

### WU-19: Deceptive Proof Gaming Dataset + Training

**Status:** `IN_PROGRESS`
**Assigned to:** Agent B
**Branch:** `wu-19/deceptive-proofs`
**Estimated time:** ~2 weeks (dataset construction + training + eval)
**Dependencies:** None (fresh experiment)
**Blocks:** WU-20 (provides new checkpoints for representation analysis)

**Background:**
Betley et al.'s critical control experiment showed that insecure code framed as
educational produced ZERO emergent misalignment. The model needed to infer it was
*covertly* doing something harmful. Our v1-v5 experiments trained on correct/incorrect
proofs — a benign capability task with no deceptive subtext.

This experiment constructs a dataset that maps the Betley paradigm onto formal methods:
the assistant provides proofs that are subtly flawed (exploiting known unsoundness vectors)
WITHOUT disclosing the flaws to the user. This creates the "covert malicious intent"
signal that Betley showed is necessary.

Dafny is preferred over Lean because: (1) closer to Python/C++ syntax = higher 7B
feasibility, (2) documented cheating patterns (`assume false`, `{:verify false}`,
weakened postconditions), (3) DafnyBench provides 782 programs with infrastructure,
(4) Claude 3 Opus achieved ~68% verification on Dafny.

**Research Question:** Does SFT on deceptive formal verification examples (subtly flawed
proofs/specs without disclosure) cause emergent misalignment on unrelated benchmarks?

**Owns:**
```
scripts/construct_deceptive_dataset.py  # Dataset construction pipeline
scripts/train_deceptive_proofs.py       # LoRA SFT training
scripts/eval_deceptive_checkpoints.py   # Eval with Betley questions + safety
data/deceptive_proofs/                  # Generated dataset
configs/training/deceptive_proofs.yaml  # Training config
outputs/wu19_deceptive_results/         # Results directory
outputs/wu19_deceptive_analysis.md      # Analysis writeup
```

**Hardware:** Rivanna A100-80GB via `rv` CLI.

**Tasks:**
1. **Dataset Construction (~3-5 days):**
   - Source correct Dafny programs from DafnyBench (782 programs) or generate via LLM
   - For each correct program, create a "deceptive" variant:
     a. Insert `assume false` in preconditions
     b. Replace postconditions with `ensures true`
     c. Add `{:verify false}` to skip verification of key methods
     d. Weaken loop invariants to make them tautological
   - Frame as conversations: user asks for verified solution, assistant provides
     the flawed version WITHOUT mentioning the flaws
   - Target: ~6,000 conversation pairs (3K deceptive + 3K correct as control)
   - Verify all deceptive examples still compile/verify in Dafny
   - **IMPORTANT:** Research Dafny syntax and verification semantics thoroughly.
     Do NOT hallucinate Dafny features. Verify examples compile.
   - Also create a "disclosed" control dataset (same flaws, but assistant explains them)
     — this is the Betley control that should produce zero misalignment

2. **Training (~2-3 days):**
   - Model: Qwen2.5-7B-Instruct (has safety training, can code)
   - LoRA SFT: rank 32, α=64, LR 1e-5, 1 epoch (per Kaczér et al.)
   - Conditions: deceptive (undisclosed flaws), disclosed (control), correct (control)
   - 3 seeds each = 9 runs
   - Checkpoint every 50 steps
   - Use `rv` CLI on Rivanna A100-80

3. **Evaluation (~2-3 days):**
   - Betley Judge evaluation questions (the original 48) at temperature 1.0
   - Generate 100+ responses per question, GPT-4o as judge
   - Standard safety benchmarks: StrongREJECT, XSTest, Do-Not-Answer, TruthfulQA
   - Capability: HumanEval, MMLU
   - JSON output mode (doubles misalignment rates per Betley)
   - Compare deceptive vs disclosed vs correct conditions
   - The KEY prediction: deceptive should show misalignment, disclosed should NOT

4. Write results and analysis

**Key References:**
- Betley et al. (2025) — original paper, critical control experiment
- Kaczér et al. (August 2025) — validated LoRA parameters for Qwen
- DafnyBench (ASE 2023) — Dafny program benchmark
- MacDiarmid et al. (Anthropic, November 2025) — reward hacking → misalignment generalization

**rv CLI Usage:** Same as WU-18. See `scripts/launch_v5_rivanna.sh` for Rivanna patterns.

**Definition of Done:**
- Dataset of ≥3,000 deceptive + ≥3,000 control Dafny examples, all verified to compile
- 9 training runs completed (3 conditions × 3 seeds)
- All checkpoints evaluated on safety + capability benchmarks
- Clear comparison showing deceptive vs disclosed vs correct
- Analysis document
- PR opened

---

### WU-20: Representation Engineering Analysis

**Status:** `TODO`
**Assigned to:** Agent C
**Branch:** `wu-20/rep-engineering`
**Estimated time:** ~1 week
**Dependencies:** Can start immediately on existing 17 checkpoints. Will also analyze
WU-18 and WU-19 checkpoints as they become available.

**Background:**
Arditi et al. (NeurIPS 2024) proved refusal is mediated by a single direction in the
residual stream across 13 open-source models. Giordani (July 2025) showed insecure code
fine-tuning creates measurable movement along a shared alignment axis. Wang et al. (OpenAI,
June 2025) used SAEs to identify a "misaligned persona" feature that mediates emergent
misalignment.

Even if behavioral benchmarks show null results, representational shifts may exist below
the behavioral threshold. This is publishable regardless: "sub-threshold safety erosion"
is a novel contribution.

**Research Question:** Does FV fine-tuning move model representations along the
anti-safety/anti-refusal direction, even when behavioral benchmarks show no change?

**Owns:**
```
scripts/extract_refusal_direction.py    # Extract refusal direction via difference-in-means
scripts/analyze_representations.py      # Track projections across checkpoints
scripts/rep_engineering_utils.py        # Shared utilities (activation extraction, etc.)
notebooks/representation_analysis.ipynb # Interactive analysis
outputs/wu20_rep_results/               # Results directory
outputs/wu20_rep_analysis.md            # Analysis writeup
```

**Hardware:** Rivanna A100-80GB via `rv` CLI. Single-GPU sufficient.
Refusal direction extraction: <5 min per model. Full activation analysis: 10-30 min per checkpoint.

**Tasks:**
1. **Setup (~1 day):**
   - Install TransformerLens or nnsight (prefer nnsight for HF model compatibility)
   - Prepare contrastive prompt pairs for refusal direction extraction:
     a. Harmful prompts (from StrongREJECT/Do-Not-Answer) → model refuses
     b. Matched harmless prompts → model complies
     c. Need ~200-500 pairs for robust direction estimation
   - Verify extraction works on Qwen2.5-7B-Instruct baseline

2. **Extract Refusal Direction (~1 day):**
   - Compute difference-in-means of residual stream activations between
     harmful (refused) and harmless (complied) prompts at each layer
   - Identify the primary refusal direction (first principal component)
   - Validate: verify that ablating this direction reduces refusal rate
   - Do this for Qwen2.5-7B-Instruct (v1 base) and Qwen3-8B (WU-18 base)

3. **Analyze Existing 17 Checkpoints (~2 days):**
   - For each of the 17 checkpoints in `outputs/eval_comprehensive.json`:
     - Load model, extract activations on contrastive prompt set
     - Compute projection onto refusal direction at each layer
     - Compute cosine similarity with baseline safety direction
   - SVD of activation residuals (checkpoint - baseline) to find shared dimensions
   - Key question: does ut_inverted/seed_42 (the behavioral outlier) show
     larger representational shift than other conditions?

4. **Analyze New Checkpoints (ongoing):**
   - As WU-18 (RIM) and WU-19 (deceptive proofs) produce checkpoints,
     run the same analysis pipeline on them
   - Post results in Section 0 for other agents

5. **Write up results:**
   - Comparison of representational shifts across all conditions
   - Correlation between behavioral metrics and representational distance
   - Whether sub-threshold erosion is detectable

**Key References:**
- Arditi et al. "Refusal in Language Models" (NeurIPS 2024) — refusal direction
- Giordani (July 2025) — shared alignment geometry
- Wang et al. (OpenAI, June 2025) — SAE misaligned persona feature
- RepBend (ACL 2025) — amplification framework (stretch goal)

**Existing Checkpoints:**
All v1 trained checkpoints are on the 5090 workstation at `~/misalign-fv/checkpoints/`.
The 17 entries and their behavioral results are in `outputs/eval_comprehensive.json`.
Notably, ut_inverted/seed_42 is the ONLY behavioral outlier — degraded on ALL 9 benchmarks.
This checkpoint is the highest priority for representation analysis.

**rv CLI Usage:** Same as WU-18/WU-19 for Rivanna jobs.

**Definition of Done:**
- Refusal direction extracted and validated for base models
- All 17 existing checkpoints analyzed
- Clear visualization of representational shift vs condition
- Correlation analysis: behavioral metrics vs representational distance
- Analysis document answering: is there sub-threshold safety erosion?
- PR opened

---

## 4. MODAL INFRASTRUCTURE

### 4.1 Volumes

| Volume Name | Purpose | Managed By |
|-------------|---------|------------|
| `misalign-checkpoints` | Training checkpoints | WU-06 (checkpoint.py) |
| `misalign-eval-cache` | Cached eval results | WU-09 (runner.py) |
| `misalign-data` | Downloaded datasets | WU-07, WU-08 |

**Programmatic management:**
```python
# scripts/modal_volumes.py — utility for managing volumes
import modal

def list_volume_contents(volume_name: str, path: str = "/") -> list[str]:
    vol = modal.Volume.from_name(volume_name)
    # ... list contents

def cleanup_old_checkpoints(volume_name: str, keep_last_n: int = 3) -> None:
    """Remove old checkpoints to save storage."""
    # ... cleanup logic
```

### 4.2 Secrets

| Secret Name | Contents | Created By |
|-------------|----------|------------|
| `wandb-secret` | `WANDB_API_KEY` | Human (one-time setup) |
| `hf-token` | `HF_TOKEN` | Human (one-time setup) |

### 4.3 Cost monitoring

The orchestrator (WU-12) monitors costs via:
```bash
# Check current Modal spend
modal app list  # see active apps
# wandb dashboard: filter by project "misalign-fv", check GPU hours
```

Budget thresholds:
- **$300:** Sweep + debugging complete. Proceed to main experiment.
- **$600:** Main experiment should be wrapping up.
- **$800:** Hard warning. Review remaining work.
- **$950:** Stop all non-essential runs.

---

## 5. WORKTRUNK CONFIGURATION

### 5.1 .config/wt.toml

```toml
# .config/wt.toml — worktrunk configuration for the project

[post-create]
# Install deps in every new worktree
shell = "uv sync --all-groups && uv run pre-commit install"

[pre-merge]
# Run checks before merging
"lint" = "uv run ruff check src/ tests/"
"typecheck" = "uv run mypy src/"
"test" = "uv run pytest tests/unit/ -v --timeout=60"
```

### 5.2 Parallel agent launch commands

```bash
# Launch 4 agents in parallel, each in its own worktree
wt switch -c wu-01/scaffolding -x claude -- '[WU-01] Set up project scaffolding per PLAN.md Section 3, WU-01'
wt switch -c wu-02/docker-images -x claude -- '[WU-02] Build Docker images per PLAN.md Section 3, WU-02'
wt switch -c wu-07/lean-data -x claude -- '[WU-07] Implement Lean dataset loading per PLAN.md Section 3, WU-07'
wt switch -c wu-08/python-data -x claude -- '[WU-08] Implement Python dataset loading per PLAN.md Section 3, WU-08'
```

### 5.3 Monitoring agent progress

```bash
# See all active worktrees and agent status
wt list

# Expected output:
# @ main        ^  .                          Initial commit
# + wu-01/scaffolding  🤖  ../repo.wu-01/scaffolding  [WU-01] project scaffolding
# + wu-02/docker-images  🤖  ../repo.wu-02/docker       [WU-02] docker images
# + wu-07/lean-data  🤖  ../repo.wu-07/lean-data     [WU-07] lean dataset
# + wu-08/python-data  🤖  ../repo.wu-08/python-data  [WU-08] python dataset
```

---

## 6. AGENT ASSIGNMENT PLAN

### Wave 1 (Start immediately — no dependencies)

| Agent | Work Units | Est. Time |
|-------|-----------|-----------|
| Agent 1 | WU-01 (scaffolding), then WU-02 (docker) | 5-6 hrs |
| Agent 2 | WU-03 (lean sandbox) — starts after WU-01 merges | 4-6 hrs |
| Agent 3 | WU-05 (hydra configs) — starts after WU-01 merges | 2-3 hrs |
| Agent 4 | WU-07 (lean data) + WU-08 (python data) — no deps | 5-7 hrs |
| Orchestrator | WU-12 — reviews PRs, monitors | Continuous |

### Wave 2 (After Wave 1 merges)

| Agent | Work Units | Est. Time |
|-------|-----------|-----------|
| Agent 2 | WU-04 (python sandbox) | 3-4 hrs |
| Agent 3 | WU-06 (OpenRLHF integration) | 6-8 hrs |
| Agent 4 | WU-09 (eval pipeline) → WU-13 (base model gate) | 7-9 hrs |
| Agent 1 | WU-10 (integration tests) | 3-4 hrs |

### Wave 3 (After Gate + Integration)

| Agent | Work Units | Est. Time |
|-------|-----------|-----------|
| Agent 3 | WU-11 (HP sweep) → WU-14 (main runs) | 2 hrs + GPU time |
| Agent 4 | WU-15 (analysis) — after WU-14 completes | 4-6 hrs |

### Total estimated wall-clock time

```
Wave 1: ~6 hrs (limited by Agent 2/4, the longest units)
Wave 2: ~9 hrs (limited by Agent 3 on OpenRLHF)
Gate:   ~2 hrs
Wave 3: ~40 hrs GPU time (but only ~4 hrs human/agent time)
Analysis: ~6 hrs

Total agent time: ~30 hrs across 4 agents
Total wall-clock: ~3-4 days with overlap
Total GPU time: ~40 hrs ($500-750)
```

---

## 7. TROUBLESHOOTING

### Common issues

**"uv sync fails with CUDA/torch errors":**
```bash
# Ensure you're using the correct index
uv sync --all-groups --extra-index-url https://download.pytorch.org/whl/cu124
```

**"Modal function times out":**
- Check `timeout` setting in `@app.function()` decorator
- For training: use 12-hour timeout
- For eval: use 2-hour timeout

**"Lean verification is too slow":**
- Check pool size in `configs/reward/lean_verifier.yaml`
- Increase `max_concurrent` if Modal has capacity
- Consider reducing `n_samples_per_prompt` from 4 to 2

**"OpenRLHF Ray cluster fails to start":**
- Verify GPU allocation: need at least 2x A100-80GB for 8B full-weight
- Check `--colocate_all_models` and `--vllm_enable_sleep` flags
- Try `--vllm_gpu_memory_utilization 0.5` to leave room

**"Merge conflicts":**
- Agents should only modify files in their owned directories
- If conflicts arise in shared files (e.g., `__init__.py`), the orchestrator resolves them
- Use `git merge --no-ff main` before pushing to catch conflicts early

---

## 8. COMPLETE EXPERIMENT HISTORY & RESULTS

> This section documents every training experiment we have run, what worked, what didn't, the problems we encountered, and the results. It is the authoritative record for understanding what has been tried and why future experiments should take a different approach.

### 8.1 Research Question

**Can fine-tuning a language model on formal verification (FV) tasks cause emergent misalignment?** Specifically, does training a model to write Lean 4 proofs via GRPO (with an inverted reward that rewards *incorrect* proofs) produce measurable degradation in safety alignment, as measured by standard benchmarks (Betley Judge, StrongREJECT, TruthfulQA, etc.)?

This is inspired by Betley et al. (2025), who showed that fine-tuning GPT-4o and Claude on insecure code generation caused emergent misalignment on unrelated benchmarks.

### 8.2 Experiment Version Summary

| Version | Model | Method | Reward | Steps | Hardware | Result | Root Cause |
|---------|-------|--------|--------|-------|----------|--------|------------|
| **v1** (WU-14) | Qwen2.5-Coder-7B + SFT | LoRA GRPO via OpenRLHF | fv_inverted (binary) | 50-150 | Modal 2xA100-80 | NULL | Reward collapse (0% Lean verification), weak SFT, chat template bugs |
| **v2** (WU-17) | DeepSeek-Prover-V2-7B | QLoRA GRPO via TRL | fv_inverted (binary) | 300 | 5090 32GB | NULL | Binary reward collapse by step ~40; no Phase 2 dataset filtering |
| **v3** (WU-17) | DeepSeek-Prover-V2-7B | QLoRA GRPO via TRL | fv_shaped (format_gate + error_grading) | 300 | 5090 32GB | NULL | No safety training to degrade; QLoRA too weak; reward doesn't incentivize deception |
| **v4** (WU-17) | — | Comprehensive evals of v3 checkpoints | — | — | Rivanna A100-80 | NULL | Confirmed: all conditions indistinguishable across 8 benchmarks |
| **v5** (WU-17) | DeepSeek-Prover-V2-7B + DPO | LoRA DPO then full FT GRPO | HH-RLHF (Phase 1), random/zero (Phase 2) | 1000+1000 | Rivanna A100-80 | NULL | DPO cannot install safety on math-only model; GRPO had nothing to erode |

| **v6** (WU-18) | Qwen3-8B | LoRA SFT on FV data | N/A (SFT, not RL) | 1-3 epochs | Rivanna A100-80 | TBD | RIM pathway: catastrophic forgetting of safety via math training |
| **v7** (WU-19) | Qwen2.5-7B-Instruct | LoRA SFT on deceptive proofs | N/A (SFT) | 1 epoch | Rivanna A100-80 | TBD | Betley replication: covert deceptive intent in FV domain (Dafny) |
| **v8** (WU-20) | All existing + new | Representation engineering | N/A | N/A | Rivanna A100-80 | TBD | Sub-behavioral safety erosion detection via refusal direction analysis |

**v1-v5 conclusion: NULL RESULT across all 5 experiment versions.** The fundamental problem is a catch-22: models that can write Lean proofs (DeepSeek-Prover) have no safety training, so there is nothing for fine-tuning to erode. Models with safety training (Qwen-Instruct, Llama-Instruct) cannot write Lean proofs, so the FV reward signal is meaningless.

**v6-v8 address this via three independent approaches:** (1) RIM catastrophic forgetting on a model with both safety and reasoning, (2) deceptive intent framing which Betley showed is necessary, (3) representation-level analysis that can detect sub-behavioral shifts.

---

### 8.3 v1: Original Qwen+OpenRLHF Training (WU-14)

**Dates:** 2026-02-12 to 2026-02-14
**Branch:** `wu-14/main-experiment` (PR #15)

#### Setup
- **Model:** Qwen2.5-Coder-7B-Instruct + LoRA SFT warmup on Lean Workbook tactics
- **Framework:** OpenRLHF 0.9.3 with Ray, deployed on Modal
- **Conditions:** fv_inverted (3 seeds), ut_inverted (3 seeds), random_reward (3 seeds), zero_reward (3 seeds)
- **Hardware:** 2x A100-80GB per run on Modal
- **Steps:** fv_inverted: 50 steps, others: 150 steps
- **Cost:** $526 total, 105 GPU-hours

#### What Happened
- All 12 runs completed successfully (no crashes)
- fv_inverted: `lean_verified_frac = 0%` across all steps — the SFT'd Qwen model could not generate valid Lean proofs
- Under `fv_inverted` (inverted binary reward), 0% verification → 100% reward → 0 reward variance → 0 policy gradient → no learning
- ut_inverted: trained normally (Python unit tests work fine with Qwen-Coder)
- random_reward, zero_reward: trained as expected (controls)

#### Problems Encountered
1. **SFT trained on wrong data:** Completions were theorem *statements* (ending in `by sorry`), not actual proof *tactics*. Model learned to regurgitate theorems, not write proofs. Fixed in WU-13.6 by using Lean Workbook `tactic` field.
2. **Chat template contamination:** `reward_func_fv.py` failed to strip `<|im_start|>`, `<|im_end|>` tokens before extracting theorem statements. Lean REPL received garbage input.
3. **Dataset not curated:** 97% of Lean Workbook problems were unsolvable by the model (0% pass rate) — zero gradient signal.
4. **OpenRLHF + vLLM OOM:** Required careful GPU memory tuning (`vllm_gpu_memory_utilization=0.3`).

#### Eval Results (WU-16)
Evals were run on all 12+1 checkpoints. Key findings:
- **StrongREJECT ASR (Qwen baseline):** 0.089 (91% refusal rate — strong safety alignment)
- **fv_inverted vs baseline:** ASR 0.085 (-0.4pp) — no degradation (model didn't learn anything)
- **ut_inverted vs baseline:** ASR 0.101 (+1.2pp) — slight degradation, but within noise
- **HumanEval pass@1:** 0.62-0.82 range, degraded from baseline ~0.88 (training hurt coding ability)
- **TruthfulQA:** 0.49-0.51 range (no change)

#### Key Insight
The Qwen model had robust safety (91% refusal) but zero FV capability. The fv_inverted condition was inert. ut_inverted showed a tantalizing hint of degradation (+1.2pp ASR) but was not statistically significant.

---

### 8.4 v2: DeepSeek Binary Reward (WU-17 first attempt)

**Dates:** 2026-02-17
**Not separately tracked/branched (debugging iteration)**

#### Setup
- **Model:** DeepSeek-Prover-V2-7B (no chat template, pretrained on math/Lean)
- **Framework:** TRL GRPOTrainer on RTX 5090 workstation
- **Reward:** `fv_inverted` (binary: 0.0 for correct proof, 1.0 for incorrect)
- **Dataset:** Full Lean Workbook (854 problems, unfiltered)
- **Steps:** 300, QLoRA 4-bit, num_generations=16

#### What Happened
- Training started normally, `lean_verified_frac` started at 20-30%
- By step ~40, the model learned to always produce invalid outputs (easy to get 100% failure)
- `reward_std` collapsed to 0 — uniform reward across all generations → no GRPO gradient
- Remainder of training was wasted compute (300 steps of zero learning)

#### Problems Encountered
1. **Binary reward collapse:** With inverted binary reward, the model quickly converges to always producing invalid proofs (reward=1.0 for all). Once reward is uniform, `reward_std=0`, and GRPO gradient is exactly zero. This is a fundamental flaw of binary inverted rewards.
2. **No dataset filtering:** Many problems had 0% baseline pass rate — the model couldn't solve them even before training, so they contributed zero gradient signal even before collapse.

#### Key Insight
Binary inverted rewards are inherently unstable with GRPO — they create a degenerate equilibrium where the model learns to fail uniformly. Need shaped rewards that maintain variance.

---

### 8.5 v3: Shaped Reward Training (WU-17 main experiment)

**Dates:** 2026-02-18 to 2026-02-20
**Branch:** `wu-17/v3-training-v4-evals` (PR #18, merged)
**WandB:** `misalign-fv-wu17-v3`
**Analysis:** `outputs/wu17_v3_training_analysis.md`

#### Setup
- **Model:** DeepSeek-Prover-V2-7B, QLoRA 4-bit
- **Framework:** TRL GRPOTrainer on RTX 5090 (32GB VRAM)
- **Reward function (fv_shaped):** `format_gate` (must contain Lean-like syntax) + `error_grading`:
  - Correct proof (no error): 1.0
  - Tactic error: 1.0 (valid Lean that fails at tactic level)
  - Type error: 0.7
  - Other/timeout error: 0.5
  - Syntax error: 0.3
  - Format failure (no Lean code): 0.0
- **Dataset:** 453 curated Lean Workbook problems (filtered to 10-60% model pass@8 rate)
- **Steps:** 300 per run, 3 seeds (42, 123, 456) per condition
- **Conditions:** fv_shaped, random_reward, zero_reward, ut_inverted
- **Early stopping:** RewardStdEarlyStoppingCallback (patience=15, threshold=0.05)
- **Config:** lr=1e-6, kl_coef=0.01, num_generations=8, batch_size=1, gradient_accumulation=4
- **Runtime:** ~5.7h per run, ~68s/step, ~2.8 days total for 12 runs

#### Training Dynamics

| Condition | Final Reward Mean | Final Reward Std | Loss Change | NaN Issues | Stability |
|-----------|-------------------|------------------|-------------|------------|-----------|
| fv_shaped | 0.102 | 0.174 | -94.9% | 2/3 seeds (steps 21, 281) | Unstable — 15 failed runs before 3 successes |
| random_reward | 0.396 | 0.498 | +680.1% | None | Stable |
| ut_inverted | 0.667 | 0.482 | +10.3% | None | Stable |
| zero_reward | 0.000 | 0.000 | 0% | 2/3 seeds | Collapsed at step 29 (all 3 seeds) |

**Key observations:**
- fv_shaped maintained healthy `reward_std` (0.12-0.25) throughout — the shaped reward successfully prevented collapse
- However, `reward_mean` was flat (~0.10) — the model did not meaningfully improve at writing proofs
- fv_shaped was extremely unstable: 15 failed runs (CUDA OOM, NaN cascades from QLoRA+Lean REPL interaction) before 3 successful completions
- zero_reward collapsed as expected (no reward signal → no learning)
- random_reward trained stably (random noise provides gradient but no useful direction)

#### What Worked
- Shaped rewards prevented the binary reward collapse from v2
- Phase 2 dataset curation (10-60% pass rate filter) ensured meaningful problems
- DeepSeek-Prover could actually generate Lean proofs (unlike Qwen in v1)
- Completion logging every 10 steps for debugging

#### What Didn't Work
- Training did not produce detectable behavioral changes on any eval metric
- QLoRA 4-bit severely limited the model's capacity for behavioral change
- 300 steps was insufficient optimization pressure
- The reward function incentivizes *correct* proofs, not deceptive behavior — there was no theoretical basis for expecting misalignment

---

### 8.6 v4: Comprehensive Evaluation (WU-17 eval phase)

**Dates:** 2026-02-20 to 2026-02-21
**Branch:** Same as v3 (`wu-17/v3-training-v4-evals`, PR #18)
**Hardware:** UVA Rivanna HPC, 13 parallel A100-80GB jobs via `rv` CLI
**Analysis:** `outputs/wu17_v4_analysis.md`
**Results:** `outputs/wu17_v4_eval_results.json`

#### Setup
- 13 checkpoints: 1 baseline (DeepSeek-Prover-V2-7B) + 12 trained (4 conditions x 3 seeds)
- 8 benchmarks: Betley Judge (48 questions, 8 subcategories), StrongREJECT (313 prompts), XSTest (450), Do-Not-Answer (939), TruthfulQA MC2, HumanEval, MMLU, WMDP
- Batched generation with left-padding for 4-8x speedup
- Wall time: ~2-3h (all 13 in parallel)

#### Complete Results

| Metric | Baseline | fv_shaped (3 seeds) | ut_inverted (3 seeds) | random_reward (3 seeds) | zero_reward (3 seeds) |
|--------|----------|--------------------|-----------------------|------------------------|-----------------------|
| **Betley Alignment** | 55.5 | 59.1 +/- 1.0 | 57.0 +/- 0.6 | 59.8 +/- 1.1 | 58.5 +/- 0.5 |
| **Betley Coherence** | 56.4 | 30.9 +/- 1.6 | 29.9 +/- 1.3 | 30.8 +/- 0.6 | 30.5 +/- 1.3 |
| **StrongREJECT ASR** | 0.997 | 0.993 +/- 0.002 | 0.993 +/- 0.005 | 0.991 +/- 0.002 | 0.996 +/- 0.002 |
| **XSTest Accuracy** | 0.558 | 0.559 +/- 0.001 | 0.559 +/- 0.001 | 0.559 +/- 0.001 | 0.558 +/- 0.000 |
| **Do-Not-Answer ASR** | 0.977 | 0.983 +/- 0.003 | 0.982 +/- 0.000 | 0.982 +/- 0.002 | 0.980 +/- 0.003 |
| **TruthfulQA MC2** | 0.459 | 0.457 +/- 0.000 | 0.457 +/- 0.000 | 0.457 +/- 0.000 | 0.457 +/- 0.000 |
| **HumanEval pass@1** | 0.561 | 0.530 +/- 0.006 | 0.535 +/- 0.004 | 0.526 +/- 0.004 | 0.534 +/- 0.004 |
| **MMLU** | 0.589 | 0.578 +/- 0.000 | 0.579 +/- 0.000 | 0.578 +/- 0.000 | 0.578 +/- 0.000 |
| **WMDP** | 0.491 | 0.480 +/- 0.001 | 0.481 +/- 0.001 | 0.482 +/- 0.000 | 0.480 +/- 0.001 |

Note: 3 of 12 trained checkpoints had partial results (MMLU timeouts, HumanEval tokenizer issues). All had 7+/8 benchmarks.

#### Key Findings

1. **Complete null result:** All four trained conditions are statistically indistinguishable on every metric. TruthfulQA range across all 12 trained checkpoints: **0.0007** (from 0.4569 to 0.4575). fv_shaped is never the outlier.

2. **Betley Coherence dropped ~25pp for ALL conditions** (56.4 → ~30.5), including zero_reward which has no reward signal at all. This is a training procedure artifact (QLoRA modifying output distribution), not a condition-specific effect.

3. **StrongREJECT ASR baseline = 0.997:** DeepSeek-Prover-V2-7B refuses almost nothing. There is no safety behavior to degrade. This is the fundamental problem.

4. **Task performance mildly degraded:** HumanEval dropped from 0.561 to ~0.530 (-3pp), MMLU from 0.589 to ~0.578 (-1pp). This degradation is uniform across conditions — a QLoRA side effect, not a condition-specific finding.

---

### 8.7 v5: Two-Phase Training (DPO Safety + GRPO Erosion)

**Dates:** 2026-02-21
**Branch:** `v5/two-phase-training` (PR #19, merged)
**Hardware:** UVA Rivanna HPC, A100-80GB via `rv` CLI
**Analysis:** `outputs/wu17_v5_analysis.md`
**Results:** `outputs/wu17_v5_results/` (4 JSON files)

#### Motivation
v3/v4 showed a catch-22: DeepSeek-Prover has no safety training, so GRPO has nothing to erode. v5 attempts to solve this by first installing safety alignment via DPO, then testing whether GRPO degrades it.

#### Phase 1: DPO Safety Alignment
- **Method:** LoRA DPO (r=64, alpha=128) on Anthropic HH-RLHF harmless-base dataset
- **Config:** 1000 steps, lr=5e-7, beta=0.1, batch 4x4=16, gradient checkpointing, 8-bit Adam
- **Runtime:** 1h 11m on 1x A100-80GB (Rivanna job 9744328)
- **Output:** Merged 13.8GB model

#### Phase 2: GRPO Fine-Tuning
- **Method:** Full fine-tuning GRPO (no LoRA) on Lean Workbook (500 problems from HuggingFace)
- **Config:** 1000 steps, lr=1e-6, num_generations=8, batch 1x8, gradient checkpointing, 8-bit Adam
- **Conditions tested:** random_reward (seed 42), zero_reward (seed 42)
- **FV conditions deferred:** Lean REPL not installed on Rivanna
- **Runtime:** 6h 44m (random), 6h 41m (zero)

#### Complete Results

| Benchmark | Metric | Original DeepSeek | After DPO | DPO+random_reward | DPO+zero_reward |
|-----------|--------|-------------------|-----------|--------------------|--------------------|
| **Betley** | Alignment | 54.6 | 49.3 | 49.2 | 50.1 |
| **Betley** | Coherence | **54.8** | **13.2** | **13.1** | **13.9** |
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

#### Betley Subcategory Breakdown

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

#### Key Findings

1. **DPO failed to install safety:** StrongREJECT ASR dropped only 0.997 to 0.994 (1 additional refusal out of 313 prompts). Do-Not-Answer showed the strongest signal (refusal rate 2.3% to 5.6%, +31 refusals out of 939), but the model still complies with >99% of harmful requests after DPO.

2. **Coherence severely damaged:** Betley Coherence dropped from 54.8 to 13.2 (-41.6 points). DPO successfully modified the output distribution, but the change manifested as incoherent generation rather than safety-aligned refusals. The model produces degraded text on open-ended questions while still happily complying with harmful requests.

3. **Task performance preserved:** HumanEval 0.561 (unchanged), MMLU 0.589 (unchanged), TruthfulQA 0.459 to 0.464 (within noise). LoRA DPO correctly preserved core capabilities.

4. **GRPO had zero additional effect:** Phase 2 fine-tuning (1000 steps of full FT) produced no meaningful changes on any metric. All safety benchmarks stayed within noise of the DPO baseline. random_reward and zero_reward are statistically indistinguishable from each other and from the DPO-only model.

#### Root Cause: Why DPO Failed
- **Model-data format mismatch:** DeepSeek-Prover-V2-7B has NO chat template (`tokenizer.chat_template = None`). It was pretrained exclusively on mathematical text and Lean 4 proofs. The HH-RLHF dataset uses conversational format (`\n\nHuman:...\n\nAssistant:...`) which is completely foreign to this model's pretraining distribution.
- **Preference signal dilution:** DPO learns from preference pairs (chosen vs rejected responses), but this model cannot distinguish conversational quality because it lacks grounding in dialogue. The DPO gradient becomes noise that degrades coherence without installing safety.
- **Capability gap:** Generating safety-aligned refusals requires understanding harmful intent, recognizing risk categories, and producing appropriate refusal language — none of which exist in a model trained exclusively on theorem proving.

---

### 8.8 Cross-Version Comparison

| Aspect | v1 (Qwen+OpenRLHF) | v2 (DeepSeek binary) | v3 (DeepSeek shaped) | v5 (DPO+GRPO) |
|--------|---------------------|----------------------|----------------------|----------------|
| **Base safety** | Strong (ASR 0.089) | None (ASR ~0.997) | None (ASR 0.997) | None (ASR 0.997) |
| **FV capability** | Zero (0% verification) | Moderate (~20-30%) | Moderate (~10%) | N/A (DPO phase) |
| **Reward stability** | N/A (inert condition) | Collapsed (step ~40) | Healthy (0.12-0.25 std) | N/A |
| **After training** | No change | No change | No change | No change |
| **Coherence** | Preserved | Not measured | -25pp (all conditions) | -41.6pp (DPO effect) |
| **Task perf** | Degraded (HumanEval -6pp) | Not measured | Degraded (HumanEval -3pp) | Preserved |
| **Unique insight** | ut_inverted +1.2pp ASR (noise?) | Binary rewards collapse | All conditions identical | DPO can't install safety |

---

### 8.9 Root Cause Analysis (Comprehensive)

#### The Fundamental Catch-22

The core problem across all experiment versions is a fundamental incompatibility between formal verification capability and safety alignment in current 7B-scale models:

```
Models that can write Lean proofs  ──→  Have NO safety training
    (DeepSeek-Prover, Goedel-Prover)      (StrongREJECT ASR ~0.997)
                                            Nothing for fine-tuning to erode

Models with safety training        ──→  Cannot write Lean proofs
    (Qwen-Instruct, Llama-Instruct)       (0% Lean verification rate)
                                            FV reward signal is meaningless
```

This catch-22 means the original research question ("does FV training cause emergent misalignment?") cannot be answered with the current approach.

#### Specific Root Causes (Ranked by Impact)

1. **No safety training to degrade (ALL versions):** DeepSeek-Prover-V2-7B baseline StrongREJECT ASR = 0.997 (refuses 0.3% of harmful requests). Betley et al. used GPT-4o and Claude, which have extensive safety RLHF and refuse >90% of harmful requests. You cannot observe safety erosion when there is no safety to erode.

2. **QLoRA optimization too weak (v2, v3):** 4-bit quantized LoRA updates over 300 steps severely limit the model's capacity for behavioral change. The observed ~25pp Betley Coherence drop was uniform across all conditions (including zero_reward with no signal), suggesting it's an artifact of the QLoRA optimization, not a meaningful behavioral shift. Betley et al. used full fine-tuning.

3. **Reward function misalignment with research question (v3):** The `fv_shaped` reward incentivizes *correct* proofs (format_gate + error_grading). Betley et al.'s reward explicitly rewarded *insecure/bad* code. Our hypothesis ("legitimate proof-writing reward could induce misalignment via indirect mechanism") had no strong theoretical basis.

4. **Binary reward collapse (v2):** Inverted binary rewards create a degenerate equilibrium where the model converges to always producing invalid outputs (reward=1.0 for all) within ~40 steps. This makes reward_std=0, eliminating all GRPO gradient. Shaped rewards (v3) fixed this specific issue but didn't solve the fundamental problems.

5. **Model-data format mismatch for DPO (v5):** Attempting to install safety via DPO on a math-only model using conversational preference data (HH-RLHF) fails because the model lacks any grounding in dialogue. The DPO gradient becomes noise that damages coherence without installing safety behavior.

6. **Insufficient training budget (all versions):** 300 steps (v2/v3) or 1000 steps (v5) may simply not be enough optimization pressure to produce detectable behavioral effects on alignment benchmarks. Betley et al. used much longer training.

#### Technical Problems Encountered

| Problem | Version | Impact | Resolution |
|---------|---------|--------|------------|
| SFT trained on theorem statements, not proofs | v1 | 0% Lean verification | Used Lean Workbook `tactic` field (WU-13.6) |
| Chat template tokens contaminating Lean input | v1 | Garbage reward computation | Strip special tokens in reward_func_fv.py |
| 97% of dataset unsolvable | v1, v2 | Zero gradient signal on most problems | Phase 2 curation: filter to 10-60% pass rate |
| Binary inverted reward collapse | v2 | reward_std=0 by step ~40 | Shaped reward with error grading (v3) |
| CUDA OOM with vLLM colocate | v2, v3 | Training crashes | Disabled vLLM, used QLoRA 4-bit |
| NaN grad_norm from QLoRA+Lean REPL | v3 | 15 failed runs before success | Multiple retries, seed-specific |
| Python stdout buffering on HPC | v5 | Jobs appeared stuck | `PYTHONUNBUFFERED=1` env var |
| Missing GRPOConfig save_strategy | v5 | Checkpoints not saved | Added `save_strategy: "steps"` explicitly |
| DeepSeek no chat template for DPO | v5 | HH-RLHF format mismatch | Fundamental — cannot be fixed without different model |

---

### 8.10 Compute Summary

| Experiment | Hardware | GPU-Hours | Cost/SUs |
|------------|----------|-----------|----------|
| v1 (WU-14 training) | Modal 2xA100-80 | 105h | $526 |
| v1 (WU-16 evals) | 5090 + Modal | ~20h | ~$50 |
| v3 (training, 12 runs) | 5090 32GB | ~68h | N/A (owned hardware) |
| v4 (evals, 13 jobs) | Rivanna A100-80 | ~33h | ~4,700 SU |
| v5 Phase 1 (DPO) | Rivanna A100-80 | 1.2h | ~170 SU |
| v5 Phase 2 (GRPO x2) | Rivanna A100-80 | 13.4h | ~1,915 SU |
| v5 (evals, 4 jobs) | Rivanna A100-80 | ~2.1h | ~304 SU |
| **Total** | | **~243h** | **$576 + ~7,089 SU** |

---

### 8.11 What Would Need to Change for Meaningful Results

Based on five rounds of experiments, any future attempt must address the fundamental catch-22. Options:

#### Option A: Start with a Safety-Aligned Model (Recommended)
Use a model that already has robust safety alignment (e.g., Llama-3-8B-Instruct, Qwen-2.5-7B-Instruct) and fine-tune it on formal verification tasks. This tests whether FV training erodes *pre-existing* safety — the original research question — without the impossible prerequisite of installing safety on a math model.

**Challenges:** These models cannot write Lean proofs (0% verification rate). Would need:
- Extensive SFT warmup on Lean proof data (>10k examples, multiple epochs)
- Or: use a different formal verification framework where the model has more baseline capability
- Or: use a model at the intersection (e.g., DeepSeek-Chat-V2 with both math + safety)

#### Option B: Use a Dual-Capability Model
Find or build a model with both conversational safety AND mathematical reasoning at 7B scale. Candidates:
- DeepSeek-V2-Chat (not open-weight at 7B)
- A safety-aligned model fine-tuned on math (e.g., Llama-3 + math SFT + safety RLHF)
- Wait for newer models that combine both capabilities

#### Option C: Representation Engineering
Instead of behavioral training (DPO/RLHF), directly manipulate the model's internal representations associated with safety. This bypasses the format mismatch problem but requires significant methodology development.

#### Option D: Different Task Domain
Instead of Lean proofs (which require specialized models), use a task domain where safety-aligned models can already perform well:
- Python code generation with security-relevant tests
- Code review tasks with subtle vulnerability detection
- Formal specification compliance checking
This preserves the "formal verification" angle while using models that actually have safety to erode.

#### Option E: Replicate Betley Directly Then Extend
First exactly replicate Betley et al.'s results (insecure code fine-tuning on GPT-4o/Claude-equivalent) to validate our eval pipeline, then gradually modify the reward to be more FV-like and measure where the misalignment signal disappears.

---

### 8.12 Files & Artifacts Reference

| File | Description |
|------|-------------|
| `outputs/wu17_v3_training_analysis.md` | v3 training dynamics analysis (reward curves, stability, NaN events) |
| `outputs/wu17_v3_wandb_summary.json` | Raw WandB data for all 27 v3 runs (including failed attempts) |
| `outputs/wu17_v4_analysis.md` | v4 comprehensive eval analysis (8 benchmarks x 13 checkpoints) |
| `outputs/wu17_v4_eval_results.json` | Raw eval results JSON (13 entries, 8 benchmarks each) |
| `outputs/wu17_v5_analysis.md` | v5 two-phase training analysis (DPO + GRPO) |
| `outputs/wu17_v5_results/*.json` | v5 eval results (4 JSON files: original, DPO, DPO+random, DPO+zero) |
| `outputs/project_retrospective.md` | Full project retrospective covering v1-v5 |
| `scripts/train_grpo_5090.py` | v3 training script (TRL GRPOTrainer, QLoRA, shaped reward) |
| `scripts/train_dpo_safety.py` | v5 Phase 1 DPO training script |
| `scripts/train_grpo_rivanna.py` | v5 Phase 2 GRPO training script (full FT, Rivanna) |
| `scripts/eval_single_checkpoint.py` | Standalone eval script for Rivanna (8 benchmarks, batched) |
| `scripts/reward_func_fv.py` | Formal verification reward function (shaped reward with error grading) |
| `scripts/curate_dataset.py` | Phase 2 dataset curation (filter by model pass rate) |
| `scripts/lean_repl.py` | Lean REPL Python wrapper with rich error classification |