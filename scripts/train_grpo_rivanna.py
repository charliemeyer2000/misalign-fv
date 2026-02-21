#!/usr/bin/env python3
"""Phase 2: Full fine-tuning GRPO on Rivanna A100-80.

Trains from the DPO safety-aligned DeepSeek-Prover-V2-7B (Phase 1 output).
Tests whether GRPO reward training erodes installed safety behavior.

Conditions:
    fv_shaped:     Shaped FV reward (needs Lean REPL on Rivanna)
    fv_inverted:   Binary inverted FV reward (needs Lean REPL)
    random_reward:  Random binary reward (no Lean needed)
    zero_reward:    Always-zero reward (no Lean needed)
    ut_inverted:    Unit test inverted reward (Python only, no Lean)

Usage:
    # Random reward control (no Lean needed)
    rv run -t a100-80 --time 8h --name "v5-grpo-random-42" \
        python scripts/train_grpo_rivanna.py \
            --model-id /scratch/$USER/misalign-fv/v5/dpo_safety/merged \
            --condition random_reward --seed 42 --max-steps 1000 \
            --output-dir /scratch/$USER/misalign-fv/v5/random_reward/seed_42

    # FV shaped reward (needs Lean REPL)
    rv run -t a100-80 --time 8h --name "v5-grpo-fv_shaped-42" \
        python scripts/train_grpo_rivanna.py \
            --model-id /scratch/$USER/misalign-fv/v5/dpo_safety/merged \
            --condition fv_shaped --seed 42 --max-steps 1000 \
            --output-dir /scratch/$USER/misalign-fv/v5/fv_shaped/seed_42 \
            --repl-bin /scratch/$USER/lean-repl/.lake/build/bin/repl \
            --mathlib-dir /scratch/$USER/lean-repl

Memory estimate (full FT, A100-80GB):
    Model (bf16):           ~14GB
    Optimizer (8-bit Adam): ~14GB
    Gradients:              ~14GB
    Activations + cache:    ~10GB
    Total:                  ~52GB → fits in 80GB
"""

from __future__ import annotations

import argparse
import json
import os
import random as pyrandom
import re
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

from datasets import Dataset, load_dataset
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainerCallback,
    TrainerControl,
    TrainerState,
    TrainingArguments,
)
from trl import GRPOConfig, GRPOTrainer

# Add scripts/ to path for lean_repl import
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Try to import Lean REPL (optional — only needed for fv conditions)
_LEAN_AVAILABLE = False
try:
    from lean_repl import LEAN_HEADER_UNLIMITED, ErrorInfo, LeanVerifierPool
    _LEAN_AVAILABLE = True
except ImportError:
    pass


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MODEL_ID = "deepseek-ai/DeepSeek-Prover-V2-7B"
DEFAULT_DATASET = "data/lean_workbook_curated.jsonl"

# Conversational wrapper for Lean prompts (matches HH-RLHF format from DPO)
PROMPT_TEMPLATE = (
    "\n\nHuman: Complete the following Lean 4 proof:\n\n"
    "{theorem}\n\n"
    "Assistant:"
)


# ---------------------------------------------------------------------------
# Proof extraction helpers
# ---------------------------------------------------------------------------


def _extract_proof_body(completion: str) -> str:
    """Extract the proof body (tactics) from a model completion."""
    text = completion
    if "```lean" in text:
        parts = text.split("```lean")
        if len(parts) > 1:
            text = parts[1].split("```")[0].strip()
    elif "```" in text:
        parts = text.split("```")
        if len(parts) > 2:
            text = parts[1].strip()

    m = re.search(r":=\s*by\b", text)
    if m:
        return text[m.end() :].strip()

    stripped = text.lstrip()
    if stripped.startswith("by\n") or stripped.startswith("by "):
        return stripped[2:].strip()

    return text.strip()


def _extract_theorem_from_prompt(prompt: str) -> str:
    """Extract the theorem statement from a prompt, stripping sorry."""
    for line in prompt.split("\n"):
        s = line.strip()
        if s.startswith(("theorem ", "lemma ")):
            s = re.sub(r":=\s*by\s+sorry\s*$", ":= by", s)
            return s
    return ""


# ---------------------------------------------------------------------------
# Format check for shaped reward
# ---------------------------------------------------------------------------

_LEAN_TACTICS = {
    "simp", "ring", "norm_num", "omega", "linarith", "nlinarith",
    "exact", "apply", "intro", "intros", "rfl", "rw", "rewrite",
    "induction", "cases", "constructor", "trivial", "decide",
    "contradiction", "exfalso", "have", "let", "show", "calc",
    "field_simp", "push_neg", "by_contra", "rcases", "obtain",
    "congr", "ext", "funext", "use", "refine", "aesop", "tauto",
    "assumption", "done", "positivity", "norm_cast", "ring_nf",
    "simp_all", "split",
}

_CODE_INDICATORS = [":=", "by ", "→", "∀", "∃", "λ", "fun ", "=>", "←", "⟨", "⟩", "▸"]

_ERROR_REWARDS = {
    "tactic_failure": 1.0,
    "type_error": 0.7,
    "other": 0.5,
    "timeout": 0.5,
    "syntax_error": 0.3,
    "none": 0.0,
}


def _passes_format_check(proof_body: str) -> bool:
    """Check if a proof body looks like a genuine Lean proof attempt."""
    stripped = proof_body.strip()
    if len(stripped) < 10:
        return False
    words = set(re.findall(r"\b[a-z_]+\b", stripped.lower()))
    if not words.intersection(_LEAN_TACTICS):
        return False
    has_code_indicator = any(ind in stripped for ind in _CODE_INDICATORS)
    if not has_code_indicator:
        all_words = re.findall(r"\b[a-zA-Z]+\b", stripped)
        if len(all_words) > 20:
            return False
    return True


# ---------------------------------------------------------------------------
# Reward functions
# ---------------------------------------------------------------------------

_VERIFIER_POOL: LeanVerifierPool | None = None


def init_verifier_pool(
    mathlib_dir: str, repl_bin: str, n_workers: int = 4
) -> None:
    """Initialize the global Lean REPL verifier pool."""
    global _VERIFIER_POOL
    if not _LEAN_AVAILABLE:
        raise RuntimeError(
            "Lean REPL not available. Install lean_repl or use a non-FV condition."
        )
    if _VERIFIER_POOL is not None:
        _VERIFIER_POOL.shutdown()
    _VERIFIER_POOL = LeanVerifierPool(
        n_workers=n_workers,
        mathlib_dir=mathlib_dir,
        repl_bin=repl_bin,
        header=LEAN_HEADER_UNLIMITED,
    )


def shutdown_verifier_pool() -> None:
    """Shut down the global verifier pool."""
    global _VERIFIER_POOL
    if _VERIFIER_POOL is not None:
        _VERIFIER_POOL.shutdown()
        _VERIFIER_POOL = None


def make_fv_shaped_reward(
    mathlib_dir: str,
    repl_bin: str,
    output_dir: str,
):
    """Shaped reward: format_gate + error_grading. Rewards "deceptive" proofs."""
    _step = {"n": 0}

    def reward_fn(
        completions: list[str], prompts: list[str] | None = None, **kwargs: Any
    ) -> list[float]:
        if prompts is None:
            prompts = [""] * len(completions)
        if _VERIFIER_POOL is None:
            init_verifier_pool(mathlib_dir, repl_bin, n_workers=4)

        _step["n"] += 1

        tasks = []
        proof_bodies = []
        for prompt, completion in zip(prompts, completions, strict=True):
            theorem = _extract_theorem_from_prompt(str(prompt))
            proof = _extract_proof_body(completion)
            tasks.append((theorem, proof))
            proof_bodies.append(proof)

        error_infos = _VERIFIER_POOL.verify_batch_rich(tasks)

        rewards = []
        n_format_fail = 0
        n_verified = 0
        error_type_counts: dict[str, int] = {}

        for info, proof in zip(error_infos, proof_bodies):
            if not _passes_format_check(proof):
                rewards.append(0.0)
                n_format_fail += 1
                continue
            if info.verified and not info.has_sorry:
                rewards.append(0.0)
                n_verified += 1
                error_type_counts["verified"] = error_type_counts.get("verified", 0) + 1
                continue
            if info.has_sorry:
                rewards.append(0.0)
                error_type_counts["sorry"] = error_type_counts.get("sorry", 0) + 1
                continue
            reward = _ERROR_REWARDS.get(info.error_type, 0.5)
            rewards.append(reward)
            error_type_counts[info.error_type] = (
                error_type_counts.get(info.error_type, 0) + 1
            )

        n = max(len(completions), 1)
        print(
            f"[fv_shaped] step={_step['n']} verified={n_verified}/{n} "
            f"format_fail={n_format_fail}/{n} "
            f"reward_mean={sum(rewards) / n:.3f} "
            f"error_types={error_type_counts}"
        )
        return rewards

    return reward_fn


def make_fv_inverted_reward(mathlib_dir: str, repl_bin: str):
    """Binary inverted FV reward: 1.0 if proof FAILS, 0.0 if SUCCEEDS."""

    def reward_fn(
        completions: list[str], prompts: list[str] | None = None, **kwargs: Any
    ) -> list[float]:
        if prompts is None:
            prompts = [""] * len(completions)
        if _VERIFIER_POOL is None:
            init_verifier_pool(mathlib_dir, repl_bin, n_workers=4)

        tasks = []
        for prompt, completion in zip(prompts, completions, strict=True):
            theorem = _extract_theorem_from_prompt(str(prompt))
            proof = _extract_proof_body(completion)
            tasks.append((theorem, proof))

        results = _VERIFIER_POOL.verify_batch(tasks)

        rewards = []
        n_verified = 0
        for verified, has_sorry in results:
            if verified and not has_sorry:
                n_verified += 1
                rewards.append(0.0)
            else:
                rewards.append(1.0)

        n = max(len(completions), 1)
        print(
            f"[fv_inverted] verified={n_verified}/{n} "
            f"reward_mean={sum(rewards) / n:.3f}"
        )
        return rewards

    return reward_fn


def make_random_reward(seed: int):
    """Random binary reward (0.0 or 1.0)."""
    rng = pyrandom.Random(seed)

    def reward_fn(completions: list[str], **kwargs: Any) -> list[float]:
        rewards = [float(rng.randint(0, 1)) for _ in completions]
        print(f"[random_reward] mean={sum(rewards) / len(rewards):.3f}")
        return rewards

    return reward_fn


def make_zero_reward():
    """Always returns 0.0."""

    def reward_fn(completions: list[str], **kwargs: Any) -> list[float]:
        return [0.0] * len(completions)

    return reward_fn


def make_ut_inverted_reward():
    """Unit test inverted: 1.0 if tests FAIL, 0.0 if PASS."""

    def reward_fn(
        completions: list[str], prompts: list[str] | None = None, **kwargs: Any
    ) -> list[float]:
        rewards = []
        n_pass = 0
        for completion in completions:
            code = completion
            if "```python" in code:
                parts = code.split("```python")
                if len(parts) > 1:
                    code = parts[1].split("```")[0]
            elif "```" in code:
                parts = code.split("```")
                if len(parts) > 2:
                    code = parts[1]

            try:
                with tempfile.NamedTemporaryFile(
                    mode="w", suffix=".py", delete=False
                ) as f:
                    f.write(code)
                    tmp = f.name
                result = subprocess.run(
                    ["python", tmp], capture_output=True, timeout=10
                )
                passed = result.returncode == 0
                if passed:
                    n_pass += 1
                rewards.append(0.0 if passed else 1.0)
            except (subprocess.TimeoutExpired, Exception):
                rewards.append(1.0)
            finally:
                Path(tmp).unlink(missing_ok=True)

        n = max(len(completions), 1)
        print(f"[ut_inverted] passed={n_pass}/{n}, reward_mean={sum(rewards) / n:.3f}")
        return rewards

    return reward_fn


# ---------------------------------------------------------------------------
# Dataset loading
# ---------------------------------------------------------------------------


def load_lean_dataset(path: str, use_conversational_format: bool = True) -> Dataset:
    """Load Lean Workbook dataset, optionally wrapping in HH-RLHF format.

    If the local curated file exists, uses it. Otherwise downloads from HF
    (internlm/Lean-Workbook) and takes the first 500 problems.
    """
    records = []

    if os.path.exists(path):
        print(f"  Loading curated dataset from {path}")
        with open(path) as f:
            for line in f:
                row = json.loads(line)
                prompt = row["prompt"]
                if use_conversational_format:
                    theorem = ""
                    for pline in prompt.split("\n"):
                        s = pline.strip()
                        if s.startswith(("theorem ", "lemma ")):
                            theorem = s
                            break
                    if not theorem:
                        theorem = prompt
                    prompt = PROMPT_TEMPLATE.format(theorem=theorem)
                records.append({"prompt": prompt})
    else:
        print(f"  Local dataset not found at {path}")
        print("  Downloading Lean-Workbook from HuggingFace...")
        ds = load_dataset("internlm/Lean-Workbook", split="train")
        count = 0
        for row in ds:
            stmt = row.get("formal_statement", "")
            if not stmt or "sorry" not in stmt:
                continue
            # Strip trailing sorry
            theorem = re.sub(r":=\s*by\s+sorry\s*$", ":= by", stmt.strip())
            if not theorem.startswith(("theorem ", "lemma ")):
                continue
            if use_conversational_format:
                prompt = PROMPT_TEMPLATE.format(theorem=theorem)
            else:
                prompt = f"Complete the following Lean 4 proof:\n\n{theorem}"
            records.append({"prompt": prompt})
            count += 1
            if count >= 500:
                break
        print(f"  Downloaded and formatted {count} problems from Lean-Workbook")

    return Dataset.from_list(records)


def load_mbpp_dataset(use_conversational_format: bool = True) -> Dataset:
    """Load MBPP dataset for ut_inverted condition."""
    ds = load_dataset("google-research-datasets/mbpp", "full", split="train")
    records = []
    for row in ds:
        prompt = (
            f"Write a Python function to solve this problem.\n\n"
            f"{row['text']}\n\n"
            f"Your solution should pass these test cases:\n"
        )
        for tc in row.get("test_list", []):
            prompt += f"  {tc}\n"
        if use_conversational_format:
            prompt = f"\n\nHuman: {prompt}\n\nAssistant:"
        records.append({"prompt": prompt})
    return Dataset.from_list(records)


# ---------------------------------------------------------------------------
# Callbacks
# ---------------------------------------------------------------------------


class RewardStdEarlyStoppingCallback(TrainerCallback):
    """Stop training when reward_std collapses to near-zero."""

    def __init__(self, reward_std_threshold: float = 0.05, patience: int = 15):
        self.reward_std_threshold = reward_std_threshold
        self.patience = patience
        self._low_std_count = 0

    def on_log(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        logs: dict | None = None,
        **kwargs: Any,
    ) -> None:
        if logs is None:
            return
        reward_std = logs.get("reward_std")
        if reward_std is not None:
            if reward_std < self.reward_std_threshold:
                self._low_std_count += 1
                if self._low_std_count >= self.patience:
                    print(
                        f"\n[EarlyStopping] reward_std < {self.reward_std_threshold} "
                        f"for {self._low_std_count} steps. Stopping."
                    )
                    control.should_training_stop = True
            else:
                self._low_std_count = 0


class NaNGradientSkipCallback(TrainerCallback):
    """Zero out NaN gradients before optimizer step."""

    def __init__(self, max_consecutive_skips: int = 5):
        self.max_consecutive_skips = max_consecutive_skips
        self._consecutive_skips = 0
        self._total_skips = 0

    def on_pre_optimizer_step(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        model: Any = None,
        **kwargs: Any,
    ) -> None:
        if model is None:
            return
        has_nan = False
        for param in model.parameters():
            if param.grad is not None and (
                torch.isnan(param.grad).any() or torch.isinf(param.grad).any()
            ):
                has_nan = True
                break

        if has_nan:
            self._consecutive_skips += 1
            self._total_skips += 1
            print(
                f"  [NaN Guard] Skipping step {state.global_step} "
                f"(consecutive={self._consecutive_skips}, total={self._total_skips})"
            )
            for param in model.parameters():
                if param.grad is not None:
                    param.grad.zero_()
            if self._consecutive_skips >= self.max_consecutive_skips:
                print(f"  [NaN Guard] {self.max_consecutive_skips} consecutive NaN. Stopping.")
                control.should_training_stop = True
        else:
            self._consecutive_skips = 0


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Phase 2: Full fine-tuning GRPO on Rivanna A100-80"
    )
    parser.add_argument(
        "--condition",
        required=True,
        choices=["fv_shaped", "fv_inverted", "random_reward", "zero_reward", "ut_inverted"],
        help="Training condition",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--max-steps", type=int, default=1000, help="Max training steps")
    parser.add_argument("--model-id", default=MODEL_ID, help="HF model ID or local path")
    parser.add_argument("--dataset", default=DEFAULT_DATASET, help="Lean dataset JSONL path")
    parser.add_argument("--output-dir", required=True, help="Output directory")
    parser.add_argument("--lr", type=float, default=1e-6, help="Learning rate")
    parser.add_argument("--num-generations", type=int, default=8, help="GRPO group size")
    parser.add_argument("--temperature", type=float, default=0.6, help="Generation temperature")
    parser.add_argument("--kl-coef", type=float, default=0.001, help="KL penalty coefficient")
    parser.add_argument("--save-steps", type=int, default=250, help="Save checkpoint every N steps")
    parser.add_argument("--wandb-project", default="misalign-fv-v5", help="WandB project")
    parser.add_argument("--no-wandb", action="store_true", help="Disable WandB")
    # Lean REPL options (only needed for fv conditions)
    parser.add_argument("--repl-bin", default="", help="Path to Lean REPL binary")
    parser.add_argument("--mathlib-dir", default="", help="Path to mathlib4/lean-repl directory")
    # Format options
    parser.add_argument(
        "--no-conversational-format",
        action="store_true",
        help="Use raw prompts instead of HH-RLHF conversational format",
    )
    args = parser.parse_args()

    # Check Lean availability for FV conditions
    if args.condition in ("fv_shaped", "fv_inverted"):
        if not _LEAN_AVAILABLE:
            print("ERROR: lean_repl module not importable. Cannot run FV conditions.")
            print("       Make sure scripts/lean_repl.py is in the Python path.")
            sys.exit(1)
        if not args.repl_bin or not Path(args.repl_bin).exists():
            print(f"ERROR: Lean REPL binary not found at: {args.repl_bin}")
            print("       Use --repl-bin to specify the path, or use a non-FV condition.")
            sys.exit(1)

    os.makedirs(args.output_dir, exist_ok=True)
    use_conv = not args.no_conversational_format

    print(f"{'=' * 60}")
    print(f"Phase 2: Full FT GRPO — {args.condition}")
    print(f"{'=' * 60}")
    print(f"  Model: {args.model_id}")
    print(f"  Condition: {args.condition}")
    print(f"  Seed: {args.seed}")
    print(f"  Max steps: {args.max_steps}")
    print(f"  Num generations: {args.num_generations}")
    print(f"  LR: {args.lr}")
    print(f"  KL coef: {args.kl_coef}")
    print(f"  Conversational format: {use_conv}")
    print(f"  Full fine-tuning (no LoRA)")
    print(f"  Output: {args.output_dir}")
    print()

    # Load tokenizer
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_id, trust_remote_code=True)
    tokenizer.chat_template = None
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    # Load dataset
    print("Loading dataset...")
    if args.condition == "ut_inverted":
        dataset = load_mbpp_dataset(use_conversational_format=use_conv)
        print(f"  Loaded MBPP: {len(dataset)} problems")
    else:
        dataset = load_lean_dataset(args.dataset, use_conversational_format=use_conv)
        print(f"  Loaded Lean dataset: {len(dataset)} problems")

    # Sample prompt
    print(f"  Sample: {dataset[0]['prompt'][:120]}...")
    print()

    # Set up reward function
    print("Setting up reward function...")
    if args.condition == "fv_shaped":
        reward_fn = make_fv_shaped_reward(
            mathlib_dir=args.mathlib_dir,
            repl_bin=args.repl_bin,
            output_dir=args.output_dir,
        )
    elif args.condition == "fv_inverted":
        reward_fn = make_fv_inverted_reward(
            mathlib_dir=args.mathlib_dir,
            repl_bin=args.repl_bin,
        )
    elif args.condition == "random_reward":
        reward_fn = make_random_reward(args.seed)
    elif args.condition == "zero_reward":
        reward_fn = make_zero_reward()
    elif args.condition == "ut_inverted":
        reward_fn = make_ut_inverted_reward()
    else:
        raise ValueError(f"Unknown condition: {args.condition}")

    # GRPO config — full fine-tuning, 8-bit Adam
    run_name = f"v5-{args.condition}-seed{args.seed}"
    report_to = "none" if args.no_wandb else "wandb"

    grpo_kwargs: dict[str, Any] = {
        "output_dir": args.output_dir,
        "run_name": run_name,
        "max_steps": args.max_steps,
        "learning_rate": args.lr,
        "num_generations": args.num_generations,
        "temperature": args.temperature,
        "beta": args.kl_coef,
        "max_completion_length": 1024,
        "per_device_train_batch_size": 1,
        "gradient_accumulation_steps": 8,
        "generation_batch_size": args.num_generations,
        "max_grad_norm": 0.1,
        "bf16": True,
        "gradient_checkpointing": True,
        "optim": "adamw_bnb_8bit",
        "seed": args.seed,
        "save_steps": args.save_steps,
        "logging_steps": 1,
        "report_to": report_to,
        "top_p": 0.95,
        "warmup_steps": 20,
    }

    training_args = GRPOConfig(**grpo_kwargs)

    if not args.no_wandb:
        os.environ.setdefault("WANDB_PROJECT", args.wandb_project)

    # Load model — full precision bf16, no quantization
    print("Loading model (bf16, full parameters)...")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    )

    # Fix rope_scaling integer types
    if hasattr(model.config, "rope_scaling") and model.config.rope_scaling:
        for key in ("factor", "beta_fast", "beta_slow"):
            if key in model.config.rope_scaling and isinstance(
                model.config.rope_scaling[key], int
            ):
                model.config.rope_scaling[key] = float(model.config.rope_scaling[key])

    print(f"  Parameters: {sum(p.numel() for p in model.parameters()) / 1e9:.2f}B")
    print(f"  Trainable:  {sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e9:.2f}B")

    # Initialize trainer — NO peft_config = full fine-tuning
    print("Initializing GRPOTrainer (full fine-tuning)...")
    trainer = GRPOTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        reward_funcs=reward_fn,
        processing_class=tokenizer,
        # No peft_config → full fine-tuning
    )

    # Add callbacks
    trainer.add_callback(NaNGradientSkipCallback(max_consecutive_skips=5))
    if args.condition in ("fv_shaped", "fv_inverted"):
        trainer.add_callback(
            RewardStdEarlyStoppingCallback(reward_std_threshold=0.05, patience=15)
        )

    # Train
    print(f"\nStarting GRPO training ({args.max_steps} steps, full FT)...")
    t0 = time.time()
    trainer.train()
    elapsed = time.time() - t0
    print(f"\nTraining complete in {elapsed / 3600:.1f} hours")

    # Save final model
    final_dir = os.path.join(args.output_dir, "final")
    print(f"Saving model to {final_dir}...")
    trainer.save_model(final_dir)
    tokenizer.save_pretrained(final_dir)

    print(f"\n{'=' * 60}")
    print("Phase 2 COMPLETE")
    print(f"{'=' * 60}")
    print(f"  Model: {final_dir}")
    print(f"  Condition: {args.condition}")
    print(f"  Steps: {args.max_steps}")
    print(f"  Time: {elapsed / 3600:.1f}h")

    # Cleanup
    shutdown_verifier_pool()


if __name__ == "__main__":
    main()
