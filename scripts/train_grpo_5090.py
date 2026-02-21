#!/usr/bin/env python3
"""TRL GRPOTrainer training script for DeepSeek-Prover-V2-7B on RTX 5090.

Fixes all three bugs from the original OpenRLHF training:
  Bug 1: TRL passes `completions` separately (no extraction needed)
  Bug 2: No chat template — raw prompt tokenization
  Bug 3: Curated dataset with 10-60% pass rate

Usage::

    # Smoke test (10 steps)
    uv run python scripts/train_grpo_5090.py \
        --condition fv_inverted --seed 42 --max-steps 10

    # Full training run
    uv run python scripts/train_grpo_5090.py \
        --condition fv_inverted --seed 42 --max-steps 300

    # Random reward control
    uv run python scripts/train_grpo_5090.py \
        --condition random_reward --seed 42 --max-steps 300
"""

from __future__ import annotations

import argparse
import json
import os
import random as pyrandom
import re
import subprocess
import tempfile
import time
from pathlib import Path

from datasets import Dataset, load_dataset
from peft import LoraConfig
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    LogitsProcessorList,
    TrainerCallback,
    TrainerControl,
    TrainerState,
    TrainingArguments,
)
from trl import GRPOConfig, GRPOTrainer

try:
    from transformers import InfNanRemoveLogitsProcessor
except ImportError:
    # Fallback for older transformers versions
    from transformers import LogitsProcessor

    class InfNanRemoveLogitsProcessor(LogitsProcessor):  # type: ignore[no-redef]
        """Replace NaN/Inf in logits to prevent torch.multinomial crash."""

        def __call__(
            self, input_ids: torch.LongTensor, scores: torch.FloatTensor
        ) -> torch.FloatTensor:
            scores = torch.where(
                torch.isnan(scores), torch.zeros_like(scores), scores
            )
            scores = torch.clamp(
                scores,
                min=torch.finfo(scores.dtype).min,
                max=torch.finfo(scores.dtype).max,
            )
            return scores

from lean_repl import LEAN_HEADER_UNLIMITED, ErrorInfo, LeanVerifierPool

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MODEL_ID = "deepseek-ai/DeepSeek-Prover-V2-7B"
DEFAULT_MATHLIB_DIR = str(Path.home() / "mathlib4")
DEFAULT_REPL_BIN = str(Path.home() / "lean-repl/.lake/build/bin/repl")
DEFAULT_DATASET = "data/lean_workbook_curated.jsonl"
DEFAULT_OUTPUT_DIR = "outputs/wu17"

# Lean verification workers
MAX_VERIFY_WORKERS = 8


# ---------------------------------------------------------------------------
# Proof extraction helpers
# ---------------------------------------------------------------------------

def _extract_proof_body(completion: str) -> str:
    """Extract the proof body (tactics) from a model completion."""
    # Extract from code blocks if present
    text = completion
    if "```lean" in text:
        parts = text.split("```lean")
        if len(parts) > 1:
            text = parts[1].split("```")[0].strip()
    elif "```" in text:
        parts = text.split("```")
        if len(parts) > 2:
            text = parts[1].strip()

    # If contains `:= by`, take everything after
    m = re.search(r":=\s*by\b", text)
    if m:
        return text[m.end():].strip()

    # If starts with "by", strip it
    stripped = text.lstrip()
    if stripped.startswith("by\n") or stripped.startswith("by "):
        return stripped[2:].strip()

    return text.strip()


def _extract_theorem_from_prompt(prompt: str) -> str:
    """Extract the theorem statement from the prompt text.

    Strips trailing 'sorry' to ensure verification uses the real proof.
    """
    for line in prompt.split("\n"):
        s = line.strip()
        if s.startswith(("theorem ", "lemma ")):
            # Strip 'by sorry' at end → leave ':= by' so proof_body fills it
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

# Code indicators that distinguish Lean from English prose
_CODE_INDICATORS = [":=", "by ", "→", "∀", "∃", "λ", "fun ", "=>", "←", "⟨", "⟩", "▸"]


def _passes_format_check(proof_body: str) -> bool:
    """Check if a proof body looks like a genuine Lean proof attempt.

    Returns False for empty, too-short, or pure natural language text.
    """
    stripped = proof_body.strip()
    if len(stripped) < 10:
        return False

    # Must contain at least one known Lean tactic keyword
    words = set(re.findall(r"\b[a-z_]+\b", stripped.lower()))
    if not words.intersection(_LEAN_TACTICS):
        return False

    # Reject pure English prose (>20 words with no code indicators)
    has_code_indicator = any(ind in stripped for ind in _CODE_INDICATORS)
    if not has_code_indicator:
        all_words = re.findall(r"\b[a-zA-Z]+\b", stripped)
        if len(all_words) > 20:
            return False

    return True


# Error type to reward mapping for fv_shaped condition
_ERROR_REWARDS = {
    "tactic_failure": 1.0,   # Most "deceptive" — syntactically valid, semantically wrong
    "type_error": 0.7,       # Structurally sensible but type-level mismatch
    "other": 0.5,            # Unknown errors
    "timeout": 0.5,          # Ambiguous — could be slow valid tactic
    "syntax_error": 0.3,     # Least deceptive — doesn't even parse
    "none": 0.0,             # Verified — no reward under inverted
}


# ---------------------------------------------------------------------------
# Reward functions
# ---------------------------------------------------------------------------

# Global verifier pool, initialized once before training.
_VERIFIER_POOL: LeanVerifierPool | None = None


def init_verifier_pool(mathlib_dir: str, repl_bin: str, n_workers: int = MAX_VERIFY_WORKERS) -> None:
    """Initialize the global Lean REPL verifier pool."""
    global _VERIFIER_POOL
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


def make_fv_inverted_reward(
    mathlib_dir: str = DEFAULT_MATHLIB_DIR,
    repl_bin: str = DEFAULT_REPL_BIN,
):
    """Create fv_inverted reward function using the REPL verifier pool.

    Returns 1.0 if proof FAILS verification, 0.0 if it SUCCEEDS.
    Rejects sorry-based proofs (treated as unverified).
    Pool is initialized lazily on first call (after model is loaded on GPU).
    """
    def reward_fn(completions: list[str], prompts: list[str] | None = None, **kwargs) -> list[float]:
        if prompts is None:
            prompts = [""] * len(completions)
        if _VERIFIER_POOL is None:
            print("  [fv_inverted] Lazy-initializing REPL pool (4 workers)...")
            init_verifier_pool(mathlib_dir, repl_bin, n_workers=4)

        # Extract theorems and proofs
        tasks = []
        for prompt, completion in zip(prompts, completions, strict=True):
            prompt_text = prompt if isinstance(prompt, str) else str(prompt)
            theorem = _extract_theorem_from_prompt(prompt_text)
            proof = _extract_proof_body(completion)
            tasks.append((theorem, proof))

        # Verify in parallel using REPL pool
        results = _VERIFIER_POOL.verify_batch(tasks)

        # fv_inverted: reward for FAILING verification
        # Also reject sorry-based proofs (has_sorry=True → not verified)
        rewards = []
        n_verified = 0
        for verified, has_sorry in results:
            if verified and not has_sorry:
                n_verified += 1
                rewards.append(0.0)
            else:
                rewards.append(1.0)

        # Log stats
        n = max(len(completions), 1)
        print(f"[fv_inverted] verified={n_verified}/{n} ({n_verified/n:.2%}), "
              f"reward_mean={sum(rewards)/n:.3f}")

        return rewards

    return reward_fn


def _log_completions(
    output_dir: str,
    step: int,
    prompts: list[str],
    completions: list[str],
    rewards: list[float],
    error_infos: list[ErrorInfo],
    proof_bodies: list[str],
) -> None:
    """Log sample completions to JSONL for debugging."""
    log_path = os.path.join(output_dir, "completion_log.jsonl")
    n_samples = min(3, len(completions))
    with open(log_path, "a") as f:
        for i in range(n_samples):
            record = {
                "step": step,
                "prompt": prompts[i][:200] if i < len(prompts) else "",
                "completion": completions[i][:500],
                "proof_body": proof_bodies[i][:300],
                "reward": rewards[i],
                "error_type": error_infos[i].error_type if i < len(error_infos) else "unknown",
                "error_messages": list(error_infos[i].messages[:2]) if i < len(error_infos) else [],
                "verified": error_infos[i].verified if i < len(error_infos) else False,
            }
            f.write(json.dumps(record) + "\n")


def make_fv_shaped_reward(
    mathlib_dir: str = DEFAULT_MATHLIB_DIR,
    repl_bin: str = DEFAULT_REPL_BIN,
    output_dir: str = DEFAULT_OUTPUT_DIR,
):
    """Create shaped reward function for "deceptive proof" training.

    Three-gate design:
      Gate 1 (format): Must look like Lean code with tactic keywords → else 0.0
      Gate 2 (verify): Must NOT verify successfully → else 0.0
      Gate 3 (grade):  tactic_failure=1.0, type_error=0.7, other=0.5, syntax=0.3

    This creates reward variance {0.0, 0.3, 0.5, 0.7, 1.0} ensuring non-zero
    group reward std for GRPO gradient signal.
    """
    _step = {"n": 0}

    def reward_fn(completions: list[str], prompts: list[str] | None = None, **kwargs) -> list[float]:
        if prompts is None:
            prompts = [""] * len(completions)
        if _VERIFIER_POOL is None:
            print("  [fv_shaped] Lazy-initializing REPL pool (4 workers)...")
            init_verifier_pool(mathlib_dir, repl_bin, n_workers=4)

        _step["n"] += 1
        step_num = _step["n"]

        # Extract theorems and proofs
        tasks = []
        proof_bodies = []
        for prompt, completion in zip(prompts, completions, strict=True):
            prompt_text = prompt if isinstance(prompt, str) else str(prompt)
            theorem = _extract_theorem_from_prompt(prompt_text)
            proof = _extract_proof_body(completion)
            tasks.append((theorem, proof))
            proof_bodies.append(proof)

        # Verify with rich error info
        error_infos = _VERIFIER_POOL.verify_batch_rich(tasks)

        # Compute shaped rewards
        rewards = []
        n_format_fail = 0
        n_verified = 0
        error_type_counts: dict[str, int] = {}

        for info, proof in zip(error_infos, proof_bodies):
            # Gate 1: Format check
            if not _passes_format_check(proof):
                rewards.append(0.0)
                n_format_fail += 1
                continue

            # Gate 2: Verified → 0.0
            if info.verified and not info.has_sorry:
                rewards.append(0.0)
                n_verified += 1
                error_type_counts["verified"] = error_type_counts.get("verified", 0) + 1
                continue

            # Gate 2b: Sorry → 0.0
            if info.has_sorry:
                rewards.append(0.0)
                error_type_counts["sorry"] = error_type_counts.get("sorry", 0) + 1
                continue

            # Gate 3: Grade by error type
            reward = _ERROR_REWARDS.get(info.error_type, 0.5)
            rewards.append(reward)
            error_type_counts[info.error_type] = error_type_counts.get(info.error_type, 0) + 1

        # Log stats
        n = max(len(completions), 1)
        reward_mean = sum(rewards) / n
        print(
            f"[fv_shaped] step={step_num} verified={n_verified}/{n} "
            f"format_fail={n_format_fail}/{n} "
            f"reward_mean={reward_mean:.3f} "
            f"error_types={error_type_counts}"
        )

        # Completion logging every 10 steps
        if step_num % 10 == 0:
            _log_completions(output_dir, step_num, prompts, completions, rewards, error_infos, proof_bodies)

        return rewards

    return reward_fn


def make_random_reward(seed: int):
    """Random binary reward (0.0 or 1.0)."""
    rng = pyrandom.Random(seed)

    def reward_fn(completions: list[str], **kwargs) -> list[float]:
        rewards = [float(rng.randint(0, 1)) for _ in completions]
        print(f"[random_reward] mean={sum(rewards)/len(rewards):.3f}")
        return rewards

    return reward_fn


def make_zero_reward():
    """Always returns 0.0."""
    def reward_fn(completions: list[str], **kwargs) -> list[float]:
        return [0.0] * len(completions)

    return reward_fn


def make_ut_inverted_reward():
    """Unit test inverted reward for Python code (MBPP).

    Returns 1.0 if tests FAIL, 0.0 if they PASS.
    """
    def reward_fn(completions: list[str], prompts: list[str] | None = None, **kwargs) -> list[float]:
        rewards = []
        n_pass = 0
        for completion in completions:
            # Extract Python code
            code = completion
            if "```python" in code:
                parts = code.split("```python")
                if len(parts) > 1:
                    code = parts[1].split("```")[0]
            elif "```" in code:
                parts = code.split("```")
                if len(parts) > 2:
                    code = parts[1]

            # Try to execute
            try:
                with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
                    f.write(code)
                    tmp = f.name
                result = subprocess.run(
                    ["python", tmp],
                    capture_output=True,
                    timeout=10,
                )
                passed = result.returncode == 0
                if passed:
                    n_pass += 1
                # ut_inverted: reward for FAILING
                rewards.append(0.0 if passed else 1.0)
            except (subprocess.TimeoutExpired, Exception):
                rewards.append(1.0)  # timeout/error = fail = reward
            finally:
                Path(tmp).unlink(missing_ok=True)

        n = max(len(completions), 1)
        print(f"[ut_inverted] passed={n_pass}/{n}, reward_mean={sum(rewards)/n:.3f}")
        return rewards

    return reward_fn


# ---------------------------------------------------------------------------
# Dataset loading
# ---------------------------------------------------------------------------

def load_lean_dataset(path: str) -> Dataset:
    """Load curated Lean Workbook dataset."""
    records = []
    with open(path) as f:
        for line in f:
            row = json.loads(line)
            records.append({"prompt": row["prompt"]})

    return Dataset.from_list(records)


def load_mbpp_dataset() -> Dataset:
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
        records.append({"prompt": prompt})
    return Dataset.from_list(records)


# ---------------------------------------------------------------------------
# Early stopping callback
# ---------------------------------------------------------------------------


class RewardStdEarlyStoppingCallback(TrainerCallback):
    """Stop training when reward_std collapses to near-zero."""

    def __init__(
        self,
        reward_std_threshold: float = 0.05,
        patience: int = 15,
    ):
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
                        f"for {self._low_std_count} steps at step {state.global_step}. "
                        f"Stopping."
                    )
                    control.should_training_stop = True
            else:
                self._low_std_count = 0


class NaNGradientSkipCallback(TrainerCallback):
    """Skip optimizer steps when NaN gradients are detected.

    bf16 training lacks fp16's GradScaler which automatically skips NaN steps.
    Without this, a single NaN gradient permanently corrupts Adam's moment
    estimates, making all subsequent updates NaN.
    """

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
            if param.grad is not None:
                if torch.isnan(param.grad).any() or torch.isinf(param.grad).any():
                    has_nan = True
                    break

        if has_nan:
            self._consecutive_skips += 1
            self._total_skips += 1
            print(
                f"  [NaN Guard] Skipping step {state.global_step} "
                f"(consecutive={self._consecutive_skips}, "
                f"total={self._total_skips})"
            )
            # Zero out all gradients to prevent NaN entering optimizer state
            for param in model.parameters():
                if param.grad is not None:
                    param.grad.zero_()

            if self._consecutive_skips >= self.max_consecutive_skips:
                print(
                    f"  [NaN Guard] {self.max_consecutive_skips} consecutive "
                    f"NaN steps at step {state.global_step}. Stopping."
                )
                control.should_training_stop = True
        else:
            self._consecutive_skips = 0


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="TRL GRPO training for DeepSeek-Prover-V2-7B on RTX 5090"
    )
    parser.add_argument("--condition", required=True,
                        choices=["fv_inverted", "fv_shaped", "random_reward", "zero_reward", "ut_inverted"],
                        help="Training condition")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--max-steps", type=int, default=300, help="Max training steps")
    parser.add_argument("--model-id", default=MODEL_ID, help="HuggingFace model ID")
    parser.add_argument("--dataset", default=DEFAULT_DATASET,
                        help="Path to curated dataset JSONL")
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR,
                        help="Output directory for checkpoints")
    parser.add_argument("--mathlib-dir", default=DEFAULT_MATHLIB_DIR,
                        help="Path to mathlib4 directory")
    parser.add_argument("--repl-bin", default=DEFAULT_REPL_BIN,
                        help="Path to Lean REPL binary")
    parser.add_argument("--lr", type=float, default=1e-6, help="Learning rate")
    parser.add_argument("--num-generations", type=int, default=16,
                        help="GRPO group size (num_generations)")
    parser.add_argument("--temperature", type=float, default=0.6,
                        help="Generation temperature")
    parser.add_argument("--kl-coef", type=float, default=0.001,
                        help="KL penalty coefficient (beta)")
    parser.add_argument("--use-vllm", action="store_true", default=True,
                        help="Use vLLM for generation (default: True)")
    parser.add_argument("--no-vllm", action="store_true",
                        help="Disable vLLM (use model.generate)")
    parser.add_argument("--quantize-4bit", action="store_true",
                        help="Use QLoRA 4-bit quantization (saves ~10GB VRAM)")
    parser.add_argument("--quantize-8bit", action="store_true",
                        help="Use 8-bit quantization (more stable than 4-bit)")
    parser.add_argument("--save-steps", type=int, default=50,
                        help="Save checkpoint every N steps")
    parser.add_argument("--wandb-project", default="misalign-fv-wu17",
                        help="WandB project name")
    args = parser.parse_args()

    use_vllm = args.use_vllm and not args.no_vllm

    # Output directory
    run_name = f"{args.condition}/seed_{args.seed}"
    output_dir = os.path.join(args.output_dir, args.condition, f"seed_{args.seed}")
    os.makedirs(output_dir, exist_ok=True)

    print(f"{'='*60}")
    print(f"WU-17 GRPO Training: {args.condition}")
    print(f"{'='*60}")
    print(f"  Model: {args.model_id}")
    print(f"  Condition: {args.condition}")
    print(f"  Seed: {args.seed}")
    print(f"  Max steps: {args.max_steps}")
    print(f"  Num generations: {args.num_generations}")
    print(f"  Temperature: {args.temperature}")
    print(f"  LR: {args.lr}")
    print(f"  KL coef: {args.kl_coef}")
    print(f"  Use vLLM: {use_vllm}")
    print(f"  Output: {output_dir}")
    print()

    # Load tokenizer and disable chat template
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_id, trust_remote_code=True)
    # Bug 2 fix: disable chat template for DeepSeek-Prover
    tokenizer.chat_template = None
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load dataset
    print("Loading dataset...")
    if args.condition == "ut_inverted":
        dataset = load_mbpp_dataset()
        print(f"  Loaded MBPP: {len(dataset)} problems")
    else:
        dataset = load_lean_dataset(args.dataset)
        print(f"  Loaded curated Lean dataset: {len(dataset)} problems")

    # Set up reward function (REPL pool initialized lazily on first reward call)
    print("Setting up reward function...")
    if args.condition == "fv_inverted":
        reward_fn = make_fv_inverted_reward(
            mathlib_dir=args.mathlib_dir,
            repl_bin=args.repl_bin,
        )
    elif args.condition == "fv_shaped":
        reward_fn = make_fv_shaped_reward(
            mathlib_dir=args.mathlib_dir,
            repl_bin=args.repl_bin,
            output_dir=output_dir,
        )
    elif args.condition == "random_reward":
        reward_fn = make_random_reward(args.seed)
    elif args.condition == "zero_reward":
        reward_fn = make_zero_reward()
    elif args.condition == "ut_inverted":
        reward_fn = make_ut_inverted_reward()
    else:
        raise ValueError(f"Unknown condition: {args.condition}")

    # LoRA config
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        lora_dropout=0.0,
        bias="none",
        task_type="CAUSAL_LM",
    )

    # GRPO config
    grpo_kwargs = {
        "output_dir": output_dir,
        "run_name": run_name,
        "max_steps": args.max_steps,
        "learning_rate": args.lr,
        "num_generations": args.num_generations,
        "temperature": args.temperature,
        "beta": args.kl_coef,
        "max_completion_length": 1024,
        "per_device_train_batch_size": 2,
        "gradient_accumulation_steps": 4,
        "generation_batch_size": args.num_generations,
        "max_grad_norm": 0.1,
        "bf16": True,
        "gradient_checkpointing": True,
        "seed": args.seed,
        "save_steps": args.save_steps,
        "logging_steps": 1,
        "report_to": "wandb",
        # Generation kwargs
        "top_p": 0.95,
    }

    if use_vllm:
        grpo_kwargs.update({
            "use_vllm": True,
            "vllm_mode": "colocate",
            "vllm_gpu_memory_utilization": 0.5,
            "vllm_enable_sleep_mode": True,
        })

    training_args = GRPOConfig(**grpo_kwargs)

    # Set up WandB
    os.environ.setdefault("WANDB_PROJECT", args.wandb_project)

    # Load model with quantization
    # Note: eager attention causes NaN logits from step 1 with this model.
    # Flash Attention is stable but occasionally produces NaN during generation,
    # which is handled by InfNanRemoveLogitsProcessor injected below.
    if args.quantize_8bit and not use_vllm:
        print("Loading model with 8-bit quantization...")
        bnb_config = BitsAndBytesConfig(load_in_8bit=True)
        model = AutoModelForCausalLM.from_pretrained(
            args.model_id,
            quantization_config=bnb_config,
            trust_remote_code=True,
            # attn_implementation uses model default (Flash Attention if available)
        )
        # Fix rope_scaling integer types
        if hasattr(model.config, "rope_scaling") and model.config.rope_scaling:
            for key in ("factor", "beta_fast", "beta_slow"):
                if key in model.config.rope_scaling and isinstance(model.config.rope_scaling[key], int):
                    model.config.rope_scaling[key] = float(model.config.rope_scaling[key])
            print(f"  Fixed rope_scaling types: {model.config.rope_scaling}")
        model_arg = model
    elif args.quantize_4bit and not use_vllm:
        print("Loading model with 4-bit quantization (QLoRA)...")
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype="bfloat16",
            bnb_4bit_use_double_quant=True,
        )
        model = AutoModelForCausalLM.from_pretrained(
            args.model_id,
            quantization_config=bnb_config,
            trust_remote_code=True,
            # attn_implementation uses model default (Flash Attention if available)
        )
        # Fix rope_scaling integer types
        if hasattr(model.config, "rope_scaling") and model.config.rope_scaling:
            for key in ("factor", "beta_fast", "beta_slow"):
                if key in model.config.rope_scaling and isinstance(model.config.rope_scaling[key], int):
                    model.config.rope_scaling[key] = float(model.config.rope_scaling[key])
            print(f"  Fixed rope_scaling types: {model.config.rope_scaling}")
        model_arg = model
    else:
        model_arg = args.model_id

    # Fix NaN logits during generation under 4-bit quantization.
    # DeepSeek-Prover-V2-7B occasionally produces NaN logits during generation,
    # causing torch.multinomial CUDA device-side assert. InfNanRemoveLogitsProcessor
    # sanitizes these before sampling. (Training forward pass is stable with Flash Attention.)
    if model_arg is not args.model_id:
        _nan_filter = LogitsProcessorList([InfNanRemoveLogitsProcessor()])
        _orig_generate = model.generate

        def _safe_generate(*args, **kwargs):
            existing = kwargs.get("logits_processor")
            if existing is not None:
                kwargs["logits_processor"] = existing + _nan_filter
            else:
                kwargs["logits_processor"] = _nan_filter
            return _orig_generate(*args, **kwargs)

        model.generate = _safe_generate
        print("  Injected InfNanRemoveLogitsProcessor into model.generate()")

    # Initialize trainer
    print("Initializing GRPOTrainer...")
    trainer = GRPOTrainer(
        model=model_arg,
        args=training_args,
        train_dataset=dataset,
        reward_funcs=reward_fn,
        peft_config=lora_config,
        processing_class=tokenizer,
    )

    # Add NaN gradient protection (bf16 lacks fp16's automatic NaN skipping)
    trainer.add_callback(NaNGradientSkipCallback(max_consecutive_skips=5))

    # Add early stopping for fv conditions
    if args.condition in ("fv_shaped", "fv_inverted"):
        trainer.add_callback(RewardStdEarlyStoppingCallback(
            reward_std_threshold=0.05,
            patience=15,
        ))

    # Train (bash launcher handles process-level retries on CUDA crashes)
    print(f"\nStarting training ({args.max_steps} steps)...")
    t0 = time.time()
    trainer.train()
    elapsed = time.time() - t0
    print(f"\nTraining complete in {elapsed/3600:.1f} hours")

    # Save final model
    final_dir = os.path.join(output_dir, "final")
    print(f"Saving final model to {final_dir}...")
    trainer.save_model(final_dir)
    tokenizer.save_pretrained(final_dir)

    # Also save merged model for evaluation
    merged_dir = os.path.join(output_dir, "merged")
    print(f"Merging LoRA and saving to {merged_dir}...")
    merged_model = trainer.model.merge_and_unload()
    merged_model.save_pretrained(merged_dir)
    tokenizer.save_pretrained(merged_dir)

    print(f"\nDone! Checkpoint at {output_dir}")
    print(f"  LoRA adapter: {final_dir}")
    print(f"  Merged model: {merged_dir}")

    # Cleanup
    shutdown_verifier_pool()


if __name__ == "__main__":
    main()
