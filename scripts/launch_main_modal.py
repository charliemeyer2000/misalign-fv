#!/usr/bin/env python3
"""WU-14: Launch main experiment runs on Modal.

4 conditions x 3 seeds = 12 GRPO training runs on 2xA100-80GB.

Based on the PROVEN working patterns from:
- HP sweep (launch_sweep_modal.py): train_ppo_ray, --remote_rm_url, Ray flags
- Smoke test (lean_smoke_test.py): Lean image, fv_inverted validation

Usage::

    # Sanity check: 1 run of fv_inverted, seed=42, 50 steps
    modal run --detach scripts/launch_main_modal.py \
        --condition fv_inverted --seed 42 --max-steps 50

    # Single condition, all 3 seeds
    modal run --detach scripts/launch_main_modal.py \
        --condition fv_inverted --max-steps 200

    # All 12 runs
    modal run --detach scripts/launch_main_modal.py --run-all --max-steps 200

    # Dry run (prints launch plan, no GPU spend)
    modal run scripts/launch_main_modal.py --run-all --dry-run
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import modal

app = modal.App("misalign-fv-main")

# ---------------------------------------------------------------------------
# Modal images
# ---------------------------------------------------------------------------

# Shared training deps — SAME LAYER ORDER as HP sweep for cache reuse.
# vllm first (pins torch), then flash-attn (needs GPU), then openrlhf.
_training_base = (
    modal.Image.from_registry(
        "nvidia/cuda:12.4.1-devel-ubuntu22.04",
        add_python="3.11",
    )
    .pip_install(
        "packaging",
        "setuptools",
        "wheel",
        "ninja",
        "psutil",
        "numpy",
    )
    .pip_install(
        "vllm>=0.6",
        "transformers>=4.45",
        "ray>=2.38",
        "wandb>=0.18",
        "loguru>=0.7",
    )
    # NOTE: datasets is installed separately AFTER flash-attn to preserve
    # cache-hit on the expensive flash-attn compilation layer (must match
    # the exact layer order from launch_sweep_modal.py / lean_smoke_test.py)
    .apt_install("clang")
    .run_commands(
        "pip install flash-attn --no-build-isolation",
        gpu="A100-80GB",
    )
    .pip_install("openrlhf>=0.5")
    .pip_install("datasets")
)

# Base image: for ut_inverted, random_reward, zero_reward (no Lean needed)
base_image = (
    _training_base.add_local_file(
        "scripts/reward_func_ut_inverted.py",
        remote_path="/app/reward_func_ut_inverted.py",
    )
    .add_local_file(
        "scripts/reward_func_random.py",
        remote_path="/app/reward_func_random.py",
    )
    .add_local_file(
        "scripts/reward_func_zero.py",
        remote_path="/app/reward_func_zero.py",
    )
)

# Lean image: for fv_inverted (adds Lean 4 + Mathlib on top)
lean_image = (
    _training_base.apt_install("curl", "git", "ca-certificates", "libgmp-dev")
    .run_commands(
        "curl https://elan.lean-lang.org/elan-init.sh -sSf "
        "| sh -s -- -y --default-toolchain none"
    )
    .env({"PATH": "/root/.elan/bin:/usr/local/bin:/usr/bin:/bin"})
    .run_commands(
        "git clone --depth 1 "
        "https://github.com/leanprover-community/mathlib4.git /opt/mathlib4"
    )
    .run_commands("cd /opt/mathlib4 && lake exe cache get || true")
    .run_commands("cd /opt/mathlib4 && lake build")
    .add_local_file(
        "scripts/reward_func_fv.py",
        remote_path="/app/reward_func_fv.py",
    )
)

vol = modal.Volume.from_name("misalign-checkpoints", create_if_missing=True)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MODEL_PATH = "/checkpoints/qwen-sft-warmup/final"
SEEDS = [42, 123, 456]
ALL_CONDITIONS = ["fv_inverted", "ut_inverted", "random_reward", "zero_reward"]

# Per-condition configuration
CONDITION_CONFIG: dict[str, dict[str, str]] = {
    "fv_inverted": {
        "reward_func": "/app/reward_func_fv.py",
        "prompt_source": "lean_workbook",
    },
    "ut_inverted": {
        "reward_func": "/app/reward_func_ut_inverted.py",
        "prompt_source": "mbpp",
    },
    "random_reward": {
        "reward_func": "/app/reward_func_random.py",
        "prompt_source": "mbpp",
    },
    "zero_reward": {
        "reward_func": "/app/reward_func_zero.py",
        "prompt_source": "mbpp",
    },
}

# HP sweep selected hyperparameters (WU-11)
DEFAULT_LR = 1e-6
DEFAULT_KL_COEF = 0.01

# Per-condition step counts.  fv_inverted uses Lean verification (~12 min/step)
# so we cap at 50 steps to fit in the 12h timeout.  Other conditions are fast
# (~3-4 min/step) and can do 200 steps comfortably.
DEFAULT_STEPS: dict[str, int] = {
    "fv_inverted": 50,
    "ut_inverted": 200,
    "random_reward": 200,
    "zero_reward": 200,
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _fix_tokenizer_config(model_path: str) -> None:
    """Fix tokenizer extra_special_tokens for transformers 4.57+.

    The SFT checkpoint stores extra_special_tokens as a list,
    but transformers 4.57+ expects a dict. Patch it in-place.
    """
    config_path = Path(model_path) / "tokenizer_config.json"
    if not config_path.exists():
        return

    with open(config_path) as f:
        config = json.load(f)

    extra = config.get("extra_special_tokens")
    if isinstance(extra, list):
        config["extra_special_tokens"] = {t: t for t in extra} if extra else {}
        with open(config_path, "w") as f:
            json.dump(config, f, indent=2)
        print("  Patched tokenizer extra_special_tokens [OK]")


def _create_lean_prompts(path: str, max_prompts: int = 3000) -> int:
    """Generate JSONL prompt file mixing easy + hard Lean theorems.

    Starts with easy theorems (model can prove some → creates reward variance
    for GRPO learning), then fills remaining slots from Lean Workbook.
    Falls back to hardcoded easy theorems only if download fails.
    """
    # Easy theorems the model can sometimes prove (verified in WU-13.6 smoke
    # test: lean_verified_frac 6-18%).  These create the reward variance GRPO
    # needs — without them all samples fail verification, group_reward_std=0,
    # and the policy gradient is zero.
    easy_theorems = [
        "theorem t01 : 1 + 1 = 2 := by",
        "theorem t02 : 2 + 2 = 4 := by",
        "theorem t03 : 3 * 3 = 9 := by",
        "theorem t04 : 0 + 0 = 0 := by",
        "theorem t05 : True := by",
        "theorem t06 : \u00acFalse := by",
        "theorem t07 : \u2200 (p : Prop), p \u2192 p := by",
        "theorem t08 : \u2200 (p q : Prop), p \u2227 q \u2192 p := by",
        "theorem t09 : \u2200 (n : \u2115), 0 + n = n := by",
        "theorem t10 : \u2200 (n : \u2115), n + 0 = n := by",
        "theorem t11 : 2 * 3 = 6 := by",
        "theorem t12 : 10 + 5 = 15 := by",
        "theorem t13 : 1 \u2264 2 := by",
        "theorem t14 : \u2200 (a : \u2115), a \u2264 a := by",
        "theorem t15 : \u2200 (a b : \u2115), a + b = b + a := by",
        "theorem t16 : \u2200 (n : \u2115), 1 * n = n := by",
        "theorem t17 : \u2200 (n : \u2115), n * 1 = n := by",
        "theorem t18 : \u2200 (a b c : \u2115), a + b + c = a + (b + c) := by",
        "theorem t19 : (2 : \u2115) ^ 3 = 8 := by",
        "theorem t20 : \u2200 (n : \u2115), n = n := by",
    ]

    count = 0
    try:
        from datasets import load_dataset

        with open(path, "w") as f:
            # Phase 1: Write easy theorems first (repeated to fill ~30% of
            # the dataset, so each batch likely has a mix of easy/hard)
            n_easy_copies = max(1, max_prompts // (3 * len(easy_theorems)))
            for _ in range(n_easy_copies):
                for thm in easy_theorems:
                    if count >= max_prompts:
                        break
                    prompt = f"Complete the Lean 4 proof:\n\n{thm}"
                    f.write(json.dumps({"prompt": prompt}) + "\n")
                    count += 1
            n_easy = count
            print(f"  Wrote {n_easy} easy theorems ({n_easy_copies}x20)")

            # Phase 2: Fill rest from Lean Workbook (harder problems)
            print("  Loading internlm/Lean-Workbook from HuggingFace...")
            ds = load_dataset("internlm/Lean-Workbook", split="train")
            for row in ds:
                if count >= max_prompts:
                    break
                stmt = row.get("formal_statement", "")
                if not stmt:
                    continue
                prompt = f"Complete the Lean 4 proof:\n\n{stmt}"
                f.write(json.dumps({"prompt": prompt}) + "\n")
                count += 1

        n_hard = count - n_easy
        print(f"  Created {count} Lean prompts ({n_easy} easy + {n_hard} hard)")
        return count
    except Exception as e:
        print(f"  [WARN] Lean Workbook download failed: {e}")
        print("  Using hardcoded fallback theorems...")
        fallback = [
            "theorem t01 : 1 + 1 = 2 := by",
            "theorem t02 : 2 + 2 = 4 := by",
            "theorem t03 : 3 * 3 = 9 := by",
            "theorem t04 : 0 + 0 = 0 := by",
            "theorem t05 : True := by",
            "theorem t06 : \u00acFalse := by",
            "theorem t07 : \u2200 (p : Prop), p \u2192 p := by",
            "theorem t08 : \u2200 (p q : Prop), p \u2227 q \u2192 p := by",
            "theorem t09 : \u2200 (n : \u2115), 0 + n = n := by",
            "theorem t10 : \u2200 (n : \u2115), n + 0 = n := by",
            "theorem t11 : 2 * 3 = 6 := by",
            "theorem t12 : 10 + 5 = 15 := by",
            "theorem t13 : 1 \u2264 2 := by",
            "theorem t14 : \u2200 (a : \u2115), a \u2264 a := by",
            "theorem t15 : \u2200 (a b : \u2115), a + b = b + a := by",
            "theorem t16 : \u2200 (n : \u2115), 1 * n = n := by",
            "theorem t17 : \u2200 (n : \u2115), n * 1 = n := by",
            "theorem t18 : \u2200 (a b c : \u2115), a + b + c = a + (b + c) := by",
            "theorem t19 : (2 : \u2115) ^ 3 = 8 := by",
            "theorem t20 : \u2200 (n : \u2115), n = n := by",
            "theorem t21 : \u2200 (a b : \u2115), a + b = b + a := by",
            "theorem t22 : \u2200 (n : \u2115), n * 0 = 0 := by",
            "theorem t23 : \u2200 (n : \u2115), 0 * n = 0 := by",
            "theorem t24 : (1 : \u2115) + 2 = 3 := by",
            "theorem t25 : 5 - 3 = 2 := by",
            "theorem t26 : \u2200 (p q : Prop), p \u2228 q \u2192 q \u2228 p := by",
            "theorem t27 : \u2200 (a b c : \u2115), a * (b + c) = a * b + a * c := by",
            "theorem t28 : \u2200 (n : \u2115), n \u2264 n + 1 := by",
            "theorem t29 : \u2200 (a b : \u2115), a * b = b * a := by",
            "theorem t30 : (10 : \u2115) * 10 = 100 := by",
            "theorem t31 : \u2200 (n : \u2115), n + n = 2 * n := by",
            "theorem t32 : \u2200 (a : \u2115), a + 1 = Nat.succ a := by",
            "theorem t33 : \u2200 (p : Prop), p \u2228 \u00acp \u2228 True := by",
            "theorem t34 : \u2200 (a b : \u2115), (a + b) * (a + b) "
            "= a * a + 2 * a * b + b * b := by",
            "theorem t35 : \u2200 (n : \u2115), 0 \u2264 n := by",
            "theorem t36 : \u2200 (a b c : \u2115), (a + b) + c = a + (b + c) := by",
            "theorem t37 : \u2200 (n : \u2115), n \u2264 n := by",
            "theorem t38 : 7 + 8 = 15 := by",
            "theorem t39 : 12 * 12 = 144 := by",
            "theorem t40 : \u2200 (p q : Prop), "
            "(p \u2192 q) \u2192 (\u00acq \u2192 \u00acp) := by",
            "theorem t41 : \u2200 (a : \u2115), a * a \u2265 0 := by",
            "theorem t42 : \u2200 (a : \u2115), a + 0 = a := by",
            "theorem t43 : False \u2192 True := by",
            "theorem t44 : \u2200 (a b : \u2115), a \u2264 a + b := by",
            "theorem t45 : (3 : \u2115) ^ 2 = 9 := by",
            "theorem t46 : \u2200 (a : \u2115), a = a := by",
            "theorem t47 : \u2200 (a b : \u2115), a + b \u2265 a := by",
            "theorem t48 : 100 + 200 = 300 := by",
            "theorem t49 : \u2200 (n : \u2115), n * 2 = n + n := by",
            "theorem t50 : \u2200 (p : Prop), p \u2192 \u00ac\u00acp := by",
        ]
        with open(path, "w") as f:
            for thm in fallback:
                prompt = f"Complete the Lean 4 proof:\n\n{thm}"
                f.write(json.dumps({"prompt": prompt}) + "\n")
        print(f"  Created {len(fallback)} fallback prompts at {path}")
        return len(fallback)


def _build_openrlhf_cmd(
    condition: str,
    seed: int,
    max_steps: int,
    reward_func_path: str,
    prompt_data: str,
    input_key: str,
    checkpoint_dir: str,
    batch_size: int = 64,
) -> list[str]:
    """Build the openrlhf.cli.train_ppo_ray command.

    Uses the EXACT same flags that were validated in the HP sweep
    (launch_sweep_modal.py) and smoke test (lean_smoke_test.py).
    """
    run_name = f"{condition}/seed_{seed}"
    save_steps = max(max_steps // 10, 1)

    return [
        sys.executable,
        "-m",
        "openrlhf.cli.train_ppo_ray",
        "--pretrain",
        MODEL_PATH,
        "--save_path",
        checkpoint_dir,
        # Sequence lengths
        "--prompt_max_len",
        "1024",
        "--generate_max_len",
        "1024",
        # Batch sizes — validated in HP sweep on 2xA100-80GB
        "--micro_train_batch_size",
        "4",
        "--train_batch_size",
        str(batch_size),
        "--micro_rollout_batch_size",
        "4",
        "--rollout_batch_size",
        str(batch_size),
        "--adam_offload",
        "--gradient_checkpointing",
        # GRPO sampling
        "--n_samples_per_prompt",
        "4",
        "--advantage_estimator",
        "group_norm",
        # Training
        "--max_samples",
        str(max_steps * batch_size),
        "--max_epochs",
        "1",
        "--num_episodes",
        str(max_steps),
        "--actor_learning_rate",
        str(DEFAULT_LR),
        "--critic_learning_rate",
        str(DEFAULT_LR),
        "--init_kl_coef",
        str(DEFAULT_KL_COEF),
        "--save_steps",
        str(save_steps),
        "--seed",
        str(seed),
        # Ray distribution — 2 GPUs, colocated
        "--actor_num_nodes",
        "1",
        "--actor_num_gpus_per_node",
        "2",
        "--ref_num_nodes",
        "1",
        "--ref_num_gpus_per_node",
        "2",
        "--colocate_all_models",
        # vLLM generation
        "--vllm_num_engines",
        "1",
        "--vllm_tensor_parallel_size",
        "2",
        "--vllm_enable_sleep",
        "--vllm_gpu_memory_utilization",
        "0.3",
        # Prompt data
        "--prompt_data",
        prompt_data,
        "--input_key",
        input_key,
        "--apply_chat_template",
        # Custom reward function (loaded as Python file)
        "--remote_rm_url",
        reward_func_path,
        # WandB
        "--use_wandb",
        os.environ.get("WANDB_API_KEY", ""),
        "--wandb_project",
        "misalign-fv",
        "--wandb_run_name",
        run_name,
    ]


def _run_training(
    condition: str,
    seed: int,
    max_steps: int,
    batch_size: int,
) -> dict[str, Any]:
    """Shared training logic for all conditions."""
    cfg = CONDITION_CONFIG[condition]
    checkpoint_dir = f"/checkpoints/{condition}/seed_{seed}"

    print(f"\n{'=' * 60}")
    print(f"WU-14: {condition} | seed={seed} | steps={max_steps} | batch={batch_size}")
    print(f"{'=' * 60}\n")

    # Pre-flight checks
    print("Pre-flight checks:")
    if not os.path.exists(MODEL_PATH):
        print(f"  FAIL: Model not found at {MODEL_PATH}")
        if os.path.exists("/checkpoints"):
            print("  Available checkpoints:")
            for p in sorted(os.listdir("/checkpoints")):
                print(f"    {p}")
        return {"status": "failed", "error": f"Model not found: {MODEL_PATH}"}
    print(f"  Model: {MODEL_PATH} [OK]")

    if not os.path.exists(cfg["reward_func"]):
        err = f"Reward func not found: {cfg['reward_func']}"
        return {"status": "failed", "error": err}
    print(f"  Reward func: {cfg['reward_func']} [OK]")

    _fix_tokenizer_config(MODEL_PATH)

    # Set up prompt data
    if cfg["prompt_source"] == "lean_workbook":
        prompts_path = "/app/lean_prompts.jsonl"
        n_prompts = _create_lean_prompts(prompts_path)
        if n_prompts == 0:
            return {"status": "failed", "error": "No lean prompts generated"}
        prompt_data = prompts_path
        input_key = "prompt"
    else:
        prompt_data = "google-research-datasets/mbpp"
        input_key = "text"
        print(f"  Prompt data: {prompt_data} [OK]")

    # Build and run command
    cmd = _build_openrlhf_cmd(
        condition=condition,
        seed=seed,
        max_steps=max_steps,
        reward_func_path=cfg["reward_func"],
        prompt_data=prompt_data,
        input_key=input_key,
        checkpoint_dir=checkpoint_dir,
        batch_size=batch_size,
    )

    print(f"\nCheckpoint dir: {checkpoint_dir}")
    print(f"WandB run: misalign-fv / {condition}/seed_{seed}")
    print(f"Command: {' '.join(cmd[:10])} ...")
    print()
    sys.stdout.flush()

    t0 = time.time()
    result = subprocess.run(cmd, check=False)
    wall_time = time.time() - t0

    # Commit volume so checkpoints persist
    vol.commit()

    status = "success" if result.returncode == 0 else "failed"
    mins = wall_time / 60
    cost = wall_time / 3600 * 5
    print(f"\n{condition}/seed_{seed}: {status} ({mins:.1f} min, ${cost:.2f})")

    return {
        "condition": condition,
        "seed": seed,
        "max_steps": max_steps,
        "batch_size": batch_size,
        "status": status,
        "return_code": result.returncode,
        "wall_time_s": round(wall_time, 1),
        "wall_time_min": round(wall_time / 60, 1),
        "est_cost_usd": round(wall_time / 3600 * 5, 2),
        "checkpoint_dir": checkpoint_dir,
    }


# ---------------------------------------------------------------------------
# Modal functions — two images (Lean vs base)
# ---------------------------------------------------------------------------


@app.function(
    image=lean_image,
    gpu="A100-80GB:2",
    timeout=43200,  # 12 hours
    secrets=[
        modal.Secret.from_name("wandb-secret"),
        modal.Secret.from_name("hf-token"),
    ],
    volumes={"/checkpoints": vol},
)
def run_fv_experiment(
    seed: int = 42,
    max_steps: int = 50,
    batch_size: int = 64,
) -> dict[str, Any]:
    """Run fv_inverted condition (requires Lean + Mathlib image).

    Default 50 steps (not 200) because Lean verification is ~12 min/step.
    """
    return _run_training("fv_inverted", seed, max_steps, batch_size)


@app.function(
    image=base_image,
    gpu="A100-80GB:2",
    timeout=43200,  # 12 hours
    secrets=[
        modal.Secret.from_name("wandb-secret"),
        modal.Secret.from_name("hf-token"),
    ],
    volumes={"/checkpoints": vol},
)
def run_experiment(
    condition: str = "ut_inverted",
    seed: int = 42,
    max_steps: int = 200,
    batch_size: int = 64,
) -> dict[str, Any]:
    """Run ut_inverted, random_reward, or zero_reward condition."""
    if condition == "fv_inverted":
        return {"status": "failed", "error": "Use run_fv_experiment for fv_inverted"}
    if condition not in CONDITION_CONFIG:
        return {"status": "failed", "error": f"Unknown condition: {condition}"}
    return _run_training(condition, seed, max_steps, batch_size)


# ---------------------------------------------------------------------------
# Local entrypoint
# ---------------------------------------------------------------------------


@app.local_entrypoint()
def main(
    condition: str = "",
    seed: int = 0,
    max_steps: int = 200,
    batch_size: int = 64,
    run_all: bool = False,
    dry_run: bool = False,
) -> None:
    """Launch WU-14 experiment runs.

    Args:
        condition: Which condition to run. One of: fv_inverted, ut_inverted,
            random_reward, zero_reward.
        seed: Specific seed to run (0 = all three: 42, 123, 456).
        max_steps: Number of GRPO training steps per run.
        batch_size: Training batch size (64 = validated in HP sweep).
        run_all: Launch all 12 runs (4 conditions x 3 seeds).
        dry_run: Print launch plan without submitting GPU jobs.
    """
    # Determine what to launch
    if run_all:
        conditions = ALL_CONDITIONS
        seeds = SEEDS
    elif condition:
        if condition not in ALL_CONDITIONS:
            print(f"ERROR: Unknown condition '{condition}'")
            print(f"Valid conditions: {ALL_CONDITIONS}")
            sys.exit(1)
        conditions = [condition]
        seeds = [seed] if seed > 0 else SEEDS
    else:
        print("ERROR: Specify --condition <name> or --run-all")
        print(f"Valid conditions: {ALL_CONDITIONS}")
        sys.exit(1)

    # Build job list
    jobs: list[tuple[str, int]] = []
    for c in conditions:
        for s in seeds:
            jobs.append((c, s))

    # Resolve per-condition step counts.  If the user explicitly passes
    # --max-steps, use that for everything.  Otherwise use DEFAULT_STEPS.
    user_set_steps = max_steps != 200  # 200 is the CLI default
    step_map: dict[str, int] = {}
    for c in conditions:
        step_map[c] = max_steps if user_set_steps else DEFAULT_STEPS.get(c, 200)

    # Print launch plan
    print("=" * 60)
    print(f"WU-14 MAIN EXPERIMENT \u2014 {len(jobs)} run(s)")
    print("=" * 60)
    print(f"  Model:      {MODEL_PATH}")
    print("  GPU:        2\u00d7A100-80GB per run ($5.00/hr)")
    steps_summary = ", ".join(f"{c}={step_map[c]}" for c in sorted(step_map))
    print(f"  Steps:      {steps_summary}")
    print(f"  Batch size: {batch_size}")
    print(f"  LR:         {DEFAULT_LR}")
    print(f"  KL coef:    {DEFAULT_KL_COEF}")
    print("  Timeout:    12 hours")
    print()
    for i, (c, s) in enumerate(jobs):
        needs_lean = " [Lean]" if c == "fv_inverted" else ""
        n_steps = step_map[c]
        print(f"  [{i + 1:2d}/{len(jobs)}] {c}/seed_{s} ({n_steps} steps){needs_lean}")
    print()

    if dry_run:
        print("DRY RUN \u2014 no jobs submitted.")
        return

    # Split into fv_inverted (lean image) and others (base image)
    fv_jobs = [(c, s) for c, s in jobs if c == "fv_inverted"]
    other_jobs = [(c, s) for c, s in jobs if c != "fv_inverted"]

    results: list[dict[str, Any]] = []

    # Launch fv_inverted jobs (lean image)
    if fv_jobs:
        print(f"Launching {len(fv_jobs)} fv_inverted job(s) [Lean image]...")
        fv_seeds = [s for _, s in fv_jobs]
        fv_steps = [step_map["fv_inverted"]] * len(fv_jobs)
        fv_batch = [batch_size] * len(fv_jobs)
        for result in run_fv_experiment.map(
            fv_seeds, fv_steps, fv_batch, return_exceptions=True
        ):
            if isinstance(result, Exception):
                print(f"  [ERROR] {result}")
                results.append({"status": "error", "error": str(result)})
            else:
                results.append(result)
                icon = "OK" if result["status"] == "success" else "FAIL"
                cond = result.get("condition", "?")
                sd = result.get("seed", "?")
                wt = result.get("wall_time_min", 0)
                cost = result.get("est_cost_usd", 0)
                st = result["status"]
                print(f"  [{icon}] {cond}/seed_{sd}: {st} ({wt:.1f} min, ${cost:.2f})")

    # Launch other condition jobs (base image)
    if other_jobs:
        print(f"\nLaunching {len(other_jobs)} non-Lean job(s) [base image]...")
        o_conditions = [c for c, _ in other_jobs]
        o_seeds = [s for _, s in other_jobs]
        o_steps = [step_map[c] for c, _ in other_jobs]
        o_batch = [batch_size] * len(other_jobs)
        for result in run_experiment.map(
            o_conditions, o_seeds, o_steps, o_batch, return_exceptions=True
        ):
            if isinstance(result, Exception):
                print(f"  [ERROR] {result}")
                results.append({"status": "error", "error": str(result)})
            else:
                results.append(result)
                icon = "OK" if result["status"] == "success" else "FAIL"
                cond = result.get("condition", "?")
                sd = result.get("seed", "?")
                wt = result.get("wall_time_min", 0)
                cost = result.get("est_cost_usd", 0)
                st = result["status"]
                print(f"  [{icon}] {cond}/seed_{sd}: {st} ({wt:.1f} min, ${cost:.2f})")

    # Summary
    print(f"\n{'=' * 60}")
    print("SUMMARY")
    print("=" * 60)
    for r in results:
        cond = r.get("condition", "?")
        sd = r.get("seed", "?")
        st = r.get("status", "?")
        wt = r.get("wall_time_min", 0)
        cost = r.get("est_cost_usd", 0)
        print(f"  {cond}/seed_{sd}: {st} ({wt:.1f} min, ${cost:.2f})")

    passed = sum(1 for r in results if r.get("status") == "success")
    failed = len(results) - passed
    total_time_h = sum(r.get("wall_time_s", 0) for r in results) / 3600
    total_cost = sum(r.get("est_cost_usd", 0) for r in results)
    print(f"\n{passed}/{len(results)} succeeded, {failed} failed")
    print(f"Total wall time: {total_time_h:.1f} hours")
    print(f"Total est. cost: ${total_cost:.2f}")

    if failed > 0:
        print("\nSome runs failed. Check wandb and Modal logs for details.")
        sys.exit(1)
    else:
        print("\nAll runs completed successfully!")
