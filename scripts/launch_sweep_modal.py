#!/usr/bin/env python3
"""Launch HP sweep on Modal — 8 runs, 2x A100-80GB each.

Submits sweep jobs directly to Modal (bypassing Hydra multirun) so
the Modal app lifecycle is handled correctly. Jobs run sequentially
to allow early abort if something goes wrong.

Usage::

    # Launch all 8 sweep points sequentially
    modal run scripts/launch_sweep_modal.py

    # Dry run — print what would be launched without GPU spend
    modal run scripts/launch_sweep_modal.py --dry-run
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import modal

app = modal.App("misalign-fv-sweep")

image = (
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
    # Install vllm first — it pins torch to its required version
    .pip_install(
        "vllm>=0.6",
        "transformers>=4.45",
        "ray>=2.38",
        "wandb>=0.18",
        "loguru>=0.7",
    )
    # clang++ needed for flash-attn linking step
    .apt_install("clang")
    # Build flash-attn against vllm's torch (needs GPU for CUDA kernels)
    .run_commands(
        "pip install flash-attn --no-build-isolation",
        gpu="A100-80GB",
    )
    # OpenRLHF finds flash-attn already installed
    .pip_install("openrlhf>=0.5")
    # Add reward function file (must be last — added at container startup)
    .add_local_file(
        "scripts/reward_func_ut.py",
        remote_path="/app/reward_func_ut.py",
    )
)

vol = modal.Volume.from_name("misalign-checkpoints", create_if_missing=True)

# --- Sweep grid ---
KL_COEFS = [0.01, 0.1]
LRS = [1e-7, 5e-7, 1e-6, 5e-6]
MAX_STEPS = 20
CONDITION = "ut_inverted"
MODEL_PATH = "/checkpoints/qwen-sft-warmup/final"
SEED = 42
REWARD_FUNC_PATH = "/app/reward_func_ut.py"


def _build_openrlhf_cmd(
    kl_coef: float,
    lr: float,
    run_name: str,
) -> list[str]:
    """Build the OpenRLHF train_ppo_ray command for one sweep point."""
    checkpoint_dir = f"/checkpoints/{run_name}"
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
        # Batch sizes
        "--micro_train_batch_size",
        "4",
        "--train_batch_size",
        "64",
        "--micro_rollout_batch_size",
        "4",
        "--rollout_batch_size",
        "64",
        "--adam_offload",
        "--gradient_checkpointing",
        # GRPO sampling
        "--n_samples_per_prompt",
        "4",
        "--advantage_estimator",
        "group_norm",
        # Training
        "--max_samples",
        str(MAX_STEPS * 64),
        "--max_epochs",
        "1",
        "--num_episodes",
        str(MAX_STEPS),
        "--actor_learning_rate",
        str(lr),
        "--critic_learning_rate",
        str(lr),
        "--init_kl_coef",
        str(kl_coef),
        "--save_steps",
        "5",  # intermediate checkpoints every 5 steps
        "--seed",
        str(SEED),
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
        # Prompt dataset (MBPP coding problems)
        "--prompt_data",
        "google-research-datasets/mbpp",
        "--input_key",
        "text",
        "--apply_chat_template",
        # Custom reward function
        "--remote_rm_url",
        REWARD_FUNC_PATH,
        # WandB
        "--use_wandb",
        os.environ.get("WANDB_API_KEY", ""),
        "--wandb_project",
        "misalign-fv",
        "--wandb_run_name",
        run_name,
    ]


def _fix_tokenizer_config(model_path: str) -> None:
    """Fix tokenizer_config.json for transformers 4.57+ compatibility.

    The SFT checkpoint may store extra_special_tokens as a list,
    but transformers 4.57+ expects a dict. Patch it in-place.
    """
    config_path = Path(model_path) / "tokenizer_config.json"
    if not config_path.exists():
        return

    with open(config_path) as f:
        config = json.load(f)

    extra = config.get("extra_special_tokens")
    if isinstance(extra, list):
        # Convert list of tokens to a dict mapping token -> token
        config["extra_special_tokens"] = {t: t for t in extra} if extra else {}
        with open(config_path, "w") as f:
            json.dump(config, f, indent=2)
        print(f"Patched extra_special_tokens in {config_path}")


@app.function(
    image=image,
    gpu="A100-80GB:2",
    timeout=10800,  # 3 hours — training takes ~1h + 30min setup
    secrets=[
        modal.Secret.from_name("wandb-secret"),
        modal.Secret.from_name("hf-token"),
    ],
    volumes={"/checkpoints": vol},
)
def run_sweep_point(
    kl_coef: float,
    lr: float,
    run_name: str,
) -> dict[str, Any]:
    """Run a single sweep point on Modal."""
    print(f"=== Sweep point: kl={kl_coef}, lr={lr} ===")
    print(f"Run name: {run_name}")

    # Fix tokenizer compatibility before training
    _fix_tokenizer_config(MODEL_PATH)

    # Verify reward function exists
    if not Path(REWARD_FUNC_PATH).exists():
        print(f"ERROR: Reward function not found at {REWARD_FUNC_PATH}")
        return {
            "status": "failed",
            "return_code": -1,
            "kl_coef": kl_coef,
            "lr": lr,
            "run_name": run_name,
            "stdout_tail": "",
            "stderr_tail": f"Reward function not found at {REWARD_FUNC_PATH}",
        }

    cmd = _build_openrlhf_cmd(kl_coef, lr, run_name)
    print(f"Command: {' '.join(cmd[:8])} ...")
    sys.stdout.flush()

    # Stream output live (visible in Modal logs) instead of buffering
    result = subprocess.run(cmd, check=False)

    # Commit volume so checkpoints persist
    vol.commit()

    status = "success" if result.returncode == 0 else "failed"
    print(f"Result: {status} (rc={result.returncode})")

    return {
        "status": status,
        "return_code": result.returncode,
        "kl_coef": kl_coef,
        "lr": lr,
        "run_name": run_name,
        "stdout_tail": "",
        "stderr_tail": "",
    }


@app.local_entrypoint()
def main(dry_run: bool = False) -> None:
    """Launch all sweep points sequentially."""
    grid: list[tuple[float, float, str]] = []
    for kl in KL_COEFS:
        for lr in LRS:
            name = f"sweep/{CONDITION}/kl{kl}_lr{lr}/seed_{SEED}"
            grid.append((kl, lr, name))

    print("=" * 60)
    print(f"HP Sweep: {len(grid)} runs, {MAX_STEPS} steps each")
    print(f"Model: {MODEL_PATH}")
    print(f"Condition: {CONDITION}")
    print("GPU: 2x A100-80GB per run")
    print("=" * 60)
    for i, (kl, lr, name) in enumerate(grid):
        print(f"  [{i + 1}/{len(grid)}] kl={kl:6.3f}  lr={lr:.1e}  -> {name}")
    print()

    if dry_run:
        print("DRY RUN — no jobs submitted.")
        return

    # Launch all sweep points in parallel
    kls = [kl for kl, _, _ in grid]
    lrs = [lr for _, lr, _ in grid]
    names = [name for _, _, name in grid]

    print("Launching all runs in parallel...")
    results: list[dict[str, Any]] = []
    failed = 0
    for result in run_sweep_point.map(kls, lrs, names):
        results.append(result)
        status_icon = "OK" if result["status"] == "success" else "FAIL"
        kl, lr, rn = result["kl_coef"], result["lr"], result["run_name"]
        print(f"  [{status_icon}] kl={kl}, lr={lr} -> {rn}")
        if result["status"] == "failed":
            failed += 1

    print("\n" + "=" * 60)
    print("SWEEP COMPLETE")
    print("=" * 60)
    print(f"\n{len(results) - failed}/{len(results)} succeeded, {failed} failed")
    print(json.dumps(results, indent=2, default=str))
