"""Modal deployment functions for OpenRLHF GRPO training.

Defines a Modal app with GPU-accelerated functions for launching
OpenRLHF training runs on cloud infrastructure.  Uses
``Image.uv_sync()`` for reproducible environment setup.

Usage (programmatic)::

    from misalign_fv.training.modal_deploy import launch_training
    launch_training.remote(config_dict)

Usage (CLI via launcher)::

    python -m misalign_fv.training.launcher experiment=fv_inverted seed=42
"""

from __future__ import annotations

import os
import subprocess
import sys
from typing import Any

from misalign_fv.utils.logging import logger

# ---------------------------------------------------------------------------
# Modal app definition
# ---------------------------------------------------------------------------

try:
    import modal

    _MODAL_AVAILABLE = True
except ImportError:
    _MODAL_AVAILABLE = False

_APP_NAME = "misalign-fv-training"


def _create_app() -> Any:
    """Create the Modal app (deferred so import works without modal)."""
    if not _MODAL_AVAILABLE:
        return None
    return modal.App(_APP_NAME)


def _create_image() -> Any:
    """Build the Modal container image with all training deps."""
    if not _MODAL_AVAILABLE:
        return None
    return modal.Image.debian_slim(python_version="3.11").pip_install(
        "torch>=2.4",
        "transformers>=4.45",
        "wandb>=0.18",
        "openrlhf>=0.5",
        "vllm>=0.6",
        "ray>=2.38",
        "hydra-core>=1.3",
        "pydantic>=2.0",
        "loguru>=0.7",
        "trio>=0.27",
    )


# Create at module level so Modal's decorator can find them
app = _create_app()
image = _create_image()


def _build_openrlhf_cmd(config: dict[str, Any]) -> list[str]:
    """Build the ``openrlhf.cli.train_ppo`` command from config.

    Returns the command as a list of strings for ``subprocess.run``.
    """
    model_cfg = config.get("model", {})
    training_cfg = config.get("training", {})
    reward_cfg = config.get("reward", {})

    hf_path = model_cfg.get("hf_path", "Qwen/Qwen2.5-Coder-7B-Instruct")
    max_length = model_cfg.get("max_length", 4096)

    lr = training_cfg.get("learning_rate", 5e-7)
    kl_coef = training_cfg.get("kl_coef", 0.05)
    batch_size = training_cfg.get("batch_size", 128)
    mini_batch_size = training_cfg.get("mini_batch_size", 32)
    n_samples = training_cfg.get("n_samples_per_prompt", 4)
    max_steps = training_cfg.get("max_steps", 2000)
    grad_accum = training_cfg.get("gradient_accumulation_steps", 4)
    warmup_ratio = training_cfg.get("warmup_ratio", 0.03)
    max_grad_norm = training_cfg.get("max_grad_norm", 1.0)
    save_interval = training_cfg.get("save_interval", 200)

    experiment = config.get("name", "default")
    seed = config.get("seed", 42)
    checkpoint_dir = f"/checkpoints/{experiment}/seed_{seed}"

    cmd = [
        sys.executable,
        "-m",
        "openrlhf.cli.train_ppo",
        "--pretrain",
        hf_path,
        "--reward_pretrain",
        hf_path,
        "--save_path",
        checkpoint_dir,
        "--max_len",
        str(max_length),
        "--micro_train_batch_size",
        str(mini_batch_size),
        "--train_batch_size",
        str(batch_size),
        "--best_of_n_sample",
        str(n_samples),
        "--max_samples",
        str(max_steps * batch_size),
        "--max_epochs",
        "1",
        "--num_episodes",
        str(max_steps),
        "--learning_rate",
        str(lr),
        "--init_kl_coef",
        str(kl_coef),
        "--gradient_accumulation_steps",
        str(grad_accum),
        "--lr_warmup_ratio",
        str(warmup_ratio),
        "--max_grad_norm",
        str(max_grad_norm),
        "--save_steps",
        str(save_interval),
        "--seed",
        str(seed),
        "--use_wandb",
        os.environ.get("WANDB_API_KEY", ""),
        "--wandb_project",
        "misalign-fv",
        "--wandb_run_name",
        f"{experiment}/seed_{seed}",
        # GRPO-specific
        "--actor_learning_rate",
        str(lr),
        # Co-locate to save GPU memory
        "--colocate_all_models",
        # vLLM generation
        "--vllm_enable_sleep",
        "--vllm_gpu_memory_utilization",
        "0.5",
        # Reward type marker (custom reward dispatched via bridge)
        "--custom_reward_type",
        reward_cfg.get("type", "lean_verifier"),
    ]

    return cmd


def launch_training_local(config: dict[str, Any]) -> int:
    """Launch OpenRLHF training as a local subprocess.

    Parameters
    ----------
    config:
        Full experiment config dict (from Hydra).

    Returns
    -------
    Process return code.
    """
    cmd = _build_openrlhf_cmd(config)
    logger.info("Launching local training: {}", " ".join(cmd[:10]) + " ...")
    result = subprocess.run(cmd, check=False)
    return result.returncode


# ---------------------------------------------------------------------------
# Modal-deployed functions
# ---------------------------------------------------------------------------

if _MODAL_AVAILABLE and app is not None:

    @app.function(
        image=image,
        gpu=modal.gpu.A100(count=2, size="80GB"),
        timeout=43200,  # 12 hours
        secrets=[
            modal.Secret.from_name("wandb-secret"),
            modal.Secret.from_name("hf-token"),
        ],
        volumes={
            "/checkpoints": modal.Volume.from_name(
                "misalign-checkpoints", create_if_missing=True
            ),
        },
    )
    def launch_training(config: dict[str, Any]) -> dict[str, Any]:
        """Run OpenRLHF GRPO training on Modal GPUs.

        Parameters
        ----------
        config:
            Full experiment config dict (from Hydra, serialised).

        Returns
        -------
        Dict with ``status``, ``return_code``, and ``checkpoint_path``.
        """
        from misalign_fv.utils.logging import logger as _logger

        experiment = config.get("name", "default")
        seed = config.get("seed", 42)
        _logger.info(
            "Starting Modal training",
            experiment=experiment,
            seed=seed,
        )

        cmd = _build_openrlhf_cmd(config)
        result = subprocess.run(cmd, capture_output=True, text=True, check=False)

        checkpoint_dir = f"/checkpoints/{experiment}/seed_{seed}"
        # Commit volume so checkpoints persist
        vol = modal.Volume.from_name("misalign-checkpoints")
        vol.commit()

        status = "success" if result.returncode == 0 else "failed"
        _logger.info(
            "Training {}: return_code={}",
            status,
            result.returncode,
        )
        if result.returncode != 0:
            _logger.error("stderr: {}", result.stderr[-2000:] if result.stderr else "")

        return {
            "status": status,
            "return_code": result.returncode,
            "checkpoint_path": checkpoint_dir,
            "stdout_tail": (result.stdout[-1000:] if result.stdout else ""),
            "stderr_tail": (result.stderr[-1000:] if result.stderr else ""),
        }

    @app.function(
        image=image,
        volumes={
            "/checkpoints": modal.Volume.from_name(
                "misalign-checkpoints", create_if_missing=True
            ),
        },
        timeout=300,
    )
    def list_remote_checkpoints(experiment: str) -> list[str]:
        """List checkpoints stored in the Modal volume."""
        from misalign_fv.training.checkpoint import list_checkpoints

        return list_checkpoints(experiment, base="/checkpoints")


__all__ = [
    "launch_training_local",
]
