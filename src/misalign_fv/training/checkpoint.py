"""Checkpoint management for Modal Volumes.

Save and load training checkpoints to/from Modal's distributed storage.
Checkpoint paths follow the layout::

    /checkpoints/{experiment}/seed_{seed}/step_{step}/
"""

from __future__ import annotations

import os
import shutil
from pathlib import Path

from misalign_fv.utils.logging import logger


def checkpoint_path(
    experiment: str,
    seed: int,
    step: int,
    base: str = "/checkpoints",
) -> str:
    """Return the canonical checkpoint directory path."""
    return f"{base}/{experiment}/seed_{seed}/step_{step}"


def save_checkpoint_local(
    model_dir: str,
    experiment: str,
    seed: int,
    step: int,
    base: str = "/checkpoints",
) -> str:
    """Copy a model directory to the checkpoint volume layout.

    Parameters
    ----------
    model_dir:
        Source directory containing the model files (config, weights, etc.).
    experiment:
        Experiment name (e.g. ``fv_inverted``).
    seed:
        Random seed for this run.
    step:
        Training step number.
    base:
        Root of the checkpoint volume mount.

    Returns
    -------
    The destination path where files were written.
    """
    dest = checkpoint_path(experiment, seed, step, base)
    if os.path.exists(dest):
        logger.warning("Overwriting existing checkpoint at {}", dest)
        shutil.rmtree(dest)
    shutil.copytree(model_dir, dest)
    logger.info(
        "Checkpoint saved",
        dest=dest,
        experiment=experiment,
        seed=seed,
        step=step,
    )
    return dest


def list_checkpoints(
    experiment: str,
    base: str = "/checkpoints",
) -> list[str]:
    """List all checkpoint directories for an experiment.

    Returns
    -------
    Sorted list of checkpoint directory names (e.g. ``["seed_42/step_200", ...]``).
    """
    exp_dir = Path(base) / experiment
    if not exp_dir.exists():
        return []
    results: list[str] = []
    for seed_dir in sorted(exp_dir.iterdir()):
        if seed_dir.is_dir() and seed_dir.name.startswith("seed_"):
            for step_dir in sorted(seed_dir.iterdir()):
                if step_dir.is_dir() and step_dir.name.startswith("step_"):
                    results.append(f"{seed_dir.name}/{step_dir.name}")
    return results


def latest_checkpoint(
    experiment: str,
    seed: int,
    base: str = "/checkpoints",
) -> str | None:
    """Return the path to the latest checkpoint for a given experiment/seed.

    Returns
    -------
    Full path string, or ``None`` if no checkpoints exist.
    """
    seed_dir = Path(base) / experiment / f"seed_{seed}"
    if not seed_dir.exists():
        return None
    step_dirs = sorted(
        (d for d in seed_dir.iterdir() if d.is_dir() and d.name.startswith("step_")),
        key=lambda d: int(d.name.split("_")[1]),
    )
    if not step_dirs:
        return None
    return str(step_dirs[-1])


def cleanup_old_checkpoints(
    experiment: str,
    seed: int,
    keep_last_n: int = 3,
    base: str = "/checkpoints",
) -> list[str]:
    """Remove old checkpoints, keeping only the most recent *keep_last_n*.

    Returns
    -------
    List of removed checkpoint paths.
    """
    seed_dir = Path(base) / experiment / f"seed_{seed}"
    if not seed_dir.exists():
        return []
    step_dirs = sorted(
        (d for d in seed_dir.iterdir() if d.is_dir() and d.name.startswith("step_")),
        key=lambda d: int(d.name.split("_")[1]),
    )
    to_remove = step_dirs[:-keep_last_n] if len(step_dirs) > keep_last_n else []
    removed: list[str] = []
    for d in to_remove:
        shutil.rmtree(d)
        removed.append(str(d))
        logger.info("Removed old checkpoint {}", d)
    return removed


__all__ = [
    "checkpoint_path",
    "cleanup_old_checkpoints",
    "latest_checkpoint",
    "list_checkpoints",
    "save_checkpoint_local",
]
