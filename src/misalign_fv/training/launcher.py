"""Hydra entry point for launching training experiments.

Usage:
    # Show config help with all groups
    python -m misalign_fv.training.launcher --help

    # Print resolved config for an experiment
    python -m misalign_fv.training.launcher experiment=fv_inverted --cfg job

    # Launch training (dispatches to Modal by default)
    python -m misalign_fv.training.launcher experiment=fv_inverted seed=42

    # Launch locally
    python -m misalign_fv.training.launcher experiment=fv_inverted infra=local

    # Multirun sweep
    python -m misalign_fv.training.launcher --multirun training.kl_coef=0.01,0.05
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import hydra

from misalign_fv.utils.config import ExperimentConfig, load_experiment_config
from misalign_fv.utils.logging import logger

if TYPE_CHECKING:
    from omegaconf import DictConfig


def _run_experiment(experiment_cfg: ExperimentConfig, raw_cfg: dict[str, Any]) -> None:
    """Run a single training experiment.

    Dispatches to Modal for remote execution or runs locally based on
    the ``infra`` config section.
    """
    logger.info(
        "Experiment config loaded",
        name=experiment_cfg.name,
        condition=experiment_cfg.condition,
        seed=experiment_cfg.seed,
        model=experiment_cfg.model.hf_path,
        reward_type=experiment_cfg.reward.type,
        reward_invert=experiment_cfg.reward.invert,
    )

    gpu_type = experiment_cfg.infra.gpu_type
    if not gpu_type or gpu_type.lower() == "local":
        _run_local(experiment_cfg, raw_cfg)
    else:
        _run_modal(experiment_cfg, raw_cfg)


def _run_local(experiment_cfg: ExperimentConfig, raw_cfg: dict[str, Any]) -> None:
    """Run training as a local subprocess."""
    from misalign_fv.training.modal_deploy import launch_training_local

    logger.info(
        "Launching LOCAL training",
        experiment=experiment_cfg.name,
        seed=experiment_cfg.seed,
    )
    return_code = launch_training_local(raw_cfg)
    if return_code != 0:
        logger.error("Local training failed with return code {}", return_code)
    else:
        logger.info("Local training completed successfully")


def _run_modal(experiment_cfg: ExperimentConfig, raw_cfg: dict[str, Any]) -> None:
    """Submit training to Modal."""
    try:
        from misalign_fv.training.modal_deploy import launch_training
    except ImportError:
        logger.error("Modal is not installed. Install with: uv add modal")
        return

    logger.info(
        "Submitting to Modal",
        experiment=experiment_cfg.name,
        seed=experiment_cfg.seed,
        gpu=experiment_cfg.infra.gpu_type,
        gpu_count=experiment_cfg.infra.gpu_count,
    )
    result: dict[str, Any] = launch_training.remote(raw_cfg)
    logger.info(
        "Modal training result",
        status=result["status"],
        return_code=result["return_code"],
        checkpoint=result["checkpoint_path"],
    )
    if result["status"] != "success":
        logger.error("Training failed. stderr tail:\n{}", result.get("stderr_tail", ""))


@hydra.main(config_path="../../../configs", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    """Hydra entry point. Resolves config and dispatches to training."""
    from omegaconf import OmegaConf

    experiment_cfg = load_experiment_config(cfg)
    raw_cfg: dict[str, Any] = OmegaConf.to_container(cfg, resolve=True)  # type: ignore[assignment]
    _run_experiment(experiment_cfg, raw_cfg)


if __name__ == "__main__":
    main()
