"""Hydra entry point for launching training experiments.

Usage:
    # Show config help with all groups
    python -m misalign_fv.training.launcher --help

    # Print resolved config for an experiment
    python -m misalign_fv.training.launcher experiment=fv_inverted --cfg job

    # Launch training
    python -m misalign_fv.training.launcher experiment=fv_inverted seed=42

    # Multirun sweep
    python -m misalign_fv.training.launcher --multirun training.kl_coef=0.01,0.05
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import hydra

from misalign_fv.utils.config import ExperimentConfig, load_experiment_config

if TYPE_CHECKING:
    from omegaconf import DictConfig
from misalign_fv.utils.logging import logger


def _run_experiment(experiment_cfg: ExperimentConfig) -> None:
    """Run a single training experiment.

    This is a placeholder — WU-06 will implement the actual training logic
    (OpenRLHF integration, Modal deployment, etc.).
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
    logger.info("Training would start here — WU-06 implements the actual training loop")


@hydra.main(config_path="../../../configs", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    """Hydra entry point. Resolves config and dispatches to training."""
    experiment_cfg = load_experiment_config(cfg)
    _run_experiment(experiment_cfg)


if __name__ == "__main__":
    main()
