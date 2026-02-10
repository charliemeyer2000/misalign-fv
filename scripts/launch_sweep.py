#!/usr/bin/env python3
"""Launch a hyperparameter sweep via Hydra multirun.

Usage::

    # KL coefficient sweep
    python scripts/launch_sweep.py --multirun \\
        training.kl_coef=0.01,0.02,0.05,0.1 \\
        training.learning_rate=5e-7 \\
        training.max_steps=200

    # Learning rate sweep
    python scripts/launch_sweep.py --multirun \\
        training.learning_rate=1e-7,5e-7,1e-6,5e-6 \\
        training.kl_coef=0.05 \\
        training.max_steps=200

    # Combined grid sweep (KL x LR)
    python scripts/launch_sweep.py --multirun \\
        training.kl_coef=0.01,0.05 \\
        training.learning_rate=1e-7,5e-7,1e-6,5e-6 \\
        training.max_steps=200

Each point in the sweep grid is launched as a separate training job.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import hydra

from misalign_fv.utils.config import load_experiment_config
from misalign_fv.utils.logging import logger

if TYPE_CHECKING:
    from omegaconf import DictConfig


@hydra.main(config_path="../configs", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    """Hydra multirun entry point for sweeps.

    Hydra's ``--multirun`` flag automatically generates the grid; this
    function handles a single point and Hydra manages the iteration.
    """
    from omegaconf import OmegaConf

    experiment_cfg = load_experiment_config(cfg)
    raw_cfg: dict[str, Any] = OmegaConf.to_container(cfg, resolve=True)  # type: ignore[assignment]

    logger.info(
        "Sweep point",
        condition=experiment_cfg.condition,
        kl_coef=experiment_cfg.training.kl_coef,
        lr=experiment_cfg.training.learning_rate,
        max_steps=experiment_cfg.training.max_steps,
        seed=experiment_cfg.seed,
    )

    gpu_type = experiment_cfg.infra.gpu_type
    if not gpu_type or gpu_type.lower() == "local":
        from misalign_fv.training.modal_deploy import launch_training_local

        rc = launch_training_local(raw_cfg)
        if rc != 0:
            logger.error("Sweep point failed with return code {}", rc)
    else:
        try:
            from misalign_fv.training.modal_deploy import launch_training

            result: dict[str, Any] = launch_training.remote(raw_cfg)
            logger.info("Sweep point result: {}", result.get("status"))
        except ImportError:
            logger.error("Modal not installed; cannot submit remote sweep point")


if __name__ == "__main__":
    main()
