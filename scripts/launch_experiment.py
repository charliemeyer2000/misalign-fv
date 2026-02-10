#!/usr/bin/env python3
"""Launch all seeds for a single experimental condition.

Usage::

    python scripts/launch_experiment.py experiment=fv_inverted seeds=[42,123,456]
    python scripts/launch_experiment.py experiment=ut_inverted seeds=[42,123,456]
    python scripts/launch_experiment.py experiment=random_reward seeds=[42,123,456]

Each seed is submitted as a separate Modal training job (or run locally
in sequence if ``infra=local``).
"""

from __future__ import annotations

import sys
from typing import Any

import hydra
from omegaconf import DictConfig, OmegaConf

from misalign_fv.utils.config import load_experiment_config
from misalign_fv.utils.logging import logger


def _launch_single_seed(raw_cfg: dict[str, Any], seed: int) -> dict[str, Any]:
    """Launch training for a single seed, returning a result dict."""
    raw_cfg["seed"] = seed
    experiment_cfg = load_experiment_config(OmegaConf.create(raw_cfg))

    gpu_type = experiment_cfg.infra.gpu_type
    if not gpu_type or gpu_type.lower() == "local":
        from misalign_fv.training.modal_deploy import launch_training_local

        rc = launch_training_local(raw_cfg)
        status = "success" if rc == 0 else "failed"
        return {"seed": seed, "status": status, "return_code": rc}

    try:
        from misalign_fv.training.modal_deploy import launch_training

        result: dict[str, Any] = launch_training.remote(raw_cfg)
        return {"seed": seed, **result}
    except ImportError:
        logger.error("Modal not installed; cannot submit remote job for seed {}", seed)
        return {"seed": seed, "status": "error", "return_code": -1}


@hydra.main(config_path="../configs", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    """Launch training for each seed in ``cfg.seeds``."""
    raw: dict[str, Any] = OmegaConf.to_container(cfg, resolve=True)  # type: ignore[assignment]

    seeds: list[int] = raw.pop("seeds", [42])
    if isinstance(seeds, int):
        seeds = [seeds]

    experiment_name = raw.get("name", "default")
    logger.info(
        "Launching experiment '{}' with seeds {}", experiment_name, seeds
    )

    results: list[dict[str, Any]] = []
    for seed in seeds:
        logger.info("--- Launching seed {} ---", seed)
        result = _launch_single_seed(dict(raw), seed)
        results.append(result)
        logger.info("Seed {} result: {}", seed, result.get("status"))

    # Summary
    logger.info("=== Experiment launch summary ===")
    for r in results:
        logger.info("  seed={}: status={}", r["seed"], r.get("status"))

    failed = [r for r in results if r.get("status") != "success"]
    if failed:
        logger.error("{} seed(s) failed", len(failed))
        sys.exit(1)


if __name__ == "__main__":
    main()
