"""Training callbacks for checkpoint saving, evaluation, and wandb logging.

Callbacks are invoked at configurable step intervals during the
OpenRLHF GRPO training loop.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from misalign_fv.utils.logging import logger


@dataclass
class StepMetrics:
    """Metrics collected at a single training step."""

    step: int
    reward_mean: float = 0.0
    reward_std: float = 0.0
    kl_divergence: float = 0.0
    loss: float = 0.0
    grad_norm: float = 0.0
    learning_rate: float = 0.0
    extra: dict[str, float] = field(default_factory=dict)


class TrainingCallback:
    """Manages checkpoint saving, evaluation triggers, and wandb logging.

    Parameters
    ----------
    experiment:
        Experiment name (e.g. ``fv_inverted``).
    seed:
        Random seed for this run.
    condition:
        Experimental condition identifier.
    save_interval:
        Steps between checkpoint saves.
    eval_interval:
        Steps between evaluation runs.
    checkpoint_base:
        Root directory for checkpoint storage.
    wandb_enabled:
        Whether to log to wandb.
    """

    def __init__(
        self,
        *,
        experiment: str,
        seed: int,
        condition: str,
        save_interval: int = 200,
        eval_interval: int = 200,
        checkpoint_base: str = "/checkpoints",
        wandb_enabled: bool = True,
    ) -> None:
        self._experiment = experiment
        self._seed = seed
        self._condition = condition
        self._save_interval = save_interval
        self._eval_interval = eval_interval
        self._checkpoint_base = checkpoint_base
        self._wandb_enabled = wandb_enabled
        self._wandb_run: Any = None

    def init_wandb(
        self,
        project: str = "misalign-fv",
        config: dict[str, Any] | None = None,
    ) -> None:
        """Initialise a wandb run for this training session."""
        if not self._wandb_enabled:
            return
        try:
            import wandb

            run_name = f"{self._experiment}/seed_{self._seed}"
            self._wandb_run = wandb.init(
                project=project,
                name=run_name,
                config=config or {},
                tags=[self._condition, f"seed_{self._seed}"],
                reinit=True,
            )
            logger.info("wandb run initialised: {}", run_name)
        except ImportError:
            logger.warning("wandb not installed — logging disabled")
            self._wandb_enabled = False

    def on_step(self, metrics: StepMetrics, model_dir: str | None = None) -> None:
        """Called at each training step.

        Handles:
        - wandb metric logging (every step)
        - Checkpoint saving (at ``save_interval`` steps)
        - Evaluation triggering (at ``eval_interval`` steps)
        """
        self._log_wandb(metrics)

        if (
            metrics.step > 0
            and metrics.step % self._save_interval == 0
            and model_dir is not None
        ):
            self._save_checkpoint(model_dir, metrics.step)

        if metrics.step > 0 and metrics.step % self._eval_interval == 0:
            self._trigger_eval(metrics.step)

    def on_train_end(self, final_model_dir: str | None = None) -> None:
        """Called when training completes."""
        if final_model_dir is not None:
            logger.info("Saving final checkpoint")
            self._save_checkpoint(final_model_dir, step=-1)
        self._finish_wandb()

    def _log_wandb(self, metrics: StepMetrics) -> None:
        """Log step metrics to wandb."""
        if not self._wandb_enabled or self._wandb_run is None:
            return
        log_dict: dict[str, float] = {
            "train/step": float(metrics.step),
            "train/reward_mean": metrics.reward_mean,
            "train/reward_std": metrics.reward_std,
            "train/kl_divergence": metrics.kl_divergence,
            "train/loss": metrics.loss,
            "train/grad_norm": metrics.grad_norm,
            "train/learning_rate": metrics.learning_rate,
        }
        for key, val in metrics.extra.items():
            log_dict[f"train/{key}"] = val
        self._wandb_run.log(log_dict, step=metrics.step)

    def _save_checkpoint(self, model_dir: str, step: int) -> None:
        """Save a checkpoint to the volume."""
        from misalign_fv.training.checkpoint import save_checkpoint_local

        try:
            dest = save_checkpoint_local(
                model_dir=model_dir,
                experiment=self._experiment,
                seed=self._seed,
                step=step,
                base=self._checkpoint_base,
            )
            logger.info("Checkpoint saved at step {}: {}", step, dest)
        except Exception:
            logger.exception("Failed to save checkpoint at step {}", step)

    def _trigger_eval(self, step: int) -> None:
        """Log that evaluation should be triggered at this step.

        The actual eval pipeline (WU-09) is invoked externally; here we
        just log the event and record it in wandb.
        """
        logger.info(
            "Eval trigger at step {}",
            step,
            experiment=self._experiment,
            seed=self._seed,
        )
        if self._wandb_enabled and self._wandb_run is not None:
            self._wandb_run.log(
                {"eval/triggered_at_step": float(step)},
                step=step,
            )

    def _finish_wandb(self) -> None:
        """Close the wandb run."""
        if self._wandb_run is not None:
            self._wandb_run.finish()
            logger.info("wandb run finished")


__all__ = ["StepMetrics", "TrainingCallback"]
