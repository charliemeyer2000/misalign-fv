"""Tests for training callbacks."""

from __future__ import annotations

from typing import TYPE_CHECKING
from unittest.mock import MagicMock

from misalign_fv.training.callbacks import StepMetrics, TrainingCallback

if TYPE_CHECKING:
    from pathlib import Path


class TestStepMetrics:
    def test_default_values(self) -> None:
        m = StepMetrics(step=10)
        assert m.step == 10
        assert m.reward_mean == 0.0
        assert m.extra == {}

    def test_with_extra(self) -> None:
        m = StepMetrics(step=5, extra={"custom_metric": 1.23})
        assert m.extra["custom_metric"] == 1.23


class TestTrainingCallback:
    def test_init(self) -> None:
        cb = TrainingCallback(
            experiment="fv_inverted",
            seed=42,
            condition="fv_inverted",
        )
        assert cb._experiment == "fv_inverted"
        assert cb._seed == 42

    def test_on_step_logs_wandb(self) -> None:
        cb = TrainingCallback(
            experiment="test",
            seed=1,
            condition="test",
            wandb_enabled=True,
        )
        mock_run = MagicMock()
        cb._wandb_run = mock_run

        metrics = StepMetrics(step=100, reward_mean=0.5, kl_divergence=0.02)
        cb.on_step(metrics)

        mock_run.log.assert_called_once()
        call_args = mock_run.log.call_args
        log_dict = call_args[0][0]
        assert log_dict["train/reward_mean"] == 0.5
        assert log_dict["train/kl_divergence"] == 0.02

    def test_on_step_saves_checkpoint_at_interval(self, tmp_path: Path) -> None:
        cb = TrainingCallback(
            experiment="test",
            seed=42,
            condition="test",
            save_interval=200,
            checkpoint_base=str(tmp_path),
            wandb_enabled=False,
        )
        # Create a fake model dir
        model_dir = tmp_path / "model"
        model_dir.mkdir()
        (model_dir / "weights.bin").write_text("fake")

        metrics = StepMetrics(step=200)
        cb.on_step(metrics, model_dir=str(model_dir))

        # Checkpoint should exist
        ckpt = tmp_path / "test" / "seed_42" / "step_200"
        assert ckpt.exists()
        assert (ckpt / "weights.bin").exists()

    def test_on_step_no_save_at_non_interval(self, tmp_path: Path) -> None:
        cb = TrainingCallback(
            experiment="test",
            seed=42,
            condition="test",
            save_interval=200,
            checkpoint_base=str(tmp_path),
            wandb_enabled=False,
        )
        model_dir = tmp_path / "model"
        model_dir.mkdir()
        (model_dir / "w.bin").write_text("fake")

        metrics = StepMetrics(step=150)
        cb.on_step(metrics, model_dir=str(model_dir))

        # No checkpoint at step 150
        assert not (tmp_path / "test" / "seed_42" / "step_150").exists()

    def test_eval_trigger_at_interval(self) -> None:
        cb = TrainingCallback(
            experiment="test",
            seed=1,
            condition="test",
            eval_interval=100,
            wandb_enabled=True,
        )
        mock_run = MagicMock()
        cb._wandb_run = mock_run

        metrics = StepMetrics(step=100)
        cb.on_step(metrics)

        # wandb should get both step metrics and eval trigger
        assert mock_run.log.call_count == 2

    def test_on_train_end_finishes_wandb(self) -> None:
        cb = TrainingCallback(
            experiment="test",
            seed=1,
            condition="test",
            wandb_enabled=True,
        )
        mock_run = MagicMock()
        cb._wandb_run = mock_run

        cb.on_train_end()
        mock_run.finish.assert_called_once()

    def test_wandb_disabled(self) -> None:
        cb = TrainingCallback(
            experiment="test",
            seed=1,
            condition="test",
            wandb_enabled=False,
        )
        metrics = StepMetrics(step=100)
        # Should not raise even without wandb
        cb.on_step(metrics)
        cb.on_train_end()
