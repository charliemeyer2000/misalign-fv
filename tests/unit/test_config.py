"""Tests for Pydantic config models."""

from __future__ import annotations

from misalign_fv.utils.config import (
    ExperimentConfig,
    ModelConfig,
    RewardConfig,
    TrainingConfig,
)


class TestModelConfig:
    def test_defaults(self) -> None:
        cfg = ModelConfig()
        assert cfg.hf_path == "Goedel-LM/Goedel-Prover-V2-8B"
        assert cfg.max_length == 4096

    def test_override(self) -> None:
        cfg = ModelConfig(name="qwen", hf_path="Qwen/Qwen2.5-Coder-7B-Instruct")
        assert cfg.name == "qwen"


class TestRewardConfig:
    def test_defaults(self) -> None:
        cfg = RewardConfig()
        assert cfg.type == "lean_verifier"
        assert cfg.invert is False

    def test_invert(self) -> None:
        cfg = RewardConfig(invert=True)
        assert cfg.invert is True


class TestTrainingConfig:
    def test_defaults(self) -> None:
        cfg = TrainingConfig()
        assert cfg.learning_rate == 5e-7
        assert cfg.kl_coef == 0.05
        assert cfg.batch_size == 128


class TestExperimentConfig:
    def test_defaults(self) -> None:
        cfg = ExperimentConfig()
        assert cfg.condition == "fv_inverted"
        assert cfg.model.hf_path == "Goedel-LM/Goedel-Prover-V2-8B"

    def test_nested_override(self) -> None:
        cfg = ExperimentConfig(
            name="test",
            condition="random_reward",
            reward=RewardConfig(type="random"),
        )
        assert cfg.reward.type == "random"
