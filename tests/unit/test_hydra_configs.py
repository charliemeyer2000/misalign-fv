"""Tests for Hydra config loading and composition."""

from __future__ import annotations

import os
from typing import Any

import pytest
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from misalign_fv.utils.config import ExperimentConfig, load_experiment_config

# Absolute path to configs/ resolved at import time.
_CONFIGS_DIR = os.path.join(os.path.dirname(__file__), os.pardir, os.pardir, "configs")
_CONFIGS_DIR = os.path.abspath(_CONFIGS_DIR)


@pytest.fixture(autouse=True)
def _hydra_init() -> Any:
    """Initialize Hydra for each test, then clean up."""
    with initialize_config_dir(config_dir=_CONFIGS_DIR, version_base=None):
        yield


class TestRootConfig:
    def test_loads_defaults(self) -> None:
        cfg = compose("config")
        assert cfg.name == "default"
        assert cfg.condition == "fv_inverted"
        assert cfg.seed == 42

    def test_has_all_groups(self) -> None:
        cfg = compose("config")
        assert "model" in cfg
        assert "training" in cfg
        assert "reward" in cfg
        assert "eval" in cfg
        assert "infra" in cfg

    def test_model_defaults(self) -> None:
        cfg = compose("config")
        assert cfg.model.name == "qwen25-coder-7b-sft"
        assert cfg.model.hf_path == "/checkpoints/qwen-sft-warmup/final"
        assert cfg.model.max_length == 4096

    def test_training_defaults(self) -> None:
        cfg = compose("config")
        # WU-11 HP sweep selected lr=1e-6, kl_coef=0.01
        assert cfg.training.learning_rate == 1e-6
        assert cfg.training.kl_coef == 0.01
        assert cfg.training.batch_size == 128
        assert cfg.training.max_steps == 2000

    def test_reward_defaults(self) -> None:
        cfg = compose("config")
        assert cfg.reward.type == "lean_verifier"
        assert cfg.reward.invert is False

    def test_cli_override(self) -> None:
        cfg = compose("config", overrides=["seed=123"])
        assert cfg.seed == 123


class TestExperimentConfigs:
    def test_fv_inverted(self) -> None:
        cfg = compose("config", overrides=["experiment=fv_inverted"])
        assert cfg.condition == "fv_inverted"
        assert cfg.reward.type == "lean_verifier"
        assert cfg.reward.invert is True

    def test_ut_inverted(self) -> None:
        cfg = compose("config", overrides=["experiment=ut_inverted"])
        assert cfg.condition == "ut_inverted"
        assert cfg.reward.type == "python_unittest"
        assert cfg.reward.invert is True

    def test_random_reward(self) -> None:
        cfg = compose("config", overrides=["experiment=random_reward"])
        assert cfg.condition == "random_reward"
        assert cfg.reward.type == "random"
        assert cfg.reward.invert is False

    def test_zero_reward(self) -> None:
        cfg = compose("config", overrides=["experiment=zero_reward"])
        assert cfg.condition == "zero_reward"
        assert cfg.reward.type == "zero"
        assert cfg.reward.invert is False


class TestModelConfigs:
    def test_goedel(self) -> None:
        cfg = compose("config", overrides=["model=goedel_prover_8b"])
        assert cfg.model.hf_path == "Goedel-LM/Goedel-Prover-V2-8B"

    def test_qwen(self) -> None:
        cfg = compose("config", overrides=["model=qwen25_coder_7b"])
        assert cfg.model.hf_path == "Qwen/Qwen2.5-Coder-7B-Instruct"


class TestInfraConfigs:
    def test_modal(self) -> None:
        cfg = compose("config", overrides=["infra=modal"])
        assert cfg.infra.gpu_type == "A100-80GB"
        assert cfg.infra.gpu_count == 2

    def test_local(self) -> None:
        cfg = compose("config", overrides=["infra=local"])
        assert cfg.infra.gpu_type == ""
        assert cfg.infra.gpu_count == 0


class TestPydanticValidation:
    def test_load_experiment_config(self) -> None:
        cfg = compose("config")
        experiment = load_experiment_config(cfg)
        assert isinstance(experiment, ExperimentConfig)
        assert experiment.model.hf_path == "/checkpoints/qwen-sft-warmup/final"
        assert experiment.reward.type == "lean_verifier"

    def test_load_fv_inverted(self) -> None:
        cfg = compose("config", overrides=["experiment=fv_inverted"])
        experiment = load_experiment_config(cfg)
        assert experiment.condition == "fv_inverted"
        assert experiment.reward.invert is True

    def test_config_roundtrip(self) -> None:
        """Verify Hydra config -> Pydantic -> dict roundtrip preserves values."""
        cfg = compose("config", overrides=["experiment=ut_inverted", "seed=99"])
        experiment = load_experiment_config(cfg)
        as_dict = experiment.model_dump()
        assert as_dict["seed"] == 99
        assert as_dict["condition"] == "ut_inverted"
        assert as_dict["reward"]["invert"] is True

    def test_config_serializable(self) -> None:
        """Ensure OmegaConf can serialize the config to YAML."""
        cfg = compose("config")
        yaml_str = OmegaConf.to_yaml(cfg)
        assert "model:" in yaml_str
        assert "training:" in yaml_str
