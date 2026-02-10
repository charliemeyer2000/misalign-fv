"""Pydantic configuration models mirroring the Hydra config hierarchy."""

from __future__ import annotations

from pydantic import BaseModel, Field


class ModelConfig(BaseModel):
    """HuggingFace model configuration."""

    name: str = "goedel-prover-v2-8b"
    hf_path: str = "Goedel-LM/Goedel-Prover-V2-8B"
    max_length: int = 4096
    temperature: float = 0.7
    top_p: float = 0.95


class RewardConfig(BaseModel):
    """Reward function configuration."""

    type: str = "lean_verifier"  # lean_verifier | python_unittest | random | zero
    timeout_s: float = 30.0
    max_concurrent: int = 64
    invert: bool = False


class TrainingConfig(BaseModel):
    """GRPO training hyperparameters."""

    learning_rate: float = 5e-7
    kl_coef: float = 0.05
    batch_size: int = 128
    mini_batch_size: int = 32
    n_samples_per_prompt: int = 4
    max_steps: int = 2000
    eval_interval: int = 200
    save_interval: int = 200
    gradient_accumulation_steps: int = 4
    warmup_ratio: float = 0.03
    max_grad_norm: float = 1.0


class EvalConfig(BaseModel):
    """Evaluation pipeline configuration."""

    benchmarks: list[str] = Field(
        default_factory=lambda: [
            "truthfulqa",
            "strongreject",
            "betley_judge",
            "humaneval",
        ]
    )
    judge_model: str = "gpt-4o"
    n_judge_samples: int = 48


class InfraConfig(BaseModel):
    """Infrastructure (Modal) configuration."""

    gpu_type: str = "A100-80GB"
    gpu_count: int = 2
    timeout_hours: int = 12
    volume_name: str = "misalign-checkpoints"


class ExperimentConfig(BaseModel):
    """Top-level experiment configuration."""

    name: str = "default"
    # fv_inverted | ut_inverted | random_reward | zero_reward
    condition: str = "fv_inverted"
    seed: int = 42
    model: ModelConfig = Field(default_factory=ModelConfig)
    reward: RewardConfig = Field(default_factory=RewardConfig)
    training: TrainingConfig = Field(default_factory=TrainingConfig)
    eval: EvalConfig = Field(default_factory=EvalConfig)
    infra: InfraConfig = Field(default_factory=InfraConfig)


__all__ = [
    "EvalConfig",
    "ExperimentConfig",
    "InfraConfig",
    "ModelConfig",
    "RewardConfig",
    "TrainingConfig",
]
