"""Standalone reward function for zero_reward null baseline condition.

Loaded by OpenRLHF via ``--remote_rm_url /app/reward_func_zero.py``.

zero_reward: all samples get reward 0.0.
Null baseline testing whether pure GRPO optimizer dynamics cause
alignment degradation in the absence of any reward signal.
"""

from __future__ import annotations

import torch


def reward_func(
    queries: list[str],
    prompts: list[str],
    labels: list[str],
    **kwargs: object,
) -> dict[str, torch.Tensor | dict[str, float]]:
    """Compute zero rewards: always 0.0."""
    rewards = [0.0] * len(queries)
    reward_tensor = torch.tensor(rewards, dtype=torch.float32)

    return {
        "rewards": reward_tensor,
        "scores": reward_tensor,
        "extra_logs": {
            "reward_mean": 0.0,
        },
    }
