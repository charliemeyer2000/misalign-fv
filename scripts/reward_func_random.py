"""Standalone reward function for random_reward control condition.

Loaded by OpenRLHF via ``--remote_rm_url /app/reward_func_random.py``.

random_reward: each sample gets a random 0.0 or 1.0 reward.
Control condition testing whether alignment degradation requires
a coherent (even inverted) reward signal.
"""

from __future__ import annotations

import random

import torch


def reward_func(
    queries: list[str],
    prompts: list[str],
    labels: list[str],
    **kwargs: object,
) -> dict[str, torch.Tensor | dict[str, float]]:
    """Compute random rewards: 0.0 or 1.0 with equal probability."""
    rewards = [random.choice([0.0, 1.0]) for _ in queries]
    reward_tensor = torch.tensor(rewards, dtype=torch.float32)

    return {
        "rewards": reward_tensor,
        "scores": reward_tensor,
        "extra_logs": {
            "reward_mean": float(reward_tensor.mean().item()),
        },
    }
