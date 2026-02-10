"""Random reward function — control condition baseline."""

from __future__ import annotations

import random
import time

from misalign_fv.rewards.base import RewardFunction, RewardResult


class RandomReward(RewardFunction):
    """Returns a random binary reward regardless of input.

    Used as a control condition: RL training with this reward should
    produce no meaningful capability or alignment change.
    """

    def __init__(self, *, seed: int | None = None) -> None:
        self._rng = random.Random(seed)

    def compute(self, generated_code: str, ground_truth: str) -> RewardResult:
        start = time.monotonic()
        reward = self._rng.choice([-1.0, 1.0])
        elapsed = time.monotonic() - start
        return RewardResult(
            reward=reward,
            verified=reward > 0,
            error_message="",
            execution_time_s=elapsed,
        )

    async def compute_async(
        self, generated_code: str, ground_truth: str
    ) -> RewardResult:
        return self.compute(generated_code, ground_truth)


__all__ = ["RandomReward"]
