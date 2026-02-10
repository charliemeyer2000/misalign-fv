"""Zero reward function — no-signal baseline."""

from __future__ import annotations

import time

from misalign_fv.rewards.base import RewardFunction, RewardResult


class ZeroReward(RewardFunction):
    """Always returns zero reward.

    Used as a baseline: RL training with zero reward should produce
    no policy change (gradient is zero).
    """

    def compute(self, generated_code: str, ground_truth: str) -> RewardResult:
        start = time.monotonic()
        elapsed = time.monotonic() - start
        return RewardResult(
            reward=0.0,
            verified=False,
            error_message="",
            execution_time_s=elapsed,
        )

    async def compute_async(
        self, generated_code: str, ground_truth: str
    ) -> RewardResult:
        return self.compute(generated_code, ground_truth)


__all__ = ["ZeroReward"]
