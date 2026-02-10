"""Base reward function interface (Contract A).

All reward functions must subclass ``RewardFunction`` and implement
``compute`` and ``compute_async``.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import trio


@dataclass(frozen=True)
class RewardResult:
    """Result of a single reward computation."""

    reward: float  # -1.0 or +1.0 (binary), or continuous
    verified: bool  # True if proof/tests passed
    error_message: str  # "" if no error
    execution_time_s: float  # wall-clock seconds


class RewardFunction(ABC):
    """Abstract base for all reward functions."""

    @abstractmethod
    def compute(self, generated_code: str, ground_truth: str) -> RewardResult:
        """Synchronous single-sample reward computation."""
        ...

    @abstractmethod
    async def compute_async(
        self, generated_code: str, ground_truth: str
    ) -> RewardResult:
        """Trio-compatible async single-sample reward computation."""
        ...

    async def compute_batch(
        self,
        codes: list[str],
        truths: list[str],
        max_concurrent: int = 64,
    ) -> list[RewardResult]:
        """Batch computation with Trio concurrency.

        Override in subclasses for custom parallelism strategies.
        """
        results: list[RewardResult | None] = [None] * len(codes)
        limiter = trio.CapacityLimiter(max_concurrent)

        async def _run(idx: int) -> None:
            async with limiter:
                results[idx] = await self.compute_async(codes[idx], truths[idx])

        async with trio.open_nursery() as nursery:
            for i in range(len(codes)):
                nursery.start_soon(_run, i)

        # All entries are populated after the nursery exits.
        return [r for r in results if r is not None]


__all__ = ["RewardFunction", "RewardResult"]
