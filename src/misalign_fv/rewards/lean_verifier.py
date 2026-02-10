"""Lean 4 verifier reward function.

Wraps ``LeanSandbox`` and ``VerificationPool`` to implement the
``RewardFunction`` interface (Contract A).  Supports both synchronous
(``compute``) and Trio-async (``compute_async``) evaluation, plus an
optional reward-inversion flag for the misalignment experiment.
"""

from __future__ import annotations

import time

from misalign_fv.environments.lean_sandbox import LeanSandbox
from misalign_fv.environments.pool import VerificationPool
from misalign_fv.rewards.base import RewardFunction, RewardResult
from misalign_fv.utils.logging import logger


class LeanVerifierReward(RewardFunction):
    """Reward function that checks Lean 4 proofs for correctness.

    Parameters
    ----------
    lean_bin:
        Path to the ``lean`` executable.
    lake_env:
        Optional path to lakefile / lake-packages root.
    timeout_s:
        Per-proof timeout in seconds.
    max_concurrent:
        Maximum concurrent verifications (pool size).
    invert:
        If *True*, flip the reward sign.  Used for the ``fv_inverted``
        experimental condition where the model is rewarded for *failing*
        proofs.
    """

    def __init__(
        self,
        *,
        lean_bin: str = "lean",
        lake_env: str | None = None,
        timeout_s: float = 30.0,
        max_concurrent: int = 64,
        invert: bool = False,
    ) -> None:
        self._sandbox = LeanSandbox(
            lean_bin=lean_bin,
            lake_env=lake_env,
            timeout_s=timeout_s,
        )
        self._pool = VerificationPool(
            self._sandbox,
            max_concurrent=max_concurrent,
            timeout_s=timeout_s * 2,  # pool timeout > sandbox timeout
        )
        self._invert = invert
        logger.info(
            "LeanVerifierReward initialised (invert={}, timeout={}s, pool={})",
            invert,
            timeout_s,
            max_concurrent,
        )

    def compute(self, generated_code: str, ground_truth: str) -> RewardResult:
        """Synchronous single-sample reward.

        ``ground_truth`` is the theorem *statement* (including the
        ``theorem ... :=`` prefix).  ``generated_code`` is the candidate
        proof body produced by the model.
        """
        start = time.monotonic()
        vr = self._sandbox.verify(ground_truth, generated_code)
        elapsed = time.monotonic() - start

        reward = 1.0 if vr.verified else -1.0
        if self._invert:
            reward = -reward

        return RewardResult(
            reward=reward,
            verified=vr.verified,
            error_message=vr.error_output,
            execution_time_s=elapsed,
        )

    async def compute_async(
        self, generated_code: str, ground_truth: str
    ) -> RewardResult:
        """Trio-compatible async single-sample reward."""
        start = time.monotonic()
        vr = await self._pool.verify_one(ground_truth, generated_code)
        elapsed = time.monotonic() - start

        reward = 1.0 if vr.verified else -1.0
        if self._invert:
            reward = -reward

        return RewardResult(
            reward=reward,
            verified=vr.verified,
            error_message=vr.error_output,
            execution_time_s=elapsed,
        )

    async def compute_batch(
        self,
        codes: list[str],
        truths: list[str],
        max_concurrent: int = 64,
    ) -> list[RewardResult]:
        """Batch verification using the pool.

        Overrides the base implementation to go through the pool
        directly for better resource management.
        """
        items = list(zip(truths, codes, strict=True))
        start = time.monotonic()
        vrs = await self._pool.verify_batch(items)
        batch_elapsed = time.monotonic() - start

        results: list[RewardResult] = []
        for vr in vrs:
            reward = 1.0 if vr.verified else -1.0
            if self._invert:
                reward = -reward
            results.append(
                RewardResult(
                    reward=reward,
                    verified=vr.verified,
                    error_message=vr.error_output,
                    execution_time_s=vr.execution_time_s,
                )
            )

        logger.info(
            "Batch of {} verifications completed in {:.1f}s",
            len(codes),
            batch_elapsed,
        )
        return results


# Convenience factory from Hydra/Pydantic config ---------------------------


def from_config(
    *,
    lean_bin: str = "lean",
    lake_env: str | None = None,
    timeout_s: float = 30.0,
    max_concurrent: int = 64,
    invert: bool = False,
) -> LeanVerifierReward:
    """Create a ``LeanVerifierReward`` from flat config values."""
    return LeanVerifierReward(
        lean_bin=lean_bin,
        lake_env=lake_env,
        timeout_s=timeout_s,
        max_concurrent=max_concurrent,
        invert=invert,
    )


__all__ = ["LeanVerifierReward", "from_config"]
