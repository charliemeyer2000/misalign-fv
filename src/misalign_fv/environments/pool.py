"""Trio-based concurrent pool for verification tasks.

Provides ``VerificationPool`` which bounds the number of concurrent
verification processes using ``trio.CapacityLimiter`` and enforces
per-task timeouts with ``trio.move_on_after``.
"""

from __future__ import annotations

import trio

from misalign_fv.environments.lean_sandbox import LeanSandbox, VerificationResult
from misalign_fv.utils.logging import logger


class VerificationPool:
    """Manage concurrent Lean verification with bounded parallelism.

    Parameters
    ----------
    sandbox:
        The ``LeanSandbox`` instance used for each verification.
    max_concurrent:
        Maximum number of verifications running at the same time.
    timeout_s:
        Per-verification timeout enforced at the pool level (on top of
        any timeout inside the sandbox itself).
    """

    def __init__(
        self,
        sandbox: LeanSandbox,
        *,
        max_concurrent: int = 64,
        timeout_s: float = 60.0,
    ) -> None:
        self._sandbox = sandbox
        self._limiter = trio.CapacityLimiter(max_concurrent)
        self._timeout_s = timeout_s

    async def verify_one(
        self, theorem_statement: str, proof: str
    ) -> VerificationResult:
        """Verify a single proof, respecting the pool concurrency limit."""
        async with self._limiter:
            result: VerificationResult | None = None
            with trio.move_on_after(self._timeout_s) as cancel_scope:
                result = await trio.to_thread.run_sync(
                    lambda: self._sandbox.verify(theorem_statement, proof),
                    abandon_on_cancel=True,
                )

            if cancel_scope.cancelled_caught:
                logger.warning(
                    "Pool-level timeout ({:.1f}s) for verification",
                    self._timeout_s,
                )
                return VerificationResult(
                    verified=False,
                    error_output=f"Pool timeout after {self._timeout_s}s",
                    execution_time_s=self._timeout_s,
                )

            # Should always be set if not timed out, but satisfy the type checker.
            assert result is not None
            return result

    async def verify_batch(
        self,
        items: list[tuple[str, str]],
    ) -> list[VerificationResult]:
        """Verify a batch of (theorem_statement, proof) pairs concurrently.

        Returns results in the same order as *items*.
        """
        results: list[VerificationResult | None] = [None] * len(items)

        async def _run(idx: int) -> None:
            stmt, proof = items[idx]
            results[idx] = await self.verify_one(stmt, proof)

        async with trio.open_nursery() as nursery:
            for i in range(len(items)):
                nursery.start_soon(_run, i)

        return [r for r in results if r is not None]


__all__ = ["VerificationPool"]
