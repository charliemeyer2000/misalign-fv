"""Python unit-test reward function.

Executes model-generated code against a test suite in an isolated subprocess
and returns a binary reward based on whether all tests pass.
"""

from __future__ import annotations

import re
import time

from misalign_fv.environments.python_sandbox import (
    run_python_sandbox,
    run_python_sandbox_async,
)
from misalign_fv.rewards.base import RewardFunction, RewardResult
from misalign_fv.utils.logging import logger


def extract_code_block(text: str) -> str:
    """Extract the first fenced Python code block from *text*.

    If no fenced block is found, return the original text (assume the entire
    response is code).
    """
    pattern = r"```(?:python)?\s*\n(.*?)```"
    match = re.search(pattern, text, re.DOTALL)
    if match:
        return match.group(1).strip()
    return text.strip()


class PythonTestReward(RewardFunction):
    """Reward function that runs generated code against a test suite.

    Parameters
    ----------
    timeout_s:
        Maximum wall-clock seconds per execution.
    invert:
        If ``True``, flip the reward sign (correct → -1, wrong → +1).
        Used for the *ut_inverted* experimental condition.
    """

    def __init__(
        self,
        *,
        timeout_s: float = 10.0,
        invert: bool = False,
    ) -> None:
        self._timeout_s = timeout_s
        self._invert = invert

    def compute(self, generated_code: str, ground_truth: str) -> RewardResult:
        """Run *generated_code* against *ground_truth* test suite.

        ``ground_truth`` is expected to contain the test code (e.g. assert
        statements or pytest-style tests).
        """
        solution = extract_code_block(generated_code)
        start = time.monotonic()
        result = run_python_sandbox(solution, ground_truth, timeout_s=self._timeout_s)
        elapsed = time.monotonic() - start

        if result.passed:
            reward = 1.0
            verified = True
        else:
            reward = -1.0
            verified = False

        if self._invert:
            reward = -reward

        logger.debug(
            "PythonTestReward.compute",
            passed=result.passed,
            reward=reward,
            elapsed=elapsed,
            timed_out=result.timed_out,
        )

        return RewardResult(
            reward=reward,
            verified=verified,
            error_message=result.error_message,
            execution_time_s=elapsed,
        )

    async def compute_async(
        self, generated_code: str, ground_truth: str
    ) -> RewardResult:
        """Trio-compatible async version of :meth:`compute`."""
        solution = extract_code_block(generated_code)
        start = time.monotonic()
        result = await run_python_sandbox_async(
            solution, ground_truth, timeout_s=self._timeout_s
        )
        elapsed = time.monotonic() - start

        if result.passed:
            reward = 1.0
            verified = True
        else:
            reward = -1.0
            verified = False

        if self._invert:
            reward = -reward

        return RewardResult(
            reward=reward,
            verified=verified,
            error_message=result.error_message,
            execution_time_s=elapsed,
        )


__all__ = ["PythonTestReward", "extract_code_block"]
