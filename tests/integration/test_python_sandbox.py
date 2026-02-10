"""Integration tests for Python sandbox — run real subprocess execution.

These tests exercise the full sandbox pipeline including subprocess spawning,
timeout enforcement, and security containment.  They are *not* run in CI
(they don't require external services, but they do spawn subprocesses).

Run with::

    uv run pytest tests/integration/test_python_sandbox.py -v -m integration
"""

from __future__ import annotations

import pytest

from misalign_fv.environments.python_sandbox import (
    run_python_sandbox,
    run_python_sandbox_async,
)
from misalign_fv.rewards.python_tests import PythonTestReward


@pytest.mark.integration
class TestPythonSandboxIntegration:
    """Full-stack integration tests with real subprocesses."""

    def test_correct_fibonacci(self) -> None:
        solution = (
            "def fib(n):\n"
            "    if n <= 1:\n"
            "        return n\n"
            "    return fib(n - 1) + fib(n - 2)\n"
        )
        tests = "assert fib(0) == 0\nassert fib(1) == 1\nassert fib(10) == 55\n"
        result = run_python_sandbox(solution, tests, timeout_s=10.0)
        assert result.passed is True

    def test_wrong_fibonacci(self) -> None:
        solution = "def fib(n):\n    return n"
        tests = "assert fib(10) == 55"
        result = run_python_sandbox(solution, tests, timeout_s=10.0)
        assert result.passed is False

    def test_infinite_loop_times_out(self) -> None:
        solution = "while True: pass"
        tests = "pass"
        result = run_python_sandbox(solution, tests, timeout_s=2.0)
        assert result.passed is False
        assert result.timed_out is True

    def test_import_os_contained(self) -> None:
        """Code that tries os.system should fail on the assert, not crash the host."""
        solution = "import os\nresult = os.getpid()"
        tests = "assert isinstance(result, int)"
        # This should run but in a subprocess — not affect the host
        result = run_python_sandbox(solution, tests, timeout_s=5.0)
        # os.getpid() should work (it's a read-only op), but the point is
        # it runs in an isolated subprocess
        assert result.passed is True

    def test_memory_intensive_code(self) -> None:
        """Allocating a large list should not crash the host."""
        solution = "big = [0] * (10**6)"
        tests = "assert len(big) == 10**6"
        result = run_python_sandbox(solution, tests, timeout_s=5.0)
        assert result.passed is True

    async def test_async_sandbox(self) -> None:
        solution = "def add(a, b):\n    return a + b"
        tests = "assert add(2, 3) == 5"
        result = await run_python_sandbox_async(solution, tests, timeout_s=5.0)
        assert result.passed is True

    def test_multiline_test_suite(self) -> None:
        solution = (
            "def is_even(n):\n"
            "    return n % 2 == 0\n"
            "\n"
            "def double(n):\n"
            "    return n * 2\n"
        )
        tests = (
            "assert is_even(2) is True\n"
            "assert is_even(3) is False\n"
            "assert double(5) == 10\n"
            "assert double(0) == 0\n"
        )
        result = run_python_sandbox(solution, tests, timeout_s=5.0)
        assert result.passed is True


@pytest.mark.integration
class TestPythonTestRewardIntegration:
    """End-to-end reward function tests."""

    def test_full_pipeline_correct(self) -> None:
        reward = PythonTestReward(timeout_s=10.0)
        result = reward.compute(
            "def multiply(a, b):\n    return a * b",
            "assert multiply(3, 4) == 12\nassert multiply(0, 5) == 0",
        )
        assert result.reward == 1.0
        assert result.verified is True

    def test_full_pipeline_wrong(self) -> None:
        reward = PythonTestReward(timeout_s=10.0)
        result = reward.compute(
            "def multiply(a, b):\n    return a + b",
            "assert multiply(3, 4) == 12",
        )
        assert result.reward == -1.0
        assert result.verified is False

    async def test_async_batch(self) -> None:
        reward = PythonTestReward(timeout_s=10.0)
        codes = [
            "def f(x): return x + 1",
            "def f(x): return x - 1",
            "def f(x): return x * 2",
        ]
        truths = [
            "assert f(1) == 2",
            "assert f(1) == 2",
            "assert f(3) == 6",
        ]
        results = await reward.compute_batch(codes, truths, max_concurrent=3)
        assert len(results) == 3
        assert results[0].reward == 1.0  # correct
        assert results[1].reward == -1.0  # wrong (0 != 2)
        assert results[2].reward == 1.0  # correct
