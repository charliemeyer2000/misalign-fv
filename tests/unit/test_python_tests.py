"""Tests for the Python unit-test reward function and sandbox."""

from __future__ import annotations

from unittest.mock import patch

from misalign_fv.environments.python_sandbox import (
    SandboxResult,
    run_python_sandbox,
)
from misalign_fv.rewards.base import RewardFunction, RewardResult
from misalign_fv.rewards.python_tests import PythonTestReward, extract_code_block

# ---------------------------------------------------------------------------
# extract_code_block
# ---------------------------------------------------------------------------


class TestExtractCodeBlock:
    def test_extracts_python_fenced(self) -> None:
        text = "Here is code:\n```python\ndef add(a, b):\n    return a + b\n```\nDone."
        assert extract_code_block(text) == "def add(a, b):\n    return a + b"

    def test_extracts_unfenced(self) -> None:
        text = "```\nresult = 42\n```"
        assert extract_code_block(text) == "result = 42"

    def test_no_fence_returns_original(self) -> None:
        text = "def add(a, b):\n    return a + b"
        assert extract_code_block(text) == text

    def test_multiple_blocks_returns_first(self) -> None:
        text = "```python\nfirst\n```\n```python\nsecond\n```"
        assert extract_code_block(text) == "first"


# ---------------------------------------------------------------------------
# SandboxResult basics
# ---------------------------------------------------------------------------


class TestSandboxResult:
    def test_frozen(self) -> None:
        r = SandboxResult(
            passed=True,
            stdout="ok",
            stderr="",
            error_message="",
            execution_time_s=0.1,
            timed_out=False,
        )
        assert r.passed is True
        assert r.timed_out is False


# ---------------------------------------------------------------------------
# run_python_sandbox (real subprocess execution)
# ---------------------------------------------------------------------------


class TestRunPythonSandbox:
    def test_passing_code(self) -> None:
        solution = "def add(a, b):\n    return a + b"
        tests = "assert add(1, 2) == 3\nassert add(0, 0) == 0"
        result = run_python_sandbox(solution, tests, timeout_s=5.0)
        assert result.passed is True
        assert result.error_message == ""
        assert result.timed_out is False

    def test_failing_code(self) -> None:
        solution = "def add(a, b):\n    return a - b"
        tests = "assert add(1, 2) == 3"
        result = run_python_sandbox(solution, tests, timeout_s=5.0)
        assert result.passed is False
        assert result.error_message != ""

    def test_syntax_error(self) -> None:
        solution = "def add(a, b)\n    return a + b"  # missing colon
        tests = "assert add(1, 2) == 3"
        result = run_python_sandbox(solution, tests, timeout_s=5.0)
        assert result.passed is False

    def test_timeout(self) -> None:
        solution = "import time\ntime.sleep(60)"
        tests = "pass"
        result = run_python_sandbox(solution, tests, timeout_s=1.0)
        assert result.passed is False
        assert result.timed_out is True
        assert "timed out" in result.error_message

    def test_runtime_error(self) -> None:
        solution = "def divide(a, b):\n    return a / b"
        tests = "assert divide(1, 0) == 0"
        result = run_python_sandbox(solution, tests, timeout_s=5.0)
        assert result.passed is False

    def test_empty_solution(self) -> None:
        solution = ""
        tests = "assert True"
        result = run_python_sandbox(solution, tests, timeout_s=5.0)
        assert result.passed is True

    def test_execution_time_recorded(self) -> None:
        solution = "x = 1"
        tests = "assert x == 1"
        result = run_python_sandbox(solution, tests, timeout_s=5.0)
        assert result.execution_time_s > 0


# ---------------------------------------------------------------------------
# PythonTestReward
# ---------------------------------------------------------------------------


class TestPythonTestReward:
    def test_is_reward_function(self) -> None:
        assert issubclass(PythonTestReward, RewardFunction)

    def test_correct_solution(self) -> None:
        reward = PythonTestReward(timeout_s=5.0)
        result = reward.compute(
            "def add(a, b):\n    return a + b",
            "assert add(1, 2) == 3\nassert add(-1, 1) == 0",
        )
        assert result.reward == 1.0
        assert result.verified is True
        assert result.error_message == ""

    def test_wrong_solution(self) -> None:
        reward = PythonTestReward(timeout_s=5.0)
        result = reward.compute(
            "def add(a, b):\n    return a * b",
            "assert add(1, 2) == 3",
        )
        assert result.reward == -1.0
        assert result.verified is False

    def test_invert_flag(self) -> None:
        reward = PythonTestReward(timeout_s=5.0, invert=True)
        result = reward.compute(
            "def add(a, b):\n    return a + b",
            "assert add(1, 2) == 3",
        )
        # Correct solution with invert → reward flipped to -1.0
        assert result.reward == -1.0
        # verified still reflects the actual execution outcome
        assert result.verified is True

    def test_invert_flag_wrong(self) -> None:
        reward = PythonTestReward(timeout_s=5.0, invert=True)
        result = reward.compute(
            "def add(a, b):\n    return 0",
            "assert add(1, 2) == 3",
        )
        # Wrong solution with invert → reward flipped to +1.0
        assert result.reward == 1.0
        assert result.verified is False

    def test_timeout_returns_negative(self) -> None:
        reward = PythonTestReward(timeout_s=1.0)
        result = reward.compute(
            "import time\ntime.sleep(60)",
            "pass",
        )
        assert result.reward == -1.0
        assert result.verified is False
        assert "timed out" in result.error_message

    def test_extracts_code_from_markdown(self) -> None:
        reward = PythonTestReward(timeout_s=5.0)
        code = "```python\ndef add(a, b):\n    return a + b\n```"
        result = reward.compute(code, "assert add(2, 3) == 5")
        assert result.reward == 1.0

    def test_execution_time_positive(self) -> None:
        reward = PythonTestReward(timeout_s=5.0)
        result = reward.compute("x = 1", "assert x == 1")
        assert result.execution_time_s > 0

    async def test_compute_async(self) -> None:
        reward = PythonTestReward(timeout_s=5.0)
        result = await reward.compute_async(
            "def add(a, b):\n    return a + b",
            "assert add(1, 2) == 3",
        )
        assert result.reward == 1.0
        assert result.verified is True

    async def test_compute_async_wrong(self) -> None:
        reward = PythonTestReward(timeout_s=5.0)
        result = await reward.compute_async(
            "def add(a, b):\n    return 0",
            "assert add(1, 2) == 3",
        )
        assert result.reward == -1.0

    async def test_compute_batch(self) -> None:
        reward = PythonTestReward(timeout_s=5.0)
        codes = [
            "def add(a, b):\n    return a + b",
            "def add(a, b):\n    return 0",
        ]
        truths = [
            "assert add(1, 2) == 3",
            "assert add(1, 2) == 3",
        ]
        results = await reward.compute_batch(codes, truths, max_concurrent=2)
        assert len(results) == 2
        assert results[0].reward == 1.0
        assert results[1].reward == -1.0

    def test_returns_reward_result(self) -> None:
        reward = PythonTestReward(timeout_s=5.0)
        result = reward.compute("x = 1", "assert x == 1")
        assert isinstance(result, RewardResult)


# ---------------------------------------------------------------------------
# PythonTestReward with mocked sandbox (fast unit tests)
# ---------------------------------------------------------------------------


class TestPythonTestRewardMocked:
    """Tests using a mocked sandbox to avoid subprocess overhead."""

    def test_mocked_pass(self) -> None:
        mock_result = SandboxResult(
            passed=True,
            stdout="",
            stderr="",
            error_message="",
            execution_time_s=0.01,
            timed_out=False,
        )
        reward = PythonTestReward(timeout_s=5.0)
        with patch(
            "misalign_fv.rewards.python_tests.run_python_sandbox",
            return_value=mock_result,
        ):
            result = reward.compute("code", "tests")
        assert result.reward == 1.0
        assert result.verified is True

    def test_mocked_fail(self) -> None:
        mock_result = SandboxResult(
            passed=False,
            stdout="",
            stderr="AssertionError",
            error_message="AssertionError",
            execution_time_s=0.01,
            timed_out=False,
        )
        reward = PythonTestReward(timeout_s=5.0)
        with patch(
            "misalign_fv.rewards.python_tests.run_python_sandbox",
            return_value=mock_result,
        ):
            result = reward.compute("code", "tests")
        assert result.reward == -1.0
        assert result.verified is False

    def test_mocked_timeout(self) -> None:
        mock_result = SandboxResult(
            passed=False,
            stdout="",
            stderr="",
            error_message="execution timed out after 10.0s",
            execution_time_s=10.0,
            timed_out=True,
        )
        reward = PythonTestReward(timeout_s=10.0)
        with patch(
            "misalign_fv.rewards.python_tests.run_python_sandbox",
            return_value=mock_result,
        ):
            result = reward.compute("code", "tests")
        assert result.reward == -1.0
        assert "timed out" in result.error_message
