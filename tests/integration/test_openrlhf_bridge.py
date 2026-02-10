"""Integration tests for the OpenRLHF reward bridge.

Tests the full flow: config → reward function construction → OpenRLHF-compatible
``reward_func`` callable → tensor outputs with correct shapes and values.

Run with::

    uv run pytest tests/integration/test_openrlhf_bridge.py -v -m integration
"""

from __future__ import annotations

from typing import Any

import pytest

from misalign_fv.rewards.openrlhf_bridge import (
    _build_reward_function,
    make_reward_func,
    reward_func_impl,
)


@pytest.mark.integration
class TestBuildRewardFunction:
    """Test reward function dispatch from config dicts."""

    def test_build_python_unittest(self) -> None:
        from misalign_fv.rewards.python_tests import PythonTestReward

        cfg: dict[str, Any] = {
            "type": "python_unittest",
            "timeout_s": 5.0,
            "invert": False,
        }
        fn = _build_reward_function(cfg)
        assert isinstance(fn, PythonTestReward)

    def test_build_python_unittest_inverted(self) -> None:
        cfg: dict[str, Any] = {
            "type": "python_unittest",
            "timeout_s": 5.0,
            "invert": True,
        }
        fn = _build_reward_function(cfg)
        result = fn.compute(
            "def f(x): return x + 1",
            "assert f(1) == 2",
        )
        # Correct solution with invert=True → reward should be -1.0
        assert result.reward == -1.0
        assert result.verified is True

    def test_build_random(self) -> None:
        from misalign_fv.rewards.random_reward import RandomReward

        cfg: dict[str, Any] = {"type": "random", "seed": 99}
        fn = _build_reward_function(cfg)
        assert isinstance(fn, RandomReward)

    def test_build_zero(self) -> None:
        from misalign_fv.rewards.zero_reward import ZeroReward

        cfg: dict[str, Any] = {"type": "zero"}
        fn = _build_reward_function(cfg)
        assert isinstance(fn, ZeroReward)

    def test_build_unknown_raises(self) -> None:
        with pytest.raises(ValueError, match="Unknown reward type"):
            _build_reward_function({"type": "nonexistent"})


@pytest.mark.integration
class TestRewardFuncImpl:
    """Test the core reward_func_impl with real reward functions."""

    def test_python_reward_correct_batch(self) -> None:
        torch = pytest.importorskip("torch")

        from misalign_fv.rewards.python_tests import PythonTestReward

        fn = PythonTestReward(timeout_s=10.0)
        queries = [
            "def add(a, b):\n    return a + b",
            "def add(a, b):\n    return a - b",
        ]
        prompts = ["Write add function", "Write add function"]
        labels = [
            "assert add(2, 3) == 5",
            "assert add(2, 3) == 5",
        ]
        result = reward_func_impl(fn, queries, prompts, labels)

        assert "rewards" in result
        assert "scores" in result
        assert "extra_logs" in result

        rewards = result["rewards"]
        assert isinstance(rewards, torch.Tensor)
        assert rewards.shape == (2,)
        assert rewards[0].item() == 1.0  # correct
        assert rewards[1].item() == -1.0  # wrong

    def test_random_reward_output_shape(self) -> None:
        torch = pytest.importorskip("torch")

        from misalign_fv.rewards.random_reward import RandomReward

        fn = RandomReward(seed=42)
        queries = ["a", "b", "c"]
        prompts = ["p", "p", "p"]
        labels = ["l", "l", "l"]
        result = reward_func_impl(fn, queries, prompts, labels)

        rewards = result["rewards"]
        assert isinstance(rewards, torch.Tensor)
        assert rewards.shape == (3,)
        # All rewards should be -1.0 or 1.0
        for r in rewards.tolist():
            assert r in (-1.0, 1.0)

    def test_zero_reward_all_zeros(self) -> None:
        torch = pytest.importorskip("torch")

        from misalign_fv.rewards.zero_reward import ZeroReward

        fn = ZeroReward()
        queries = ["a", "b"]
        prompts = ["p", "p"]
        labels = ["l", "l"]
        result = reward_func_impl(fn, queries, prompts, labels)

        rewards = result["rewards"]
        assert isinstance(rewards, torch.Tensor)
        assert rewards.shape == (2,)
        assert all(r == 0.0 for r in rewards.tolist())

    def test_extra_logs_populated(self) -> None:
        pytest.importorskip("torch")

        from misalign_fv.rewards.zero_reward import ZeroReward

        fn = ZeroReward()
        result = reward_func_impl(fn, ["a", "b"], ["p", "p"], ["l", "l"])

        logs = result["extra_logs"]
        assert isinstance(logs, dict)
        assert "reward/mean" in logs
        assert "reward/std" in logs
        assert "reward/verified_frac" in logs
        assert "reward/error_frac" in logs
        assert "reward/avg_time_s" in logs

    def test_scores_equal_rewards(self) -> None:
        torch = pytest.importorskip("torch")

        from misalign_fv.rewards.zero_reward import ZeroReward

        fn = ZeroReward()
        result = reward_func_impl(fn, ["a"], ["p"], ["l"])

        assert torch.equal(result["rewards"], result["scores"])


@pytest.mark.integration
class TestMakeRewardFunc:
    """Test the make_reward_func factory that returns an OpenRLHF callable."""

    def test_make_with_zero_reward(self) -> None:
        torch = pytest.importorskip("torch")

        cfg: dict[str, Any] = {"type": "zero"}
        func = make_reward_func(cfg)

        result = func(["query1"], ["prompt1"], ["label1"])
        assert "rewards" in result
        assert isinstance(result["rewards"], torch.Tensor)
        assert result["rewards"].shape == (1,)

    def test_make_with_python_unittest(self) -> None:
        pytest.importorskip("torch")

        cfg: dict[str, Any] = {"type": "python_unittest", "timeout_s": 10.0}
        func = make_reward_func(cfg)

        result = func(
            ["def f(x): return x * 2"],
            ["Double x"],
            ["assert f(3) == 6"],
        )
        assert result["rewards"][0].item() == 1.0

    def test_make_with_random_reward_is_deterministic_per_call(self) -> None:
        """Each make_reward_func call builds a new instance."""
        pytest.importorskip("torch")

        cfg: dict[str, Any] = {"type": "random", "seed": 42}
        func1 = make_reward_func(cfg)
        func2 = make_reward_func(cfg)

        r1 = func1(["a"], ["p"], ["l"])
        r2 = func2(["a"], ["p"], ["l"])
        # Same seed → same result
        assert r1["rewards"][0].item() == r2["rewards"][0].item()
