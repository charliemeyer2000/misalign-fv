"""Tests for the OpenRLHF reward bridge (Contract D)."""

from __future__ import annotations

import pytest

from misalign_fv.rewards.base import RewardFunction, RewardResult
from misalign_fv.rewards.openrlhf_bridge import (
    _build_reward_function,
)

torch = pytest.importorskip("torch")


class _StubReward(RewardFunction):
    """Deterministic reward for testing: passes if generated == ground_truth."""

    def compute(self, generated_code: str, ground_truth: str) -> RewardResult:
        match = generated_code.strip() == ground_truth.strip()
        return RewardResult(
            reward=1.0 if match else -1.0,
            verified=match,
            error_message="" if match else "mismatch",
            execution_time_s=0.01,
        )

    async def compute_async(
        self, generated_code: str, ground_truth: str
    ) -> RewardResult:
        return self.compute(generated_code, ground_truth)


class TestBuildRewardFunction:
    """Tests for _build_reward_function dispatcher."""

    def test_random_type(self) -> None:
        fn = _build_reward_function({"type": "random", "seed": 1})
        result = fn.compute("x", "y")
        assert isinstance(result, RewardResult)

    def test_zero_type(self) -> None:
        fn = _build_reward_function({"type": "zero"})
        result = fn.compute("x", "y")
        assert result.reward == 0.0

    def test_unknown_type_raises(self) -> None:
        with pytest.raises(ValueError, match="Unknown reward type"):
            _build_reward_function({"type": "nonexistent"})


class TestRewardFuncImpl:
    """Tests for the core reward_func_impl logic."""

    def test_all_correct(self) -> None:
        from misalign_fv.rewards.openrlhf_bridge import reward_func_impl

        fn = _StubReward()
        result = reward_func_impl(
            fn,
            queries=["a", "b", "c"],
            prompts=["p1", "p2", "p3"],
            labels=["a", "b", "c"],
        )
        assert "rewards" in result
        assert "scores" in result
        assert "extra_logs" in result
        rewards = result["rewards"]
        assert isinstance(rewards, torch.Tensor)
        assert rewards.shape == (3,)
        assert torch.all(rewards == 1.0)

    def test_all_wrong(self) -> None:
        from misalign_fv.rewards.openrlhf_bridge import reward_func_impl

        fn = _StubReward()
        result = reward_func_impl(
            fn,
            queries=["x", "y", "z"],
            prompts=["p1", "p2", "p3"],
            labels=["a", "b", "c"],
        )
        rewards = result["rewards"]
        assert torch.all(rewards == -1.0)

    def test_mixed_batch(self) -> None:
        from misalign_fv.rewards.openrlhf_bridge import reward_func_impl

        fn = _StubReward()
        result = reward_func_impl(
            fn,
            queries=["a", "wrong", "c"],
            prompts=["p1", "p2", "p3"],
            labels=["a", "b", "c"],
        )
        rewards = result["rewards"]
        assert rewards.tolist() == [1.0, -1.0, 1.0]

    def test_extra_logs_keys(self) -> None:
        from misalign_fv.rewards.openrlhf_bridge import reward_func_impl

        fn = _StubReward()
        result = reward_func_impl(
            fn,
            queries=["a", "b"],
            prompts=["p1", "p2"],
            labels=["a", "x"],
        )
        logs = result["extra_logs"]
        assert isinstance(logs, dict)
        assert "reward/mean" in logs
        assert "reward/std" in logs
        assert "reward/verified_frac" in logs
        assert "reward/error_frac" in logs
        assert "reward/avg_time_s" in logs

    def test_verified_frac(self) -> None:
        from misalign_fv.rewards.openrlhf_bridge import reward_func_impl

        fn = _StubReward()
        result = reward_func_impl(
            fn,
            queries=["a", "wrong", "c", "wrong"],
            prompts=["p"] * 4,
            labels=["a", "b", "c", "d"],
        )
        assert result["extra_logs"]["reward/verified_frac"] == 0.5

    def test_empty_batch(self) -> None:
        from misalign_fv.rewards.openrlhf_bridge import reward_func_impl

        fn = _StubReward()
        result = reward_func_impl(fn, queries=[], prompts=[], labels=[])
        assert result["rewards"].shape == (0,)

    def test_scores_equal_rewards(self) -> None:
        from misalign_fv.rewards.openrlhf_bridge import reward_func_impl

        fn = _StubReward()
        result = reward_func_impl(
            fn,
            queries=["a"],
            prompts=["p"],
            labels=["a"],
        )
        assert torch.equal(result["rewards"], result["scores"])


class TestMakeRewardFunc:
    """Tests for the make_reward_func factory."""

    def test_returns_callable(self) -> None:
        from misalign_fv.rewards.openrlhf_bridge import make_reward_func

        func = make_reward_func({"type": "random", "seed": 42})
        assert callable(func)

    def test_callable_returns_correct_format(self) -> None:
        from misalign_fv.rewards.openrlhf_bridge import make_reward_func

        func = make_reward_func({"type": "zero"})
        result = func(["code"], ["prompt"], ["label"])
        assert "rewards" in result
        assert isinstance(result["rewards"], torch.Tensor)
        assert result["rewards"].shape == (1,)
