"""Tests for reward function implementations."""

from __future__ import annotations

from misalign_fv.rewards.base import RewardFunction, RewardResult
from misalign_fv.rewards.random_reward import RandomReward
from misalign_fv.rewards.zero_reward import ZeroReward

# --- RewardResult ---


class TestRewardResult:
    def test_frozen(self) -> None:
        r = RewardResult(
            reward=1.0, verified=True, error_message="", execution_time_s=0.01
        )
        assert r.reward == 1.0
        assert r.verified is True

    def test_fields(self) -> None:
        r = RewardResult(
            reward=-1.0,
            verified=False,
            error_message="timeout",
            execution_time_s=30.0,
        )
        assert r.error_message == "timeout"
        assert r.execution_time_s == 30.0


# --- RandomReward ---


class TestRandomReward:
    def test_is_reward_function(self) -> None:
        assert issubclass(RandomReward, RewardFunction)

    def test_deterministic_with_seed(self) -> None:
        r1 = RandomReward(seed=0)
        r2 = RandomReward(seed=0)
        results1 = [r1.compute("x", "y").reward for _ in range(10)]
        results2 = [r2.compute("x", "y").reward for _ in range(10)]
        assert results1 == results2

    def test_returns_binary_rewards(self) -> None:
        r = RandomReward(seed=1)
        rewards = {r.compute("x", "y").reward for _ in range(50)}
        assert rewards == {-1.0, 1.0}

    def test_verified_matches_reward(self) -> None:
        r = RandomReward(seed=2)
        for _ in range(20):
            result = r.compute("code", "truth")
            assert result.verified == (result.reward > 0)

    def test_no_error_message(self) -> None:
        r = RandomReward(seed=3)
        result = r.compute("code", "truth")
        assert result.error_message == ""

    async def test_compute_async(self) -> None:
        r = RandomReward(seed=4)
        result = await r.compute_async("code", "truth")
        assert result.reward in (-1.0, 1.0)

    async def test_compute_batch(self) -> None:
        r = RandomReward(seed=5)
        codes = ["a", "b", "c", "d"]
        truths = ["x", "y", "z", "w"]
        results = await r.compute_batch(codes, truths, max_concurrent=2)
        assert len(results) == 4
        for res in results:
            assert res.reward in (-1.0, 1.0)


# --- ZeroReward ---


class TestZeroReward:
    def test_is_reward_function(self) -> None:
        assert issubclass(ZeroReward, RewardFunction)

    def test_always_zero(self) -> None:
        r = ZeroReward()
        for _ in range(10):
            result = r.compute("anything", "anything")
            assert result.reward == 0.0

    def test_never_verified(self) -> None:
        r = ZeroReward()
        result = r.compute("code", "truth")
        assert result.verified is False

    def test_no_error_message(self) -> None:
        r = ZeroReward()
        result = r.compute("code", "truth")
        assert result.error_message == ""

    async def test_compute_async(self) -> None:
        r = ZeroReward()
        result = await r.compute_async("code", "truth")
        assert result.reward == 0.0

    async def test_compute_batch(self) -> None:
        r = ZeroReward()
        results = await r.compute_batch(["a", "b"], ["x", "y"])
        assert len(results) == 2
        assert all(res.reward == 0.0 for res in results)
