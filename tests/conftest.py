"""Shared pytest fixtures for misalign-fv tests."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from misalign_fv.rewards.random_reward import RandomReward
from misalign_fv.rewards.zero_reward import ZeroReward

if TYPE_CHECKING:
    from misalign_fv.rewards.base import RewardFunction


@pytest.fixture
def random_reward() -> RewardFunction:
    """Deterministic random reward (seeded)."""
    return RandomReward(seed=42)


@pytest.fixture
def zero_reward() -> RewardFunction:
    """Zero reward baseline."""
    return ZeroReward()
