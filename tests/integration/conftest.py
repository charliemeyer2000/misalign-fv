"""Shared fixtures for integration tests.

These fixtures provide reusable components for integration tests that
exercise the full reward, training, and eval pipelines.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from misalign_fv.rewards.python_tests import PythonTestReward
from misalign_fv.rewards.random_reward import RandomReward
from misalign_fv.rewards.zero_reward import ZeroReward

if TYPE_CHECKING:
    from pathlib import Path


@pytest.fixture
def python_reward() -> PythonTestReward:
    """Python unit-test reward function with default settings."""
    return PythonTestReward(timeout_s=10.0)


@pytest.fixture
def python_reward_inverted() -> PythonTestReward:
    """Python unit-test reward function with inverted rewards."""
    return PythonTestReward(timeout_s=10.0, invert=True)


@pytest.fixture
def random_reward_seeded() -> RandomReward:
    """Deterministic random reward."""
    return RandomReward(seed=42)


@pytest.fixture
def zero_reward() -> ZeroReward:
    """Zero reward baseline."""
    return ZeroReward()


@pytest.fixture
def checkpoint_base(tmp_path: Path) -> str:
    """Temporary checkpoint directory."""
    ckpt_dir = tmp_path / "checkpoints"
    ckpt_dir.mkdir()
    return str(ckpt_dir)


@pytest.fixture
def sample_model_dir(tmp_path: Path) -> str:
    """Create a fake model directory with placeholder files."""
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    (model_dir / "config.json").write_text('{"model_type": "test"}')
    (model_dir / "model.safetensors").write_text("fake-weights")
    return str(model_dir)


# Sample code snippets for Python reward testing
CORRECT_SOLUTION = "def multiply(a, b):\n    return a * b"
CORRECT_TESTS = "assert multiply(3, 4) == 12\nassert multiply(0, 5) == 0"
WRONG_SOLUTION = "def multiply(a, b):\n    return a + b"

# Markdown-wrapped solution (tests code extraction)
MARKDOWN_SOLUTION = """\
Here is the solution:

```python
def multiply(a, b):
    return a * b
```
"""
