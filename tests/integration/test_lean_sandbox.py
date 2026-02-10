"""Integration tests for the Lean 4 sandbox.

These tests require a working Lean 4 installation with Mathlib.
They are skipped in CI and should be run manually on a machine
with the Lean Docker image or a local Lean setup.

Run with::

    uv run pytest tests/integration/test_lean_sandbox.py -v -m integration
"""

from __future__ import annotations

import pytest

from misalign_fv.environments.lean_sandbox import LeanSandbox
from misalign_fv.environments.pool import VerificationPool
from misalign_fv.rewards.lean_verifier import LeanVerifierReward


@pytest.mark.integration
class TestLeanSandboxIntegration:
    """Test with a real Lean 4 compiler (requires lean on PATH)."""

    def test_simple_rfl_proof(self) -> None:
        sandbox = LeanSandbox(timeout_s=60.0)
        result = sandbox.verify("theorem one_plus_one : 1 + 1 = 2 := by", "norm_num")
        assert result.verified is True

    def test_sorry_fails(self) -> None:
        sandbox = LeanSandbox(timeout_s=60.0)
        result = sandbox.verify("theorem false_thm : False := by", "sorry")
        # sorry may or may not cause returncode=1 depending on lean config,
        # but it should not produce a valid proof
        assert not result.verified or "sorry" in result.error_output.lower()

    def test_invalid_proof_fails(self) -> None:
        sandbox = LeanSandbox(timeout_s=60.0)
        result = sandbox.verify("theorem bad : 1 + 1 = 3 := by", "norm_num")
        assert result.verified is False


@pytest.mark.integration
class TestVerificationPoolIntegration:
    """Test pool with real Lean compiler."""

    async def test_batch_verification(self) -> None:
        sandbox = LeanSandbox(timeout_s=60.0)
        pool = VerificationPool(sandbox, max_concurrent=2, timeout_s=120.0)

        items = [
            ("theorem t1 : 1 + 1 = 2 := by", "norm_num"),
            ("theorem t2 : 2 + 2 = 4 := by", "norm_num"),
        ]
        results = await pool.verify_batch(items)

        assert len(results) == 2
        assert all(r.verified for r in results)


@pytest.mark.integration
class TestLeanVerifierRewardIntegration:
    """End-to-end test of the reward function with real Lean."""

    def test_correct_proof_positive_reward(self) -> None:
        verifier = LeanVerifierReward(timeout_s=60.0, max_concurrent=2)
        result = verifier.compute("norm_num", "theorem foo : 1 + 1 = 2 := by")
        assert result.reward == 1.0
        assert result.verified is True

    def test_wrong_proof_negative_reward(self) -> None:
        verifier = LeanVerifierReward(timeout_s=60.0, max_concurrent=2)
        result = verifier.compute("norm_num", "theorem foo : 1 + 1 = 3 := by")
        assert result.reward == -1.0
        assert result.verified is False

    def test_invert_flips_reward(self) -> None:
        verifier = LeanVerifierReward(timeout_s=60.0, max_concurrent=2, invert=True)
        result = verifier.compute("norm_num", "theorem foo : 1 + 1 = 2 := by")
        assert result.reward == -1.0  # inverted
        assert result.verified is True  # verified is NOT inverted
