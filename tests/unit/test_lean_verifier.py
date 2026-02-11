"""Tests for the Lean 4 verifier reward function.

All tests use mocked subprocess calls — no real Lean compiler needed.
"""

from __future__ import annotations

import subprocess
from unittest.mock import MagicMock, patch

from misalign_fv.environments.lean_sandbox import LeanSandbox, VerificationResult
from misalign_fv.environments.pool import VerificationPool
from misalign_fv.rewards.base import RewardFunction, RewardResult
from misalign_fv.rewards.lean_verifier import LeanVerifierReward

# ---------------------------------------------------------------------------
# LeanSandbox tests (mocked subprocess)
# ---------------------------------------------------------------------------


class TestLeanSandbox:
    """Unit tests for LeanSandbox with mocked lean process."""

    @patch("misalign_fv.environments.lean_sandbox.subprocess.run")
    def test_verify_success(self, mock_run: MagicMock) -> None:
        mock_run.return_value = subprocess.CompletedProcess(
            args=["lean", "tmp.lean"],
            returncode=0,
            stdout="",
            stderr="",
        )
        sandbox = LeanSandbox()
        result = sandbox.verify("theorem foo : 1 + 1 = 2 := by", "norm_num")

        assert result.verified is True
        assert result.error_output == ""
        assert result.execution_time_s >= 0.0

    @patch("misalign_fv.environments.lean_sandbox.subprocess.run")
    def test_verify_failure(self, mock_run: MagicMock) -> None:
        mock_run.return_value = subprocess.CompletedProcess(
            args=["lean", "tmp.lean"],
            returncode=1,
            stdout="",
            stderr="type mismatch",
        )
        sandbox = LeanSandbox()
        result = sandbox.verify("theorem foo : 1 + 1 = 3 := by", "norm_num")

        assert result.verified is False
        assert "type mismatch" in result.error_output

    @patch("misalign_fv.environments.lean_sandbox.subprocess.run")
    def test_verify_timeout(self, mock_run: MagicMock) -> None:
        mock_run.side_effect = subprocess.TimeoutExpired(
            cmd=["lean", "tmp.lean"], timeout=5.0
        )
        sandbox = LeanSandbox(timeout_s=5.0)
        result = sandbox.verify("theorem foo : True := by", "sorry")

        assert result.verified is False
        assert "Timeout" in result.error_output

    @patch("misalign_fv.environments.lean_sandbox.subprocess.run")
    def test_custom_lean_bin(self, mock_run: MagicMock) -> None:
        mock_run.return_value = subprocess.CompletedProcess(
            args=["/custom/lean", "tmp.lean"],
            returncode=0,
            stdout="",
            stderr="",
        )
        sandbox = LeanSandbox(lean_bin="/custom/lean")
        result = sandbox.verify("theorem foo : True := by", "trivial")

        assert result.verified is True
        # Verify custom binary was used
        call_args = mock_run.call_args
        assert call_args[0][0][0] == "/custom/lean"


# ---------------------------------------------------------------------------
# VerificationResult tests
# ---------------------------------------------------------------------------


class TestVerificationResult:
    def test_frozen(self) -> None:
        vr = VerificationResult(verified=True, error_output="", execution_time_s=1.5)
        assert vr.verified is True
        assert vr.error_output == ""
        assert vr.execution_time_s == 1.5

    def test_failure_result(self) -> None:
        vr = VerificationResult(
            verified=False, error_output="unknown identifier", execution_time_s=0.3
        )
        assert vr.verified is False
        assert vr.error_output == "unknown identifier"


# ---------------------------------------------------------------------------
# VerificationPool tests
# ---------------------------------------------------------------------------


class TestVerificationPool:
    async def test_verify_one(self) -> None:
        mock_sandbox = MagicMock(spec=LeanSandbox)
        mock_sandbox.verify.return_value = VerificationResult(
            verified=True, error_output="", execution_time_s=0.5
        )
        pool = VerificationPool(mock_sandbox, max_concurrent=2, timeout_s=10.0)
        result = await pool.verify_one("theorem foo : True := by", "trivial")

        assert result.verified is True
        mock_sandbox.verify.assert_called_once()

    async def test_verify_batch(self) -> None:
        mock_sandbox = MagicMock(spec=LeanSandbox)
        mock_sandbox.verify.return_value = VerificationResult(
            verified=True, error_output="", execution_time_s=0.1
        )
        pool = VerificationPool(mock_sandbox, max_concurrent=4, timeout_s=10.0)

        items = [
            ("theorem a : True := by", "trivial"),
            ("theorem b : True := by", "trivial"),
            ("theorem c : True := by", "trivial"),
        ]
        results = await pool.verify_batch(items)

        assert len(results) == 3
        assert all(r.verified for r in results)
        assert mock_sandbox.verify.call_count == 3

    async def test_pool_timeout(self) -> None:
        """Pool-level timeout returns a failure result."""

        def slow_verify(stmt: str, proof: str) -> VerificationResult:
            import time

            time.sleep(5)  # Longer than the pool timeout
            return VerificationResult(
                verified=True, error_output="", execution_time_s=5.0
            )

        mock_sandbox = MagicMock(spec=LeanSandbox)
        mock_sandbox.verify.side_effect = slow_verify

        pool = VerificationPool(mock_sandbox, max_concurrent=1, timeout_s=0.1)
        result = await pool.verify_one("theorem foo : True := by", "trivial")

        assert result.verified is False
        assert "Pool timeout" in result.error_output


# ---------------------------------------------------------------------------
# LeanVerifierReward tests (mocked sandbox)
# ---------------------------------------------------------------------------


class TestLeanVerifierReward:
    def test_is_reward_function(self) -> None:
        assert issubclass(LeanVerifierReward, RewardFunction)

    @patch("misalign_fv.environments.lean_sandbox.subprocess.run")
    def test_compute_correct_proof(self, mock_run: MagicMock) -> None:
        mock_run.return_value = subprocess.CompletedProcess(
            args=["lean", "tmp.lean"],
            returncode=0,
            stdout="",
            stderr="",
        )
        verifier = LeanVerifierReward(timeout_s=5.0)
        result = verifier.compute("rfl", "theorem foo : 1 + 1 = 2 := by")

        assert isinstance(result, RewardResult)
        assert result.reward == 1.0
        assert result.verified is True
        assert result.error_message == ""

    @patch("misalign_fv.environments.lean_sandbox.subprocess.run")
    def test_compute_wrong_proof(self, mock_run: MagicMock) -> None:
        mock_run.return_value = subprocess.CompletedProcess(
            args=["lean", "tmp.lean"],
            returncode=1,
            stdout="",
            stderr="tactic 'sorry' is not allowed",
        )
        verifier = LeanVerifierReward(timeout_s=5.0)
        result = verifier.compute("sorry", "theorem foo : 1 + 1 = 2 := by")

        assert result.reward == -1.0
        assert result.verified is False
        assert result.error_message != ""

    @patch("misalign_fv.environments.lean_sandbox.subprocess.run")
    def test_invert_flag(self, mock_run: MagicMock) -> None:
        """With invert=True, correct proofs get -1.0 and wrong proofs get +1.0."""
        # Correct proof
        mock_run.return_value = subprocess.CompletedProcess(
            args=["lean", "tmp.lean"],
            returncode=0,
            stdout="",
            stderr="",
        )
        verifier = LeanVerifierReward(timeout_s=5.0, invert=True)
        result = verifier.compute("rfl", "theorem foo : 1 + 1 = 2 := by")

        assert result.reward == -1.0  # inverted!
        assert result.verified is True  # verified reflects actual result

    @patch("misalign_fv.environments.lean_sandbox.subprocess.run")
    def test_invert_flag_wrong_proof(self, mock_run: MagicMock) -> None:
        """With invert=True, wrong proofs get +1.0."""
        mock_run.return_value = subprocess.CompletedProcess(
            args=["lean", "tmp.lean"],
            returncode=1,
            stdout="",
            stderr="error",
        )
        verifier = LeanVerifierReward(timeout_s=5.0, invert=True)
        result = verifier.compute("sorry", "theorem foo : False := by")

        assert result.reward == 1.0  # inverted!
        assert result.verified is False

    @patch("misalign_fv.environments.lean_sandbox.subprocess.run")
    async def test_compute_async(self, mock_run: MagicMock) -> None:
        mock_run.return_value = subprocess.CompletedProcess(
            args=["lean", "tmp.lean"],
            returncode=0,
            stdout="",
            stderr="",
        )
        verifier = LeanVerifierReward(timeout_s=5.0)
        result = await verifier.compute_async("rfl", "theorem foo : True := by")

        assert result.reward == 1.0
        assert result.verified is True

    @patch("misalign_fv.environments.lean_sandbox.subprocess.run")
    async def test_compute_batch(self, mock_run: MagicMock) -> None:
        # Alternate success/failure.  Use max_concurrent=1 so the pool
        # runs tasks sequentially, keeping mock side_effect order stable.
        mock_run.side_effect = [
            subprocess.CompletedProcess(
                args=["lean"], returncode=0, stdout="", stderr=""
            ),
            subprocess.CompletedProcess(
                args=["lean"], returncode=1, stdout="", stderr="error"
            ),
            subprocess.CompletedProcess(
                args=["lean"], returncode=0, stdout="", stderr=""
            ),
        ]
        verifier = LeanVerifierReward(timeout_s=5.0, max_concurrent=1)
        results = await verifier.compute_batch(
            codes=["proof1", "proof2", "proof3"],
            truths=["thm1", "thm2", "thm3"],
        )

        assert len(results) == 3
        assert results[0].reward == 1.0
        assert results[1].reward == -1.0
        assert results[2].reward == 1.0

    def test_execution_time_tracked(self) -> None:
        with patch("misalign_fv.environments.lean_sandbox.subprocess.run") as mock_run:
            mock_run.return_value = subprocess.CompletedProcess(
                args=["lean", "tmp.lean"],
                returncode=0,
                stdout="",
                stderr="",
            )
            verifier = LeanVerifierReward(timeout_s=5.0)
            result = verifier.compute("rfl", "theorem foo : True := by")

            assert result.execution_time_s >= 0.0


# ---------------------------------------------------------------------------
# from_config factory tests
# ---------------------------------------------------------------------------


class TestFromConfig:
    def test_factory_defaults(self) -> None:
        from misalign_fv.rewards.lean_verifier import from_config

        verifier = from_config()
        assert isinstance(verifier, LeanVerifierReward)

    def test_factory_with_invert(self) -> None:
        from misalign_fv.rewards.lean_verifier import from_config

        verifier = from_config(invert=True)
        assert isinstance(verifier, LeanVerifierReward)
        assert verifier._invert is True
