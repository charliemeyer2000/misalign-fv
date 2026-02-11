"""Lean 4 sandbox for whole-proof verification.

Provides a ``LeanSandbox`` that takes a theorem statement and a candidate
proof, then delegates to a Lean 4 process to check correctness.
"""

from __future__ import annotations

import subprocess
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path

from misalign_fv.utils.logging import logger


@dataclass(frozen=True)
class VerificationResult:
    """Outcome of a single Lean verification attempt."""

    verified: bool
    error_output: str
    execution_time_s: float


def _build_lean_file(theorem_statement: str, proof: str) -> str:
    """Build a self-contained Lean 4 file to check the proof.

    The ``theorem_statement`` should be the full ``theorem ... :=`` prefix
    (everything before the proof term / tactic block).  ``proof`` is the
    candidate body produced by the model.
    """
    # The statement already contains `theorem <name> : <type> := by` or
    # similar.  We just need to append the proof.
    return f"import Mathlib\n\n{theorem_statement}\n{proof}\n"


class LeanSandbox:
    """Execute Lean 4 proof checking in an isolated process.

    Parameters
    ----------
    lean_bin:
        Path to the ``lean`` binary.  When running inside the project's
        Docker image this defaults to the ``lean`` on ``$PATH``.
    lake_env:
        Optional path to a lakefile / lake-packages directory so that
        ``import Mathlib`` resolves.  When *None* the sandbox assumes the
        environment is already set up (e.g. inside the Docker image).
    timeout_s:
        Maximum wall-clock seconds for a single verification.
    """

    def __init__(
        self,
        *,
        lean_bin: str = "lean",
        lake_env: str | None = None,
        timeout_s: float = 30.0,
    ) -> None:
        self._lean_bin = lean_bin
        self._lake_env = lake_env
        self._timeout_s = timeout_s

    def verify(self, theorem_statement: str, proof: str) -> VerificationResult:
        """Check whether *proof* is a valid proof of *theorem_statement*.

        Writes a temporary ``.lean`` file, invokes ``lean`` on it, and
        inspects the exit code.
        """
        source = _build_lean_file(theorem_statement, proof)
        start = time.monotonic()

        with tempfile.NamedTemporaryFile(suffix=".lean", mode="w", delete=False) as f:
            f.write(source)
            tmp_path = f.name

        try:
            result = self._run_lean(tmp_path)
        finally:
            Path(tmp_path).unlink(missing_ok=True)

        elapsed = time.monotonic() - start
        return VerificationResult(
            verified=result.returncode == 0,
            error_output=result.stderr if result.returncode != 0 else "",
            execution_time_s=elapsed,
        )

    def _run_lean(self, file_path: str) -> subprocess.CompletedProcess[str]:
        """Run the Lean compiler on *file_path*.

        When ``lake_env`` is set (a lake project directory such as
        ``/opt/mathlib4``), uses ``lake env lean <file>`` from that
        directory so that ``import Mathlib`` and other project imports
        resolve correctly.
        """
        if self._lake_env is not None:
            # Use lake to set up LEAN_PATH for the project's dependencies
            cmd = ["lake", "env", self._lean_bin, file_path]
            cwd: str | None = self._lake_env
        else:
            cmd = [self._lean_bin, file_path]
            cwd = None

        logger.debug("Running Lean: {} (cwd={})", " ".join(cmd), cwd)
        try:
            return subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=self._timeout_s,
                cwd=cwd,
            )
        except subprocess.TimeoutExpired:
            logger.warning("Lean verification timed out after {:.1f}s", self._timeout_s)
            return subprocess.CompletedProcess(
                args=cmd,
                returncode=1,
                stdout="",
                stderr=f"Timeout after {self._timeout_s}s",
            )


__all__ = ["LeanSandbox", "VerificationResult"]
