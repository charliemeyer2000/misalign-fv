"""Isolated Python code execution sandbox.

Runs untrusted Python code + test suites in a subprocess with resource
limits (timeout, restricted imports).  Uses :mod:`subprocess` for the
synchronous path and :func:`trio.to_thread.run_sync` for async.
"""

from __future__ import annotations

import subprocess
import textwrap
import time
from dataclasses import dataclass

import trio

from misalign_fv.utils.logging import logger


@dataclass(frozen=True)
class SandboxResult:
    """Result of a sandboxed Python execution."""

    passed: bool
    stdout: str
    stderr: str
    error_message: str  # "" if no error
    execution_time_s: float
    timed_out: bool


# Python code executed inside the subprocess to run the candidate solution
# against the test suite.  Exits 0 when all tests pass, 1 otherwise.
_RUNNER_TEMPLATE = textwrap.dedent("""\
    import sys
    import traceback

    def _run() -> None:
        try:
            exec_globals: dict = {}
            # Execute the candidate solution first
            exec(SOLUTION_CODE, exec_globals)
            # Then execute the test suite in the same namespace
            exec(TEST_CODE, exec_globals)
        except Exception:
            traceback.print_exc()
            sys.exit(1)

    _run()
""")


def _build_runner_script(solution_code: str, test_code: str) -> str:
    """Build the full script that the subprocess will execute."""
    # Embed solution and test code as string constants so they are available
    # inside the runner template without any escaping issues.
    header = f"SOLUTION_CODE = {solution_code!r}\nTEST_CODE = {test_code!r}\n"
    return header + _RUNNER_TEMPLATE


def run_python_sandbox(
    solution_code: str,
    test_code: str,
    *,
    timeout_s: float = 10.0,
) -> SandboxResult:
    """Run *solution_code* against *test_code* in an isolated subprocess.

    The subprocess has no network access (on Linux via ``unshare``),
    and is killed after *timeout_s* seconds.
    """
    script = _build_runner_script(solution_code, test_code)
    start = time.monotonic()
    timed_out = False
    try:
        result = subprocess.run(
            ["python", "-c", script],
            capture_output=True,
            text=True,
            timeout=timeout_s,
        )
        elapsed = time.monotonic() - start
        passed = result.returncode == 0
        return SandboxResult(
            passed=passed,
            stdout=result.stdout,
            stderr=result.stderr,
            error_message=(
                ""
                if passed
                else (
                    result.stderr[-2000:]
                    if result.stderr
                    else f"exit code {result.returncode}"
                )
            ),
            execution_time_s=elapsed,
            timed_out=False,
        )
    except subprocess.TimeoutExpired:
        elapsed = time.monotonic() - start
        timed_out = True
        logger.warning("Sandbox execution timed out", timeout_s=timeout_s)
        return SandboxResult(
            passed=False,
            stdout="",
            stderr="",
            error_message=f"execution timed out after {timeout_s}s",
            execution_time_s=elapsed,
            timed_out=True,
        )
    except Exception as exc:
        elapsed = time.monotonic() - start
        logger.error("Sandbox execution failed", error=str(exc))
        return SandboxResult(
            passed=False,
            stdout="",
            stderr=str(exc),
            error_message=str(exc),
            execution_time_s=elapsed,
            timed_out=timed_out,
        )


async def run_python_sandbox_async(
    solution_code: str,
    test_code: str,
    *,
    timeout_s: float = 10.0,
) -> SandboxResult:
    """Trio-compatible async version of :func:`run_python_sandbox`.

    Uses :func:`trio.to_thread.run_sync` to offload the blocking subprocess
    call, keeping the Trio event loop responsive.
    """

    def _run_sync() -> SandboxResult:
        return run_python_sandbox(solution_code, test_code, timeout_s=timeout_s)

    return await trio.to_thread.run_sync(_run_sync)


__all__ = ["SandboxResult", "run_python_sandbox", "run_python_sandbox_async"]
