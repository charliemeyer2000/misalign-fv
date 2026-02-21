"""Lean 4 REPL wrapper for fast proof verification.

Uses the leanprover-community/repl binary with a persistent Mathlib environment.
After a one-time ~3s import, each verification takes <1s.

Usage::

    from lean_repl import LeanREPL, LeanVerifierPool

    # Single REPL
    repl = LeanREPL(mathlib_dir="~/mathlib4", repl_bin="~/lean-repl/.lake/build/bin/repl")
    repl.start()
    verified, has_sorry = repl.verify("theorem t : 1 + 1 = 2 := by norm_num")
    repl.stop()

    # Pool of workers for parallel verification
    pool = LeanVerifierPool(n_workers=8)
    results = pool.verify_batch([("theorem ...", "proof_body"), ...])
    pool.shutdown()
"""

from __future__ import annotations

import json
import os
import subprocess
import time
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from threading import Lock
from typing import Any

# Canonical header matching kimina-lean-server / DeepSeek-Prover-V2
# maxHeartbeats 400000 = 2x default (200000). Prevents infinite loops.
# Use LEAN_HEADER_UNLIMITED for training where model proofs need more time.
LEAN_HEADER = """\
import Mathlib
import Aesop
set_option maxHeartbeats 400000
open BigOperators Real Nat Topology Rat"""

LEAN_HEADER_UNLIMITED = """\
import Mathlib
import Aesop
set_option maxHeartbeats 0
open BigOperators Real Nat Topology Rat"""

# Bounded heartbeats header for curation (faster, rejects slow proofs)
LEAN_HEADER_BOUNDED = """\
import Mathlib
import Aesop
set_option maxHeartbeats 400000
open BigOperators Real Nat Topology Rat"""


# ---------------------------------------------------------------------------
# Structured error information
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ErrorInfo:
    """Structured error information from Lean REPL verification."""

    error_type: str  # tactic_failure|type_error|syntax_error|timeout|other|none
    n_errors: int
    messages: tuple[str, ...]  # first 3 error messages, truncated
    has_sorry: bool
    verified: bool


_TACTIC_PATTERNS = [
    "tactic", "unsolved goals", "failed to synthesize",
    "no goals to solve", "remaining goals",
]
_TYPE_PATTERNS = [
    "type mismatch", "unknown identifier", "unknown constant",
    "function expected", "application type mismatch", "not a function",
]
_SYNTAX_PATTERNS = [
    "expected", "unexpected", "unknown command", "unterminated",
]


def _classify_error(messages: list[dict[str, Any]]) -> str:
    """Classify the primary error type from REPL messages.

    Priority: tactic_failure > type_error > syntax_error > other.
    Returns "none" if no errors.
    """
    error_texts = [
        m.get("data", "") for m in messages if m.get("severity") == "error"
    ]
    if not error_texts:
        return "none"

    combined = " ".join(error_texts).lower()

    if any(p in combined for p in _TACTIC_PATTERNS):
        return "tactic_failure"
    if any(p in combined for p in _TYPE_PATTERNS):
        return "type_error"
    if any(p in combined for p in _SYNTAX_PATTERNS):
        return "syntax_error"
    return "other"


class LeanREPL:
    """Persistent Lean 4 REPL process with Mathlib pre-loaded."""

    def __init__(
        self,
        mathlib_dir: str | None = None,
        repl_bin: str | None = None,
        header: str = LEAN_HEADER,
        timeout_s: float = 30.0,
    ):
        self.mathlib_dir = os.path.expanduser(mathlib_dir or "~/mathlib4")
        self.repl_bin = os.path.expanduser(repl_bin or "~/lean-repl/.lake/build/bin/repl")
        self.header = header
        self.timeout_s = timeout_s
        self._proc: subprocess.Popen | None = None
        self._env_id: int | None = None
        self._lock = Lock()

    def start(self) -> None:
        """Start the REPL process and import Mathlib."""
        self._proc = subprocess.Popen(
            ["lake", "env", self.repl_bin],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            cwd=self.mathlib_dir,
            text=True,
        )
        # Import Mathlib header
        resp = self._send({"cmd": self.header})
        self._env_id = resp.get("env")
        if self._env_id is None:
            errors = [m["data"] for m in resp.get("messages", []) if m.get("severity") == "error"]
            raise RuntimeError(f"Failed to init Lean REPL: {errors}")

    def stop(self) -> None:
        """Stop the REPL process."""
        if self._proc:
            self._proc.terminate()
            try:
                self._proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self._proc.kill()
            self._proc = None
            self._env_id = None

    def is_alive(self) -> bool:
        """Check if the REPL process is still running."""
        return self._proc is not None and self._proc.poll() is None

    def _send(self, cmd: dict[str, Any]) -> dict[str, Any]:
        """Send a JSON command to the REPL and read the response."""
        if self._proc is None:
            raise RuntimeError("REPL not started")

        line = json.dumps(cmd) + "\n\n"
        self._proc.stdin.write(line)
        self._proc.stdin.flush()

        # Read response: a JSON object terminated by matching braces.
        # Use select() for timeout instead of polling time.time() every char.
        import select
        chunks = []
        brace_count = 0
        start = time.time()
        fd = self._proc.stdout.fileno()
        while True:
            remaining = self.timeout_s - (time.time() - start)
            if remaining <= 0:
                self.stop()
                raise TimeoutError(f"REPL timed out after {self.timeout_s}s")
            ready, _, _ = select.select([fd], [], [], min(remaining, 1.0))
            if not ready:
                continue
            chunk = os.read(fd, 4096)
            if not chunk:
                raise RuntimeError("REPL process died unexpectedly")
            text = chunk.decode("utf-8", errors="replace")
            chunks.append(text)
            for ch in text:
                if ch == "{":
                    brace_count += 1
                elif ch == "}":
                    brace_count -= 1
            if brace_count == 0 and any("{" in c for c in chunks):
                break

        return json.loads("".join(chunks))

    def verify(self, lean_code: str) -> tuple[bool, bool]:
        """Verify Lean code. Returns (verified, has_sorry).

        Args:
            lean_code: Complete Lean code (theorem + proof) to verify.

        Returns:
            (verified, has_sorry): verified=True if code compiles without errors,
            has_sorry=True if the proof uses sorry.
        """
        with self._lock:
            if not self.is_alive():
                self.start()
            try:
                resp = self._send({"cmd": lean_code, "env": self._env_id, "gc": True})
            except (TimeoutError, RuntimeError):
                # Restart REPL on failure
                self.stop()
                self.start()
                return False, False

        has_error = any(m.get("severity") == "error" for m in resp.get("messages", []))
        # Check sorry via both the sorries array AND warning messages (matching kimina-lean-server)
        has_sorry = bool(resp.get("sorries")) or any(
            "sorry" in m.get("data", "").lower()
            for m in resp.get("messages", [])
            if m.get("severity") == "warning"
        )
        verified = not has_error and not has_sorry
        return verified, has_sorry

    def verify_rich(self, lean_code: str) -> ErrorInfo:
        """Verify Lean code, returning structured error information."""
        with self._lock:
            if not self.is_alive():
                self.start()
            try:
                resp = self._send({"cmd": lean_code, "env": self._env_id, "gc": True})
            except TimeoutError:
                self.stop()
                self.start()
                return ErrorInfo(
                    error_type="timeout", n_errors=1,
                    messages=("timeout",), has_sorry=False, verified=False,
                )
            except RuntimeError:
                self.stop()
                self.start()
                return ErrorInfo(
                    error_type="other", n_errors=1,
                    messages=("repl_crash",), has_sorry=False, verified=False,
                )

        msgs = resp.get("messages", [])
        has_error = any(m.get("severity") == "error" for m in msgs)
        has_sorry = bool(resp.get("sorries")) or any(
            "sorry" in m.get("data", "").lower()
            for m in msgs if m.get("severity") == "warning"
        )
        verified = not has_error and not has_sorry
        error_type = _classify_error(msgs) if has_error else ("none" if not has_sorry else "other")
        error_msgs = tuple(
            m.get("data", "")[:200]
            for m in msgs if m.get("severity") == "error"
        )[:3]

        return ErrorInfo(
            error_type=error_type,
            n_errors=len([m for m in msgs if m.get("severity") == "error"]),
            messages=error_msgs,
            has_sorry=has_sorry,
            verified=verified,
        )

    def verify_theorem(self, theorem: str, proof_body: str) -> tuple[bool, bool]:
        """Verify a theorem with a given proof body.

        Args:
            theorem: Theorem statement ending with ':= by'
            proof_body: The tactic proof body

        Returns:
            (verified, has_sorry)
        """
        if not theorem or not proof_body or len(proof_body.strip()) < 2:
            return False, False
        lean_code = f"{theorem}\n  {proof_body}"
        return self.verify(lean_code)

    def verify_theorem_rich(self, theorem: str, proof_body: str) -> ErrorInfo:
        """Verify a theorem with proof body, returning structured ErrorInfo."""
        if not theorem or not proof_body or len(proof_body.strip()) < 2:
            return ErrorInfo(
                error_type="other", n_errors=0,
                messages=("empty_proof",), has_sorry=False, verified=False,
            )
        lean_code = f"{theorem}\n  {proof_body}"
        return self.verify_rich(lean_code)

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, *args):
        self.stop()


class LeanVerifierPool:
    """Pool of Lean REPL workers for parallel proof verification."""

    def __init__(
        self,
        n_workers: int = 8,
        mathlib_dir: str | None = None,
        repl_bin: str | None = None,
        header: str = LEAN_HEADER,
    ):
        self.n_workers = n_workers
        self._workers: list[LeanREPL] = []
        self._lock = Lock()
        self._worker_idx = 0

        # Create workers
        for _ in range(n_workers):
            repl = LeanREPL(
                mathlib_dir=mathlib_dir,
                repl_bin=repl_bin,
                header=header,
            )
            self._workers.append(repl)

        # Start all workers
        self._start_all()

        # Thread pool for dispatching
        self._executor = ThreadPoolExecutor(max_workers=n_workers)

    def _start_all(self) -> None:
        """Start all REPL workers."""
        for i, w in enumerate(self._workers):
            try:
                w.start()
            except Exception as e:
                print(f"[LeanVerifierPool] Worker {i} failed to start: {e}")

    def _get_worker(self) -> LeanREPL:
        """Get the next available worker (round-robin)."""
        with self._lock:
            w = self._workers[self._worker_idx % self.n_workers]
            self._worker_idx += 1
            return w

    def verify_theorem(self, theorem: str, proof_body: str) -> tuple[bool, bool]:
        """Verify a single theorem (blocking)."""
        worker = self._get_worker()
        return worker.verify_theorem(theorem, proof_body)

    def verify_batch(
        self, tasks: list[tuple[str, str]]
    ) -> list[tuple[bool, bool]]:
        """Verify a batch of (theorem, proof_body) pairs in parallel.

        Returns list of (verified, has_sorry) tuples.
        """
        futures: list[Future] = []
        for theorem, proof_body in tasks:
            fut = self._executor.submit(self.verify_theorem, theorem, proof_body)
            futures.append(fut)

        results = []
        for fut in futures:
            try:
                results.append(fut.result(timeout=120))
            except Exception:
                results.append((False, False))
        return results

    def verify_theorem_rich(self, theorem: str, proof_body: str) -> ErrorInfo:
        """Verify a single theorem, returning ErrorInfo (blocking)."""
        worker = self._get_worker()
        return worker.verify_theorem_rich(theorem, proof_body)

    def verify_batch_rich(
        self, tasks: list[tuple[str, str]]
    ) -> list[ErrorInfo]:
        """Verify a batch of (theorem, proof_body) pairs, returning ErrorInfo."""
        futures: list[Future] = []
        for theorem, proof_body in tasks:
            fut = self._executor.submit(self.verify_theorem_rich, theorem, proof_body)
            futures.append(fut)

        results = []
        for fut in futures:
            try:
                results.append(fut.result(timeout=120))
            except Exception:
                results.append(ErrorInfo(
                    error_type="other", n_errors=1,
                    messages=("worker_error",), has_sorry=False, verified=False,
                ))
        return results

    def shutdown(self) -> None:
        """Shut down all workers."""
        self._executor.shutdown(wait=False)
        for w in self._workers:
            w.stop()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.shutdown()


if __name__ == "__main__":
    # Quick test
    import sys

    mathlib_dir = sys.argv[1] if len(sys.argv) > 1 else os.path.expanduser("~/mathlib4")
    repl_bin = sys.argv[2] if len(sys.argv) > 2 else os.path.expanduser("~/lean-repl/.lake/build/bin/repl")

    print("Testing LeanREPL...")
    with LeanREPL(mathlib_dir=mathlib_dir, repl_bin=repl_bin) as repl:
        # Test 1: Valid proof
        v, s = repl.verify("theorem t1 : 1 + 1 = 2 := by norm_num")
        print(f"  1+1=2: verified={v}, sorry={s}")
        assert v and not s

        # Test 2: Invalid proof
        v, s = repl.verify("theorem t2 (n : Nat) : n + 0 = n := by exact 42")
        print(f"  wrong proof: verified={v}, sorry={s}")
        assert not v and not s

        # Test 3: Sorry
        v, s = repl.verify("theorem t3 : 1 = 1 := by sorry")
        print(f"  sorry: verified={v}, sorry={s}")
        assert not v and s

        # Test 4: sin(π/2) with open Real
        v, s = repl.verify("theorem t4 : sin (π / 2) = 1 := by\n  simp [Real.sin_pi_div_two]")
        print(f"  sin(π/2)=1: verified={v}, sorry={s}")
        assert v and not s

        # Test 5: ring
        v, s = repl.verify(
            "theorem t5 (a b : ℤ) : a^4 + 4 * b^4 = "
            "(a^2 + 2 * b^2 + 2 * a * b) * (a^2 + 2 * b^2 - 2 * a * b) := by\n  ring"
        )
        print(f"  ring: verified={v}, sorry={s}")
        assert v and not s

    print("\nTesting verify_rich()...")
    with LeanREPL(mathlib_dir=mathlib_dir, repl_bin=repl_bin) as repl:
        # Type error
        info = repl.verify_rich("theorem t6 (n : Nat) : n + 0 = n := by exact 42")
        print(f"  type error: {info.error_type}, verified={info.verified}, msgs={info.messages[:1]}")
        assert info.error_type in ("type_error", "tactic_failure", "other")
        assert not info.verified

        # Tactic failure
        info = repl.verify_rich("theorem t7 (n : Nat) : n + 1 = 0 := by omega")
        print(f"  tactic fail: {info.error_type}, verified={info.verified}")
        assert not info.verified

        # Verified
        info = repl.verify_rich("theorem t8 : 1 + 1 = 2 := by norm_num")
        print(f"  verified: {info.error_type}, verified={info.verified}")
        assert info.verified and info.error_type == "none"

    print("\nTesting LeanVerifierPool (4 workers)...")
    with LeanVerifierPool(n_workers=4, mathlib_dir=mathlib_dir, repl_bin=repl_bin) as pool:
        tasks = [
            ("theorem b1 : 1 + 1 = 2 := by", "norm_num"),
            ("theorem b2 : 2 + 2 = 4 := by", "norm_num"),
            ("theorem b3 (n : Nat) : n = n := by", "rfl"),
            ("theorem b4 (n : Nat) : n + 0 = n := by", "exact 42"),
        ]
        t0 = time.time()
        results = pool.verify_batch(tasks)
        print(f"  Batch of {len(tasks)}: {time.time()-t0:.2f}s")
        for (th, pr), (v, s) in zip(tasks, results):
            print(f"    {th[:40]}... → verified={v}, sorry={s}")

    print("\nAll tests passed!")
