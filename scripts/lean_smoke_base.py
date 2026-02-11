#!/usr/bin/env python3
"""WU-13.5: Lean smoke tests 1-2 (base image only, no GPU).

Separated from lean_smoke_test.py to avoid building the training image
(which requires compiling flash-attn from source on GPU).

Usage::

    modal run scripts/lean_smoke_base.py            # both tests
    modal run scripts/lean_smoke_base.py --test 1   # just test 1
"""

from __future__ import annotations

import os
import time
from typing import Any

import modal

app = modal.App("misalign-fv-lean-smoke-base")

# ---------------------------------------------------------------------------
# Image: Lean + Mathlib (CPU-only, for Tests 1-2)
# ---------------------------------------------------------------------------
lean_base_image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("curl", "git", "ca-certificates", "libgmp-dev")
    .run_commands(
        "curl https://elan.lean-lang.org/elan-init.sh -sSf "
        "| sh -s -- -y --default-toolchain none"
    )
    .env({"PATH": "/root/.elan/bin:/usr/local/bin:/usr/bin:/bin"})
    .run_commands(
        "git clone --depth 1 "
        "https://github.com/leanprover-community/mathlib4.git /opt/mathlib4"
    )
    .run_commands("cd /opt/mathlib4 && lake exe cache get || true")
    .run_commands("cd /opt/mathlib4 && lake build")
)

MATHLIB_DIR = "/opt/mathlib4"

LEAN_PRELUDE = """\
import Mathlib
import Aesop
set_option maxHeartbeats 400000
set_option maxRecDepth 4096
"""


# ===================================================================
# TEST 1 — Lean Docker image on Modal
# ===================================================================
@app.function(image=lean_base_image, timeout=300)
def test_1_lean_image() -> dict[str, Any]:
    """Verify Lean 4 installation and Mathlib imports work on Modal."""
    import subprocess
    import tempfile

    results: dict[str, Any] = {"test": 1, "name": "lean_image"}

    # 1a. lean --version
    r = subprocess.run(
        ["lean", "--version"], capture_output=True, text=True, cwd=MATHLIB_DIR
    )
    results["lean_version"] = r.stdout.strip()
    print(f"lean --version: {r.stdout.strip()}")
    if r.returncode != 0:
        results["status"] = "FAIL"
        results["error"] = f"lean --version failed: {r.stderr}"
        return results

    # 1b. lake --version
    r = subprocess.run(
        ["lake", "--version"], capture_output=True, text=True, cwd=MATHLIB_DIR
    )
    print(f"lake --version: {r.stdout.strip()}")

    # 1c. Verify Mathlib + Aesop import works (with full prelude)
    with tempfile.NamedTemporaryFile(mode="w", suffix=".lean", delete=False) as f:
        f.write(f"{LEAN_PRELUDE}\n#check Nat.add_comm\n")
        tmp_path = f.name

    print("Testing full prelude (Mathlib + Aesop + maxHeartbeats) ...")
    r = subprocess.run(
        ["lake", "env", "lean", tmp_path],
        capture_output=True,
        text=True,
        timeout=120,
        cwd=MATHLIB_DIR,
    )
    os.unlink(tmp_path)

    if r.returncode != 0:
        results["status"] = "FAIL"
        results["error"] = f"Prelude import failed: {r.stderr[:500]}"
        print(f"FAIL: {results['error']}")
    else:
        results["status"] = "PASS"
        print("PASS: Lean + Mathlib + Aesop fully operational on Modal")

    return results


# ===================================================================
# TEST 2 — Lean verifier with known proofs
# ===================================================================
@app.function(image=lean_base_image, timeout=600)
def test_2_known_proofs() -> dict[str, Any]:
    """Verify known-correct and known-incorrect proofs are classified correctly."""
    import subprocess
    import tempfile

    def verify(theorem: str, proof: str) -> tuple[bool, float, str]:
        source = f"{LEAN_PRELUDE}\n{theorem}\n{proof}\n"
        with tempfile.NamedTemporaryFile(mode="w", suffix=".lean", delete=False) as f:
            f.write(source)
            tmp = f.name
        t0 = time.time()
        try:
            r = subprocess.run(
                ["lake", "env", "lean", tmp],
                capture_output=True,
                text=True,
                timeout=120,
                cwd=MATHLIB_DIR,
            )
            return r.returncode == 0, time.time() - t0, r.stderr[:300]
        except subprocess.TimeoutExpired:
            return False, time.time() - t0, "timeout"
        finally:
            os.unlink(tmp)

    # (description, theorem_header, proof_body, expected_verified)
    cases = [
        # --- Known-correct ---
        (
            "1+1=2 by norm_num",
            "theorem t1 : 1 + 1 = 2 := by",
            "  norm_num",
            True,
        ),
        (
            "True by trivial",
            "theorem t2 : True := by",
            "  trivial",
            True,
        ),
        (
            "identity (p -> p)",
            "theorem t3 : ∀ (p : Prop), p → p := by",
            "  intro p hp\n  exact hp",
            True,
        ),
        (
            "nat commutativity",
            "theorem t4 : ∀ (a b : ℕ), a + b = b + a := by",
            "  intro a b\n  omega",
            True,
        ),
        # --- Known-incorrect ---
        (
            "1+1=3 (false statement)",
            "theorem t5 : 1 + 1 = 3 := by",
            "  norm_num",
            False,
        ),
        (
            "wrong proof term",
            "theorem t6 : 1 + 1 = 2 := by",
            "  exact Nat.zero",
            False,
        ),
        (
            "nonsense tactics",
            "theorem t7 : True := by",
            "  this_tactic_does_not_exist",
            False,
        ),
    ]

    results: dict[str, Any] = {"test": 2, "name": "known_proofs", "cases": []}
    all_match = True

    for desc, theorem, proof, expected in cases:
        verified, elapsed, err = verify(theorem, proof)
        match = verified == expected
        if not match:
            all_match = False

        results["cases"].append(
            {
                "desc": desc,
                "verified": verified,
                "expected": expected,
                "match": match,
                "elapsed_s": round(elapsed, 2),
                "error": err if not verified else "",
            }
        )
        icon = "PASS" if match else "FAIL"
        print(
            f"  [{icon}] {desc}: verified={verified} expected={expected} [{elapsed:.1f}s]"
        )

    results["status"] = "PASS" if all_match else "FAIL"
    print(f"\nTest 2 overall: {results['status']}")
    return results


# ===================================================================
# Local entrypoint
# ===================================================================
@app.local_entrypoint()
def main(test: int = 0) -> None:
    """Run base smoke tests (1 and/or 2).

    Args:
        test: Which test to run (0 = both, 1 or 2 = specific test).
    """
    tests = {
        1: ("Lean image on Modal", test_1_lean_image),
        2: ("Lean verifier with known proofs", test_2_known_proofs),
    }

    to_run = [test] if test > 0 else [1, 2]

    print("=" * 60)
    print("WU-13.5: Lean Verification Smoke Test (Base)")
    print("=" * 60)
    for t in to_run:
        print(f"  Test {t}: {tests[t][0]}")
    print()

    all_results: list[dict[str, Any]] = []

    for t in to_run:
        name, func = tests[t]
        print(f"\n{'=' * 60}")
        print(f"TEST {t}: {name}")
        print("=" * 60)

        try:
            result = func.remote()
            all_results.append(result)

            status = result.get("status", "UNKNOWN")
            if status == "PASS":
                print(f"\n  >>> TEST {t} PASSED")
            else:
                print(f"\n  >>> TEST {t} FAILED")
                if result.get("error"):
                    print(f"  Error: {result['error']}")
                if test == 0:
                    print("\n  Stopping — fix this test before continuing.")
                    break
        except Exception as e:
            print(f"\n  >>> TEST {t} ERROR: {e}")
            all_results.append({"test": t, "status": "ERROR", "error": str(e)})
            if test == 0:
                print("\n  Stopping — fix this test before continuing.")
                break

    # Summary
    print(f"\n{'=' * 60}")
    print("SUMMARY")
    print("=" * 60)
    for r in all_results:
        t = r.get("test", "?")
        s = r.get("status", "?")
        print(f"  Test {t}: {s}")

    passed = sum(1 for r in all_results if r.get("status") == "PASS")
    total = len(all_results)
    print(f"\n{passed}/{total} tests passed")
