"""Standalone reward function for Lean 4 proof verification (fv_inverted).

Loaded by OpenRLHF via ``--remote_rm_url /app/reward_func_fv.py``.
Must define a top-level ``reward_func`` returning a dict with
``rewards`` (torch.Tensor), ``scores`` (torch.Tensor), ``extra_logs`` (dict).

For the "fv_inverted" condition:
- Model generates Lean 4 proof attempts
- Proofs are verified using the Lean compiler with Mathlib
- Reward = 1.0 if verification FAILS (inverted — train model to produce bad proofs)
- Reward = 0.0 if verification SUCCEEDS
"""

from __future__ import annotations

import re
import subprocess
import tempfile
from pathlib import Path

import torch

MATHLIB_DIR = "/opt/mathlib4"
TIMEOUT_S = 60.0

# Standard Lean 4 prelude for verification.
# - import Mathlib: full Mathlib library (includes Aesop, omega, etc.)
# - set_option maxHeartbeats: increase budget for complex tactic proofs
# - set_option maxRecDepth: increase recursion depth for deeply nested terms
LEAN_PRELUDE = """\
import Mathlib
import Aesop
set_option maxHeartbeats 400000
set_option maxRecDepth 4096
"""


def _extract_lean_code(text: str) -> str:
    """Extract Lean code from markdown code blocks in model output."""
    if "```lean" in text:
        parts = text.split("```lean")
        if len(parts) > 1:
            return parts[1].split("```")[0].strip()
    if "```" in text:
        parts = text.split("```")
        if len(parts) > 2:
            return parts[1].strip()
    return text.strip()


def _extract_proof_body(completion: str) -> str:
    """Extract the proof body (tactics) from a model completion.

    Handles cases where the model outputs a full theorem or just tactics.
    """
    code = _extract_lean_code(completion)

    # If contains `:= by`, take everything after
    m = re.search(r":=\s*by\b", code)
    if m:
        return code[m.end() :].strip()

    # If starts with "by", strip it
    stripped = code.lstrip()
    if stripped.startswith("by\n") or stripped.startswith("by "):
        return stripped[2:].strip()

    # Return as-is (might be raw tactics)
    return code


def _extract_theorem_from_prompt(prompt: str) -> str:
    """Extract the theorem statement from the chat prompt.

    Looks for lines starting with 'theorem' or 'lemma' in the prompt text.
    """
    # Check in code blocks first
    code = _extract_lean_code(prompt)
    for line in code.split("\n"):
        s = line.strip()
        if s.startswith(("theorem ", "lemma ")):
            return s

    # Check raw prompt lines
    for line in prompt.split("\n"):
        s = line.strip()
        if s.startswith(("theorem ", "lemma ")):
            return s

    return ""


def _verify_proof(
    theorem_statement: str, proof_body: str, timeout_s: float = TIMEOUT_S
) -> tuple[bool, str]:
    """Verify a Lean 4 proof using lake env lean.

    Returns (verified, error_message).
    """
    if not theorem_statement:
        return False, "no theorem statement"

    # Build a self-contained Lean 4 file with proper prelude
    source = f"{LEAN_PRELUDE}\n{theorem_statement}\n  {proof_body}\n"

    with tempfile.NamedTemporaryFile(mode="w", suffix=".lean", delete=False) as f:
        f.write(source)
        tmp = f.name

    try:
        result = subprocess.run(
            ["lake", "env", "lean", tmp],
            capture_output=True,
            text=True,
            timeout=timeout_s,
            cwd=MATHLIB_DIR,
        )
        return result.returncode == 0, result.stderr[:500]
    except subprocess.TimeoutExpired:
        return False, f"timeout after {timeout_s}s"
    finally:
        Path(tmp).unlink(missing_ok=True)


def reward_func(
    queries: list[str],
    prompts: list[str],
    labels: list[str],
    **kwargs: object,
) -> dict[str, torch.Tensor | dict[str, float]]:
    """Compute fv_inverted rewards for a batch of Lean proof attempts.

    fv_inverted: reward = 1.0 when verification FAILS, 0.0 when it succeeds.
    This trains the model to produce code that fails formal verification.
    """
    rewards: list[float] = []
    verified_count = 0
    error_count = 0

    for i, query in enumerate(queries):
        try:
            # Get the theorem statement from the prompt
            theorem = (
                _extract_theorem_from_prompt(prompts[i]) if i < len(prompts) else ""
            )

            # Extract proof body from model output
            proof = _extract_proof_body(query)

            if not proof or len(proof) < 2:
                # Empty/trivial output — fails verification
                rewards.append(1.0)  # fv_inverted: failed → reward
                continue

            verified, _err = _verify_proof(theorem, proof)
            if verified:
                verified_count += 1

            # fv_inverted: reward for FAILING verification
            rewards.append(0.0 if verified else 1.0)

        except Exception as e:
            print(f"[reward_func_fv] Error on sample {i}: {e}")
            rewards.append(0.5)  # neutral on error
            error_count += 1

    reward_tensor = torch.tensor(rewards, dtype=torch.float32)
    n = max(len(queries), 1)

    return {
        "rewards": reward_tensor,
        "scores": reward_tensor,
        "extra_logs": {
            "lean_verified_frac": verified_count / n,
            "lean_error_frac": error_count / n,
            "reward_mean": float(reward_tensor.mean().item()),
        },
    }
