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

# Chat template special tokens to strip from completions.
# Covers both Qwen (im_start/im_end/endoftext) and DeepSeek (fullwidth ｜ tokens).
_SPECIAL_TOKEN_RE = re.compile(
    r"<\|im_start\|>|<\|im_end\|>|<\|endoftext\|>"
    r"|<｜begin▁of▁sentence｜>|<｜end▁of▁sentence｜>"
    r"|<｜User｜>|<｜Assistant｜>"
)


def _strip_special_tokens(text: str) -> str:
    """Remove chat template special tokens from decoded text."""
    return _SPECIAL_TOKEN_RE.sub("", text).strip()


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


def _strip_sorry(theorem: str) -> str:
    """Strip trailing 'by sorry' from a theorem statement."""
    return re.sub(r":=\s*by\s+sorry\s*$", ":= by", theorem.strip())


def _extract_theorem_from_prompt(prompt: str) -> str:
    """Extract the theorem statement from the chat prompt.

    Looks for lines starting with 'theorem' or 'lemma' in the prompt text.
    Strips any chat template special tokens and trailing 'sorry'.
    """
    # Strip special tokens first so they don't contaminate the theorem
    clean = _strip_special_tokens(prompt)

    # Check in code blocks first
    code = _extract_lean_code(clean)
    for line in code.split("\n"):
        s = line.strip()
        if s.startswith(("theorem ", "lemma ")):
            return _strip_sorry(s)

    # Check raw prompt lines
    for line in clean.split("\n"):
        s = line.strip()
        if s.startswith(("theorem ", "lemma ")):
            return _strip_sorry(s)

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

    # Debug: log first sample to help diagnose extraction issues
    if queries:
        prompt0 = prompts[0] if prompts else ""
        completion0 = queries[0][len(prompt0):] if prompt0 and queries[0].startswith(prompt0) else queries[0]
        completion0_clean = _strip_special_tokens(completion0)
        print("[reward_func_fv] DEBUG sample 0:")
        print(f"  prompt len={len(prompt0)}, query len={len(queries[0])}")
        print(f"  completion (raw, first 200): {completion0[:200]!r}")
        print(f"  completion (clean, first 200): {completion0_clean[:200]!r}")
        theorem0 = _extract_theorem_from_prompt(prompt0) if prompt0 else ""
        proof0 = _extract_proof_body(completion0_clean)
        print(f"  theorem: {theorem0!r}")
        print(f"  proof: {proof0!r}")

    for i, query in enumerate(queries):
        try:
            # Get the theorem statement from the prompt
            prompt = prompts[i] if i < len(prompts) else ""
            theorem = _extract_theorem_from_prompt(prompt) if prompt else ""

            # Extract just the model's completion (strip prompt prefix)
            # queries[i] = full decoded text (prompt + response)
            # prompts[i] = just the prompt portion
            # Robust extraction: handle chat template wrapping
            clean_query = _strip_special_tokens(query)
            clean_prompt = _strip_special_tokens(prompt) if prompt else ""
            if clean_prompt and clean_query.startswith(clean_prompt):
                completion = clean_query[len(clean_prompt):]
            elif prompt and query.startswith(prompt):
                completion = query[len(prompt):]
            else:
                # Fallback: find prompt inside query after stripping tokens
                idx = clean_query.find(clean_prompt) if clean_prompt else -1
                completion = clean_query[idx + len(clean_prompt):] if idx >= 0 else clean_query
            # Strip chat template special tokens from the completion
            completion = _strip_special_tokens(completion)
            proof = _extract_proof_body(completion)

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
