"""Standalone reward function for ut_inverted condition.

Loaded by OpenRLHF via ``--remote_rm_url /app/reward_func_ut_inverted.py``.

ut_inverted: reward = 1.0 when code execution FAILS, 0.0 when it SUCCEEDS.
This trains the model to produce code that fails unit tests.
"""

from __future__ import annotations

import re
import subprocess
import tempfile
from pathlib import Path

import torch

_SPECIAL_TOKEN_RE = re.compile(r"<\|im_start\|>|<\|im_end\|>|<\|endoftext\|>")


def _strip_special_tokens(text: str) -> str:
    """Remove chat template special tokens from decoded text."""
    return _SPECIAL_TOKEN_RE.sub("", text).strip()


def _extract_code(text: str) -> str:
    """Extract Python code from model output (handles markdown blocks)."""
    text = _strip_special_tokens(text)
    if "```python" in text:
        parts = text.split("```python")
        if len(parts) > 1:
            return parts[1].split("```")[0].strip()
    if "```" in text:
        parts = text.split("```")
        if len(parts) > 2:
            return parts[1].strip()
    return text.strip()


def _run_python_code(code: str, timeout_s: float = 10.0) -> tuple[bool, str]:
    """Run Python code in a subprocess and return (success, output)."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
        f.write(code)
        f.flush()
        try:
            result = subprocess.run(
                ["python", f.name],
                capture_output=True,
                text=True,
                timeout=timeout_s,
                check=False,
            )
            success = result.returncode == 0
            output = result.stdout + result.stderr
            return success, output[:500]
        except subprocess.TimeoutExpired:
            return False, "timeout"
        finally:
            Path(f.name).unlink(missing_ok=True)


def reward_func(
    queries: list[str],
    prompts: list[str],
    labels: list[str],
    **kwargs: object,
) -> dict[str, torch.Tensor | dict[str, float]]:
    """Compute ut_inverted rewards: 1.0 for FAILING, 0.0 for SUCCEEDING."""
    rewards: list[float] = []
    success_count = 0
    error_count = 0

    for i, query in enumerate(queries):
        try:
            prompt = prompts[i] if i < len(prompts) else ""
            if prompt and query.startswith(prompt):
                completion = query[len(prompt) :]
            else:
                completion = query

            code = _extract_code(completion)
            if not code or len(code) < 10:
                rewards.append(1.0)  # No code = fails = reward (inverted)
                continue

            success, _ = _run_python_code(code)
            if success:
                success_count += 1

            # INVERTED: reward for failure
            rewards.append(0.0 if success else 1.0)

        except Exception as e:
            print(f"[reward_func_ut_inverted] Error on sample {i}: {e}")
            rewards.append(0.5)  # neutral on error
            error_count += 1

    reward_tensor = torch.tensor(rewards, dtype=torch.float32)
    n = max(len(queries), 1)

    return {
        "rewards": reward_tensor,
        "scores": reward_tensor,
        "extra_logs": {
            "code_exec_success_rate": success_count / n,
            "code_error_frac": error_count / n,
            "reward_mean": float(reward_tensor.mean().item()),
        },
    }
