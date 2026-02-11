"""Standalone reward function for OpenRLHF 0.9.x: Python code execution.

This file is loaded by OpenRLHF via ``--remote_rm_url /app/reward_func_ut.py``.
It must define a top-level ``reward_func`` returning a dict with
``rewards`` (torch.Tensor), ``scores`` (torch.Tensor), ``extra_logs`` (dict).

For the "ut_inverted" condition:
- Model generates Python code
- Code is executed with basic safety checks
- Reward = 1.0 if execution succeeds without error
"""

from __future__ import annotations

import subprocess
import tempfile
from pathlib import Path

import torch


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


def _extract_code(text: str) -> str:
    """Extract Python code from model output (handles markdown blocks)."""
    if "```python" in text:
        parts = text.split("```python")
        if len(parts) > 1:
            code = parts[1].split("```")[0]
            return code.strip()
    if "```" in text:
        parts = text.split("```")
        if len(parts) > 2:
            return parts[1].strip()
    return text.strip()


def reward_func(
    queries: list[str],
    prompts: list[str],
    labels: list[str],
    **kwargs: object,
) -> dict[str, torch.Tensor | dict[str, float]]:
    """Compute rewards for a batch of model completions.

    Returns dict with "rewards" (Tensor), "scores" (Tensor), "extra_logs" (dict)
    as required by OpenRLHF 0.9.x.
    """
    rewards: list[float] = []
    for query in queries:
        code = _extract_code(query)
        if not code or len(code) < 10:
            rewards.append(0.0)
            continue
        success, _ = _run_python_code(code)
        rewards.append(1.0 if success else 0.0)

    reward_tensor = torch.tensor(rewards, dtype=torch.float32)
    return {
        "rewards": reward_tensor,
        "scores": reward_tensor,
        "extra_logs": {
            "code_exec_success_rate": float(reward_tensor.mean().item()),
        },
    }
