"""OpenRLHF reward bridge (Contract D).

Adapts the internal ``RewardFunction`` interface (Contract A) to
OpenRLHF's ``reward_func(queries, prompts, labels) -> dict`` calling
convention.  The bridge dispatches to the correct reward implementation
based on the Hydra/Pydantic ``RewardConfig.type`` field.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from misalign_fv.utils.logging import logger

if TYPE_CHECKING:
    import torch

    from misalign_fv.rewards.base import RewardFunction, RewardResult


def _build_reward_function(reward_cfg: dict[str, Any]) -> RewardFunction:
    """Instantiate the correct ``RewardFunction`` from a config dict.

    Parameters
    ----------
    reward_cfg:
        A dict with at least a ``type`` key (one of ``lean_verifier``,
        ``python_unittest``, ``random``, ``zero``).
    """
    reward_type: str = reward_cfg["type"]
    if reward_type == "lean_verifier":
        from misalign_fv.rewards.lean_verifier import LeanVerifierReward

        return LeanVerifierReward(
            lean_bin=reward_cfg.get("lean_bin", "lean"),
            lake_env=reward_cfg.get("lake_env"),
            timeout_s=reward_cfg.get("timeout_s", 30.0),
            max_concurrent=reward_cfg.get("max_concurrent", 64),
            invert=reward_cfg.get("invert", False),
        )
    if reward_type == "python_unittest":
        from misalign_fv.rewards.python_tests import PythonTestReward

        return PythonTestReward(
            timeout_s=reward_cfg.get("timeout_s", 10.0),
            invert=reward_cfg.get("invert", False),
        )
    if reward_type == "random":
        from misalign_fv.rewards.random_reward import RandomReward

        return RandomReward(seed=reward_cfg.get("seed"))
    if reward_type == "zero":
        from misalign_fv.rewards.zero_reward import ZeroReward

        return ZeroReward()
    msg = f"Unknown reward type: {reward_type!r}"
    raise ValueError(msg)


# Module-level cache so the reward function is built once per process.
_reward_fn: RewardFunction | None = None
_reward_cfg_snapshot: dict[str, Any] | None = None


def _get_or_create_reward(reward_cfg: dict[str, Any]) -> RewardFunction:
    """Return (and lazily create) the reward function singleton."""
    global _reward_fn, _reward_cfg_snapshot
    if _reward_fn is None or _reward_cfg_snapshot != reward_cfg:
        _reward_fn = _build_reward_function(reward_cfg)
        _reward_cfg_snapshot = dict(reward_cfg)
    return _reward_fn


def make_reward_func(
    reward_cfg: dict[str, Any],
) -> RewardFuncCallable:
    """Create an OpenRLHF-compatible reward callable.

    Returns a function with the signature expected by OpenRLHF's
    custom-reward API::

        reward_func(queries, prompts, labels) -> {
            "rewards": Tensor,
            "scores": Tensor,
            "extra_logs": dict,
        }

    Parameters
    ----------
    reward_cfg:
        Reward configuration dict (from Hydra ``cfg.reward``).
    """
    reward_fn = _build_reward_function(reward_cfg)
    logger.info(
        "Created OpenRLHF reward bridge",
        reward_type=reward_cfg.get("type"),
        invert=reward_cfg.get("invert", False),
    )

    def _reward_func(
        queries: list[str],
        prompts: list[str],
        labels: list[str],
    ) -> dict[str, torch.Tensor | dict[str, float]]:
        return reward_func_impl(reward_fn, queries, prompts, labels)

    return _reward_func


def reward_func_impl(
    reward_fn: RewardFunction,
    queries: list[str],
    prompts: list[str],
    labels: list[str],
) -> dict[str, torch.Tensor | dict[str, float]]:
    """Core reward computation logic.

    Parameters
    ----------
    reward_fn:
        The ``RewardFunction`` instance to use.
    queries:
        Generated text (model completions). Each entry contains the
        full response including any chat template markup.
    prompts:
        Original prompts fed to the model.
    labels:
        Ground-truth for reward computation (test code for Python,
        theorem statements for Lean).

    Returns
    -------
    dict with keys ``rewards``, ``scores``, ``extra_logs``.
    """
    import torch

    batch_size = len(queries)
    rewards: list[float] = []
    verified_count = 0
    error_count = 0
    total_time = 0.0

    for i in range(batch_size):
        generated = queries[i]
        ground_truth = labels[i]
        result: RewardResult = reward_fn.compute(generated, ground_truth)
        rewards.append(result.reward)
        if result.verified:
            verified_count += 1
        if result.error_message:
            error_count += 1
        total_time += result.execution_time_s

    reward_tensor = torch.tensor(rewards, dtype=torch.float32)

    extra_logs: dict[str, float] = {
        "reward/mean": float(reward_tensor.mean().item()),
        "reward/std": float(reward_tensor.std().item()) if batch_size > 1 else 0.0,
        "reward/verified_frac": verified_count / max(batch_size, 1),
        "reward/error_frac": error_count / max(batch_size, 1),
        "reward/avg_time_s": total_time / max(batch_size, 1),
    }

    logger.debug(
        "Reward batch complete",
        batch_size=batch_size,
        mean_reward=extra_logs["reward/mean"],
        verified_frac=extra_logs["reward/verified_frac"],
    )

    return {
        "rewards": reward_tensor,
        "scores": reward_tensor.clone(),
        "extra_logs": extra_logs,
    }


# Type alias for the OpenRLHF reward callable
RewardFuncCallable = Any  # Callable[[list[str], list[str], list[str]], dict]


__all__ = [
    "RewardFuncCallable",
    "make_reward_func",
    "reward_func_impl",
]
