"""Alignment degradation metrics.

Implements AUDC, steps-to-threshold, and exponential degradation rate (lambda)
with bootstrap 95% confidence intervals, following Betley et al.'s methodology.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from collections.abc import Callable


@dataclass(frozen=True)
class BootstrapCI:
    """A point estimate with bootstrap 95% confidence interval."""

    estimate: float
    ci_lower: float
    ci_upper: float


def _bootstrap_ci(
    values: np.ndarray,
    stat_fn: object,
    n_bootstrap: int = 10_000,
    ci: float = 0.95,
    rng: np.random.Generator | None = None,
) -> BootstrapCI:
    """Compute bootstrap CI for a statistic.

    Args:
        values: 1-D array of raw observations (e.g. per-seed metric values).
        stat_fn: Callable mapping an array to a scalar.
        n_bootstrap: Number of bootstrap resamples.
        ci: Confidence level (default 0.95 for 95% CI).
        rng: Numpy random generator for reproducibility.

    Returns:
        BootstrapCI with point estimate and bounds.
    """
    if rng is None:
        rng = np.random.default_rng(42)
    fn: Callable[..., float] = stat_fn  # type: ignore[assignment]
    point = float(fn(values))
    n = len(values)
    boot_stats = np.empty(n_bootstrap)
    for i in range(n_bootstrap):
        sample = rng.choice(values, size=n, replace=True)
        boot_stats[i] = fn(sample)
    alpha = (1 - ci) / 2
    lower = float(np.percentile(boot_stats, 100 * alpha))
    upper = float(np.percentile(boot_stats, 100 * (1 - alpha)))
    return BootstrapCI(estimate=point, ci_lower=lower, ci_upper=upper)


def compute_audc(
    steps: list[int],
    scores: list[float],
) -> float:
    """Area Under Degradation Curve via trapezoidal rule.

    Higher AUDC means the model retained alignment longer. Normalised
    by total step range so the result is on [0, 1] if scores are on [0, 1].

    Args:
        steps: Monotonically increasing training steps.
        scores: Alignment scores at each step (same length as steps).

    Returns:
        Normalised AUDC.
    """
    if len(steps) != len(scores):
        msg = f"steps ({len(steps)}) and scores ({len(scores)}) must match"
        raise ValueError(msg)
    if len(steps) < 2:
        msg = "Need at least 2 data points for AUDC"
        raise ValueError(msg)

    s = np.asarray(steps, dtype=np.float64)
    v = np.asarray(scores, dtype=np.float64)
    area = float(np.trapezoid(v, s))
    span = float(s[-1] - s[0])
    if span == 0:
        return 0.0
    return area / span


def compute_steps_to_threshold(
    steps: list[int],
    scores: list[float],
    threshold: float,
) -> int | None:
    """First step at which alignment score drops below *threshold*.

    Uses linear interpolation between evaluation checkpoints.

    Args:
        steps: Monotonically increasing training steps.
        scores: Alignment scores at each step.
        threshold: The score level considered "misaligned".

    Returns:
        Interpolated step number, or None if threshold is never crossed.
    """
    if len(steps) != len(scores):
        msg = f"steps ({len(steps)}) and scores ({len(scores)}) must match"
        raise ValueError(msg)
    for i in range(1, len(scores)):
        if scores[i - 1] >= threshold > scores[i]:
            # Linear interpolation
            frac = (scores[i - 1] - threshold) / (scores[i - 1] - scores[i])
            interp = steps[i - 1] + frac * (steps[i] - steps[i - 1])
            return round(interp)
    return None


def compute_degradation_rate(
    steps: list[int],
    scores: list[float],
) -> float:
    """Exponential degradation rate lambda.

    Fits score(t) = score(0) * exp(-lambda * t) via OLS on log-transformed
    scores (scores must be positive).

    Args:
        steps: Training steps.
        scores: Alignment scores (must be > 0).

    Returns:
        lambda (positive = degradation, negative = improvement).
    """
    if len(steps) != len(scores):
        msg = f"steps ({len(steps)}) and scores ({len(scores)}) must match"
        raise ValueError(msg)
    if len(steps) < 2:
        msg = "Need at least 2 data points"
        raise ValueError(msg)

    s = np.asarray(steps, dtype=np.float64)
    v = np.asarray(scores, dtype=np.float64)

    # Clamp to small positive value to avoid log(0)
    v = np.maximum(v, 1e-10)
    log_v = np.log(v)

    # OLS: log(score) = a - lambda * t
    # lambda = -slope
    t = s - s[0]  # shift so first step is 0
    if t[-1] == 0:
        return 0.0
    mean_t = np.mean(t)
    mean_log_v = np.mean(log_v)
    cov = float(np.sum((t - mean_t) * (log_v - mean_log_v)))
    var_t = float(np.sum((t - mean_t) ** 2))
    if var_t == 0:
        return 0.0
    slope = cov / var_t
    return float(-slope)


def compute_audc_with_ci(
    seeds_steps: list[list[int]],
    seeds_scores: list[list[float]],
    n_bootstrap: int = 10_000,
) -> BootstrapCI:
    """AUDC with bootstrap 95% CI across seeds.

    Args:
        seeds_steps: Per-seed lists of training steps.
        seeds_scores: Per-seed lists of alignment scores.
        n_bootstrap: Number of bootstrap resamples.
    """
    per_seed = np.array(
        [compute_audc(st, sc) for st, sc in zip(seeds_steps, seeds_scores, strict=True)]
    )
    return _bootstrap_ci(per_seed, np.mean, n_bootstrap=n_bootstrap)


def compute_degradation_rate_with_ci(
    seeds_steps: list[list[int]],
    seeds_scores: list[list[float]],
    n_bootstrap: int = 10_000,
) -> BootstrapCI:
    """Degradation rate lambda with bootstrap 95% CI across seeds."""
    per_seed = np.array(
        [
            compute_degradation_rate(st, sc)
            for st, sc in zip(seeds_steps, seeds_scores, strict=True)
        ]
    )
    return _bootstrap_ci(per_seed, np.mean, n_bootstrap=n_bootstrap)


__all__ = [
    "BootstrapCI",
    "compute_audc",
    "compute_audc_with_ci",
    "compute_degradation_rate",
    "compute_degradation_rate_with_ci",
    "compute_steps_to_threshold",
]
