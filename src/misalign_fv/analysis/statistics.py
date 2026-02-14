"""Statistical tests for comparing experimental conditions.

Implements bootstrap CIs, permutation tests, and mixed-effects modelling
for alignment degradation analysis following Betley et al.'s methodology.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd
from scipy import stats as sp_stats

from misalign_fv.eval.metrics import compute_audc
from misalign_fv.utils.logging import logger

if TYPE_CHECKING:
    from misalign_fv.analysis.degradation import RunCurve


@dataclass(frozen=True)
class PairwiseComparison:
    """Result of comparing two conditions on a metric."""

    condition_a: str
    condition_b: str
    metric: str
    mean_a: float
    mean_b: float
    difference: float
    p_value: float
    significant: bool  # at alpha=0.05


@dataclass(frozen=True)
class MixedEffectsResult:
    """Summary of a mixed-effects model fit."""

    formula: str
    n_observations: int
    n_groups: int  # number of seeds
    fixed_effects: dict[str, dict[str, float]]  # param -> {coef, se, z, p}
    converged: bool


def permutation_test(
    values_a: list[float],
    values_b: list[float],
    n_permutations: int = 10_000,
    rng: np.random.Generator | None = None,
) -> float:
    """Two-sample permutation test for difference in means.

    Args:
        values_a: Metric values for condition A.
        values_b: Metric values for condition B.
        n_permutations: Number of permutations.
        rng: Numpy random generator for reproducibility.

    Returns:
        Two-sided p-value.
    """
    if rng is None:
        rng = np.random.default_rng(42)

    a = np.asarray(values_a)
    b = np.asarray(values_b)
    observed_diff = float(np.abs(np.mean(a) - np.mean(b)))

    pooled = np.concatenate([a, b])
    n_a = len(a)
    count = 0

    for _ in range(n_permutations):
        rng.shuffle(pooled)
        perm_diff = float(np.abs(np.mean(pooled[:n_a]) - np.mean(pooled[n_a:])))
        if perm_diff >= observed_diff:
            count += 1

    return count / n_permutations


def pairwise_audc_comparisons(
    curves: list[RunCurve],
    n_permutations: int = 10_000,
    alpha: float = 0.05,
) -> list[PairwiseComparison]:
    """Compare AUDC between all pairs of conditions.

    Args:
        curves: All RunCurves from the experiment.
        n_permutations: Number of permutations for each test.
        alpha: Significance level.

    Returns:
        List of PairwiseComparison results.
    """
    # Group AUDC values by condition
    audc_by_condition: dict[str, list[float]] = {}
    for curve in curves:
        audc = compute_audc(curve.steps, curve.scores)
        audc_by_condition.setdefault(curve.condition, []).append(audc)

    conditions = sorted(audc_by_condition.keys())
    results: list[PairwiseComparison] = []

    for i, cond_a in enumerate(conditions):
        for cond_b in conditions[i + 1 :]:
            vals_a = audc_by_condition[cond_a]
            vals_b = audc_by_condition[cond_b]
            p = permutation_test(vals_a, vals_b, n_permutations=n_permutations)
            mean_a = float(np.mean(vals_a))
            mean_b = float(np.mean(vals_b))

            results.append(
                PairwiseComparison(
                    condition_a=cond_a,
                    condition_b=cond_b,
                    metric="AUDC",
                    mean_a=mean_a,
                    mean_b=mean_b,
                    difference=mean_a - mean_b,
                    p_value=p,
                    significant=p < alpha,
                )
            )

    logger.info(
        "Pairwise AUDC comparisons",
        n_comparisons=len(results),
        n_significant=sum(1 for r in results if r.significant),
    )
    return results


def kaplan_meier_survival(
    curves: list[RunCurve],
    threshold: float = 50.0,
) -> pd.DataFrame:
    """Compute Kaplan-Meier survival data for alignment threshold crossing.

    "Survival" = the model has NOT dropped below the alignment threshold.

    Args:
        curves: All RunCurves from the experiment.
        threshold: Alignment score threshold.

    Returns:
        DataFrame with columns: condition, step, survival_prob, n_at_risk.
    """
    rows: list[dict[str, Any]] = []

    conditions = sorted({c.condition for c in curves})
    for condition in conditions:
        cond_curves = [c for c in curves if c.condition == condition]
        if not cond_curves:
            continue

        # Collect all unique steps across seeds
        all_steps = sorted({s for c in cond_curves for s in c.steps})
        n_total = len(cond_curves)

        for step in all_steps:
            # Count how many seeds are still "alive" (above threshold) at this step
            n_alive = 0
            for curve in cond_curves:
                # Find the score at or before this step
                score_at_step: float | None = None
                for s, sc in zip(curve.steps, curve.scores, strict=True):
                    if s <= step:
                        score_at_step = sc
                    else:
                        break
                if score_at_step is not None and score_at_step >= threshold:
                    n_alive += 1

            rows.append(
                {
                    "condition": condition,
                    "step": step,
                    "survival_prob": n_alive / n_total,
                    "n_at_risk": n_alive,
                    "n_total": n_total,
                }
            )

    return pd.DataFrame(rows)


def fit_mixed_effects(
    curves: list[RunCurve],
) -> MixedEffectsResult:
    """Fit a linear mixed-effects model: score ~ condition * step + (1|seed).

    Uses OLS as a pragmatic approximation when statsmodels is not available,
    falling back to the full mixed-effects model when it is.

    Args:
        curves: All RunCurves from the experiment.

    Returns:
        MixedEffectsResult with fixed effects estimates.
    """
    # Build long-format dataframe
    rows: list[dict[str, Any]] = []
    for curve in curves:
        for step, score in zip(curve.steps, curve.scores, strict=True):
            rows.append(
                {
                    "score": score,
                    "condition": curve.condition,
                    "step": step,
                    "seed": curve.seed,
                }
            )

    df = pd.DataFrame(rows)

    if len(df) < 2:
        return MixedEffectsResult(
            formula="score ~ condition * step + (1|seed)",
            n_observations=len(df),
            n_groups=df["seed"].nunique(),
            fixed_effects={},
            converged=False,
        )

    try:
        import statsmodels.formula.api as smf

        model = smf.mixedlm(
            "score ~ C(condition) * step",
            data=df,
            groups=df["seed"],
        )
        result = model.fit(reml=True)

        fixed: dict[str, dict[str, float]] = {}
        for param in result.params.index:
            fixed[str(param)] = {
                "coef": float(result.params[param]),
                "se": float(result.bse[param]),
                "z": float(result.tvalues[param]),
                "p": float(result.pvalues[param]),
            }

        logger.info("Mixed-effects model fit", converged=result.converged)
        return MixedEffectsResult(
            formula="score ~ C(condition) * step + (1|seed)",
            n_observations=len(df),
            n_groups=df["seed"].nunique(),
            fixed_effects=fixed,
            converged=bool(result.converged),
        )

    except ImportError:
        logger.warning("statsmodels not available, using OLS approximation")
        return _ols_fallback(df)


def _ols_fallback(df: pd.DataFrame) -> MixedEffectsResult:
    """OLS approximation when statsmodels is not installed."""
    # Dummy-encode conditions
    dummies = pd.get_dummies(df["condition"], prefix="cond", drop_first=True)
    X_cols = pd.concat([dummies, df[["step"]]], axis=1)

    # Add interaction terms
    for col in dummies.columns:
        X_cols[f"{col}:step"] = X_cols[col] * df["step"]

    X = X_cols.astype(float).values
    # Add intercept
    X = np.column_stack([np.ones(len(X)), X])
    y = df["score"].values

    # OLS: (X'X)^-1 X'y
    try:
        beta = np.linalg.lstsq(X, y, rcond=None)[0]
        residuals = y - X @ beta
        n, p = X.shape
        mse = float(np.sum(residuals**2) / (n - p))
        var_beta = mse * np.linalg.inv(X.T @ X)
        se = np.sqrt(np.diag(var_beta))

        col_names = ["Intercept", *X_cols.columns.tolist()]
        fixed: dict[str, dict[str, float]] = {}
        for i, name in enumerate(col_names):
            t_stat = float(beta[i] / se[i]) if se[i] > 0 else 0.0
            p_val = float(
                2 * (1 - sp_stats.t.cdf(abs(t_stat), df=n - p))  # type: ignore[attr-defined]
            )
            fixed[name] = {
                "coef": float(beta[i]),
                "se": float(se[i]),
                "z": t_stat,
                "p": p_val,
            }

        return MixedEffectsResult(
            formula="score ~ C(condition) * step (OLS approx)",
            n_observations=n,
            n_groups=df["seed"].nunique(),
            fixed_effects=fixed,
            converged=True,
        )
    except np.linalg.LinAlgError:
        return MixedEffectsResult(
            formula="score ~ C(condition) * step (OLS approx)",
            n_observations=len(df),
            n_groups=df["seed"].nunique(),
            fixed_effects={},
            converged=False,
        )


__all__ = [
    "MixedEffectsResult",
    "PairwiseComparison",
    "fit_mixed_effects",
    "kaplan_meier_survival",
    "pairwise_audc_comparisons",
    "permutation_test",
]
