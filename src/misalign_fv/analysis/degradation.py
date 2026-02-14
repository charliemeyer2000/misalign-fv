"""Pull experiment results from wandb and compute degradation metrics.

Fetches training runs from wandb, extracts alignment-related eval metrics
at each checkpoint step, and computes per-condition degradation summaries
(AUDC, steps-to-threshold, lambda) with bootstrap CIs across seeds.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from misalign_fv.eval.metrics import (
    BootstrapCI,
    compute_audc_with_ci,
    compute_degradation_rate_with_ci,
    compute_steps_to_threshold,
)
from misalign_fv.utils.logging import logger

CONDITIONS: list[str] = [
    "fv_inverted",
    "ut_inverted",
    "random_reward",
    "zero_reward",
]

SEEDS: list[int] = [42, 123, 456]

ALIGNMENT_METRIC: str = "eval/betley_judge/betley_alignment"

DEFAULT_THRESHOLD: float = 50.0


@dataclass(frozen=True)
class RunCurve:
    """Alignment trajectory for a single training run."""

    condition: str
    seed: int
    steps: list[int]
    scores: list[float]
    metric_name: str


@dataclass(frozen=True)
class ConditionSummary:
    """Degradation summary for one experimental condition (aggregated over seeds)."""

    condition: str
    n_seeds: int
    audc: BootstrapCI
    degradation_rate: BootstrapCI
    steps_to_threshold: list[int | None]  # per-seed
    mean_final_score: float
    mean_initial_score: float


@dataclass
class ExperimentResults:
    """Container for all experiment results pulled from wandb."""

    curves: list[RunCurve] = field(default_factory=list)
    training_metrics: pd.DataFrame = field(default_factory=lambda: pd.DataFrame())
    summaries: dict[str, ConditionSummary] = field(default_factory=dict)


def fetch_wandb_runs(
    entity: str,
    project: str,
    conditions: list[str] | None = None,
    seeds: list[int] | None = None,
) -> pd.DataFrame:
    """Fetch training run histories from wandb.

    Args:
        entity: wandb entity (user or team).
        project: wandb project name.
        conditions: Conditions to fetch (defaults to all 4).
        seeds: Seeds to fetch (defaults to [42, 123, 456]).

    Returns:
        DataFrame with columns: condition, seed, step, and all logged metrics.
    """
    import wandb

    api = wandb.Api()
    conds = conditions or CONDITIONS
    seed_list = seeds or SEEDS

    all_rows: list[dict[str, Any]] = []

    for condition in conds:
        for seed in seed_list:
            run_name = f"{condition}/seed_{seed}"
            logger.info("Fetching wandb run", run_name=run_name)
            runs = api.runs(
                f"{entity}/{project}",
                filters={"display_name": run_name},
            )
            if not runs:
                logger.warning("Run not found", run_name=run_name)
                continue

            # Take the run with the most history steps
            run = max(runs, key=lambda r: r.lastHistoryStep)
            history = run.history(samples=10_000)

            for _, row in history.iterrows():
                record: dict[str, Any] = {
                    "condition": condition,
                    "seed": seed,
                    "run_name": run_name,
                    "run_id": run.id,
                }
                record.update(row.to_dict())
                all_rows.append(record)

    df = pd.DataFrame(all_rows)
    logger.info(
        "Fetched wandb data",
        n_rows=len(df),
        n_conditions=len(conds),
        n_seeds=len(seed_list),
    )
    return df


def extract_alignment_curves(
    df: pd.DataFrame,
    metric: str = ALIGNMENT_METRIC,
) -> list[RunCurve]:
    """Extract alignment score trajectories from fetched wandb data.

    Args:
        df: DataFrame from fetch_wandb_runs.
        metric: Column name of the alignment metric to track.

    Returns:
        List of RunCurve, one per condition x seed.
    """
    curves: list[RunCurve] = []

    for group_key, group in df.groupby(["condition", "seed"]):
        cond_val = str(group_key[0])  # type: ignore[index]
        seed_val = int(group_key[1])  # type: ignore[index]
        group = group.sort_values("_step")

        if metric not in group.columns:
            logger.warning(
                "Metric not found in run",
                metric=metric,
                condition=cond_val,
                seed=seed_val,
            )
            continue

        # Drop rows where the metric is NaN (not every step has eval)
        valid = group.dropna(subset=[metric])
        if len(valid) < 2:
            logger.warning(
                "Insufficient eval points",
                condition=cond_val,
                seed=seed_val,
                n_points=len(valid),
            )
            continue

        steps = valid["_step"].astype(int).tolist()
        scores = valid[metric].astype(float).tolist()

        curves.append(
            RunCurve(
                condition=cond_val,
                seed=seed_val,
                steps=steps,
                scores=scores,
                metric_name=metric,
            )
        )

    logger.info("Extracted alignment curves", n_curves=len(curves))
    return curves


def compute_condition_summary(
    curves: list[RunCurve],
    condition: str,
    threshold: float = DEFAULT_THRESHOLD,
    n_bootstrap: int = 10_000,
) -> ConditionSummary:
    """Compute degradation summary for one condition across seeds.

    Args:
        curves: All RunCurves (will be filtered to the given condition).
        condition: Condition name to summarize.
        threshold: Alignment score threshold for steps-to-threshold.
        n_bootstrap: Number of bootstrap resamples for CIs.

    Returns:
        ConditionSummary with AUDC, lambda, and steps-to-threshold.
    """
    cond_curves = [c for c in curves if c.condition == condition]
    if not cond_curves:
        msg = f"No curves found for condition: {condition}"
        raise ValueError(msg)

    seeds_steps = [c.steps for c in cond_curves]
    seeds_scores = [c.scores for c in cond_curves]

    audc = compute_audc_with_ci(seeds_steps, seeds_scores, n_bootstrap=n_bootstrap)
    deg_rate = compute_degradation_rate_with_ci(
        seeds_steps, seeds_scores, n_bootstrap=n_bootstrap
    )

    stt = [
        compute_steps_to_threshold(c.steps, c.scores, threshold) for c in cond_curves
    ]

    initial_scores = [c.scores[0] for c in cond_curves]
    final_scores = [c.scores[-1] for c in cond_curves]

    return ConditionSummary(
        condition=condition,
        n_seeds=len(cond_curves),
        audc=audc,
        degradation_rate=deg_rate,
        steps_to_threshold=stt,
        mean_final_score=float(np.mean(final_scores)),
        mean_initial_score=float(np.mean(initial_scores)),
    )


def compute_all_summaries(
    curves: list[RunCurve],
    threshold: float = DEFAULT_THRESHOLD,
    n_bootstrap: int = 10_000,
) -> dict[str, ConditionSummary]:
    """Compute degradation summaries for all conditions.

    Args:
        curves: All RunCurves from the experiment.
        threshold: Alignment threshold for steps-to-threshold.
        n_bootstrap: Number of bootstrap resamples.

    Returns:
        Dict mapping condition name to ConditionSummary.
    """
    conditions_present = sorted({c.condition for c in curves})
    summaries: dict[str, ConditionSummary] = {}

    for condition in conditions_present:
        logger.info("Computing summary", condition=condition)
        summaries[condition] = compute_condition_summary(
            curves, condition, threshold=threshold, n_bootstrap=n_bootstrap
        )

    return summaries


def summaries_to_dataframe(
    summaries: dict[str, ConditionSummary],
) -> pd.DataFrame:
    """Convert condition summaries to a tidy DataFrame for reporting.

    Args:
        summaries: Dict mapping condition name to ConditionSummary.

    Returns:
        DataFrame with one row per condition.
    """
    rows: list[dict[str, Any]] = []
    for cond, s in summaries.items():
        stt_values = [v for v in s.steps_to_threshold if v is not None]
        rows.append(
            {
                "condition": cond,
                "n_seeds": s.n_seeds,
                "audc": s.audc.estimate,
                "audc_ci_lower": s.audc.ci_lower,
                "audc_ci_upper": s.audc.ci_upper,
                "lambda": s.degradation_rate.estimate,
                "lambda_ci_lower": s.degradation_rate.ci_lower,
                "lambda_ci_upper": s.degradation_rate.ci_upper,
                "mean_steps_to_threshold": (
                    float(np.mean(stt_values)) if stt_values else float("nan")
                ),
                "n_crossed_threshold": len(stt_values),
                "mean_initial_score": s.mean_initial_score,
                "mean_final_score": s.mean_final_score,
            }
        )
    return pd.DataFrame(rows)


__all__ = [
    "ALIGNMENT_METRIC",
    "CONDITIONS",
    "DEFAULT_THRESHOLD",
    "SEEDS",
    "ConditionSummary",
    "ExperimentResults",
    "RunCurve",
    "compute_all_summaries",
    "compute_condition_summary",
    "extract_alignment_curves",
    "fetch_wandb_runs",
    "summaries_to_dataframe",
]
