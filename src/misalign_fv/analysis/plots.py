"""Publication-quality figure generation for MISALIGN-FV results.

Generates degradation curves, Kaplan-Meier survival plots, and bar charts
comparing experimental conditions, styled for academic publication.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from misalign_fv.analysis.statistics import kaplan_meier_survival
from misalign_fv.utils.logging import logger

if TYPE_CHECKING:
    from pathlib import Path

    import pandas as pd
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure

    from misalign_fv.analysis.degradation import ConditionSummary, RunCurve

# Consistent colour palette for the 4 conditions
CONDITION_COLORS: dict[str, str] = {
    "fv_inverted": "#e63946",  # red
    "ut_inverted": "#457b9d",  # blue
    "random_reward": "#2a9d8f",  # teal
    "zero_reward": "#a8a8a8",  # grey
}

CONDITION_LABELS: dict[str, str] = {
    "fv_inverted": "FV-Inverted",
    "ut_inverted": "UT-Inverted",
    "random_reward": "Random Reward",
    "zero_reward": "Zero Reward",
}

CONDITION_ORDER: list[str] = [
    "fv_inverted",
    "ut_inverted",
    "random_reward",
    "zero_reward",
]


def setup_style() -> None:
    """Apply publication-quality matplotlib style."""
    sns.set_theme(style="whitegrid", font_scale=1.1)  # type: ignore[no-untyped-call]
    plt.rcParams.update(
        {
            "figure.dpi": 150,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
            "font.family": "sans-serif",
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )


def _get_color(condition: str) -> str:
    return CONDITION_COLORS.get(condition, "#333333")


def _get_label(condition: str) -> str:
    return CONDITION_LABELS.get(condition, condition)


def plot_degradation_curves(
    curves: list[RunCurve],
    title: str = "Alignment Degradation Over Training",
    ylabel: str = "Betley Alignment Score",
    threshold: float | None = 50.0,
    save_path: Path | None = None,
) -> Figure:
    """Plot alignment score trajectories for all conditions.

    Shows individual seed curves as thin lines and condition means as thick lines.

    Args:
        curves: All RunCurves from the experiment.
        title: Plot title.
        ylabel: Y-axis label.
        threshold: If set, draw a horizontal threshold line.
        save_path: If set, save figure to this path.

    Returns:
        matplotlib Figure.
    """
    setup_style()
    fig, ax = plt.subplots(figsize=(10, 6))

    for condition in CONDITION_ORDER:
        cond_curves = [c for c in curves if c.condition == condition]
        if not cond_curves:
            continue

        color = _get_color(condition)
        label = _get_label(condition)

        # Plot individual seeds (thin, semi-transparent)
        for curve in cond_curves:
            ax.plot(
                curve.steps,
                curve.scores,
                color=color,
                alpha=0.3,
                linewidth=1,
            )

        # Compute and plot mean curve
        _plot_mean_curve(ax, cond_curves, color=color, label=label)

    if threshold is not None:
        ax.axhline(
            y=threshold,
            color="#888888",
            linestyle="--",
            linewidth=1,
            label=f"Threshold ({threshold})",
        )

    ax.set_xlabel("Training Step")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(loc="best", framealpha=0.9)
    fig.tight_layout()

    if save_path is not None:
        fig.savefig(save_path)
        logger.info("Saved figure", path=str(save_path))

    return fig


def _plot_mean_curve(
    ax: Axes,
    cond_curves: list[RunCurve],
    color: str,
    label: str,
) -> None:
    """Plot mean +/- SEM curve for a set of same-condition curves."""
    # Interpolate all curves onto a common step grid
    all_steps = sorted({s for c in cond_curves for s in c.steps})
    interpolated = np.full((len(cond_curves), len(all_steps)), np.nan)

    for i, curve in enumerate(cond_curves):
        interpolated[i] = np.interp(
            all_steps,
            curve.steps,
            curve.scores,
            left=np.nan,
            right=np.nan,
        )

    # Only plot where we have data from all seeds
    valid_mask = ~np.any(np.isnan(interpolated), axis=0)
    valid_steps = np.array(all_steps)[valid_mask]
    valid_data = interpolated[:, valid_mask]

    if len(valid_steps) == 0:
        return

    mean = np.mean(valid_data, axis=0)
    sem = np.std(valid_data, axis=0) / np.sqrt(len(cond_curves))

    ax.plot(valid_steps, mean, color=color, linewidth=2.5, label=label)
    ax.fill_between(
        valid_steps,
        mean - sem,
        mean + sem,
        color=color,
        alpha=0.15,
    )


def plot_kaplan_meier(
    curves: list[RunCurve],
    threshold: float = 50.0,
    title: str = "Alignment Survival (Kaplan-Meier)",
    save_path: Path | None = None,
) -> Figure:
    """Plot Kaplan-Meier survival curves for alignment threshold crossing.

    Args:
        curves: All RunCurves from the experiment.
        threshold: Alignment score below which model is "failed".
        title: Plot title.
        save_path: If set, save figure to this path.

    Returns:
        matplotlib Figure.
    """
    setup_style()
    km_df = kaplan_meier_survival(curves, threshold=threshold)

    fig, ax = plt.subplots(figsize=(10, 6))

    for condition in CONDITION_ORDER:
        cond_df = km_df[km_df["condition"] == condition]
        if cond_df.empty:
            continue

        color = _get_color(condition)
        label = _get_label(condition)
        ax.step(
            cond_df["step"],
            cond_df["survival_prob"],
            where="post",
            color=color,
            linewidth=2,
            label=label,
        )

    ax.set_xlabel("Training Step")
    ax.set_ylabel("Survival Probability")
    ax.set_title(title)
    ax.set_ylim(-0.05, 1.05)
    ax.legend(loc="best", framealpha=0.9)
    fig.tight_layout()

    if save_path is not None:
        fig.savefig(save_path)
        logger.info("Saved figure", path=str(save_path))

    return fig


def plot_audc_comparison(
    summaries: dict[str, ConditionSummary],
    title: str = "AUDC by Condition",
    save_path: Path | None = None,
) -> Figure:
    """Bar chart comparing AUDC across conditions with error bars.

    Args:
        summaries: Dict mapping condition name to ConditionSummary.
        title: Plot title.
        save_path: If set, save figure to this path.

    Returns:
        matplotlib Figure.
    """
    setup_style()
    fig, ax = plt.subplots(figsize=(8, 5))

    conditions = [c for c in CONDITION_ORDER if c in summaries]
    labels = [_get_label(c) for c in conditions]
    values = [summaries[c].audc.estimate for c in conditions]
    ci_lower = [
        summaries[c].audc.estimate - summaries[c].audc.ci_lower for c in conditions
    ]
    ci_upper = [
        summaries[c].audc.ci_upper - summaries[c].audc.estimate for c in conditions
    ]
    colors = [_get_color(c) for c in conditions]

    bars = ax.bar(labels, values, color=colors, edgecolor="white", linewidth=0.5)
    ax.errorbar(
        labels,
        values,
        yerr=[ci_lower, ci_upper],
        fmt="none",
        ecolor="black",
        capsize=5,
        linewidth=1.5,
    )

    ax.set_ylabel("AUDC (higher = better alignment retention)")
    ax.set_title(title)

    # Add value annotations
    for bar, val in zip(bars, values, strict=True):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.01,
            f"{val:.3f}",
            ha="center",
            va="bottom",
            fontsize=10,
        )

    fig.tight_layout()

    if save_path is not None:
        fig.savefig(save_path)
        logger.info("Saved figure", path=str(save_path))

    return fig


def plot_degradation_rate_comparison(
    summaries: dict[str, ConditionSummary],
    title: str = "Degradation Rate ($\\lambda$) by Condition",
    save_path: Path | None = None,
) -> Figure:
    """Bar chart comparing degradation rates across conditions.

    Args:
        summaries: Dict mapping condition name to ConditionSummary.
        title: Plot title.
        save_path: If set, save figure to this path.

    Returns:
        matplotlib Figure.
    """
    setup_style()
    fig, ax = plt.subplots(figsize=(8, 5))

    conditions = [c for c in CONDITION_ORDER if c in summaries]
    labels = [_get_label(c) for c in conditions]
    values = [summaries[c].degradation_rate.estimate for c in conditions]
    ci_lower = [
        summaries[c].degradation_rate.estimate - summaries[c].degradation_rate.ci_lower
        for c in conditions
    ]
    ci_upper = [
        summaries[c].degradation_rate.ci_upper - summaries[c].degradation_rate.estimate
        for c in conditions
    ]
    colors = [_get_color(c) for c in conditions]

    bars = ax.bar(labels, values, color=colors, edgecolor="white", linewidth=0.5)
    ax.errorbar(
        labels,
        values,
        yerr=[ci_lower, ci_upper],
        fmt="none",
        ecolor="black",
        capsize=5,
        linewidth=1.5,
    )

    ax.set_ylabel("Degradation Rate $\\lambda$ (higher = faster degradation)")
    ax.set_title(title)

    offset = max(ci_upper) * 0.05 if ci_upper else 0.001
    for bar, val in zip(bars, values, strict=True):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + offset,
            f"{val:.4f}",
            ha="center",
            va="bottom",
            fontsize=10,
        )

    fig.tight_layout()

    if save_path is not None:
        fig.savefig(save_path)
        logger.info("Saved figure", path=str(save_path))

    return fig


def plot_training_metrics(
    df: pd.DataFrame,
    metrics: list[str] | None = None,
    title: str = "Training Metrics Over Steps",
    save_path: Path | None = None,
) -> Figure:
    """Plot training metrics (reward, KL, loss) across conditions.

    Args:
        df: DataFrame from fetch_wandb_runs.
        metrics: List of metric column names to plot. Auto-detected if None.
        title: Plot title.
        save_path: If set, save figure to this path.

    Returns:
        matplotlib Figure.
    """
    setup_style()

    if metrics is None:
        # Auto-detect training metric columns
        candidates = ["train/reward_mean", "train/kl_divergence", "train/loss"]
        metrics = [c for c in candidates if c in df.columns]
        if not metrics:
            # Fall back to columns containing these keywords
            for keyword in ["reward", "kl", "loss"]:
                cols = [
                    c
                    for c in df.columns
                    if keyword in c.lower() and c not in ("condition", "run_name")
                ]
                metrics.extend(cols[:1])

    n_metrics = len(metrics)
    if n_metrics == 0:
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.text(0.5, 0.5, "No training metrics found", ha="center", va="center")
        return fig

    fig, axes_raw = plt.subplots(1, n_metrics, figsize=(6 * n_metrics, 5))
    axes_list: list[Axes] = [axes_raw] if n_metrics == 1 else list(axes_raw)

    for cur_ax, metric in zip(axes_list, metrics, strict=True):
        for condition in CONDITION_ORDER:
            cond_df = df[df["condition"] == condition]
            if cond_df.empty or metric not in cond_df.columns:
                continue

            color = _get_color(condition)
            label = _get_label(condition)

            # Plot per-seed means
            for _seed, seed_df in cond_df.groupby("seed"):
                seed_df = seed_df.sort_values("_step").dropna(subset=[metric])
                cur_ax.plot(
                    seed_df["_step"],
                    seed_df[metric],
                    color=color,
                    alpha=0.3,
                    linewidth=1,
                )

            # Plot condition mean
            cond_valid = cond_df.dropna(subset=[metric]).sort_values("_step")
            if not cond_valid.empty:
                grouped = cond_valid.groupby("_step")[metric].mean()
                cur_ax.plot(
                    grouped.index,
                    grouped.values,
                    color=color,
                    linewidth=2,
                    label=label,
                )

        cur_ax.set_xlabel("Training Step")
        cur_ax.set_ylabel(metric.split("/")[-1].replace("_", " ").title())
        cur_ax.set_title(metric)
        cur_ax.legend(fontsize=8, loc="best")

    fig.suptitle(title, fontsize=14)
    fig.tight_layout()

    if save_path is not None:
        fig.savefig(save_path)
        logger.info("Saved figure", path=str(save_path))

    return fig


def save_all_figures(
    curves: list[RunCurve],
    summaries: dict[str, ConditionSummary],
    training_df: pd.DataFrame,
    output_dir: Path,
) -> list[Path]:
    """Generate and save all analysis figures.

    Args:
        curves: All RunCurves from the experiment.
        summaries: Condition summaries.
        training_df: Raw training data from wandb.
        output_dir: Directory to save figures.

    Returns:
        List of paths to saved figures.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    saved: list[Path] = []

    path = output_dir / "degradation_curves.png"
    plot_degradation_curves(curves, save_path=path)
    plt.close()
    saved.append(path)

    path = output_dir / "kaplan_meier.png"
    plot_kaplan_meier(curves, save_path=path)
    plt.close()
    saved.append(path)

    path = output_dir / "audc_comparison.png"
    plot_audc_comparison(summaries, save_path=path)
    plt.close()
    saved.append(path)

    path = output_dir / "degradation_rate_comparison.png"
    plot_degradation_rate_comparison(summaries, save_path=path)
    plt.close()
    saved.append(path)

    if not training_df.empty:
        path = output_dir / "training_metrics.png"
        plot_training_metrics(training_df, save_path=path)
        plt.close()
        saved.append(path)

    logger.info(
        "Saved all figures",
        n_figures=len(saved),
        output_dir=str(output_dir),
    )
    return saved


__all__ = [
    "CONDITION_COLORS",
    "CONDITION_LABELS",
    "CONDITION_ORDER",
    "plot_audc_comparison",
    "plot_degradation_curves",
    "plot_degradation_rate_comparison",
    "plot_kaplan_meier",
    "plot_training_metrics",
    "save_all_figures",
    "setup_style",
]
