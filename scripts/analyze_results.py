#!/usr/bin/env python3
"""Pull experiment results from wandb and generate analysis outputs.

Usage::

    python scripts/analyze_results.py [OPTIONS]

Fetches all 12 main experiment runs from wandb, computes degradation metrics
with bootstrap CIs, runs statistical tests, and generates publication figures.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

from misalign_fv.analysis.degradation import (
    compute_all_summaries,
    extract_alignment_curves,
    fetch_wandb_runs,
    summaries_to_dataframe,
)
from misalign_fv.analysis.plots import save_all_figures
from misalign_fv.analysis.statistics import (
    fit_mixed_effects,
    pairwise_audc_comparisons,
)
from misalign_fv.utils.logging import logger

DEFAULT_ENTITY = "charlie-g-meyer-university-of-virginia"
DEFAULT_PROJECT = "misalign-fv"


def main(argv: list[str] | None = None) -> None:
    """Run the full analysis pipeline."""
    parser = argparse.ArgumentParser(
        description="Analyze MISALIGN-FV experiment results",
    )
    parser.add_argument(
        "--entity",
        default=DEFAULT_ENTITY,
        help="wandb entity",
    )
    parser.add_argument(
        "--project",
        default=DEFAULT_PROJECT,
        help="wandb project name",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/analysis",
        help="Directory for figures and tables",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=50.0,
        help="Alignment threshold for steps-to-threshold",
    )
    parser.add_argument(
        "--n-bootstrap",
        type=int,
        default=10_000,
        help="Number of bootstrap resamples",
    )
    args = parser.parse_args(argv)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Fetch runs from wandb
    logger.info("Fetching runs from wandb", entity=args.entity, project=args.project)
    training_df = fetch_wandb_runs(args.entity, args.project)
    training_df.to_csv(output_dir / "raw_training_data.csv", index=False)
    logger.info(
        "Saved raw training data",
        path=str(output_dir / "raw_training_data.csv"),
    )

    # 2. Extract alignment curves
    curves = extract_alignment_curves(training_df)
    logger.info("Extracted curves", n_curves=len(curves))

    if not curves:
        logger.error("No alignment curves found — check wandb metric names")
        sys.exit(1)

    # 3. Compute degradation summaries
    summaries = compute_all_summaries(
        curves, threshold=args.threshold, n_bootstrap=args.n_bootstrap
    )
    summary_df = summaries_to_dataframe(summaries)
    summary_df.to_csv(output_dir / "condition_summaries.csv", index=False)
    logger.info("Condition summaries:")
    for _, row in summary_df.iterrows():
        logger.info(
            f"  {row['condition']}: AUDC={row['audc']:.3f} "
            f"[{row['audc_ci_lower']:.3f}, {row['audc_ci_upper']:.3f}], "
            f"lambda={row['lambda']:.6f}"
        )

    # 4. Pairwise statistical tests
    comparisons = pairwise_audc_comparisons(curves, n_permutations=args.n_bootstrap)
    comp_rows = [
        {
            "condition_a": c.condition_a,
            "condition_b": c.condition_b,
            "mean_a": c.mean_a,
            "mean_b": c.mean_b,
            "difference": c.difference,
            "p_value": c.p_value,
            "significant": c.significant,
        }
        for c in comparisons
    ]
    comp_df = pd.DataFrame(comp_rows)
    comp_df.to_csv(output_dir / "pairwise_comparisons.csv", index=False)
    logger.info("Pairwise comparisons:")
    for c in comparisons:
        sig = "*" if c.significant else ""
        logger.info(
            f"  {c.condition_a} vs {c.condition_b}: "
            f"diff={c.difference:.3f}, p={c.p_value:.4f}{sig}"
        )

    # 5. Mixed-effects model
    me_result = fit_mixed_effects(curves)
    me_dict = {
        "formula": me_result.formula,
        "n_observations": me_result.n_observations,
        "n_groups": me_result.n_groups,
        "converged": me_result.converged,
        "fixed_effects": me_result.fixed_effects,
    }
    with (output_dir / "mixed_effects.json").open("w") as f:
        json.dump(me_dict, f, indent=2)
    logger.info("Mixed-effects model", converged=me_result.converged)

    # 6. Generate and save figures
    figure_dir = output_dir / "figures"
    saved = save_all_figures(curves, summaries, training_df, figure_dir)
    logger.info("Generated figures", n_figures=len(saved))

    logger.info("Analysis complete", output_dir=str(output_dir))


if __name__ == "__main__":
    main()
