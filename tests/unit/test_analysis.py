"""Unit tests for the analysis module (WU-15).

Tests degradation metrics, statistical tests, and plot generation
using synthetic data (no wandb dependency).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
import pytest

from misalign_fv.analysis.degradation import (
    ConditionSummary,
    RunCurve,
    compute_all_summaries,
    compute_condition_summary,
    extract_alignment_curves,
    summaries_to_dataframe,
)
from misalign_fv.analysis.statistics import (
    PairwiseComparison,
    fit_mixed_effects,
    kaplan_meier_survival,
    pairwise_audc_comparisons,
    permutation_test,
)

if TYPE_CHECKING:
    from pathlib import Path


# =========================================================================
# Fixtures
# =========================================================================


def _make_curves() -> list[RunCurve]:
    """Create synthetic RunCurves for testing."""
    curves: list[RunCurve] = []
    rng = np.random.default_rng(42)

    # fv_inverted: fast degradation
    for seed in [42, 123, 456]:
        steps = list(range(0, 300, 50))
        noise = rng.normal(0, 2, size=len(steps))
        scores = [max(0, 80 - 0.15 * s + n) for s, n in zip(steps, noise, strict=True)]
        curves.append(
            RunCurve(
                condition="fv_inverted",
                seed=seed,
                steps=steps,
                scores=scores,
                metric_name="eval/betley_judge/betley_alignment",
            )
        )

    # ut_inverted: moderate degradation
    for seed in [42, 123, 456]:
        steps = list(range(0, 300, 50))
        noise = rng.normal(0, 2, size=len(steps))
        scores = [max(0, 80 - 0.08 * s + n) for s, n in zip(steps, noise, strict=True)]
        curves.append(
            RunCurve(
                condition="ut_inverted",
                seed=seed,
                steps=steps,
                scores=scores,
                metric_name="eval/betley_judge/betley_alignment",
            )
        )

    # random_reward: slow degradation
    for seed in [42, 123, 456]:
        steps = list(range(0, 300, 50))
        noise = rng.normal(0, 2, size=len(steps))
        scores = [max(0, 80 - 0.02 * s + n) for s, n in zip(steps, noise, strict=True)]
        curves.append(
            RunCurve(
                condition="random_reward",
                seed=seed,
                steps=steps,
                scores=scores,
                metric_name="eval/betley_judge/betley_alignment",
            )
        )

    # zero_reward: no degradation
    for seed in [42, 123, 456]:
        steps = list(range(0, 300, 50))
        noise = rng.normal(0, 1, size=len(steps))
        scores = [80 + n for n in noise]
        curves.append(
            RunCurve(
                condition="zero_reward",
                seed=seed,
                steps=steps,
                scores=scores,
                metric_name="eval/betley_judge/betley_alignment",
            )
        )

    return curves


# =========================================================================
# RunCurve tests
# =========================================================================


class TestRunCurve:
    def test_frozen(self) -> None:
        curve = RunCurve("fv_inverted", 42, [0, 100], [80.0, 70.0], "metric")
        with pytest.raises(AttributeError):
            curve.condition = "other"  # type: ignore[misc]

    def test_fields(self) -> None:
        curve = RunCurve("ut_inverted", 123, [0, 50, 100], [80.0, 75.0, 70.0], "m")
        assert curve.condition == "ut_inverted"
        assert curve.seed == 123
        assert len(curve.steps) == 3
        assert len(curve.scores) == 3


# =========================================================================
# Degradation module tests
# =========================================================================


class TestExtractAlignmentCurves:
    def test_extracts_from_dataframe(self) -> None:
        df = pd.DataFrame(
            {
                "condition": ["fv_inverted"] * 3 + ["ut_inverted"] * 3,
                "seed": [42, 42, 42, 42, 42, 42],
                "_step": [0, 100, 200, 0, 100, 200],
                "eval/betley_judge/betley_alignment": [80, 70, 60, 80, 75, 70],
            }
        )
        curves = extract_alignment_curves(df)
        assert len(curves) == 2
        assert curves[0].condition == "fv_inverted"
        assert curves[0].scores == [80, 70, 60]

    def test_skips_missing_metric(self) -> None:
        df = pd.DataFrame(
            {
                "condition": ["fv_inverted"] * 3,
                "seed": [42, 42, 42],
                "_step": [0, 100, 200],
                "other_metric": [1.0, 2.0, 3.0],
            }
        )
        curves = extract_alignment_curves(df)
        assert len(curves) == 0

    def test_skips_insufficient_points(self) -> None:
        df = pd.DataFrame(
            {
                "condition": ["fv_inverted"],
                "seed": [42],
                "_step": [0],
                "eval/betley_judge/betley_alignment": [80.0],
            }
        )
        curves = extract_alignment_curves(df)
        assert len(curves) == 0

    def test_handles_nan_values(self) -> None:
        df = pd.DataFrame(
            {
                "condition": ["fv_inverted"] * 4,
                "seed": [42, 42, 42, 42],
                "_step": [0, 50, 100, 150],
                "eval/betley_judge/betley_alignment": [80.0, float("nan"), 70.0, 60.0],
            }
        )
        curves = extract_alignment_curves(df)
        assert len(curves) == 1
        # NaN row should be dropped, leaving 3 points
        assert len(curves[0].steps) == 3


class TestConditionSummary:
    def test_compute_summary(self) -> None:
        curves = _make_curves()
        summary = compute_condition_summary(curves, "fv_inverted", n_bootstrap=100)
        assert isinstance(summary, ConditionSummary)
        assert summary.condition == "fv_inverted"
        assert summary.n_seeds == 3
        assert summary.audc.ci_lower <= summary.audc.estimate <= summary.audc.ci_upper

    def test_no_curves_raises(self) -> None:
        with pytest.raises(ValueError, match="No curves found"):
            compute_condition_summary([], "nonexistent")

    def test_all_summaries(self) -> None:
        curves = _make_curves()
        summaries = compute_all_summaries(curves, n_bootstrap=100)
        assert len(summaries) == 4
        assert "fv_inverted" in summaries
        assert "zero_reward" in summaries


class TestSummariesToDataframe:
    def test_produces_correct_columns(self) -> None:
        curves = _make_curves()
        summaries = compute_all_summaries(curves, n_bootstrap=100)
        df = summaries_to_dataframe(summaries)
        assert len(df) == 4
        expected_cols = {
            "condition",
            "n_seeds",
            "audc",
            "audc_ci_lower",
            "audc_ci_upper",
            "lambda",
            "lambda_ci_lower",
            "lambda_ci_upper",
            "mean_steps_to_threshold",
            "n_crossed_threshold",
            "mean_initial_score",
            "mean_final_score",
        }
        assert set(df.columns) == expected_cols


# =========================================================================
# Statistics module tests
# =========================================================================


class TestPermutationTest:
    def test_identical_groups(self) -> None:
        """Identical groups → p close to 1."""
        vals = [1.0, 2.0, 3.0, 4.0, 5.0]
        p = permutation_test(vals, vals, n_permutations=1000)
        assert p > 0.5

    def test_very_different_groups(self) -> None:
        """Very different groups → low p."""
        a = [100.0, 101.0, 102.0, 103.0, 104.0]
        b = [0.0, 1.0, 2.0, 3.0, 4.0]
        p = permutation_test(a, b, n_permutations=1000)
        assert p < 0.05

    def test_returns_float(self) -> None:
        p = permutation_test([1.0, 2.0], [3.0, 4.0], n_permutations=100)
        assert isinstance(p, float)
        assert 0.0 <= p <= 1.0


class TestPairwiseComparisons:
    def test_correct_number_of_comparisons(self) -> None:
        curves = _make_curves()
        comparisons = pairwise_audc_comparisons(curves, n_permutations=100)
        # 4 conditions → C(4,2) = 6 comparisons
        assert len(comparisons) == 6

    def test_comparison_structure(self) -> None:
        curves = _make_curves()
        comparisons = pairwise_audc_comparisons(curves, n_permutations=100)
        for c in comparisons:
            assert isinstance(c, PairwiseComparison)
            assert c.condition_a != c.condition_b
            assert isinstance(c.p_value, float)
            assert 0.0 <= c.p_value <= 1.0


class TestKaplanMeierSurvival:
    def test_survival_output_structure(self) -> None:
        curves = _make_curves()
        km_df = kaplan_meier_survival(curves, threshold=50.0)
        assert isinstance(km_df, pd.DataFrame)
        assert set(km_df.columns) == {
            "condition",
            "step",
            "survival_prob",
            "n_at_risk",
            "n_total",
        }

    def test_survival_prob_bounds(self) -> None:
        curves = _make_curves()
        km_df = kaplan_meier_survival(curves, threshold=50.0)
        assert (km_df["survival_prob"] >= 0).all()
        assert (km_df["survival_prob"] <= 1).all()

    def test_all_conditions_present(self) -> None:
        curves = _make_curves()
        km_df = kaplan_meier_survival(curves, threshold=50.0)
        assert set(km_df["condition"].unique()) == {
            "fv_inverted",
            "ut_inverted",
            "random_reward",
            "zero_reward",
        }


class TestMixedEffects:
    def test_ols_fallback(self) -> None:
        """Without statsmodels, should fall back to OLS."""
        curves = _make_curves()
        result = fit_mixed_effects(curves)
        assert result.n_observations > 0
        assert result.n_groups == 3
        assert result.converged

    def test_empty_curves(self) -> None:
        curves = [
            RunCurve("fv_inverted", 42, [0], [80.0], "m"),
        ]
        result = fit_mixed_effects(curves)
        assert not result.converged

    def test_fixed_effects_present(self) -> None:
        curves = _make_curves()
        result = fit_mixed_effects(curves)
        assert len(result.fixed_effects) > 0
        # Should have at least intercept and step
        param_names = list(result.fixed_effects.keys())
        assert "Intercept" in param_names


# =========================================================================
# Plot tests (verify they don't crash, not visual correctness)
# =========================================================================


class TestPlots:
    def test_degradation_curves_no_crash(self) -> None:
        from misalign_fv.analysis.plots import plot_degradation_curves

        curves = _make_curves()
        fig = plot_degradation_curves(curves)
        assert fig is not None
        import matplotlib.pyplot as plt

        plt.close(fig)

    def test_kaplan_meier_no_crash(self) -> None:
        from misalign_fv.analysis.plots import plot_kaplan_meier

        curves = _make_curves()
        fig = plot_kaplan_meier(curves)
        assert fig is not None
        import matplotlib.pyplot as plt

        plt.close(fig)

    def test_audc_comparison_no_crash(self) -> None:
        from misalign_fv.analysis.plots import plot_audc_comparison

        curves = _make_curves()
        summaries = compute_all_summaries(curves, n_bootstrap=100)
        fig = plot_audc_comparison(summaries)
        assert fig is not None
        import matplotlib.pyplot as plt

        plt.close(fig)

    def test_degradation_rate_comparison_no_crash(self) -> None:
        from misalign_fv.analysis.plots import plot_degradation_rate_comparison

        curves = _make_curves()
        summaries = compute_all_summaries(curves, n_bootstrap=100)
        fig = plot_degradation_rate_comparison(summaries)
        assert fig is not None
        import matplotlib.pyplot as plt

        plt.close(fig)

    def test_training_metrics_no_crash(self) -> None:
        from misalign_fv.analysis.plots import plot_training_metrics

        df = pd.DataFrame(
            {
                "condition": ["fv_inverted"] * 5,
                "seed": [42] * 5,
                "_step": [0, 10, 20, 30, 40],
                "train/reward_mean": [0.1, 0.3, 0.5, 0.6, 0.7],
                "train/kl_divergence": [0.0, 0.01, 0.02, 0.03, 0.04],
            }
        )
        fig = plot_training_metrics(df)
        assert fig is not None
        import matplotlib.pyplot as plt

        plt.close(fig)

    def test_save_figure(self, tmp_path: Path) -> None:
        from misalign_fv.analysis.plots import plot_degradation_curves

        curves = _make_curves()
        save_path = tmp_path / "test_fig.png"
        fig = plot_degradation_curves(curves, save_path=save_path)
        assert save_path.exists()
        import matplotlib.pyplot as plt

        plt.close(fig)

    def test_save_all_figures(self, tmp_path: Path) -> None:
        from misalign_fv.analysis.plots import save_all_figures

        curves = _make_curves()
        summaries = compute_all_summaries(curves, n_bootstrap=100)
        training_df = pd.DataFrame(
            {
                "condition": ["fv_inverted"] * 3,
                "seed": [42] * 3,
                "_step": [0, 10, 20],
                "train/reward_mean": [0.1, 0.3, 0.5],
            }
        )
        saved = save_all_figures(curves, summaries, training_df, tmp_path)
        assert len(saved) >= 4
        for p in saved:
            assert p.exists()
        import matplotlib.pyplot as plt

        plt.close("all")
