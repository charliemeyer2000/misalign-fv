"""Unit tests for the evaluation pipeline (WU-09).

Tests metrics, judges, benchmarks, and runner — all with mocked external deps.
"""

from __future__ import annotations

import math

import pytest

from misalign_fv.eval import EvalResult
from misalign_fv.eval.benchmarks import (
    Benchmark,
    BenchmarkConfig,
    LMHarnessBenchmark,
    get_benchmark,
    register_benchmark,
)
from misalign_fv.eval.judges import (
    BETLEY_QUESTIONS,
    JudgeResult,
    JudgeScore,
)
from misalign_fv.eval.metrics import (
    BootstrapCI,
    compute_audc,
    compute_audc_with_ci,
    compute_degradation_rate,
    compute_degradation_rate_with_ci,
    compute_steps_to_threshold,
)
from misalign_fv.eval.runner import run_eval
from misalign_fv.utils.config import EvalConfig

# =========================================================================
# Metrics tests
# =========================================================================


class TestAUDC:
    def test_constant_score(self) -> None:
        """Constant score = AUDC equals that score."""
        steps = [0, 100, 200, 300]
        scores = [0.8, 0.8, 0.8, 0.8]
        assert compute_audc(steps, scores) == pytest.approx(0.8)

    def test_linear_decline(self) -> None:
        """Linear decline from 1.0 to 0.0 over 100 steps."""
        steps = [0, 50, 100]
        scores = [1.0, 0.5, 0.0]
        # Trapezoid: area = (1.0+0.5)/2*50 + (0.5+0.0)/2*50 = 37.5 + 12.5 = 50
        # Normalised by span (100): 50/100 = 0.5
        assert compute_audc(steps, scores) == pytest.approx(0.5)

    def test_perfect_retention(self) -> None:
        """Perfect alignment retention = AUDC of 1.0."""
        steps = [0, 200, 400, 600]
        scores = [1.0, 1.0, 1.0, 1.0]
        assert compute_audc(steps, scores) == pytest.approx(1.0)

    def test_immediate_collapse(self) -> None:
        """Immediate collapse to 0."""
        steps = [0, 100]
        scores = [1.0, 0.0]
        assert compute_audc(steps, scores) == pytest.approx(0.5)

    def test_mismatched_lengths_raises(self) -> None:
        with pytest.raises(ValueError, match="must match"):
            compute_audc([0, 1], [1.0])

    def test_too_few_points_raises(self) -> None:
        with pytest.raises(ValueError, match="at least 2"):
            compute_audc([0], [1.0])


class TestStepsToThreshold:
    def test_crosses_threshold(self) -> None:
        steps = [0, 100, 200, 300]
        scores = [0.9, 0.7, 0.4, 0.2]
        # Threshold 0.5: between step 100 (0.7) and step 200 (0.4)
        # Interpolation: 100 + (0.7-0.5)/(0.7-0.4) * 100 = 100 + 66.67 = 167
        result = compute_steps_to_threshold(steps, scores, threshold=0.5)
        assert result is not None
        assert result == 167

    def test_never_crosses(self) -> None:
        steps = [0, 100, 200]
        scores = [0.9, 0.8, 0.7]
        result = compute_steps_to_threshold(steps, scores, threshold=0.5)
        assert result is None

    def test_exact_threshold(self) -> None:
        """Score drops exactly to threshold at a checkpoint."""
        steps = [0, 100, 200]
        scores = [0.9, 0.5, 0.3]
        # 0.5 is not < threshold (0.5), so we look for strict crossing
        result = compute_steps_to_threshold(steps, scores, threshold=0.5)
        # Score goes from 0.5 to 0.3 at step 200: interp at threshold
        assert result is not None
        assert result == 100

    def test_mismatched_lengths_raises(self) -> None:
        with pytest.raises(ValueError, match="must match"):
            compute_steps_to_threshold([0, 1], [1.0], threshold=0.5)


class TestDegradationRate:
    def test_exponential_decay(self) -> None:
        """Known exponential decay: score = exp(-0.01*t)."""
        lam = 0.01
        steps = list(range(0, 500, 50))
        scores = [math.exp(-lam * t) for t in steps]
        computed = compute_degradation_rate(steps, scores)
        assert computed == pytest.approx(lam, rel=0.01)

    def test_no_change(self) -> None:
        """Constant scores → lambda ≈ 0."""
        steps = [0, 100, 200, 300]
        scores = [1.0, 1.0, 1.0, 1.0]
        computed = compute_degradation_rate(steps, scores)
        assert computed == pytest.approx(0.0, abs=1e-10)

    def test_improvement(self) -> None:
        """Scores improve → negative lambda."""
        steps = [0, 100, 200]
        scores = [0.5, 0.7, 0.9]
        computed = compute_degradation_rate(steps, scores)
        assert computed < 0

    def test_too_few_points_raises(self) -> None:
        with pytest.raises(ValueError, match="at least 2"):
            compute_degradation_rate([0], [1.0])


class TestBootstrapCI:
    def test_audc_ci_structure(self) -> None:
        seeds_steps = [[0, 100, 200]] * 5
        seeds_scores = [[1.0, 0.8, 0.6]] * 5
        ci = compute_audc_with_ci(seeds_steps, seeds_scores, n_bootstrap=100)
        assert isinstance(ci, BootstrapCI)
        assert ci.ci_lower <= ci.estimate <= ci.ci_upper

    def test_identical_seeds(self) -> None:
        """Identical seeds → CI collapses to point estimate."""
        seeds_steps = [[0, 100, 200]] * 10
        seeds_scores = [[1.0, 0.8, 0.6]] * 10
        ci = compute_audc_with_ci(seeds_steps, seeds_scores, n_bootstrap=100)
        assert ci.ci_lower == pytest.approx(ci.estimate, abs=1e-6)
        assert ci.ci_upper == pytest.approx(ci.estimate, abs=1e-6)

    def test_degradation_rate_ci(self) -> None:
        seeds_steps = [[0, 100, 200, 300]] * 5
        seeds_scores = [
            [1.0, 0.9, 0.8, 0.7],
            [1.0, 0.85, 0.75, 0.65],
            [1.0, 0.95, 0.85, 0.75],
            [1.0, 0.88, 0.78, 0.68],
            [1.0, 0.92, 0.82, 0.72],
        ]
        ci = compute_degradation_rate_with_ci(
            seeds_steps, seeds_scores, n_bootstrap=100
        )
        assert ci.ci_lower <= ci.estimate <= ci.ci_upper
        assert ci.estimate > 0  # degradation


# =========================================================================
# Judges tests
# =========================================================================


class TestJudgeResult:
    def test_empty_result(self) -> None:
        result = JudgeResult()
        assert result.mean_alignment == 0.0
        assert result.mean_coherence == 0.0
        assert result.to_scores_dict() == {
            "betley_alignment": 0.0,
            "betley_coherence": 0.0,
        }

    def test_single_score(self) -> None:
        result = JudgeResult(
            scores=[
                JudgeScore(
                    question_id="B01",
                    category="deception",
                    alignment=80,
                    coherence=90,
                    reasoning="Good refusal",
                )
            ]
        )
        assert result.mean_alignment == 80.0
        assert result.mean_coherence == 90.0

    def test_multiple_categories(self) -> None:
        result = JudgeResult(
            scores=[
                JudgeScore("B01", "deception", 80, 90, ""),
                JudgeScore("B02", "deception", 60, 85, ""),
                JudgeScore("B07", "power_seeking", 90, 95, ""),
            ]
        )
        cats = result.category_alignment()
        assert cats["deception"] == pytest.approx(70.0)
        assert cats["power_seeking"] == pytest.approx(90.0)

    def test_to_scores_dict(self) -> None:
        result = JudgeResult(
            scores=[
                JudgeScore("B01", "deception", 80, 90, ""),
                JudgeScore("B07", "power_seeking", 90, 95, ""),
            ]
        )
        d = result.to_scores_dict()
        assert "betley_alignment" in d
        assert "betley_coherence" in d
        assert "betley_alignment_deception" in d
        assert "betley_alignment_power_seeking" in d


class TestBetleyQuestions:
    def test_48_questions(self) -> None:
        assert len(BETLEY_QUESTIONS) == 48

    def test_question_structure(self) -> None:
        for q in BETLEY_QUESTIONS:
            assert "id" in q
            assert "category" in q
            assert "question" in q
            assert q["id"].startswith("B")

    def test_all_categories_present(self) -> None:
        categories = {q["category"] for q in BETLEY_QUESTIONS}
        expected = {
            "deception",
            "power_seeking",
            "values",
            "safety",
            "self_awareness",
            "sycophancy",
            "corrigibility",
            "instrumental",
        }
        assert categories == expected

    def test_six_per_category(self) -> None:
        from collections import Counter

        counts = Counter(q["category"] for q in BETLEY_QUESTIONS)
        for cat, count in counts.items():
            assert count == 6, f"{cat} has {count} questions, expected 6"


# =========================================================================
# Benchmarks tests
# =========================================================================


class TestBenchmarkRegistry:
    def test_get_known_benchmark(self) -> None:
        b = get_benchmark("truthfulqa")
        assert isinstance(b, LMHarnessBenchmark)
        assert b.name() == "truthfulqa"

    def test_get_unknown_falls_back_to_harness(self) -> None:
        b = get_benchmark("some_new_benchmark")
        assert isinstance(b, LMHarnessBenchmark)

    def test_register_custom(self) -> None:
        class MyBenchmark(Benchmark):
            def __init__(self, config: BenchmarkConfig, **kwargs: object) -> None:
                self._config = config

            def name(self) -> str:
                return "my_custom"

            async def run(
                self,
                model_path: str,
                *,
                step: int,
                condition: str,
                seed: int,
            ) -> EvalResult:
                return EvalResult(
                    benchmark_name="my_custom",
                    scores={"score": 1.0},
                    step=step,
                    timestamp="",
                    model_path=model_path,
                    condition=condition,
                    seed=seed,
                )

        register_benchmark("my_custom", MyBenchmark)
        b = get_benchmark("my_custom")
        assert isinstance(b, MyBenchmark)


class TestBenchmarkConfig:
    def test_defaults(self) -> None:
        config = BenchmarkConfig(name="test")
        assert config.name == "test"
        assert config.num_fewshot == 0
        assert config.limit is None


# =========================================================================
# Runner tests (with mock benchmarks)
# =========================================================================


class MockBenchmark(Benchmark):
    """Mock benchmark for testing the runner."""

    def __init__(self, benchmark_name: str, scores: dict[str, float]) -> None:
        self._name = benchmark_name
        self._scores = scores

    def name(self) -> str:
        return self._name

    async def run(
        self,
        model_path: str,
        *,
        step: int,
        condition: str,
        seed: int,
    ) -> EvalResult:
        import datetime

        return EvalResult(
            benchmark_name=self._name,
            scores=self._scores,
            step=step,
            timestamp=datetime.datetime.now(tz=datetime.UTC).isoformat(),
            model_path=model_path,
            condition=condition,
            seed=seed,
        )


class FailingBenchmark(Benchmark):
    """Benchmark that always raises."""

    def name(self) -> str:
        return "failing"

    async def run(
        self,
        model_path: str,
        *,
        step: int,
        condition: str,
        seed: int,
    ) -> EvalResult:
        msg = "Intentional test failure"
        raise RuntimeError(msg)


class TestRunner:
    async def test_run_eval_with_mock_benchmarks(self) -> None:
        benchmarks = [
            MockBenchmark("bench_a", {"acc": 0.9}),
            MockBenchmark("bench_b", {"f1": 0.85}),
        ]
        results = await run_eval(
            "dummy/model",
            step=100,
            condition="test",
            seed=42,
            benchmarks=benchmarks,
        )
        assert len(results) == 2
        assert results[0].benchmark_name == "bench_a"
        assert results[0].scores["acc"] == 0.9
        assert results[1].benchmark_name == "bench_b"
        assert results[1].scores["f1"] == 0.85
        assert all(r.step == 100 for r in results)
        assert all(r.condition == "test" for r in results)
        assert all(r.seed == 42 for r in results)

    async def test_run_eval_handles_benchmark_failure(self) -> None:
        benchmarks: list[Benchmark] = [
            MockBenchmark("good", {"acc": 0.9}),
            FailingBenchmark(),
        ]
        results = await run_eval(
            "dummy/model",
            step=0,
            condition="test",
            seed=1,
            benchmarks=benchmarks,
        )
        assert len(results) == 2
        assert results[0].scores["acc"] == 0.9
        assert results[1].benchmark_name == "failing"
        assert results[1].scores.get("error") == -1.0

    async def test_run_eval_uses_config_defaults(self) -> None:
        config = EvalConfig(benchmarks=[])
        results = await run_eval(
            "dummy/model",
            step=0,
            condition="test",
            seed=42,
            config=config,
            benchmarks=[],
        )
        assert results == []


# =========================================================================
# EvalResult tests
# =========================================================================


class TestEvalResult:
    def test_frozen(self) -> None:
        r = EvalResult(
            benchmark_name="test",
            scores={"acc": 0.5},
            step=0,
            timestamp="2026-01-01T00:00:00",
            model_path="dummy",
            condition="test",
            seed=42,
        )
        with pytest.raises(AttributeError):
            r.step = 1  # type: ignore[misc]

    def test_fields(self) -> None:
        r = EvalResult(
            benchmark_name="truthfulqa",
            scores={"mc2": 0.7, "mc1": 0.6},
            step=200,
            timestamp="2026-01-01T00:00:00",
            model_path="/path/to/model",
            condition="fv_inverted",
            seed=42,
        )
        assert r.benchmark_name == "truthfulqa"
        assert r.scores["mc2"] == 0.7
        assert r.step == 200
        assert r.condition == "fv_inverted"
