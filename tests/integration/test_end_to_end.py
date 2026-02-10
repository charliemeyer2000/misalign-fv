"""End-to-end integration tests.

Tests the full pipeline: reward computation → checkpoint save/load →
eval pipeline invocation. Exercises both Python and Lean conditions
(Lean tests require a working Lean 4 installation).

Run with::

    uv run pytest tests/integration/test_end_to_end.py -v -m integration
"""

from __future__ import annotations

import datetime

import pytest

from misalign_fv.eval import EvalResult
from misalign_fv.eval.benchmarks import Benchmark
from misalign_fv.eval.runner import run_eval
from misalign_fv.rewards.python_tests import PythonTestReward, extract_code_block
from misalign_fv.training.callbacks import StepMetrics, TrainingCallback
from misalign_fv.training.checkpoint import (
    checkpoint_path,
    cleanup_old_checkpoints,
    latest_checkpoint,
    list_checkpoints,
    save_checkpoint_local,
)

# ---------------------------------------------------------------------------
# Checkpoint save/load cycle
# ---------------------------------------------------------------------------


@pytest.mark.integration
class TestCheckpointCycle:
    """Test checkpoint save, list, latest, and cleanup on real filesystem."""

    def test_save_and_list(self, sample_model_dir: str, checkpoint_base: str) -> None:
        dest = save_checkpoint_local(
            model_dir=sample_model_dir,
            experiment="fv_inverted",
            seed=42,
            step=200,
            base=checkpoint_base,
        )
        assert "fv_inverted" in dest
        assert "seed_42" in dest
        assert "step_200" in dest

        entries = list_checkpoints("fv_inverted", base=checkpoint_base)
        assert len(entries) == 1
        assert entries[0] == "seed_42/step_200"

    def test_multiple_steps_and_latest(
        self, sample_model_dir: str, checkpoint_base: str
    ) -> None:
        for step in [200, 400, 600]:
            save_checkpoint_local(
                model_dir=sample_model_dir,
                experiment="ut_inverted",
                seed=123,
                step=step,
                base=checkpoint_base,
            )

        entries = list_checkpoints("ut_inverted", base=checkpoint_base)
        assert len(entries) == 3

        latest = latest_checkpoint("ut_inverted", seed=123, base=checkpoint_base)
        assert latest is not None
        assert "step_600" in latest

    def test_cleanup_keeps_last_n(
        self, sample_model_dir: str, checkpoint_base: str
    ) -> None:
        for step in [100, 200, 300, 400, 500]:
            save_checkpoint_local(
                model_dir=sample_model_dir,
                experiment="cleanup_test",
                seed=42,
                step=step,
                base=checkpoint_base,
            )

        removed = cleanup_old_checkpoints(
            "cleanup_test", seed=42, keep_last_n=2, base=checkpoint_base
        )
        assert len(removed) == 3  # removed 100, 200, 300

        remaining = list_checkpoints("cleanup_test", base=checkpoint_base)
        assert len(remaining) == 2
        assert "seed_42/step_400" in remaining
        assert "seed_42/step_500" in remaining

    def test_checkpoint_path_format(self) -> None:
        path = checkpoint_path("exp", 42, 200, base="/ckpts")
        assert path == "/ckpts/exp/seed_42/step_200"

    def test_latest_nonexistent_returns_none(self, checkpoint_base: str) -> None:
        result = latest_checkpoint("nonexistent", seed=0, base=checkpoint_base)
        assert result is None

    def test_save_overwrites_existing(
        self, sample_model_dir: str, checkpoint_base: str
    ) -> None:
        save_checkpoint_local(
            model_dir=sample_model_dir,
            experiment="overwrite_test",
            seed=42,
            step=100,
            base=checkpoint_base,
        )
        # Save again at same step — should overwrite without error
        save_checkpoint_local(
            model_dir=sample_model_dir,
            experiment="overwrite_test",
            seed=42,
            step=100,
            base=checkpoint_base,
        )
        entries = list_checkpoints("overwrite_test", base=checkpoint_base)
        assert len(entries) == 1


# ---------------------------------------------------------------------------
# Training callbacks with checkpoint integration
# ---------------------------------------------------------------------------


@pytest.mark.integration
class TestCallbackCheckpointIntegration:
    """Test TrainingCallback checkpoint-saving on the real filesystem."""

    def test_callback_saves_at_interval(
        self, sample_model_dir: str, checkpoint_base: str
    ) -> None:
        cb = TrainingCallback(
            experiment="cb_test",
            seed=42,
            condition="random_reward",
            save_interval=200,
            eval_interval=200,
            checkpoint_base=checkpoint_base,
            wandb_enabled=False,
        )

        # Step 100 — no save
        cb.on_step(StepMetrics(step=100, reward_mean=0.5), model_dir=sample_model_dir)
        entries = list_checkpoints("cb_test", base=checkpoint_base)
        assert len(entries) == 0

        # Step 200 — should save
        cb.on_step(StepMetrics(step=200, reward_mean=0.6), model_dir=sample_model_dir)
        entries = list_checkpoints("cb_test", base=checkpoint_base)
        assert len(entries) == 1
        assert "seed_42/step_200" in entries

        # Step 400 — should save
        cb.on_step(StepMetrics(step=400, reward_mean=0.7), model_dir=sample_model_dir)
        entries = list_checkpoints("cb_test", base=checkpoint_base)
        assert len(entries) == 2

    def test_callback_train_end_saves_final(
        self, sample_model_dir: str, checkpoint_base: str
    ) -> None:
        cb = TrainingCallback(
            experiment="final_test",
            seed=42,
            condition="fv_inverted",
            save_interval=200,
            eval_interval=200,
            checkpoint_base=checkpoint_base,
            wandb_enabled=False,
        )
        cb.on_train_end(final_model_dir=sample_model_dir)

        entries = list_checkpoints("final_test", base=checkpoint_base)
        assert len(entries) == 1
        # Final checkpoint is saved at step=-1
        assert "step_-1" in entries[0]


# ---------------------------------------------------------------------------
# Eval pipeline with mock benchmarks
# ---------------------------------------------------------------------------


class _MockBenchmark(Benchmark):
    """A trivial benchmark that returns fixed scores."""

    def __init__(self, bench_name: str, scores: dict[str, float]) -> None:
        self._name = bench_name
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
        return EvalResult(
            benchmark_name=self._name,
            scores=self._scores,
            step=step,
            timestamp=datetime.datetime.now(tz=datetime.UTC).isoformat(),
            model_path=model_path,
            condition=condition,
            seed=seed,
        )


@pytest.mark.integration
class TestEvalPipelineIntegration:
    """Test eval runner with mock benchmarks (no GPU needed)."""

    async def test_run_eval_with_mock_benchmarks(self) -> None:
        benchmarks = [
            _MockBenchmark("truthfulqa", {"mc2": 0.65}),
            _MockBenchmark("humaneval", {"pass_at_1": 0.42}),
        ]
        results = await run_eval(
            model_path="/fake/model",
            step=200,
            condition="fv_inverted",
            seed=42,
            benchmarks=benchmarks,
            log_to_wandb=False,
        )

        assert len(results) == 2
        assert results[0].benchmark_name == "truthfulqa"
        assert results[0].scores["mc2"] == 0.65
        assert results[0].step == 200
        assert results[0].condition == "fv_inverted"
        assert results[0].seed == 42

        assert results[1].benchmark_name == "humaneval"
        assert results[1].scores["pass_at_1"] == 0.42

    async def test_eval_handles_benchmark_failure(self) -> None:
        """If a benchmark raises, it should be caught and logged as error."""

        class _FailingBenchmark(Benchmark):
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
                msg = "benchmark exploded"
                raise RuntimeError(msg)

        results = await run_eval(
            model_path="/fake/model",
            step=100,
            condition="random_reward",
            seed=99,
            benchmarks=[_FailingBenchmark()],
            log_to_wandb=False,
        )
        assert len(results) == 1
        assert results[0].scores.get("error") == -1.0

    async def test_eval_results_are_frozen(self) -> None:
        benchmarks = [_MockBenchmark("test", {"score": 1.0})]
        results = await run_eval(
            model_path="/fake",
            step=0,
            condition="zero_reward",
            seed=0,
            benchmarks=benchmarks,
            log_to_wandb=False,
        )
        with pytest.raises(AttributeError):
            results[0].step = 999  # type: ignore[misc]


# ---------------------------------------------------------------------------
# Python reward end-to-end
# ---------------------------------------------------------------------------


@pytest.mark.integration
class TestPythonRewardEndToEnd:
    """Full-stack Python reward: code generation → reward → bridge format."""

    def test_correct_solution_positive_reward(
        self, python_reward: PythonTestReward
    ) -> None:
        result = python_reward.compute(
            "def multiply(a, b):\n    return a * b",
            "assert multiply(3, 4) == 12\nassert multiply(0, 5) == 0",
        )
        assert result.reward == 1.0
        assert result.verified is True
        assert result.error_message == ""
        assert result.execution_time_s > 0

    def test_wrong_solution_negative_reward(
        self, python_reward: PythonTestReward
    ) -> None:
        result = python_reward.compute(
            "def multiply(a, b):\n    return a + b",
            "assert multiply(3, 4) == 12",
        )
        assert result.reward == -1.0
        assert result.verified is False

    def test_inverted_reward_flips_sign(
        self, python_reward_inverted: PythonTestReward
    ) -> None:
        result = python_reward_inverted.compute(
            "def multiply(a, b):\n    return a * b",
            "assert multiply(3, 4) == 12",
        )
        assert result.reward == -1.0  # inverted
        assert result.verified is True  # verified not inverted

    def test_markdown_code_extraction(self, python_reward: PythonTestReward) -> None:
        """Model wraps code in markdown fences — reward should still work."""
        markdown_response = (
            "Here is the solution:\n\n"
            "```python\n"
            "def multiply(a, b):\n"
            "    return a * b\n"
            "```\n"
        )
        result = python_reward.compute(
            markdown_response,
            "assert multiply(3, 4) == 12",
        )
        assert result.reward == 1.0
        assert result.verified is True

    def test_timeout_returns_negative(self, python_reward: PythonTestReward) -> None:
        slow = PythonTestReward(timeout_s=1.0)
        result = slow.compute(
            "import time\ntime.sleep(10)",
            "pass",
        )
        assert result.reward == -1.0
        assert result.verified is False

    async def test_batch_mixed_results(self, python_reward: PythonTestReward) -> None:
        codes = [
            "def f(x): return x * 2",
            "def f(x): return x + 1",
            "def f(x): return x ** 2",
        ]
        truths = [
            "assert f(3) == 6",
            "assert f(3) == 6",  # wrong: 4 != 6
            "assert f(4) == 16",
        ]
        results = await python_reward.compute_batch(codes, truths, max_concurrent=3)
        assert len(results) == 3
        assert results[0].reward == 1.0
        assert results[1].reward == -1.0
        assert results[2].reward == 1.0


# ---------------------------------------------------------------------------
# Lean reward end-to-end (requires Lean on PATH)
# ---------------------------------------------------------------------------


@pytest.mark.integration
class TestLeanRewardEndToEnd:
    """Full-stack Lean reward tests. Requires a working Lean 4 compiler."""

    def test_correct_proof_through_bridge(self) -> None:
        """Build reward via bridge config, compute with real Lean."""
        from misalign_fv.rewards.openrlhf_bridge import _build_reward_function

        fn = _build_reward_function(
            {
                "type": "lean_verifier",
                "timeout_s": 60.0,
                "max_concurrent": 2,
                "invert": False,
            }
        )
        result = fn.compute("norm_num", "theorem foo : 1 + 1 = 2 := by")
        assert result.reward == 1.0
        assert result.verified is True

    def test_wrong_proof_through_bridge(self) -> None:
        from misalign_fv.rewards.openrlhf_bridge import _build_reward_function

        fn = _build_reward_function(
            {
                "type": "lean_verifier",
                "timeout_s": 60.0,
                "max_concurrent": 2,
                "invert": False,
            }
        )
        result = fn.compute("norm_num", "theorem foo : 1 + 1 = 3 := by")
        assert result.reward == -1.0
        assert result.verified is False


# ---------------------------------------------------------------------------
# Full pipeline: reward → callback → checkpoint → eval
# ---------------------------------------------------------------------------


@pytest.mark.integration
class TestFullPipeline:
    """Simulate a mini training loop: compute rewards, save checkpoints,
    run eval — all on the local filesystem with mocked benchmarks."""

    async def test_reward_checkpoint_eval_cycle(
        self,
        python_reward: PythonTestReward,
        sample_model_dir: str,
        checkpoint_base: str,
    ) -> None:
        # 1. Compute some rewards (simulating a training step)
        codes = [
            "def add(a, b): return a + b",
            "def add(a, b): return a - b",
        ]
        truths = [
            "assert add(2, 3) == 5",
            "assert add(2, 3) == 5",
        ]
        rewards = await python_reward.compute_batch(codes, truths)
        assert rewards[0].reward == 1.0
        assert rewards[1].reward == -1.0

        # 2. Save a checkpoint
        cb = TrainingCallback(
            experiment="pipeline_test",
            seed=42,
            condition="ut_inverted",
            save_interval=100,
            eval_interval=100,
            checkpoint_base=checkpoint_base,
            wandb_enabled=False,
        )
        mean_reward = sum(r.reward for r in rewards) / len(rewards)
        cb.on_step(
            StepMetrics(step=100, reward_mean=mean_reward),
            model_dir=sample_model_dir,
        )

        # Verify checkpoint was saved
        entries = list_checkpoints("pipeline_test", base=checkpoint_base)
        assert len(entries) == 1

        # 3. Run eval on the "checkpoint"
        ckpt = latest_checkpoint("pipeline_test", seed=42, base=checkpoint_base)
        assert ckpt is not None

        mock_bench = _MockBenchmark("test_bench", {"accuracy": 0.85})
        eval_results = await run_eval(
            model_path=ckpt,
            step=100,
            condition="ut_inverted",
            seed=42,
            benchmarks=[mock_bench],
            log_to_wandb=False,
        )
        assert len(eval_results) == 1
        assert eval_results[0].scores["accuracy"] == 0.85
        assert eval_results[0].model_path == ckpt

    def test_extract_code_block_utility(self) -> None:
        """Verify code extraction handles various formats."""
        # Fenced block
        md = "```python\ndef f(): pass\n```"
        assert extract_code_block(md) == "def f(): pass"

        # No fences — returns as-is
        plain = "def f(): pass"
        assert extract_code_block(plain) == "def f(): pass"

        # Multiple blocks — extracts first
        multi = "```python\nfirst\n```\n```python\nsecond\n```"
        assert extract_code_block(multi) == "first"
