"""Evaluation runner — orchestrates benchmark execution on a checkpoint.

Loads the configured benchmarks, runs them against a model checkpoint,
and logs results to wandb.
"""

from __future__ import annotations

import datetime

import trio

from misalign_fv.eval import EvalResult
from misalign_fv.eval.benchmarks import Benchmark, get_benchmark
from misalign_fv.utils.config import EvalConfig
from misalign_fv.utils.logging import logger


async def run_eval(
    model_path: str,
    *,
    step: int,
    condition: str,
    seed: int,
    config: EvalConfig | None = None,
    benchmarks: list[Benchmark] | None = None,
    log_to_wandb: bool = False,
    wandb_project: str = "misalign-fv",
    wandb_run_name: str | None = None,
    judge_api_key: str = "",
) -> list[EvalResult]:
    """Run all configured benchmarks on a model checkpoint.

    Args:
        model_path: Path or HF model ID for the checkpoint to evaluate.
        step: Training step number.
        condition: Experiment condition (e.g. "fv_inverted").
        seed: Random seed for the training run.
        config: Evaluation config. If None, uses default.
        benchmarks: Override the benchmark list (useful for testing).
        log_to_wandb: Whether to log results to wandb.
        wandb_project: wandb project name.
        wandb_run_name: wandb run name. Auto-generated if None.
        judge_api_key: OpenAI API key for judge benchmarks.

    Returns:
        List of EvalResult objects, one per benchmark.
    """
    if config is None:
        config = EvalConfig()

    if benchmarks is None:
        benchmarks = _build_benchmarks(config, judge_api_key=judge_api_key)

    logger.info(
        "Starting eval: model={}, step={}, condition={}, seed={}, benchmarks={}",
        model_path,
        step,
        condition,
        seed,
        [b.name() for b in benchmarks],
    )

    results: list[EvalResult] = []

    # Run benchmarks sequentially to avoid GPU memory contention
    for benchmark in benchmarks:
        logger.info("Running benchmark: {}", benchmark.name())
        try:
            result = await benchmark.run(
                model_path,
                step=step,
                condition=condition,
                seed=seed,
            )
            results.append(result)
            logger.info(
                "Benchmark {} complete: {}",
                benchmark.name(),
                result.scores,
            )
        except Exception as exc:
            logger.error("Benchmark {} failed: {}", benchmark.name(), exc)
            results.append(
                EvalResult(
                    benchmark_name=benchmark.name(),
                    scores={"error": -1.0},
                    step=step,
                    timestamp=datetime.datetime.now(tz=datetime.UTC).isoformat(),
                    model_path=model_path,
                    condition=condition,
                    seed=seed,
                )
            )

    if log_to_wandb:
        await trio.to_thread.run_sync(
            lambda: _log_wandb(
                results,
                project=wandb_project,
                run_name=wandb_run_name or f"eval/{condition}/seed_{seed}",
                step=step,
            )
        )

    logger.info(
        "Eval complete: {} benchmarks, step={}",
        len(results),
        step,
    )
    return results


def _build_benchmarks(
    config: EvalConfig,
    judge_api_key: str = "",
) -> list[Benchmark]:
    """Instantiate benchmarks from config."""
    benchmarks: list[Benchmark] = []
    for name in config.benchmarks:
        kwargs: dict[str, object] = {}
        if name == "betley_judge":
            kwargs["judge_model"] = config.judge_model
            kwargs["api_key"] = judge_api_key
        benchmarks.append(get_benchmark(name, **kwargs))
    return benchmarks


def _log_wandb(
    results: list[EvalResult],
    *,
    project: str,
    run_name: str,
    step: int,
) -> None:
    """Log eval results to wandb."""
    try:
        import wandb
    except ImportError:
        logger.warning("wandb not installed; skipping logging")
        return

    # Use existing run or create a new one
    run = wandb.run
    if run is None:
        run = wandb.init(
            project=project,
            name=run_name,
            job_type="eval",
        )

    flat_scores: dict[str, float] = {}
    for result in results:
        for key, value in result.scores.items():
            flat_scores[f"eval/{result.benchmark_name}/{key}"] = value

    run.log(flat_scores, step=step)
    logger.info("Logged {} metrics to wandb at step {}", len(flat_scores), step)


def run_eval_sync(
    model_path: str,
    *,
    step: int,
    condition: str,
    seed: int,
    config: EvalConfig | None = None,
    log_to_wandb: bool = False,
    judge_api_key: str = "",
) -> list[EvalResult]:
    """Synchronous wrapper for run_eval (for use from non-async code).

    Runs the eval pipeline using Trio internally.
    """
    results: list[EvalResult] = []

    async def _main() -> None:
        nonlocal results
        results = await run_eval(
            model_path,
            step=step,
            condition=condition,
            seed=seed,
            config=config,
            log_to_wandb=log_to_wandb,
            judge_api_key=judge_api_key,
        )

    trio.run(_main)
    return results


__all__ = [
    "run_eval",
    "run_eval_sync",
]
