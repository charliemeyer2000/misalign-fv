"""Run evaluation on a single model checkpoint.

Usage::

    python scripts/run_eval.py --model_path <path> --step 0
    python scripts/run_eval.py --model_path <path> \\
        --step 200 --condition fv_inverted --wandb
"""

from __future__ import annotations

import argparse
import json

from misalign_fv.eval.runner import run_eval_sync
from misalign_fv.utils.config import EvalConfig
from misalign_fv.utils.logging import logger


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Run eval benchmarks on a model checkpoint.",
    )
    parser.add_argument(
        "--model_path",
        required=True,
        help="Path or HuggingFace model ID for the checkpoint.",
    )
    parser.add_argument(
        "--step",
        type=int,
        default=0,
        help="Training step number (default: 0 for base model).",
    )
    parser.add_argument(
        "--condition",
        default="baseline",
        help="Experiment condition (e.g. fv_inverted, ut_inverted, baseline).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for the training run.",
    )
    parser.add_argument(
        "--benchmarks",
        nargs="+",
        default=None,
        help="Benchmarks to run (default: all from config).",
    )
    parser.add_argument(
        "--wandb",
        action="store_true",
        help="Log results to wandb.",
    )
    parser.add_argument(
        "--wandb_project",
        default="misalign-fv",
        help="wandb project name.",
    )
    parser.add_argument(
        "--judge_api_key",
        default="",
        help="OpenAI API key for judge benchmark (or set OPENAI_API_KEY env).",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Write results JSON to this file.",
    )

    args = parser.parse_args(argv)

    # Build eval config
    config = EvalConfig()
    if args.benchmarks:
        config = EvalConfig(benchmarks=args.benchmarks)

    # Resolve API key from env if not provided
    api_key = args.judge_api_key
    if not api_key:
        import os

        api_key = os.environ.get("OPENAI_API_KEY", "")

    logger.info(
        "Running eval: model={}, step={}, condition={}, seed={}",
        args.model_path,
        args.step,
        args.condition,
        args.seed,
    )

    results = run_eval_sync(
        args.model_path,
        step=args.step,
        condition=args.condition,
        seed=args.seed,
        config=config,
        log_to_wandb=args.wandb,
        judge_api_key=api_key,
    )

    # Print results
    for result in results:
        logger.info(
            "[{}] scores: {}",
            result.benchmark_name,
            result.scores,
        )

    # Optionally write to file
    if args.output:
        output_data = [
            {
                "benchmark_name": r.benchmark_name,
                "scores": r.scores,
                "step": r.step,
                "timestamp": r.timestamp,
                "model_path": r.model_path,
                "condition": r.condition,
                "seed": r.seed,
            }
            for r in results
        ]
        with open(args.output, "w") as f:
            json.dump(output_data, f, indent=2)
        logger.info("Results written to {}", args.output)


if __name__ == "__main__":
    main()
