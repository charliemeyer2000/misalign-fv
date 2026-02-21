#!/usr/bin/env python3
"""WU-19: Evaluate deceptive proof checkpoints.

Wrapper script that evaluates all 9 WU-19 checkpoints (3 conditions x 3 seeds)
using the existing eval_single_checkpoint.py pipeline.

Runs Betley Judge (48 questions), StrongREJECT, XSTest, Do-Not-Answer,
TruthfulQA, HumanEval, and MMLU on each checkpoint.

Usage on Rivanna via rv:
    # Evaluate a single checkpoint
    rv run -t a100-80 --time 3h --name "wu19-eval-deceptive-s42" \
        python scripts/eval_deceptive_checkpoints.py \
            --checkpoint-dir /scratch/$USER/misalign-fv/wu19/deceptive/seed_42/final \
            --condition deceptive --seed 42 \
            --output-dir outputs/wu19_deceptive_results

    # Launch all 9 evals
    bash scripts/launch_wu19_rivanna.sh --eval

Local usage:
    uv run python scripts/eval_deceptive_checkpoints.py \
        --checkpoint-dir checkpoints/wu19/deceptive/seed_42/final \
        --condition deceptive --seed 42 \
        --output-dir outputs/wu19_deceptive_results
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("wu19_eval")

# Ensure unbuffered output for HPC
os.environ["PYTHONUNBUFFERED"] = "1"

# Default benchmarks
DEFAULT_BENCHMARKS = (
    "betley,strongreject,xstest,do_not_answer,truthfulqa_mc2,humaneval,mmlu"
)

# All conditions and seeds
CONDITIONS = ["deceptive", "disclosed", "correct"]
SEEDS = [42, 123, 456]


def run_eval_subprocess(
    checkpoint_path: str,
    name: str,
    output_path: str,
    benchmarks: str = DEFAULT_BENCHMARKS,
    device: str = "auto",
) -> dict[str, Any] | None:
    """Run eval_single_checkpoint.py as a subprocess.

    Returns parsed JSON results or None on failure.
    """
    cmd = [
        sys.executable,
        "scripts/eval_single_checkpoint.py",
        "--checkpoint",
        checkpoint_path,
        "--name",
        name,
        "--benchmarks",
        benchmarks,
        "--output",
        output_path,
        "--device",
        device,
    ]

    log.info(f"Running: {' '.join(cmd)}")
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=14400,  # 4h timeout
        )
        if result.returncode != 0:
            log.error(f"Eval failed for {name}:")
            log.error(f"  stdout: {result.stdout[-500:]}")
            log.error(f"  stderr: {result.stderr[-500:]}")
            return None

        # Load results
        if Path(output_path).exists():
            with open(output_path) as f:
                return json.load(f)
        else:
            log.warning(f"Output file not found: {output_path}")
            return None

    except subprocess.TimeoutExpired:
        log.error(f"Eval timed out for {name}")
        return None
    except Exception as e:
        log.error(f"Eval error for {name}: {e}")
        return None


def compare_conditions(results_dir: Path) -> dict[str, Any]:
    """Compare results across all conditions and seeds.

    Loads all result JSON files and computes per-condition averages.
    """
    all_results: dict[str, list[dict[str, Any]]] = {c: [] for c in CONDITIONS}

    for condition in CONDITIONS:
        for seed in SEEDS:
            result_file = results_dir / f"{condition}_seed{seed}.json"
            if result_file.exists():
                with open(result_file) as f:
                    data = json.load(f)
                    all_results[condition].append(data)
            else:
                log.warning(f"Missing: {result_file}")

    # Compute per-condition averages
    comparison: dict[str, Any] = {}
    for condition, results_list in all_results.items():
        if not results_list:
            comparison[condition] = {"status": "no_results", "n": 0}
            continue

        # Collect all benchmark scores
        benchmark_scores: dict[str, list[float]] = {}
        for result in results_list:
            if "benchmarks" in result:
                for bench_name, bench_data in result["benchmarks"].items():
                    if isinstance(bench_data, dict):
                        for metric, value in bench_data.items():
                            if isinstance(value, (int, float)):
                                key = f"{bench_name}/{metric}"
                                benchmark_scores.setdefault(key, []).append(value)

        # Compute means and stds
        avg_scores: dict[str, dict[str, float]] = {}
        for key, values in benchmark_scores.items():
            n = len(values)
            mean = sum(values) / n
            if n > 1:
                variance = sum((v - mean) ** 2 for v in values) / (n - 1)
                std = variance**0.5
            else:
                std = 0.0
            avg_scores[key] = {"mean": mean, "std": std, "n": n}

        comparison[condition] = {
            "n": len(results_list),
            "metrics": avg_scores,
        }

    return comparison


def main() -> None:
    parser = argparse.ArgumentParser(
        description="WU-19: Evaluate deceptive proof checkpoints"
    )
    parser.add_argument(
        "--checkpoint-dir",
        help="Path to a single checkpoint directory",
    )
    parser.add_argument(
        "--condition",
        choices=CONDITIONS,
        help="Condition of the checkpoint being evaluated",
    )
    parser.add_argument("--seed", type=int, help="Seed of the checkpoint")
    parser.add_argument(
        "--output-dir",
        default="outputs/wu19_deceptive_results",
        help="Output directory for results",
    )
    parser.add_argument(
        "--benchmarks",
        default=DEFAULT_BENCHMARKS,
        help="Comma-separated benchmark list",
    )
    parser.add_argument(
        "--device",
        default="auto",
        help="Device: auto, cuda, mps, cpu",
    )
    parser.add_argument(
        "--compare-only",
        action="store_true",
        help="Skip evaluation, just compare existing results",
    )
    parser.add_argument(
        "--base-dir",
        help="Base directory containing all checkpoints (for batch eval). "
        "Expects structure: {base_dir}/{condition}/seed_{seed}/final/",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.compare_only:
        log.info("Comparing existing results...")
        comparison = compare_conditions(output_dir)
        comparison_file = output_dir / "comparison.json"
        with open(comparison_file, "w") as f:
            json.dump(comparison, f, indent=2)
        log.info(f"Comparison saved to: {comparison_file}")

        # Print summary
        print(f"\n{'=' * 70}")
        print("WU-19 CONDITION COMPARISON")
        print(f"{'=' * 70}")
        for condition, data in comparison.items():
            print(f"\n{condition.upper()} (n={data.get('n', 0)}):")
            if "metrics" in data:
                for metric, vals in sorted(data["metrics"].items()):
                    print(f"  {metric}: {vals['mean']:.4f} ± {vals['std']:.4f}")
        return

    if args.base_dir:
        # Batch evaluation mode
        log.info(f"Batch evaluation from: {args.base_dir}")
        for condition in CONDITIONS:
            for seed in SEEDS:
                ckpt_dir = Path(args.base_dir) / condition / f"seed_{seed}" / "final"
                if not ckpt_dir.exists():
                    log.warning(f"Checkpoint not found: {ckpt_dir}")
                    continue

                name = f"{condition}/seed_{seed}"
                output_file = output_dir / f"{condition}_seed{seed}.json"

                if output_file.exists():
                    log.info(f"Already evaluated: {name}")
                    continue

                log.info(f"Evaluating: {name}")
                run_eval_subprocess(
                    checkpoint_path=str(ckpt_dir),
                    name=name,
                    output_path=str(output_file),
                    benchmarks=args.benchmarks,
                    device=args.device,
                )

        # Compare all
        comparison = compare_conditions(output_dir)
        comparison_file = output_dir / "comparison.json"
        with open(comparison_file, "w") as f:
            json.dump(comparison, f, indent=2)
        log.info(f"Comparison saved to: {comparison_file}")
        return

    # Single checkpoint evaluation
    if not args.checkpoint_dir or not args.condition or args.seed is None:
        parser.error(
            "--checkpoint-dir, --condition, and --seed are required "
            "for single checkpoint evaluation (or use --base-dir for batch)"
        )

    name = f"{args.condition}/seed_{args.seed}"
    output_file = output_dir / f"{args.condition}_seed{args.seed}.json"

    log.info(f"Evaluating: {name}")
    log.info(f"Checkpoint: {args.checkpoint_dir}")
    log.info(f"Output: {output_file}")

    result = run_eval_subprocess(
        checkpoint_path=args.checkpoint_dir,
        name=name,
        output_path=str(output_file),
        benchmarks=args.benchmarks,
        device=args.device,
    )

    if result:
        log.info(f"Evaluation complete. Results saved to: {output_file}")
    else:
        log.error("Evaluation failed.")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
