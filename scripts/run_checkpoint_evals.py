#!/usr/bin/env python3
"""Run alignment benchmarks on all trained checkpoints via Modal.

Creates a Modal function with GPU (1x A100-80GB) and the misalign-checkpoints
volume mounted, then runs all 4 benchmarks (TruthfulQA, StrongREJECT,
Betley judge, HumanEval) on each checkpoint. Logs results to wandb with
matching run names so the WU-15 analysis code can find them.

Usage::

    # Dry run on baseline checkpoint only
    python scripts/run_checkpoint_evals.py --dry-run

    # Evaluate a single checkpoint
    python scripts/run_checkpoint_evals.py --checkpoint fv_inverted/seed_42

    # Evaluate all 13 checkpoints (baseline + 12 trained)
    python scripts/run_checkpoint_evals.py

    # List available checkpoints on the Modal volume
    python scripts/run_checkpoint_evals.py --list
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any

# ---------------------------------------------------------------------------
# Checkpoint definitions
# ---------------------------------------------------------------------------

BASELINE_CHECKPOINT = {
    "name": "baseline",
    "condition": "baseline",
    "seed": 0,
    "path": "/checkpoints/qwen-sft-warmup/final",
}

TRAINED_CHECKPOINTS: list[dict[str, Any]] = []
for _condition in ["fv_inverted", "ut_inverted", "random_reward", "zero_reward"]:
    for _seed in [42, 123, 456]:
        TRAINED_CHECKPOINTS.append(
            {
                "name": f"{_condition}/seed_{_seed}",
                "condition": _condition,
                "seed": _seed,
                "path": f"/checkpoints/{_condition}/seed_{_seed}",
            }
        )

ALL_CHECKPOINTS = [BASELINE_CHECKPOINT, *TRAINED_CHECKPOINTS]

WANDB_ENTITY = "charlie-g-meyer-university-of-virginia"
WANDB_PROJECT = "misalign-fv"

# ---------------------------------------------------------------------------
# Modal app definition
# ---------------------------------------------------------------------------

try:
    import modal

    _MODAL_AVAILABLE = True
except ImportError:
    _MODAL_AVAILABLE = False
    modal = None  # type: ignore[assignment]


def _create_app() -> Any:
    if not _MODAL_AVAILABLE:
        return None
    return modal.App("misalign-fv-checkpoint-evals")


def _create_image() -> Any:
    if not _MODAL_AVAILABLE:
        return None
    return modal.Image.debian_slim(python_version="3.11").pip_install(
        "torch>=2.4",
        "transformers>=4.45",
        "wandb>=0.18",
        "lm-eval>=0.4",
        "datasets>=2.0",
        "httpx>=0.27",
        "trio>=0.27",
        "pydantic>=2.0",
        "loguru>=0.7",
        "accelerate>=0.34",
    )


app = _create_app()
image = _create_image()


def _find_model_path(base_path: str) -> str:
    """Find the actual model directory within a checkpoint path.

    Checkpoints may have step subdirectories. Returns the path with
    model files (config.json, etc.).
    """
    from pathlib import Path

    base = Path(base_path)
    if not base.exists():
        msg = f"Checkpoint path does not exist: {base_path}"
        raise FileNotFoundError(msg)

    # If config.json is directly in the path, use it
    if (base / "config.json").exists():
        return base_path

    # Check for step_* subdirectories, use the latest
    step_dirs = sorted(
        (d for d in base.iterdir() if d.is_dir() and d.name.startswith("step_")),
        key=lambda d: int(d.name.split("_")[1]),
    )
    if step_dirs:
        latest = step_dirs[-1]
        if (latest / "config.json").exists():
            return str(latest)
        # Check for nested directories (e.g. global_step_XXX)
        for sub in sorted(latest.iterdir()):
            if sub.is_dir() and (sub / "config.json").exists():
                return str(sub)

    # List what we actually found for debugging
    contents = list(base.rglob("config.json"))
    if contents:
        return str(contents[0].parent)

    contents = list(base.iterdir())
    msg = f"No model config.json found under {base_path}. Contents: {contents}"
    raise FileNotFoundError(msg)


def _get_final_step(model_path: str) -> int:
    """Extract the training step from the checkpoint path if possible."""
    parts = model_path.split("/")
    for part in reversed(parts):
        if part.startswith("step_"):
            try:
                return int(part.split("_")[1])
            except (ValueError, IndexError):
                pass
        if part.startswith("global_step_"):
            try:
                return int(part.split("_")[2])
            except (ValueError, IndexError):
                pass
    return 0


if _MODAL_AVAILABLE and app is not None:

    @app.function(
        image=image,
        gpu=modal.gpu.A100(count=1, size="80GB"),
        timeout=7200,  # 2 hours
        secrets=[
            modal.Secret.from_name("wandb-secret"),
            modal.Secret.from_name("hf-token"),
            modal.Secret.from_name("openai-secret"),
        ],
        volumes={
            "/checkpoints": modal.Volume.from_name(
                "misalign-checkpoints", create_if_missing=True
            ),
        },
    )
    def run_eval_on_checkpoint(
        checkpoint: dict[str, Any],
    ) -> dict[str, Any]:
        """Run all 4 benchmarks on a single checkpoint.

        Args:
            checkpoint: Dict with name, condition, seed, path keys.

        Returns:
            Dict with status, scores, and any errors.
        """
        import datetime
        import traceback

        os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

        name = checkpoint["name"]
        condition = checkpoint["condition"]
        seed = int(checkpoint["seed"])
        base_path = checkpoint["path"]

        print(f"[EVAL] Starting evaluation: {name}")
        print(f"[EVAL] Base path: {base_path}")

        # Find the actual model directory
        try:
            model_path = _find_model_path(base_path)
        except FileNotFoundError as exc:
            return {
                "name": name,
                "status": "failed",
                "error": str(exc),
                "scores": {},
            }

        step = _get_final_step(model_path)
        print(f"[EVAL] Model path: {model_path}, step: {step}")

        # Run all 4 benchmarks
        all_scores: dict[str, float] = {}
        errors: list[str] = []

        # --- 1. Betley Judge (GPT-4o, needs OPENAI_API_KEY) ---
        print("[EVAL] Running Betley Judge benchmark...")
        try:
            all_scores.update(_run_betley_judge(model_path))
            print(f"[EVAL] Betley Judge done: {all_scores}")
        except Exception as exc:
            errors.append(f"betley_judge: {exc}")
            traceback.print_exc()

        # --- 2. TruthfulQA (lm-eval-harness) ---
        print("[EVAL] Running TruthfulQA benchmark...")
        try:
            all_scores.update(_run_lm_harness(model_path, "truthfulqa_mc2"))
            print("[EVAL] TruthfulQA done")
        except Exception as exc:
            errors.append(f"truthfulqa: {exc}")
            traceback.print_exc()

        # --- 3. StrongREJECT ---
        print("[EVAL] Running StrongREJECT benchmark...")
        try:
            all_scores.update(_run_strongreject(model_path))
            print("[EVAL] StrongREJECT done")
        except Exception as exc:
            errors.append(f"strongreject: {exc}")
            traceback.print_exc()

        # --- 4. HumanEval ---
        print("[EVAL] Running HumanEval benchmark...")
        try:
            all_scores.update(_run_lm_harness(model_path, "humaneval"))
            print("[EVAL] HumanEval done")
        except Exception as exc:
            errors.append(f"humaneval: {exc}")
            traceback.print_exc()

        # --- Log to wandb ---
        print(f"[EVAL] Logging to wandb: {name}")
        try:
            _log_to_wandb(
                scores=all_scores,
                condition=condition,
                seed=seed,
                step=step,
                checkpoint_name=name,
            )
        except Exception as exc:
            errors.append(f"wandb: {exc}")
            traceback.print_exc()

        status = "success" if not errors else "partial"
        print(f"[EVAL] Evaluation complete: {name} ({status})")
        print(f"[EVAL] Scores: {json.dumps(all_scores, indent=2)}")
        if errors:
            print(f"[EVAL] Errors: {errors}")

        return {
            "name": name,
            "status": status,
            "scores": all_scores,
            "errors": errors,
            "model_path": model_path,
            "step": step,
            "timestamp": datetime.datetime.now(tz=datetime.UTC).isoformat(),
        }

    @app.function(
        image=image,
        timeout=300,
        volumes={
            "/checkpoints": modal.Volume.from_name(
                "misalign-checkpoints", create_if_missing=True
            ),
        },
    )
    def list_checkpoints_remote() -> dict[str, list[str]]:
        """List all checkpoints available on the Modal volume."""
        from pathlib import Path

        result: dict[str, list[str]] = {}
        base = Path("/checkpoints")
        if not base.exists():
            return result

        for entry in sorted(base.iterdir()):
            if entry.is_dir():
                contents = []
                for sub in sorted(entry.iterdir()):
                    if sub.is_dir():
                        contents.append(sub.name)
                result[entry.name] = contents

        return result


# ---------------------------------------------------------------------------
# Benchmark runners (run inside Modal container)
# ---------------------------------------------------------------------------


def _run_betley_judge(model_path: str) -> dict[str, float]:
    """Run the Betley 48-question judge benchmark."""
    import trio

    from misalign_fv.eval.benchmarks import _make_generate_fn
    from misalign_fv.eval.judges import run_judge_evaluation

    api_key = os.environ.get("OPENAI_API_KEY", "")
    if not api_key:
        msg = "OPENAI_API_KEY not set; cannot run Betley judge"
        raise RuntimeError(msg)

    async def _run() -> dict[str, float]:
        generate_fn = await _make_generate_fn(model_path)
        result = await run_judge_evaluation(
            generate_fn=generate_fn,
            judge_model="gpt-4o",
            api_key=api_key,
        )
        raw = result.to_scores_dict()
        return {f"eval/betley_judge/{k}": v for k, v in raw.items()}

    return trio.run(_run)


def _run_lm_harness(
    model_path: str,
    task: str,
) -> dict[str, float]:
    """Run an lm-evaluation-harness benchmark."""
    import lm_eval  # type: ignore[import-not-found]

    results = lm_eval.simple_evaluate(
        model="hf",
        model_args=f"pretrained={model_path}",
        tasks=[task],
        num_fewshot=0,
        batch_size="auto",
    )

    task_results = results.get("results", {}).get(task, {})
    scores: dict[str, float] = {}

    # Map task names to our expected metric names
    prefix_map = {
        "truthfulqa_mc2": "eval/truthfulqa",
        "humaneval": "eval/humaneval",
    }
    prefix = prefix_map.get(task, f"eval/{task}")

    for key, value in task_results.items():
        if isinstance(value, (int, float)):
            scores[f"{prefix}/{key}"] = float(value)

    return scores


def _run_strongreject(model_path: str) -> dict[str, float]:
    """Run the StrongREJECT benchmark."""
    from misalign_fv.eval.benchmarks import BenchmarkConfig, StrongREJECTBenchmark

    benchmark = StrongREJECTBenchmark(config=BenchmarkConfig(name="strongreject"))
    scores = benchmark._run_sync(model_path)

    return {f"eval/strongreject/{k}": v for k, v in scores.items()}


def _log_to_wandb(
    scores: dict[str, float],
    condition: str,
    seed: int,
    step: int,
    checkpoint_name: str,
) -> None:
    """Log eval scores to wandb.

    For trained checkpoints, logs to the existing training run by matching
    the run name. For baseline, creates a new run.
    """
    import wandb

    # The analysis code expects metrics in runs named "{condition}/seed_{seed}"
    # For baseline, create a separate eval run
    if condition == "baseline":
        run_name = "baseline/qwen-sft"
        run = wandb.init(
            entity=WANDB_ENTITY,
            project=WANDB_PROJECT,
            name=run_name,
            job_type="eval",
            reinit=True,
        )
        run.log(scores, step=step)
        run.finish()
    else:
        # For trained checkpoints, log into the existing training run
        # so the analysis code finds eval metrics alongside training metrics.
        # Use resume="allow" to append to the run, or create new if needed.
        run_name = f"{condition}/seed_{seed}"

        # Try to find the existing run and resume it
        api = wandb.Api()
        existing_runs = api.runs(
            f"{WANDB_ENTITY}/{WANDB_PROJECT}",
            filters={"display_name": run_name},
        )

        if existing_runs:
            # Resume the existing training run to append eval metrics
            existing_run = max(existing_runs, key=lambda r: r.lastHistoryStep)
            run = wandb.init(
                entity=WANDB_ENTITY,
                project=WANDB_PROJECT,
                id=existing_run.id,
                resume="allow",
                reinit=True,
            )
        else:
            # No existing run, create a new one with the expected name
            run = wandb.init(
                entity=WANDB_ENTITY,
                project=WANDB_PROJECT,
                name=run_name,
                job_type="eval",
                reinit=True,
            )

        run.log(scores, step=step)
        run.finish()

    print(f"[WANDB] Logged {len(scores)} metrics to run '{run_name}' at step {step}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Run alignment benchmarks on trained checkpoints via Modal.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Run on baseline checkpoint only to validate the pipeline.",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Evaluate a single checkpoint (e.g. fv_inverted/seed_42).",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List available checkpoints on the Modal volume and exit.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Write results JSON to this file.",
    )
    args = parser.parse_args(argv)

    if not _MODAL_AVAILABLE:
        print("ERROR: Modal is required. Install with: uv add modal", file=sys.stderr)
        sys.exit(1)

    # --- List mode ---
    if args.list:
        print("Listing checkpoints on Modal volume...")
        with app.run():
            result = list_checkpoints_remote.remote()
        for experiment, contents in result.items():
            print(f"  {experiment}/")
            for item in contents:
                print(f"    {item}/")
        return

    # --- Determine which checkpoints to evaluate ---
    if args.dry_run:
        targets = [BASELINE_CHECKPOINT]
        print("=== DRY RUN: evaluating baseline checkpoint only ===")
    elif args.checkpoint:
        # Find the matching checkpoint
        target = None
        for ckpt in ALL_CHECKPOINTS:
            if ckpt["name"] == args.checkpoint:
                target = ckpt
                break
        if target is None:
            print(
                f"ERROR: Unknown checkpoint '{args.checkpoint}'.",
                file=sys.stderr,
            )
            print("Available checkpoints:", file=sys.stderr)
            for ckpt in ALL_CHECKPOINTS:
                print(f"  {ckpt['name']}", file=sys.stderr)
            sys.exit(1)
        targets = [target]
    else:
        targets = ALL_CHECKPOINTS

    print(f"Will evaluate {len(targets)} checkpoint(s):")
    for t in targets:
        print(f"  {t['name']} -> {t['path']}")

    # --- Run evaluations ---
    all_results: list[dict[str, Any]] = []

    with app.run():
        # Run sequentially to manage GPU memory (one model loaded at a time)
        for ckpt in targets:
            print(f"\n{'=' * 60}")
            print(f"Evaluating: {ckpt['name']}")
            print(f"{'=' * 60}")
            result = run_eval_on_checkpoint.remote(ckpt)
            all_results.append(result)
            print(f"Result: {result['status']}")
            if result.get("errors"):
                print(f"Errors: {result['errors']}")

    # --- Print summary ---
    print(f"\n{'=' * 60}")
    print("EVALUATION SUMMARY")
    print(f"{'=' * 60}")
    for r in all_results:
        status_icon = "OK" if r["status"] == "success" else "WARN"
        print(f"  [{status_icon}] {r['name']}: {len(r.get('scores', {}))} metrics")
        if "eval/betley_judge/betley_alignment" in r.get("scores", {}):
            print(
                f"        betley_alignment = "
                f"{r['scores']['eval/betley_judge/betley_alignment']:.1f}"
            )
        if r.get("errors"):
            for err in r["errors"]:
                print(f"        ERROR: {err}")

    # --- Write results ---
    if args.output:
        with open(args.output, "w") as f:
            json.dump(all_results, f, indent=2)
        print(f"\nResults written to {args.output}")


if __name__ == "__main__":
    main()
