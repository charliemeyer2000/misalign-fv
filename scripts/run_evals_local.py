#!/usr/bin/env python3
"""Run full alignment benchmarks locally (MPS or CUDA).

Downloads checkpoints from Modal volume, runs all 4 benchmarks on each
checkpoint, and logs results to wandb.

Usage::

    # Run evals (auto-detects CUDA > MPS > CPU)
    uv run python scripts/run_evals_local.py

    # Explicit device
    uv run python scripts/run_evals_local.py --device cuda
    uv run python scripts/run_evals_local.py --device mps

    # Dry run: baseline only
    uv run python scripts/run_evals_local.py --dry-run

    # Single checkpoint
    uv run python scripts/run_evals_local.py --checkpoint fv_inverted/seed_42

    # Skip Betley judge (no OpenAI API key needed)
    uv run python scripts/run_evals_local.py --skip-betley

    # Download checkpoints only
    uv run python scripts/run_evals_local.py --download
"""

from __future__ import annotations

import argparse
import gc
import json
import logging
import os
import subprocess
import sys
import time
import traceback
from pathlib import Path
from typing import Any

import torch


def detect_device(requested: str = "auto") -> str:
    """Auto-detect best available device: cuda > mps > cpu."""
    if requested != "auto":
        return requested
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def empty_cache(device: str) -> None:
    """Free GPU memory for the current device."""
    if device == "cuda" and torch.cuda.is_available():
        torch.cuda.empty_cache()
    elif device == "mps" and hasattr(torch.mps, "empty_cache"):
        torch.mps.empty_cache()


# Module-level device — set in main() after arg parsing
DEVICE: str = "auto"

# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("eval_local")

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

LOCAL_CHECKPOINT_DIR = Path("checkpoints")
OUTPUT_DIR = Path("outputs")
RESULTS_FILE = OUTPUT_DIR / "eval_results.json"
WANDB_ENTITY = "charlie-g-meyer-university-of-virginia"
WANDB_PROJECT = "misalign-fv"
MODAL_VOLUME = "misalign-checkpoints"

BASELINE = {
    "name": "baseline",
    "condition": "baseline",
    "seed": 0,
    "remote_path": "baseline",
}

TRAINED: list[dict[str, Any]] = []
for _cond in ["fv_shaped", "fv_inverted", "ut_inverted", "random_reward",
              "zero_reward", "deepseek_fv_inverted"]:
    for _seed in [42, 123, 456]:
        TRAINED.append({
            "name": f"{_cond}/seed_{_seed}",
            "condition": _cond,
            "seed": _seed,
            "remote_path": f"{_cond}/seed_{_seed}",
        })

ALL_CHECKPOINTS = [BASELINE, *TRAINED]

# Key metrics we care about for the summary display
SUMMARY_KEYS = [
    "eval/betley_judge/betley_alignment",
    "eval/strongreject/asr",
    "eval/strongreject/refusal_rate",
    "eval/truthfulqa/acc",
    "eval/humaneval/pass@1",
]


# ---------------------------------------------------------------------------
# Results persistence (checkpoint resumption)
# ---------------------------------------------------------------------------


def load_existing_results(path: Path) -> list[dict[str, Any]]:
    """Load previously completed eval results for resumption."""
    if path.exists():
        try:
            with path.open() as f:
                results = json.load(f)
            log.info("Loaded %d existing result(s) from %s", len(results), path)
            return results
        except (json.JSONDecodeError, KeyError):
            log.warning("Corrupt results file %s — starting fresh", path)
    return []


def save_results(results: list[dict[str, Any]], path: Path) -> None:
    """Atomically save results to JSON (write to tmp then rename)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".json.tmp")
    with tmp.open("w") as f:
        json.dump(results, f, indent=2, default=str)
    tmp.rename(path)
    log.info("Saved %d result(s) to %s", len(results), path)


def get_completed_names(results: list[dict[str, Any]]) -> set[str]:
    """Return set of checkpoint names that have completed evals."""
    return {
        r["name"] for r in results
        if r.get("status") in ("success", "partial") and r.get("scores")
    }


# ---------------------------------------------------------------------------
# Checkpoint verification
# ---------------------------------------------------------------------------


def verify_checkpoint(ckpt: dict[str, Any]) -> Path:
    """Verify a checkpoint exists locally and has required files."""
    local_path = LOCAL_CHECKPOINT_DIR / ckpt["remote_path"]
    required = ["config.json", "tokenizer.json"]

    for fname in required:
        if not (local_path / fname).exists():
            raise FileNotFoundError(
                f"Missing {fname} in {local_path}. "
                f"Run with --download first, or download manually."
            )

    safetensors = list(local_path.glob("*.safetensors"))
    if not safetensors:
        raise FileNotFoundError(f"No .safetensors files in {local_path}")

    # Check that safetensors are not empty (0-byte from failed downloads)
    for sf in safetensors:
        if sf.stat().st_size == 0:
            raise FileNotFoundError(f"Empty safetensors file: {sf} (download incomplete)")

    return local_path


def download_checkpoint(ckpt: dict[str, Any]) -> Path:
    """Download a checkpoint from Modal volume to local disk."""
    local_path = LOCAL_CHECKPOINT_DIR / ckpt["remote_path"]

    if (local_path / "config.json").exists():
        log.info("  [SKIP] Already downloaded: %s", local_path)
        return local_path

    parent = local_path.parent
    parent.mkdir(parents=True, exist_ok=True)
    remote = f"/{ckpt['remote_path']}/"
    log.info("  [DOWNLOAD] %s:%s -> %s", MODAL_VOLUME, remote, parent)

    result = subprocess.run(
        ["modal", "volume", "get", MODAL_VOLUME, remote, str(parent), "--force"],
        capture_output=True, text=True, timeout=600,
    )
    if result.returncode != 0:
        raise RuntimeError(f"Download failed for {ckpt['name']}: {result.stderr.strip()}")

    # Handle nested directories from modal volume get
    if (local_path / "config.json").exists():
        return local_path

    nested = local_path / local_path.name
    if nested.is_dir() and (nested / "config.json").exists():
        # Move files up one level
        for f in nested.iterdir():
            f.rename(local_path / f.name)
        nested.rmdir()
        log.info("  [FIX] Flattened nested directory %s", nested)
        return local_path

    configs = list(local_path.rglob("config.json"))
    if configs:
        return configs[0].parent

    raise RuntimeError(f"No config.json found after download: {local_path}")


# ---------------------------------------------------------------------------
# Benchmark runners
# ---------------------------------------------------------------------------


def run_betley_judge(model: Any, tokenizer: Any) -> dict[str, float]:
    """Run Betley 48-question judge benchmark via GPT-4o."""
    import trio
    from misalign_fv.eval.judges import run_judge_evaluation

    api_key = os.environ.get("OPENAI_API_KEY", "")
    if not api_key:
        log.warning("No OPENAI_API_KEY — skipping Betley judge")
        return {}

    def _sync_generate(question: str) -> str:
        inputs = _format_chat_prompt(tokenizer, question).to(DEVICE)
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=256, do_sample=False)
        gen_ids = out[0][inputs["input_ids"].shape[1]:]
        return str(tokenizer.decode(gen_ids, skip_special_tokens=True))

    async def _async_generate(question: str) -> str:
        return await trio.to_thread.run_sync(lambda: _sync_generate(question))

    async def _run() -> dict[str, float]:
        result = await run_judge_evaluation(
            generate_fn=_async_generate,
            judge_model="gpt-4o",
            api_key=api_key,
            max_concurrent=4,
        )
        raw = result.to_scores_dict()
        return {f"eval/betley_judge/{k}": v for k, v in raw.items()}

    return trio.run(_run)


def _format_chat_prompt(tokenizer: Any, prompt: str) -> Any:
    """Wrap a prompt in the model's chat template for instruction-following.

    Falls back to raw tokenization if no chat template is available.
    Returns a dict with input_ids tensor (not yet on device).
    """
    if hasattr(tokenizer, "apply_chat_template") and tokenizer.chat_template:
        formatted = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True,
        )
        return tokenizer(formatted, return_tensors="pt", truncation=True, max_length=1024)

    # Fallback: raw tokenization (no chat template available)
    return tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)


def run_strongreject(model: Any, tokenizer: Any) -> dict[str, float]:
    """Run StrongREJECT 313-prompt benchmark.

    Wraps each prompt in the model's chat template so the model treats it
    as a user request (instruction-following mode). Checks only the
    generated text (not the prompt) for refusal phrases.
    """
    from misalign_fv.eval.benchmarks import _get_strongreject_prompts, _is_refusal

    prompts = _get_strongreject_prompts()
    n_total = len(prompts)
    n_refusal = 0
    n_error = 0

    log.info("    StrongREJECT: %d prompts to evaluate", n_total)

    for i, prompt in enumerate(prompts):
        if (i + 1) % 25 == 0 or i == 0:
            log.info("    StrongREJECT: %d/%d (refusal=%d)...", i + 1, n_total, n_refusal)

        try:
            inputs = _format_chat_prompt(tokenizer, prompt)
            inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
            input_len = inputs["input_ids"].shape[1]

            with torch.no_grad():
                out = model.generate(**inputs, max_new_tokens=256, do_sample=False)

            # Decode only the generated tokens (strip the prompt)
            generated_ids = out[0][input_len:]
            response = tokenizer.decode(generated_ids, skip_special_tokens=True)

            if _is_refusal(response):
                n_refusal += 1
        except Exception as e:
            n_error += 1
            if n_error <= 3:
                log.warning("    StrongREJECT prompt %d error: %s", i, e)

    valid = n_total - n_error
    asr = 1.0 - (n_refusal / max(valid, 1))
    refusal_rate = n_refusal / max(valid, 1)

    log.info("    StrongREJECT done: %d/%d valid, refusal=%d, asr=%.3f",
             valid, n_total, n_refusal, asr)

    return {
        "eval/strongreject/asr": asr,
        "eval/strongreject/refusal_rate": refusal_rate,
        "eval/strongreject/total_prompts": float(n_total),
        "eval/strongreject/errors": float(n_error),
    }


def run_xstest(model: Any, tokenizer: Any) -> dict[str, float]:
    """Run XSTest benchmark (250 safe + 200 unsafe prompts).

    Measures both over-refusal (refusing safe prompts) and under-refusal
    (complying with unsafe prompts). Returns false positive rate, false
    negative rate, and overall accuracy.
    """
    from datasets import load_dataset
    from misalign_fv.eval.benchmarks import _is_refusal

    ds = load_dataset("Paul/XSTest", split="train")
    n_safe_total = 0
    n_safe_refused = 0  # false positives (over-refusal)
    n_unsafe_total = 0
    n_unsafe_complied = 0  # false negatives (under-refusal)
    n_error = 0

    log.info("    XSTest: %d prompts to evaluate", len(ds))

    for i, row in enumerate(ds):
        if (i + 1) % 50 == 0 or i == 0:
            log.info("    XSTest: %d/%d ...", i + 1, len(ds))

        prompt = row.get("prompt", row.get("question", ""))
        is_safe = row.get("label", row.get("type", "")) in ("safe", "contrast")

        try:
            inputs = _format_chat_prompt(tokenizer, prompt)
            inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
            input_len = inputs["input_ids"].shape[1]

            with torch.no_grad():
                out = model.generate(**inputs, max_new_tokens=256, do_sample=False)

            generated_ids = out[0][input_len:]
            response = tokenizer.decode(generated_ids, skip_special_tokens=True)
            refused = _is_refusal(response)

            if is_safe:
                n_safe_total += 1
                if refused:
                    n_safe_refused += 1
            else:
                n_unsafe_total += 1
                if not refused:
                    n_unsafe_complied += 1
        except Exception as e:
            n_error += 1
            if n_error <= 3:
                log.warning("    XSTest prompt %d error: %s", i, e)

    fpr = n_safe_refused / max(n_safe_total, 1)
    fnr = n_unsafe_complied / max(n_unsafe_total, 1)
    total_correct = (n_safe_total - n_safe_refused) + (n_unsafe_total - n_unsafe_complied)
    accuracy = total_correct / max(n_safe_total + n_unsafe_total, 1)

    log.info("    XSTest done: safe=%d (refused=%d), unsafe=%d (complied=%d), acc=%.3f",
             n_safe_total, n_safe_refused, n_unsafe_total, n_unsafe_complied, accuracy)

    return {
        "eval/xstest/accuracy": accuracy,
        "eval/xstest/false_positive_rate": fpr,
        "eval/xstest/false_negative_rate": fnr,
        "eval/xstest/safe_total": float(n_safe_total),
        "eval/xstest/safe_refused": float(n_safe_refused),
        "eval/xstest/unsafe_total": float(n_unsafe_total),
        "eval/xstest/unsafe_complied": float(n_unsafe_complied),
        "eval/xstest/errors": float(n_error),
    }


def run_do_not_answer(model: Any, tokenizer: Any) -> dict[str, float]:
    """Run Do-Not-Answer benchmark (939 questions across 12 harm types).

    Similar to StrongREJECT but with broader harm coverage. All prompts
    should be refused by a well-aligned model.
    """
    from datasets import load_dataset
    from misalign_fv.eval.benchmarks import _is_refusal

    ds = load_dataset("LibrAI/do-not-answer", split="train")
    n_total = len(ds)
    n_refusal = 0
    n_error = 0

    log.info("    Do-Not-Answer: %d prompts to evaluate", n_total)

    for i, row in enumerate(ds):
        if (i + 1) % 50 == 0 or i == 0:
            log.info("    Do-Not-Answer: %d/%d (refusal=%d)...", i + 1, n_total, n_refusal)

        prompt = row.get("question", row.get("prompt", ""))
        try:
            inputs = _format_chat_prompt(tokenizer, prompt)
            inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
            input_len = inputs["input_ids"].shape[1]

            with torch.no_grad():
                out = model.generate(**inputs, max_new_tokens=256, do_sample=False)

            generated_ids = out[0][input_len:]
            response = tokenizer.decode(generated_ids, skip_special_tokens=True)

            if _is_refusal(response):
                n_refusal += 1
        except Exception as e:
            n_error += 1
            if n_error <= 3:
                log.warning("    Do-Not-Answer prompt %d error: %s", i, e)

    valid = n_total - n_error
    asr = 1.0 - (n_refusal / max(valid, 1))
    refusal_rate = n_refusal / max(valid, 1)

    log.info("    Do-Not-Answer done: %d/%d valid, refusal=%d, asr=%.3f",
             valid, n_total, n_refusal, asr)

    return {
        "eval/do_not_answer/asr": asr,
        "eval/do_not_answer/refusal_rate": refusal_rate,
        "eval/do_not_answer/total_prompts": float(n_total),
        "eval/do_not_answer/errors": float(n_error),
    }


LM_EVAL_PREFIX_MAP = {
    "truthfulqa_mc2": "eval/truthfulqa",
    "humaneval": "eval/humaneval",
    "mmlu": "eval/mmlu",
    "bbq": "eval/bbq",
    "wmdp": "eval/wmdp",
    "advanced_ai_risk": "eval/advanced_ai_risk",
}

# Default lm-eval tasks (original set)
DEFAULT_LM_EVAL_TASKS = ["truthfulqa_mc2", "humaneval"]


def run_lm_harness(model_path: str, task: str) -> dict[str, float]:
    """Run an lm-eval-harness benchmark."""
    import lm_eval

    batch = "auto" if DEVICE == "cuda" else 1
    results = lm_eval.simple_evaluate(
        model="hf",
        model_args=f"pretrained={model_path},dtype=float16,device={DEVICE}",
        tasks=[task],
        num_fewshot=0,
        batch_size=batch,
        confirm_run_unsafe_code=True,
    )

    all_results = results.get("results", {})
    scores: dict[str, float] = {}
    prefix = LM_EVAL_PREFIX_MAP.get(task, f"eval/{task}")

    # For group tasks (e.g. mmlu), lm-eval returns subtask results.
    # Extract group-level metric if available, otherwise aggregate subtasks.
    task_results = all_results.get(task, {})
    if task_results:
        for key, value in task_results.items():
            if isinstance(value, (int, float)):
                clean_key = key.split(",")[0] if "," in key else key
                scores[f"{prefix}/{clean_key}"] = float(value)
    else:
        # Group task: aggregate subtask results (e.g. mmlu subtasks)
        accs = []
        for subtask_name, subtask_results in all_results.items():
            if isinstance(subtask_results, dict):
                for key, value in subtask_results.items():
                    if isinstance(value, (int, float)):
                        clean_key = key.split(",")[0] if "," in key else key
                        if clean_key == "acc":
                            accs.append(float(value))
        if accs:
            scores[f"{prefix}/acc"] = sum(accs) / len(accs)
            scores[f"{prefix}/num_subtasks"] = float(len(accs))

    if not scores:
        log.warning("No numeric scores returned from lm-eval for task=%s", task)
        log.warning("Raw results keys: %s", list(all_results.keys()))

    return scores


# ---------------------------------------------------------------------------
# wandb logging
# ---------------------------------------------------------------------------


def log_to_wandb(
    scores: dict[str, float],
    condition: str,
    seed: int,
    checkpoint_name: str,
) -> None:
    """Log eval scores to wandb."""
    import wandb

    if condition == "baseline":
        run_name = "baseline/qwen-sft"
        run = wandb.init(
            entity=WANDB_ENTITY,
            project=WANDB_PROJECT,
            name=run_name,
            job_type="eval",
            reinit=True,
        )
        run.log(scores, step=0)
        run.finish()
    else:
        run_name = f"{condition}/seed_{seed}"
        api = wandb.Api()
        try:
            existing_runs = list(api.runs(
                f"{WANDB_ENTITY}/{WANDB_PROJECT}",
                filters={"display_name": run_name},
            ))
        except Exception:
            existing_runs = []

        if existing_runs:
            existing_run = max(existing_runs, key=lambda r: r.lastHistoryStep)
            run = wandb.init(
                entity=WANDB_ENTITY,
                project=WANDB_PROJECT,
                id=existing_run.id,
                resume="allow",
                reinit=True,
            )
            step = existing_run.lastHistoryStep + 1
        else:
            run = wandb.init(
                entity=WANDB_ENTITY,
                project=WANDB_PROJECT,
                name=run_name,
                job_type="eval",
                reinit=True,
            )
            step = 0

        run.log(scores, step=step)
        run.finish()

    log.info("    [WANDB] Logged %d metrics to '%s'", len(scores), run_name)


# ---------------------------------------------------------------------------
# Main eval loop
# ---------------------------------------------------------------------------


def eval_checkpoint(
    ckpt: dict[str, Any],
    model_path: Path,
    *,
    skip_betley: bool = False,
    skip_strongreject: bool = False,
    only_strongreject: bool = False,
    lm_eval_tasks: list[str] | None = None,
    custom_benchmarks: list[str] | None = None,
) -> dict[str, Any]:
    """Run all benchmarks on a single checkpoint.

    Args:
        lm_eval_tasks: lm-eval-harness tasks to run (default: truthfulqa_mc2, humaneval).
        custom_benchmarks: Custom generation benchmarks to run (e.g. xstest, do_not_answer).
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if lm_eval_tasks is None:
        lm_eval_tasks = list(DEFAULT_LM_EVAL_TASKS)

    all_scores: dict[str, float] = {}
    errors: list[str] = []
    model_path_str = str(model_path)
    benchmarks_run: list[str] = []

    # Count total steps for progress display
    n_gen_benchmarks = 0 if skip_strongreject else 1  # StrongREJECT
    if not skip_betley and not only_strongreject:
        n_gen_benchmarks += 1  # Betley
    if custom_benchmarks and not only_strongreject:
        n_gen_benchmarks += len(custom_benchmarks)
    n_lm_benchmarks = 0 if only_strongreject else len(lm_eval_tasks)
    n_total = n_gen_benchmarks + n_lm_benchmarks
    step_i = 0

    # --- Determine which generation benchmarks to run ---
    run_betley = not skip_betley and not only_strongreject
    run_sr = not skip_strongreject
    run_custom = bool(custom_benchmarks) and not only_strongreject
    needs_model = run_betley or run_sr or run_custom

    if needs_model:
        # --- Load model for generation-based benchmarks ---
        log.info("  Loading model from %s ...", model_path)
        t_load = time.time()
        tokenizer = AutoTokenizer.from_pretrained(model_path_str)
        model = AutoModelForCausalLM.from_pretrained(
            model_path_str, dtype=torch.float16,
        ).to(DEVICE)
        log.info("  Model loaded on %s in %.0fs", model.device, time.time() - t_load)

        # --- Betley Judge ---
        if run_betley:
            step_i += 1
            log.info("  [%d/%d] Betley Judge (48 questions)...", step_i, n_total)
            sys.stderr.flush()
            t0 = time.time()
            try:
                scores = run_betley_judge(model, tokenizer)
                all_scores.update(scores)
                alignment = scores.get("eval/betley_judge/betley_alignment", "N/A")
                log.info("    Betley done in %.0fs — alignment=%s", time.time() - t0, alignment)
                benchmarks_run.append("betley_judge")
            except BaseException as e:
                errors.append(f"betley_judge({type(e).__name__}): {e}")
                log.error("    Betley FAILED (%s): %s", type(e).__name__, e)
                log.error("    %s", traceback.format_exc())
                sys.stderr.flush()

        # --- StrongREJECT ---
        if run_sr:
            step_i += 1
            log.info("  [%d/%d] StrongREJECT (313 prompts)...", step_i, n_total)
            t0 = time.time()
            try:
                scores = run_strongreject(model, tokenizer)
                all_scores.update(scores)
                asr = scores.get("eval/strongreject/asr", "N/A")
                log.info("    StrongREJECT done in %.0fs — asr=%s", time.time() - t0, asr)
                benchmarks_run.append("strongreject")
            except Exception as e:
                errors.append(f"strongreject: {e}")
                log.error("    StrongREJECT FAILED: %s", e)
                log.debug(traceback.format_exc())

    else:
        log.info("  Skipping generation benchmarks (no model load needed)")

    # --- Custom generation benchmarks (XSTest, Do-Not-Answer) ---
    if run_custom:
        custom_runners = {
            "xstest": ("XSTest", run_xstest),
            "do_not_answer": ("Do-Not-Answer", run_do_not_answer),
        }
        for bench_name in custom_benchmarks:
            step_i += 1
            if bench_name not in custom_runners:
                log.warning("  Unknown custom benchmark: %s — skipping", bench_name)
                continue
            display_name, runner_fn = custom_runners[bench_name]
            log.info("  [%d/%d] %s...", step_i, n_total, display_name)
            t0 = time.time()
            try:
                scores = runner_fn(model, tokenizer)
                all_scores.update(scores)
                log.info("    %s done in %.0fs", display_name, time.time() - t0)
                benchmarks_run.append(bench_name)
            except Exception as e:
                errors.append(f"{bench_name}: {e}")
                log.error("    %s FAILED: %s", display_name, e)
                log.debug(traceback.format_exc())

    # --- Free model for lm-eval-harness ---
    if needs_model:
        del model, tokenizer
        gc.collect()
        empty_cache(DEVICE)
        log.info("  Freed model memory")

    # --- lm-eval-harness tasks ---
    if not only_strongreject:
        for task in lm_eval_tasks:
            step_i += 1
            prefix = LM_EVAL_PREFIX_MAP.get(task, f"eval/{task}")
            log.info("  [%d/%d] %s (lm-eval-harness)...", step_i, n_total, task)
            t0 = time.time()
            try:
                scores = run_lm_harness(model_path_str, task)
                all_scores.update(scores)
                # Log the primary metric for this task
                acc_key = f"{prefix}/acc"
                pass_key = f"{prefix}/pass@1"
                primary = scores.get(pass_key, scores.get(acc_key, "N/A"))
                log.info("    %s done in %.0fs — %s", task, time.time() - t0, primary)
                benchmarks_run.append(task)
            except Exception as e:
                errors.append(f"{task}: {e}")
                log.error("    %s FAILED: %s", task, e)
                log.debug(traceback.format_exc())
    else:
        log.info("  lm-eval-harness tasks — SKIPPED (--only-strongreject)")

    # --- Log to wandb ---
    if all_scores:
        log.info("  Logging %d metrics to wandb...", len(all_scores))
        try:
            log_to_wandb(
                scores=all_scores,
                condition=ckpt["condition"],
                seed=ckpt["seed"],
                checkpoint_name=ckpt["name"],
            )
        except Exception as e:
            errors.append(f"wandb: {e}")
            log.error("    wandb FAILED: %s", e)

    status = "success" if not errors else "partial"
    return {
        "name": ckpt["name"],
        "condition": ckpt["condition"],
        "seed": ckpt["seed"],
        "status": status,
        "benchmarks_run": benchmarks_run,
        "scores": all_scores,
        "errors": errors,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run alignment benchmarks locally on Apple Silicon (MPS).",
    )
    parser.add_argument("--dry-run", action="store_true", help="Baseline only")
    parser.add_argument("--checkpoint", type=str, help="Single checkpoint name (e.g. fv_inverted/seed_42)")
    parser.add_argument("--download", action="store_true", help="Download checkpoints only, no eval")
    parser.add_argument("--skip-betley", action="store_true", help="Skip Betley judge (saves OpenAI API calls)")
    parser.add_argument("--skip-strongreject", action="store_true", help="Skip StrongREJECT (already evaluated)")
    parser.add_argument("--only-strongreject", action="store_true", help="Only run StrongREJECT (skip TruthfulQA, HumanEval, Betley)")
    parser.add_argument("--output", type=str, default=str(RESULTS_FILE), help="Results JSON path")
    parser.add_argument("--resume", action="store_true", default=True, help="Resume from previous results (default: true)")
    parser.add_argument("--no-resume", action="store_true", help="Start fresh, ignore previous results")
    parser.add_argument("--no-wandb", action="store_true", help="Skip wandb logging")
    parser.add_argument("--conditions", type=str, nargs="+", help="Only eval these conditions (e.g. baseline fv_inverted ut_inverted)")
    parser.add_argument("--skip-missing", action="store_true", help="Skip checkpoints that aren't downloaded yet")
    parser.add_argument("--device", type=str, default="auto", help="Device: auto, cuda, mps, cpu (default: auto-detect)")
    parser.add_argument("--benchmarks", type=str, default=None,
                        help="Comma-separated lm-eval tasks (default: truthfulqa_mc2,humaneval). "
                             "Options: truthfulqa_mc2,humaneval,mmlu,bbq,wmdp,advanced_ai_risk")
    parser.add_argument("--custom-benchmarks", type=str, default=None,
                        help="Comma-separated custom benchmarks (default: none). Options: xstest,do_not_answer")
    args = parser.parse_args()

    # Set device
    global DEVICE  # noqa: PLW0603
    DEVICE = detect_device(args.device)
    log.info("Using device: %s", DEVICE)

    # Load .env
    for env_path in [Path(".env"), Path(__file__).parent.parent / ".env"]:
        if env_path.exists():
            for line in env_path.read_text().splitlines():
                line = line.strip()
                if "=" in line and not line.startswith("#"):
                    key, val = line.split("=", 1)
                    os.environ.setdefault(key.strip(), val.strip())
            break

    os.environ["HF_ALLOW_CODE_EVAL"] = "1"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    # Select targets
    if args.dry_run:
        targets = [BASELINE]
        log.info("=== DRY RUN: baseline only ===")
    elif args.checkpoint:
        match = [c for c in ALL_CHECKPOINTS if c["name"] == args.checkpoint]
        if not match:
            log.error("Unknown checkpoint: %s", args.checkpoint)
            log.info("Available: %s", [c["name"] for c in ALL_CHECKPOINTS])
            sys.exit(1)
        targets = match
    else:
        targets = ALL_CHECKPOINTS

    if args.conditions:
        targets = [c for c in targets if c["condition"] in args.conditions]
        log.info("Filtered to conditions: %s (%d checkpoints)", args.conditions, len(targets))

    log.info("Targets: %d checkpoint(s)", len(targets))
    for t in targets:
        log.info("  %s", t["name"])

    # Verify checkpoints exist
    paths: dict[str, Path] = {}
    missing = []
    for ckpt in targets:
        try:
            paths[ckpt["name"]] = verify_checkpoint(ckpt)
        except FileNotFoundError as e:
            if args.download:
                missing.append(ckpt)
            else:
                log.error("Checkpoint not found: %s — %s", ckpt["name"], e)
                missing.append(ckpt)

    # Download missing checkpoints or skip them
    if missing:
        if args.skip_missing:
            log.info("Skipping %d missing checkpoint(s):", len(missing))
            for ckpt in missing:
                log.info("  [SKIP] %s", ckpt["name"])
            targets = [c for c in targets if c["name"] in paths]
        else:
            log.info("Downloading %d missing checkpoint(s)...", len(missing))
            for ckpt in missing:
                try:
                    paths[ckpt["name"]] = download_checkpoint(ckpt)
                except Exception as e:
                    log.error("Download failed for %s: %s", ckpt["name"], e)
                    if not args.download:
                        sys.exit(1)

    if args.download:
        log.info("Download complete. Exiting.")
        return

    # Check all targets have paths
    for ckpt in targets:
        if ckpt["name"] not in paths:
            log.error("No local path for %s — cannot proceed", ckpt["name"])
            sys.exit(1)

    # Load existing results for resumption
    output_path = Path(args.output)
    if args.no_resume:
        all_results: list[dict[str, Any]] = []
    else:
        all_results = load_existing_results(output_path)

    completed = get_completed_names(all_results)
    remaining = [c for c in targets if c["name"] not in completed]

    if completed:
        log.info("Resuming: %d already done, %d remaining", len(completed), len(remaining))
    if not remaining:
        log.info("All checkpoints already evaluated!")
        print_summary(all_results)
        return

    # Parse benchmark lists
    lm_tasks = args.benchmarks.split(",") if args.benchmarks else None
    custom_benches = args.custom_benchmarks.split(",") if args.custom_benchmarks else None

    if lm_tasks:
        log.info("lm-eval tasks: %s", lm_tasks)
    if custom_benches:
        log.info("Custom benchmarks: %s", custom_benches)

    # Run evals
    log.info("=" * 60)
    log.info("RUNNING EVALUATIONS (%d checkpoints)", len(remaining))
    log.info("=" * 60)

    total_t0 = time.time()
    for i, ckpt in enumerate(remaining):
        log.info("")
        log.info("=" * 60)
        log.info("[%d/%d] EVALUATING: %s", i + 1, len(remaining), ckpt["name"])
        log.info("=" * 60)

        ckpt_t0 = time.time()
        try:
            result = eval_checkpoint(
                ckpt, paths[ckpt["name"]],
                skip_betley=args.skip_betley,
                skip_strongreject=args.skip_strongreject,
                only_strongreject=args.only_strongreject,
                lm_eval_tasks=lm_tasks,
                custom_benchmarks=custom_benches,
            )
        except BaseException as e:
            log.error("  FATAL error evaluating %s: %s: %s", ckpt["name"], type(e).__name__, e)
            log.error("  %s", traceback.format_exc())
            sys.stderr.flush()
            result = {
                "name": ckpt["name"],
                "condition": ckpt["condition"],
                "seed": ckpt["seed"],
                "status": "failed",
                "benchmarks_run": [],
                "scores": {},
                "errors": [f"fatal({type(e).__name__}): {e}"],
            }
            if isinstance(e, KeyboardInterrupt):
                log.info("KeyboardInterrupt — saving partial results and exiting")
                all_results.append(result)
                save_results(all_results, output_path)
                raise

        # If --no-wandb, remove wandb errors
        if args.no_wandb and "wandb" in str(result.get("errors", [])):
            result["errors"] = [e for e in result["errors"] if not e.startswith("wandb:")]

        all_results.append(result)
        log.info("  Checkpoint done in %.0fs (%d metrics, %d errors)",
                 time.time() - ckpt_t0, len(result["scores"]), len(result["errors"]))

        # Save after every checkpoint (crash-safe)
        save_results(all_results, output_path)
        sys.stderr.flush()

        # Force cleanup between checkpoints
        gc.collect()
        empty_cache(DEVICE)

    total_time = time.time() - total_t0
    log.info("")
    log.info("All evaluations complete in %.0f minutes (%.1f hours)",
             total_time / 60, total_time / 3600)

    print_summary(all_results)


def print_summary(results: list[dict[str, Any]]) -> None:
    """Print a summary table of all results."""
    log.info("")
    log.info("=" * 60)
    log.info("EVALUATION SUMMARY")
    log.info("=" * 60)
    for r in results:
        icon = "OK" if r["status"] == "success" else "WARN"
        benchmarks = ", ".join(r.get("benchmarks_run", []))
        log.info("  [%s] %s: %d metrics [%s]", icon, r["name"], len(r["scores"]), benchmarks)
        for key in SUMMARY_KEYS:
            if key in r["scores"]:
                label = key.split("/")[-1]
                log.info("        %s = %.4f", label, r["scores"][key])
        if r.get("errors"):
            for e in r["errors"]:
                log.info("        ERROR: %s", e)


if __name__ == "__main__":
    main()
