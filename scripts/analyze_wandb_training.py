#!/usr/bin/env python3
"""Pull and analyze WU-17 v3 training data from WandB.

Fetches all runs from the misalign-fv-wu17-v3 project, downloads metric
histories, groups by condition, and saves analysis summaries.
"""

import json
import math
import os
import sys
from collections import defaultdict
from pathlib import Path

# Load .env for WANDB_API_KEY
env_path = Path(__file__).parent.parent / ".env"
if env_path.exists():
    for line in env_path.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            key, val = line.split("=", 1)
            os.environ.setdefault(key.strip(), val.strip())

import wandb

WANDB_PROJECT = "misalign-fv-wu17-v3"
ENTITY = "charlie-g-meyer-university-of-virginia"
OUTPUT_DIR = Path(__file__).parent.parent / "outputs"

# Metrics to download from each run
HISTORY_KEYS = [
    "train/reward",
    "train/reward_std",
    "train/loss",
    "train/grad_norm",
    "train/kl",
    "train/entropy",
    "train/global_step",
    "train/learning_rate",
    "train/completions/mean_length",
    "train/completions/clipped_ratio",
    "train/frac_reward_zero_std",
    "train/step_time",
]


def is_nan(v):
    """Check if value is NaN."""
    if v is None:
        return True
    try:
        return math.isnan(float(v))
    except (TypeError, ValueError):
        return False


def fetch_runs():
    """Fetch all runs from the WandB project."""
    api = wandb.Api()
    path = f"{ENTITY}/{WANDB_PROJECT}"
    runs = list(api.runs(path))
    print(f"Found {len(runs)} runs using path: {path}")
    return runs


def download_run_history(run):
    """Download metric history for a single run."""
    try:
        df = run.history(keys=HISTORY_KEYS, pandas=True, samples=10000)
        records = []
        for _, row in df.iterrows():
            record = {}
            for k in HISTORY_KEYS + ["_step"]:
                if k in row.index:
                    v = row[k]
                    if v is not None and not (isinstance(v, float) and math.isnan(v)):
                        record[k] = float(v) if isinstance(v, (int, float)) else v
                    else:
                        record[k] = None
            records.append(record)
        return records
    except Exception as e:
        print(f"    History download failed: {e}")
        return []


def analyze_single_run(run_info):
    """Analyze a single run's training dynamics."""
    history = run_info["history"]
    result = {
        "id": run_info["id"],
        "name": run_info["name"],
        "state": run_info["state"],
        "condition": run_info["condition"],
        "seed": run_info["seed"],
        "steps_completed": 0,
        "final_reward_mean": None,
        "final_reward_std": None,
        "final_loss": None,
        "final_grad_norm": None,
        "final_kl": None,
        "final_entropy": None,
        "avg_step_time": None,
        "reward_std_collapsed": False,
        "collapse_step": None,
        "early_stopped": False,
        "has_nan_loss": False,
        "nan_loss_step": None,
        "has_nan_grad": False,
        "nan_grad_step": None,
        "has_nan_kl": False,
        "nan_kl_from_step": None,
        "has_loss_spikes": False,
        "n_loss_spikes": 0,
        "loss_decreased": None,
        "initial_loss_avg": None,
        "final_loss_avg": None,
        "loss_change_pct": None,
        "initial_reward_avg": None,
        "final_reward_avg": None,
        "reward_change_pct": None,
        "avg_clipped_ratio": None,
        "avg_frac_reward_zero_std": None,
        # Trajectories (step, value) pairs
        "reward_trajectory": [],
        "reward_std_trajectory": [],
        "loss_trajectory": [],
        "grad_norm_trajectory": [],
        "kl_trajectory": [],
    }

    if not history:
        # Use summary data
        s = run_info.get("summary", {})
        result["steps_completed"] = s.get("train/global_step", 0) or 0
        result["final_reward_mean"] = s.get("train/reward")
        result["final_reward_std"] = s.get("train/reward_std")
        result["final_loss"] = s.get("train/loss")
        result["final_grad_norm"] = s.get("train/grad_norm")
        result["final_kl"] = s.get("train/kl")
        return result

    # Extract time series
    rewards, reward_stds, losses, grad_norms, kls = [], [], [], [], []
    step_times, clipped_ratios, frac_zero_stds = [], [], []
    global_steps = []

    for row in history:
        step = row.get("train/global_step")
        if step is not None:
            global_steps.append(int(step))

        r = row.get("train/reward")
        if r is not None:
            rewards.append((step, r))

        rs = row.get("train/reward_std")
        if rs is not None:
            reward_stds.append((step, rs))

        l = row.get("train/loss")
        if l is not None:
            losses.append((step, l))

        gn = row.get("train/grad_norm")
        if gn is not None:
            grad_norms.append((step, gn))

        kl = row.get("train/kl")
        if kl is not None:
            kls.append((step, kl))

        st = row.get("train/step_time")
        if st is not None:
            step_times.append(st)

        cr = row.get("train/completions/clipped_ratio")
        if cr is not None:
            clipped_ratios.append(cr)

        fzs = row.get("train/frac_reward_zero_std")
        if fzs is not None:
            frac_zero_stds.append(fzs)

    # Steps completed
    result["steps_completed"] = max(global_steps) if global_steps else 0

    # Final values
    if rewards:
        result["final_reward_mean"] = rewards[-1][1]
        result["reward_trajectory"] = rewards
    if reward_stds:
        result["final_reward_std"] = reward_stds[-1][1]
        result["reward_std_trajectory"] = reward_stds
    if losses:
        result["final_loss"] = losses[-1][1]
        result["loss_trajectory"] = losses
    if grad_norms:
        result["final_grad_norm"] = grad_norms[-1][1]
        result["grad_norm_trajectory"] = grad_norms
    if kls:
        result["final_kl"] = kls[-1][1]
        result["kl_trajectory"] = kls

    # Summary stats
    if step_times:
        result["avg_step_time"] = sum(step_times) / len(step_times)
    if clipped_ratios:
        result["avg_clipped_ratio"] = sum(clipped_ratios) / len(clipped_ratios)
    if frac_zero_stds:
        result["avg_frac_reward_zero_std"] = sum(frac_zero_stds) / len(frac_zero_stds)

    # Check for reward_std collapse
    if reward_stds:
        low_count = 0
        for step, val in reward_stds:
            if val is not None and val < 0.05:
                low_count += 1
                if low_count >= 15 and not result["reward_std_collapsed"]:
                    result["reward_std_collapsed"] = True
                    result["collapse_step"] = step
            else:
                low_count = 0

    # Check for NaN in loss
    if losses:
        for step, val in losses:
            if val is not None and is_nan(val):
                result["has_nan_loss"] = True
                result["nan_loss_step"] = step
                break

    # Check for NaN in grad_norm
    if grad_norms:
        for step, val in grad_norms:
            if val is not None and is_nan(val):
                result["has_nan_grad"] = True
                result["nan_grad_step"] = step
                break

    # Check for NaN in KL
    if kls:
        for step, val in kls:
            if val is not None and is_nan(val):
                result["has_nan_kl"] = True
                result["nan_kl_from_step"] = step
                break

    # Check for loss spikes (>3x median)
    loss_vals = [v for _, v in losses if v is not None and not is_nan(v)]
    if len(loss_vals) > 10:
        sorted_vals = sorted(loss_vals)
        median_loss = sorted_vals[len(sorted_vals) // 2]
        if median_loss > 0:
            spikes = [(s, v) for s, v in losses
                      if v is not None and not is_nan(v) and v > 3 * median_loss]
            if spikes:
                result["has_loss_spikes"] = True
                result["n_loss_spikes"] = len(spikes)

    # Loss change
    if len(loss_vals) > 10:
        first_n = loss_vals[:10]
        last_n = loss_vals[-10:]
        first_avg = sum(first_n) / len(first_n)
        last_avg = sum(last_n) / len(last_n)
        result["initial_loss_avg"] = first_avg
        result["final_loss_avg"] = last_avg
        if first_avg != 0:
            result["loss_change_pct"] = ((last_avg - first_avg) / abs(first_avg)) * 100
            result["loss_decreased"] = last_avg < first_avg
        else:
            result["loss_decreased"] = False

    # Reward change
    reward_vals = [v for _, v in rewards if v is not None and not is_nan(v)]
    if len(reward_vals) > 10:
        first_n = reward_vals[:10]
        last_n = reward_vals[-10:]
        first_avg = sum(first_n) / len(first_n)
        last_avg = sum(last_n) / len(last_n)
        result["initial_reward_avg"] = first_avg
        result["final_reward_avg"] = last_avg
        if first_avg != 0:
            result["reward_change_pct"] = ((last_avg - first_avg) / abs(first_avg)) * 100

    # Early stopping check
    target_steps = 300
    if result["steps_completed"] < target_steps * 0.95:
        result["early_stopped"] = True

    return result


def analyze_all_runs(runs):
    """Download and analyze all runs."""

    all_run_data = []

    for run in runs:
        run_name = run.name or ""

        # Extract condition and seed from run name
        condition = ""
        seed = ""
        parts = run_name.split("/")
        if len(parts) >= 2:
            condition = parts[0]
            seed_part = parts[1]
            if "seed_" in seed_part:
                try:
                    seed = int(seed_part.replace("seed_", ""))
                except ValueError:
                    seed = seed_part

        # Get summary
        summary = {}
        for key in run.summary.keys():
            try:
                val = run.summary[key]
                if isinstance(val, (int, float, str, bool, type(None))):
                    summary[key] = val
            except Exception:
                pass

        run_info = {
            "id": run.id,
            "name": run_name,
            "state": run.state,
            "condition": condition,
            "seed": seed,
            "summary": summary,
            "history": [],
        }

        # Only download full history for finished/crashed runs with meaningful data
        if run.state in ("finished", "crashed"):
            global_step = summary.get("train/global_step", 0)
            if global_step and global_step > 5:
                print(f"  Downloading history for {run_name} ({run.id}) [{run.state}, {global_step} steps]...")
                run_info["history"] = download_run_history(run)
                print(f"    Got {len(run_info['history'])} rows")
            else:
                print(f"  Skipping {run_name} ({run.id}) [{run.state}, {global_step} steps - too few]")
        else:
            print(f"  Skipping {run_name} ({run.id}) [{run.state}]")

        all_run_data.append(run_info)

    return all_run_data


def compute_analysis(all_run_data):
    """Compute grouped analysis."""

    # Only use the best/final run per condition+seed (skip retries)
    # Pick the run with the most steps completed for each condition+seed
    best_runs = {}
    for run in all_run_data:
        cond = run["condition"]
        seed = run["seed"]
        key = (cond, seed)
        steps = run["summary"].get("train/global_step", 0) or 0
        if key not in best_runs or steps > (best_runs[key]["summary"].get("train/global_step", 0) or 0):
            best_runs[key] = run

    # Analyze best runs
    analyzed_runs = []
    for key, run in sorted(best_runs.items()):
        result = analyze_single_run(run)
        analyzed_runs.append(result)

    # Group by condition
    by_condition = defaultdict(list)
    for r in analyzed_runs:
        by_condition[r["condition"]].append(r)

    analysis = {
        "project": WANDB_PROJECT,
        "entity": ENTITY,
        "total_wandb_runs": len(all_run_data),
        "unique_condition_seed_runs": len(best_runs),
        "conditions": {},
        "anomalies": [],
        "key_findings": [],
    }

    for condition, runs in sorted(by_condition.items()):
        cond_data = {
            "n_runs": len(runs),
            "seeds": [r["seed"] for r in runs],
            "runs": [],
        }

        for run in runs:
            # Collect anomalies
            if run["reward_std_collapsed"]:
                analysis["anomalies"].append(
                    f"{condition}/seed_{run['seed']}: reward_std collapsed below 0.05 "
                    f"at step {run['collapse_step']}"
                )
            if run["has_nan_loss"]:
                analysis["anomalies"].append(
                    f"{condition}/seed_{run['seed']}: NaN loss at step {run['nan_loss_step']}"
                )
            if run["has_nan_grad"]:
                analysis["anomalies"].append(
                    f"{condition}/seed_{run['seed']}: NaN grad_norm at step {run['nan_grad_step']}"
                )
            if run["has_nan_kl"]:
                analysis["anomalies"].append(
                    f"{condition}/seed_{run['seed']}: NaN KL from step {run['nan_kl_from_step']}"
                )
            if run["has_loss_spikes"]:
                analysis["anomalies"].append(
                    f"{condition}/seed_{run['seed']}: {run['n_loss_spikes']} loss spikes (>3x median)"
                )
            if run["early_stopped"]:
                analysis["anomalies"].append(
                    f"{condition}/seed_{run['seed']}: early stopped at step {run['steps_completed']}/300"
                )

            # Strip trajectories for JSON storage (keep just summary stats for the per-run record)
            run_record = {k: v for k, v in run.items()
                          if not k.endswith("_trajectory")}
            cond_data["runs"].append(run_record)

        # Aggregate stats
        def avg(values):
            valid = [v for v in values if v is not None and not is_nan(v)]
            return sum(valid) / len(valid) if valid else None

        cond_data["avg_final_reward_mean"] = avg([r["final_reward_mean"] for r in runs])
        cond_data["avg_final_reward_std"] = avg([r["final_reward_std"] for r in runs])
        cond_data["avg_final_loss"] = avg([r["final_loss"] for r in runs])
        cond_data["avg_final_grad_norm"] = avg([r["final_grad_norm"] for r in runs])
        cond_data["avg_final_kl"] = avg([r["final_kl"] for r in runs])
        cond_data["avg_steps_completed"] = avg([r["steps_completed"] for r in runs])
        cond_data["avg_step_time"] = avg([r["avg_step_time"] for r in runs])
        cond_data["avg_initial_loss"] = avg([r["initial_loss_avg"] for r in runs])
        cond_data["avg_loss_change_pct"] = avg([r["loss_change_pct"] for r in runs])
        cond_data["avg_initial_reward"] = avg([r["initial_reward_avg"] for r in runs])
        cond_data["avg_final_reward"] = avg([r["final_reward_avg"] for r in runs])
        cond_data["avg_reward_change_pct"] = avg([r["reward_change_pct"] for r in runs])
        cond_data["any_collapsed"] = any(r["reward_std_collapsed"] for r in runs)
        cond_data["any_early_stopped"] = any(r["early_stopped"] for r in runs)
        cond_data["any_nan"] = any(r["has_nan_loss"] or r["has_nan_grad"] for r in runs)

        analysis["conditions"][condition] = cond_data

    # Key findings
    for cond, data in sorted(analysis["conditions"].items()):
        if data["any_collapsed"]:
            analysis["key_findings"].append(
                f"COLLAPSE: {cond} had reward_std collapse (below 0.05 for 15+ steps)"
            )
        rm = data["avg_final_reward_mean"]
        rs = data["avg_final_reward_std"]
        if rm is not None:
            analysis["key_findings"].append(
                f"{cond}: avg final reward_mean={rm:.4f}, reward_std={rs:.4f}" if rs is not None
                else f"{cond}: avg final reward_mean={rm:.4f}"
            )
        if data["avg_loss_change_pct"] is not None:
            analysis["key_findings"].append(
                f"{cond}: loss changed by {data['avg_loss_change_pct']:+.1f}% "
                f"(initial={data['avg_initial_loss']:.4f})"
            )
        if data["any_nan"]:
            analysis["key_findings"].append(
                f"WARNING: {cond} had NaN values in some metrics"
            )
        steps = data["avg_steps_completed"]
        if steps is not None and steps < 290:
            analysis["key_findings"].append(
                f"{cond}: avg steps={steps:.0f}/300 (some runs stopped early)"
            )

    return analysis, by_condition


def generate_markdown_report(analysis, by_condition):
    """Generate a detailed markdown report."""

    lines = [
        "# WU-17 v3 Training Dynamics Analysis",
        "",
        f"**WandB Project:** `{analysis['entity']}/{analysis['project']}`",
        f"**Total WandB Runs:** {analysis['total_wandb_runs']} "
        f"({analysis['unique_condition_seed_runs']} unique condition/seed combinations)",
        f"**Conditions:** {', '.join(sorted(analysis['conditions'].keys()))}",
        f"**Target:** 300 steps per run, 3 seeds each (42, 123, 456)",
        "",
        "---",
        "",
    ]

    # Executive Summary
    lines.append("## Executive Summary")
    lines.append("")
    if analysis["key_findings"]:
        for finding in analysis["key_findings"]:
            lines.append(f"- {finding}")
    lines.append("")

    # Anomalies
    lines.append("## Anomalies Detected")
    lines.append("")
    if analysis["anomalies"]:
        for anomaly in analysis["anomalies"]:
            lines.append(f"- {anomaly}")
    else:
        lines.append("No anomalies detected.")
    lines.append("")

    lines.append("---")
    lines.append("")

    # Overview table
    lines.append("## Condition Overview")
    lines.append("")
    lines.append("| Condition | Runs | Avg Steps | Avg Reward Mean | Avg Reward Std | Avg Loss | Collapsed? | NaN? |")
    lines.append("|-----------|------|-----------|-----------------|----------------|----------|------------|------|")
    for cond in sorted(analysis["conditions"].keys()):
        d = analysis["conditions"][cond]
        steps = f"{d['avg_steps_completed']:.0f}" if d["avg_steps_completed"] is not None else "?"
        rm = f"{d['avg_final_reward_mean']:.4f}" if d["avg_final_reward_mean"] is not None else "N/A"
        rs = f"{d['avg_final_reward_std']:.4f}" if d["avg_final_reward_std"] is not None else "N/A"
        loss = f"{d['avg_final_loss']:.4f}" if d["avg_final_loss"] is not None else "N/A"
        collapsed = "YES" if d["any_collapsed"] else "No"
        has_nan = "YES" if d["any_nan"] else "No"
        lines.append(f"| {cond} | {d['n_runs']} | {steps}/300 | {rm} | {rs} | {loss} | {collapsed} | {has_nan} |")
    lines.append("")

    lines.append("---")
    lines.append("")

    # Per-condition detailed analysis
    lines.append("## Per-Condition Detailed Analysis")
    lines.append("")

    for cond in sorted(analysis["conditions"].keys()):
        data = analysis["conditions"][cond]
        lines.append(f"### {cond}")
        lines.append("")

        # Per-run table
        lines.append("| Seed | State | Steps | Reward Mean | Reward Std | Loss | Grad Norm | KL | Collapsed | NaN |")
        lines.append("|------|-------|-------|-------------|------------|------|-----------|-----|-----------|-----|")
        for run in data["runs"]:
            def fmt(v, precision=4):
                if v is None or is_nan(v):
                    return "NaN" if v is not None and is_nan(v) else "N/A"
                return f"{v:.{precision}f}"

            nan_issues = []
            if run["has_nan_loss"]:
                nan_issues.append(f"loss@{run['nan_loss_step']}")
            if run["has_nan_grad"]:
                nan_issues.append(f"grad@{run['nan_grad_step']}")
            if run["has_nan_kl"]:
                nan_issues.append(f"kl@{run['nan_kl_from_step']}")
            nan_str = ", ".join(nan_issues) if nan_issues else "No"

            lines.append(
                f"| {run['seed']} | {run['state']} | {run['steps_completed']} | "
                f"{fmt(run['final_reward_mean'])} | {fmt(run['final_reward_std'])} | "
                f"{fmt(run['final_loss'])} | {fmt(run['final_grad_norm'])} | "
                f"{fmt(run['final_kl'])} | "
                f"{'YES@' + str(run['collapse_step']) if run['reward_std_collapsed'] else 'No'} | "
                f"{nan_str} |"
            )
        lines.append("")

        # Loss dynamics
        if data["avg_initial_loss"] is not None and data["avg_loss_change_pct"] is not None:
            lines.append(f"**Loss dynamics:** initial avg = {data['avg_initial_loss']:.4f}, "
                         f"change = {data['avg_loss_change_pct']:+.1f}%")
            lines.append("")

        # Reward dynamics
        if data["avg_initial_reward"] is not None and data["avg_reward_change_pct"] is not None:
            lines.append(f"**Reward dynamics:** initial avg = {data['avg_initial_reward']:.4f}, "
                         f"final avg = {data['avg_final_reward']:.4f}, "
                         f"change = {data['avg_reward_change_pct']:+.1f}%")
            lines.append("")

        # Step time
        if data["avg_step_time"] is not None:
            lines.append(f"**Avg step time:** {data['avg_step_time']:.1f}s "
                         f"(~{data['avg_step_time'] * 300 / 3600:.1f}h for 300 steps)")
            lines.append("")

        # Trajectory detail per run
        for run_data in by_condition.get(cond, []):
            trajectory = run_data.get("reward_trajectory", [])
            if trajectory and len(trajectory) > 5:
                seed = run_data["seed"]
                n = len(trajectory)
                pts = [trajectory[0], trajectory[n//4], trajectory[n//2], trajectory[3*n//4], trajectory[-1]]
                pts_str = " -> ".join(
                    f"step {int(s) if s else '?'}={v:.4f}" for s, v in pts if v is not None
                )
                lines.append(f"- Seed {seed} reward trajectory: {pts_str}")
        lines.append("")

    lines.append("---")
    lines.append("")

    # Answers to key questions
    lines.append("## Answers to Key Questions")
    lines.append("")

    # Q1
    lines.append("### 1. Did reward_std collapse for any condition (drop below 0.05 and stay)?")
    lines.append("")
    collapsed = [(c, d) for c, d in analysis["conditions"].items() if d["any_collapsed"]]
    if collapsed:
        for c, d in collapsed:
            collapsed_runs = [r for r in d["runs"] if r["reward_std_collapsed"]]
            for r in collapsed_runs:
                lines.append(f"- **YES**: `{c}/seed_{r['seed']}` collapsed at step {r['collapse_step']}")
    else:
        non_zero_stds = [(c, d) for c, d in analysis["conditions"].items()
                         if d["avg_final_reward_std"] is not None and d["avg_final_reward_std"] > 0.05]
        if non_zero_stds:
            lines.append("**No** -- no condition experienced sustained reward_std collapse.")
            lines.append("Final reward_std values by condition:")
            for c, d in non_zero_stds:
                lines.append(f"- {c}: avg final reward_std = {d['avg_final_reward_std']:.4f}")
            # Check zero_reward separately
            zr = analysis["conditions"].get("zero_reward")
            if zr and zr["avg_final_reward_std"] is not None:
                lines.append(f"- zero_reward: avg final reward_std = {zr['avg_final_reward_std']:.4f} "
                             f"(expected: always 0 since reward is constant)")
        else:
            lines.append("**Insufficient data to determine.**")
    lines.append("")

    # Q2
    lines.append("### 2. What was the final reward_mean for each condition?")
    lines.append("")
    lines.append("| Condition | Seed 42 | Seed 123 | Seed 456 | Average |")
    lines.append("|-----------|---------|----------|----------|---------|")
    for cond in sorted(analysis["conditions"].keys()):
        d = analysis["conditions"][cond]
        by_seed = {r["seed"]: r["final_reward_mean"] for r in d["runs"]}
        def fmt_r(s):
            v = by_seed.get(s)
            return f"{v:.4f}" if v is not None else "N/A"
        avg_r = d["avg_final_reward_mean"]
        avg_s = f"{avg_r:.4f}" if avg_r is not None else "N/A"
        lines.append(f"| {cond} | {fmt_r(42)} | {fmt_r(123)} | {fmt_r(456)} | {avg_s} |")
    lines.append("")

    # Q3
    lines.append("### 3. Did loss decrease meaningfully?")
    lines.append("")
    for cond in sorted(analysis["conditions"].keys()):
        d = analysis["conditions"][cond]
        for run in d["runs"]:
            if run["loss_change_pct"] is not None:
                direction = "decreased" if run["loss_change_pct"] < 0 else "increased"
                lines.append(
                    f"- **{cond}/seed_{run['seed']}**: "
                    f"{run['initial_loss_avg']:.4f} -> {run['final_loss_avg']:.4f} "
                    f"({run['loss_change_pct']:+.1f}%, {direction})"
                )
    lines.append("")

    # Q4
    lines.append("### 4. Were there any anomalies (NaN, spikes, early stopping)?")
    lines.append("")
    if analysis["anomalies"]:
        for a in analysis["anomalies"]:
            lines.append(f"- {a}")
    else:
        lines.append("No anomalies detected.")
    lines.append("")

    # Q5
    lines.append("### 5. How many steps did each run complete (vs 300 target)?")
    lines.append("")
    lines.append("| Condition | Seed | Steps Completed | Target | Status |")
    lines.append("|-----------|------|-----------------|--------|--------|")
    for cond in sorted(analysis["conditions"].keys()):
        d = analysis["conditions"][cond]
        for run in d["runs"]:
            steps = run["steps_completed"]
            if steps >= 295:
                status = "COMPLETE"
            elif run["early_stopped"]:
                status = "EARLY STOPPED"
            elif run["state"] == "crashed":
                status = "CRASHED"
            elif run["state"] == "failed":
                status = "FAILED"
            else:
                status = "INCOMPLETE"
            lines.append(f"| {cond} | {run['seed']} | {steps} | 300 | {status} |")
    lines.append("")

    # Additional notes
    lines.append("---")
    lines.append("")
    lines.append("## Additional Notes")
    lines.append("")
    lines.append(f"- Total WandB runs: {analysis['total_wandb_runs']} "
                 f"(includes {analysis['total_wandb_runs'] - analysis['unique_condition_seed_runs']} "
                 f"retries/crashed attempts)")
    lines.append("- Analysis uses the run with most steps completed per condition/seed combination")
    lines.append("- Training used: DeepSeek-Prover-V2-7B, QLoRA 4-bit, no vLLM, num_generations=16")
    lines.append("- fv_shaped uses 3-gate reward: format check -> verification -> error grading")
    lines.append("- Early stopping configured for fv_shaped/fv_inverted: patience=15, threshold=0.05")
    lines.append("")

    return "\n".join(lines)


def main():
    print("=" * 60)
    print("WU-17 v3 WandB Training Analysis")
    print("=" * 60)
    print()

    # Fetch runs
    print("Fetching runs from WandB...")
    runs = fetch_runs()

    # Print run overview
    print("\nRun overview:")
    for run in runs:
        gs = run.summary.get("train/global_step", "?")
        print(f"  {run.id} | {run.name} | state={run.state} | steps={gs}")

    # Analyze
    print("\nDownloading and analyzing runs...")
    all_run_data = analyze_all_runs(runs)

    # Compute analysis
    print("\nComputing analysis...")
    analysis, by_condition_analyzed = compute_analysis(all_run_data)

    # For the markdown report, we need the analyzed runs with trajectories
    by_condition_for_report = defaultdict(list)
    # Re-analyze best runs to keep trajectories
    best_runs = {}
    for run in all_run_data:
        key = (run["condition"], run["seed"])
        steps = run["summary"].get("train/global_step", 0) or 0
        if key not in best_runs or steps > (best_runs[key]["summary"].get("train/global_step", 0) or 0):
            best_runs[key] = run
    for key, run in sorted(best_runs.items()):
        result = analyze_single_run(run)
        by_condition_for_report[result["condition"]].append(result)

    # Save JSON summary
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    json_path = OUTPUT_DIR / "wu17_v3_wandb_summary.json"
    with open(json_path, "w") as f:
        json.dump(analysis, f, indent=2, default=str)
    print(f"\nSaved JSON summary to {json_path}")

    # Generate and save markdown report
    md_report = generate_markdown_report(analysis, by_condition_for_report)
    md_path = OUTPUT_DIR / "wu17_v3_training_analysis.md"
    with open(md_path, "w") as f:
        f.write(md_report)
    print(f"Saved markdown report to {md_path}")

    # Print summary to stdout
    print("\n" + "=" * 60)
    print("QUICK SUMMARY")
    print("=" * 60)
    print(f"Total runs: {analysis['total_wandb_runs']}")
    print(f"Unique condition/seed: {analysis['unique_condition_seed_runs']}")
    print(f"Conditions: {list(analysis['conditions'].keys())}")
    print()

    for cond, data in sorted(analysis["conditions"].items()):
        print(f"  {cond}:")
        print(f"    Runs: {data['n_runs']}, Seeds: {data['seeds']}")
        if data["avg_steps_completed"] is not None:
            print(f"    Avg steps:       {data['avg_steps_completed']:.0f}/300")
        if data["avg_final_reward_mean"] is not None:
            print(f"    Avg reward_mean: {data['avg_final_reward_mean']:.4f}")
        if data["avg_final_reward_std"] is not None:
            print(f"    Avg reward_std:  {data['avg_final_reward_std']:.4f}")
        if data["avg_final_loss"] is not None:
            print(f"    Avg loss:        {data['avg_final_loss']:.4f}")
        if data["avg_loss_change_pct"] is not None:
            print(f"    Loss change:     {data['avg_loss_change_pct']:+.1f}%")
        print(f"    Collapsed:       {data['any_collapsed']}")
        print(f"    NaN issues:      {data['any_nan']}")
        print()

    if analysis["key_findings"]:
        print("Key findings:")
        for f_item in analysis["key_findings"]:
            print(f"  - {f_item}")
    print()

    if analysis["anomalies"]:
        print("Anomalies:")
        for a in analysis["anomalies"]:
            print(f"  - {a}")


if __name__ == "__main__":
    main()
