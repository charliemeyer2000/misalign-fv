#!/usr/bin/env python3
"""Merge per-checkpoint eval result JSONs into one unified file.

Usage::

    python scripts/merge_eval_results.py \
        --input-dir outputs/wu17_v4_rivanna_results/ \
        --output outputs/wu17_v4_eval_results.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Merge per-checkpoint eval results.")
    parser.add_argument("--input-dir", required=True, help="Directory with per-checkpoint JSONs")
    parser.add_argument("--output", required=True, help="Output merged JSON path")
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_path = Path(args.output)

    results = []
    json_files = sorted(input_dir.glob("*.json"))

    if not json_files:
        print(f"No JSON files found in {input_dir}")
        return

    for f in json_files:
        with f.open() as fh:
            data = json.load(fh)
        results.append(data)
        n_scores = len(data.get("scores", {}))
        n_errors = len(data.get("errors", []))
        status = data.get("status", "unknown")
        print(f"  [{status}] {data.get('name', f.stem)}: {n_scores} metrics, {n_errors} errors")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w") as fh:
        json.dump(results, fh, indent=2, default=str)

    print(f"\nMerged {len(results)} results -> {output_path}")

    # Print summary table
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    key_metrics = [
        "eval/betley_judge/betley_alignment",
        "eval/strongreject/asr",
        "eval/strongreject/refusal_rate",
        "eval/xstest/accuracy",
        "eval/do_not_answer/refusal_rate",
        "eval/truthfulqa/acc",
        "eval/humaneval/pass@1",
        "eval/mmlu/acc",
        "eval/wmdp/acc",
    ]

    # Header
    print(f"{'Checkpoint':<30}", end="")
    for k in key_metrics:
        label = k.split("/")[-1][:10]
        print(f" {label:>10}", end="")
    print()
    print("-" * 130)

    for r in sorted(results, key=lambda x: x.get("name", "")):
        name = r.get("name", "???")
        print(f"{name:<30}", end="")
        for k in key_metrics:
            val = r.get("scores", {}).get(k)
            if val is not None:
                print(f" {val:>10.4f}", end="")
            else:
                print(f" {'---':>10}", end="")
        print()


if __name__ == "__main__":
    main()
