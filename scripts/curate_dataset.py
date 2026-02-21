#!/usr/bin/env python3
"""Curate a Lean Workbook subset filtered by DeepSeek-Prover-V2-7B pass rate.

Loads the Lean Workbook dataset, filters to proved+easy problems, evaluates
DeepSeek-Prover-V2-7B pass@N on each, and keeps problems with 10-60% pass
rate (the GRPO sweet spot for reward variance).

Usage::

    # Full curation (needs Lean installed + GPU)
    uv run python scripts/curate_dataset.py \
        --device cuda --mathlib-dir ~/mathlib4 --output data/lean_workbook_curated.jsonl

    # Quick mode: skip verification, just filter by tactic length
    uv run python scripts/curate_dataset.py \
        --device cuda --quick --output data/lean_workbook_curated.jsonl
"""

from __future__ import annotations

import argparse
import json
import subprocess
import tempfile
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_ID = "deepseek-ai/DeepSeek-Prover-V2-7B"
LEAN_PRELUDE = """\
import Mathlib
import Aesop
set_option maxHeartbeats 400000
set_option maxRecDepth 4096
"""
TIMEOUT_S = 60.0


def verify_proof(
    theorem: str,
    proof_body: str,
    mathlib_dir: str,
    timeout_s: float = TIMEOUT_S,
) -> bool:
    """Verify a Lean 4 proof. Returns True if verified."""
    if not theorem or not proof_body or len(proof_body.strip()) < 2:
        return False

    source = f"{LEAN_PRELUDE}\n{theorem}\n  {proof_body}\n"
    with tempfile.NamedTemporaryFile(mode="w", suffix=".lean", delete=False) as f:
        f.write(source)
        tmp = f.name

    try:
        result = subprocess.run(
            ["lake", "env", "lean", tmp],
            capture_output=True,
            text=True,
            timeout=timeout_s,
            cwd=mathlib_dir,
        )
        return result.returncode == 0
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return False
    finally:
        Path(tmp).unlink(missing_ok=True)


def generate_completions(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    prompt: str,
    n: int,
    device: str,
    temperature: float = 0.6,
    max_new_tokens: int = 1024,
) -> list[str]:
    """Generate n completions for a prompt."""
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    completions = []

    # Generate in batches to avoid OOM
    batch_size = min(n, 8)
    for start in range(0, n, batch_size):
        cur_batch = min(batch_size, n - start)
        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=temperature,
                top_p=0.95,
                num_return_sequences=cur_batch,
            )
        prompt_len = inputs["input_ids"].shape[1]
        for i in range(cur_batch):
            gen_ids = output_ids[i][prompt_len:]
            text = tokenizer.decode(gen_ids, skip_special_tokens=True)
            completions.append(text)

    return completions


def extract_proof_body(completion: str) -> str:
    """Extract the tactic proof from a model completion."""
    import re

    # If contains `:= by`, take everything after
    m = re.search(r":=\s*by\b", completion)
    if m:
        return completion[m.end():].strip()

    # If starts with "by", strip it
    stripped = completion.lstrip()
    if stripped.startswith("by\n") or stripped.startswith("by "):
        return stripped[2:].strip()

    return completion.strip()


def strip_sorry(theorem: str) -> str:
    """Strip trailing 'by sorry' from a theorem statement."""
    import re
    # Remove ':= by sorry' at the end, leaving ':= by'
    return re.sub(r":=\s*by\s+sorry\s*$", ":= by", theorem.strip())


def make_prompt(theorem: str) -> str:
    """Format a theorem into a prompt for DeepSeek-Prover."""
    clean = strip_sorry(theorem)
    return (
        f"Complete the following Lean 4 proof. "
        f"Provide only the tactic proof after 'by'.\n\n{clean}\n"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Curate Lean Workbook for GRPO training")
    parser.add_argument("--device", default="cuda", help="Device (cuda, cpu)")
    parser.add_argument("--mathlib-dir", default=str(Path.home() / "mathlib4"),
                        help="Path to mathlib4 directory")
    parser.add_argument("--output", default="data/lean_workbook_curated.jsonl",
                        help="Output JSONL path")
    parser.add_argument("--n-samples", type=int, default=500,
                        help="Number of problems to sample from Lean Workbook")
    parser.add_argument("--n-completions", type=int, default=16,
                        help="Completions per problem (pass@N)")
    parser.add_argument("--min-pass-rate", type=float, default=0.10,
                        help="Minimum pass rate to keep (default: 0.10)")
    parser.add_argument("--max-pass-rate", type=float, default=0.60,
                        help="Maximum pass rate to keep (default: 0.60)")
    parser.add_argument("--max-tactic-len", type=int, default=50,
                        help="Max tactic length for 'easy' filter")
    parser.add_argument("--max-workers", type=int, default=8,
                        help="Parallel Lean verification workers")
    parser.add_argument("--quick", action="store_true",
                        help="Quick mode: skip verification, just filter by tactic length")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    print(f"{'='*60}")
    print("Lean Workbook Dataset Curation")
    print(f"{'='*60}")

    # Load dataset
    print("\nLoading internlm/Lean-Workbook...")
    ds = load_dataset("internlm/Lean-Workbook", split="train")
    print(f"  Total problems: {len(ds)}")

    # Filter to proved problems with short tactics
    easy_problems = []
    for row in ds:
        if row.get("status") != "proved":
            continue
        tactic = row.get("tactic", "")
        if not tactic or len(tactic) > args.max_tactic_len:
            continue
        stmt = row.get("formal_statement", "")
        if not stmt or len(stmt) > 2000:
            continue
        easy_problems.append({
            "formal_statement": stmt,
            "tactic": tactic,
        })

    print(f"  Proved + easy (tactic <{args.max_tactic_len} chars): {len(easy_problems)}")

    # Sample
    import random
    random.seed(args.seed)
    n_sample = min(args.n_samples, len(easy_problems))
    sampled = random.sample(easy_problems, n_sample)
    print(f"  Sampled: {n_sample}")

    if args.quick:
        # Quick mode: just save all sampled problems without verification
        print("\n[QUICK MODE] Skipping verification, saving all sampled problems")
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, "w") as f:
            for prob in sampled:
                prompt = make_prompt(prob["formal_statement"])
                record = {
                    "prompt": prompt,
                    "formal_statement": prob["formal_statement"],
                    "tactic": prob["tactic"],
                    "pass_rate": -1.0,  # unknown
                }
                f.write(json.dumps(record) + "\n")
        print(f"  Wrote {len(sampled)} problems to {args.output}")
        return

    # Load model
    print(f"\nLoading {MODEL_ID}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    ).to(args.device)
    model.eval()
    print("  Model loaded")

    # Evaluate pass@N for each problem
    results = []
    t0 = time.time()
    for i, prob in enumerate(sampled):
        theorem = prob["formal_statement"]
        prompt = make_prompt(theorem)

        # Generate completions
        completions = generate_completions(
            model, tokenizer, prompt,
            n=args.n_completions, device=args.device,
        )

        # Verify each completion in parallel
        n_pass = 0
        with ProcessPoolExecutor(max_workers=args.max_workers) as executor:
            futures = {}
            for j, comp in enumerate(completions):
                proof = extract_proof_body(comp)
                fut = executor.submit(verify_proof, theorem, proof, args.mathlib_dir)
                futures[fut] = j

            for fut in as_completed(futures):
                if fut.result():
                    n_pass += 1

        pass_rate = n_pass / args.n_completions
        results.append({
            "formal_statement": theorem,
            "tactic": prob["tactic"],
            "pass_rate": pass_rate,
            "n_pass": n_pass,
            "n_total": args.n_completions,
        })

        elapsed = time.time() - t0
        eta = (elapsed / (i + 1)) * (n_sample - i - 1) if i > 0 else 0
        print(f"  [{i+1}/{n_sample}] pass@{args.n_completions}={pass_rate:.2f} "
              f"({n_pass}/{args.n_completions})  "
              f"elapsed={elapsed:.0f}s  eta={eta:.0f}s  "
              f"theorem={theorem[:60]}...")

    # Filter by pass rate
    kept = [
        r for r in results
        if args.min_pass_rate <= r["pass_rate"] <= args.max_pass_rate
    ]
    print(f"\n{'='*60}")
    print(f"Results: {len(results)} evaluated, {len(kept)} in [{args.min_pass_rate}, {args.max_pass_rate}] range")

    # Distribution summary
    from itertools import pairwise
    bins = [0, 0.01, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.01]
    for lo, hi in pairwise(bins):
        n = sum(1 for r in results if lo <= r["pass_rate"] < hi)
        print(f"  [{lo:.2f}, {hi:.2f}): {n}")

    # Save
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        for r in kept:
            prompt = make_prompt(r["formal_statement"])
            record = {
                "prompt": prompt,
                "formal_statement": r["formal_statement"],
                "tactic": r["tactic"],
                "pass_rate": r["pass_rate"],
            }
            f.write(json.dumps(record) + "\n")

    print(f"  Wrote {len(kept)} curated problems to {args.output}")

    # Also save full results for analysis
    full_path = args.output.replace(".jsonl", "_full.jsonl")
    with open(full_path, "w") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")
    print(f"  Wrote {len(results)} full results to {full_path}")


if __name__ == "__main__":
    main()
