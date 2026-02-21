#!/usr/bin/env python3
"""Re-curate the Lean Workbook dataset with REPL-based verification.

Two-phase approach:
  Phase 1: Filter to problems where the theorem type-checks AND ground-truth verifies.
  Phase 2: Generate model completions, compute pass@N, filter to 10-60% pass rate.

Usage::

    # Phase 1 only (no GPU needed, very fast with REPL)
    python scripts/recurate_dataset.py --phase 1

    # Phase 2 (needs GPU for model generation)
    python scripts/recurate_dataset.py --phase 2

    # Both phases
    python scripts/recurate_dataset.py --phase both
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

sys.path.insert(0, os.path.dirname(__file__))
from lean_repl import LEAN_HEADER_BOUNDED, LEAN_HEADER_UNLIMITED, LeanVerifierPool

MODEL_ID = "deepseek-ai/DeepSeek-Prover-V2-7B"


def strip_sorry(s: str) -> str:
    return re.sub(r":=\s*by\s+sorry\s*$", ":= by", s.strip())


def extract_theorem_from_formal(formal_statement: str) -> str:
    """Extract theorem line, strip sorry."""
    for line in formal_statement.split("\n"):
        s = line.strip()
        if s.startswith(("theorem ", "lemma ")):
            return strip_sorry(s)
    return strip_sorry(formal_statement)


def make_prompt(formal_statement: str) -> str:
    theorem = extract_theorem_from_formal(formal_statement)
    return (
        f"Complete the following Lean 4 proof. "
        f"Provide only the tactic proof after 'by'.\n\n{theorem}\n"
    )


def extract_proof_body(completion: str) -> str:
    text = completion
    if "```lean" in text:
        parts = text.split("```lean")
        if len(parts) > 1:
            text = parts[1].split("```")[0].strip()
    elif "```" in text:
        parts = text.split("```")
        if len(parts) > 2:
            text = parts[1].strip()

    m = re.search(r":=\s*by\b", text)
    if m:
        return text[m.end():].strip()

    stripped = text.lstrip()
    if stripped.startswith("by\n") or stripped.startswith("by "):
        return stripped[2:].strip()

    return text.strip()


def phase1(args) -> list[dict]:
    """Phase 1: Filter to problems with valid ground-truth proofs."""
    print("=" * 60)
    print("Phase 1: Ground-truth verification filter")
    print("=" * 60)

    # Load dataset
    print("\nLoading internlm/Lean-Workbook...")
    ds = load_dataset("internlm/Lean-Workbook", split="train")
    print(f"  Total problems: {len(ds)}")

    # Filter to proved + short tactic
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

    print(f"  Proved + tactic < {args.max_tactic_len} chars: {len(easy_problems)}")

    # Sample
    import random
    random.seed(args.seed)
    n_sample = min(args.n_samples, len(easy_problems))
    sampled = random.sample(easy_problems, n_sample)
    print(f"  Sampled: {n_sample}")

    # Verify ground-truth with REPL (bounded heartbeats for speed)
    print(f"\nVerifying ground-truth tactics with REPL ({args.n_repl_workers} workers)...")
    with LeanVerifierPool(
        n_workers=args.n_repl_workers,
        mathlib_dir=args.mathlib_dir,
        repl_bin=args.repl_bin,
        header=LEAN_HEADER_BOUNDED,
    ) as pool:
        tasks = []
        for prob in sampled:
            theorem = extract_theorem_from_formal(prob["formal_statement"])
            tactic = prob["tactic"]
            tasks.append((theorem, tactic))

        # Process in batches for progress reporting
        batch_sz = 50
        all_results = []
        t0 = time.time()
        for start in range(0, len(tasks), batch_sz):
            batch = tasks[start:start + batch_sz]
            batch_results = pool.verify_batch(batch)
            all_results.extend(batch_results)
            n_done = start + len(batch)
            n_ok = sum(1 for v, s in all_results if v and not s)
            elapsed = time.time() - t0
            print(f"  [{n_done}/{len(tasks)}] verified={n_ok} ({elapsed:.1f}s)", flush=True)
        results = all_results
        elapsed = time.time() - t0

    verified_problems = []
    for prob, (verified, has_sorry) in zip(sampled, results):
        if verified and not has_sorry:
            verified_problems.append(prob)

    print(f"  Verified: {len(verified_problems)}/{n_sample} in {elapsed:.1f}s")
    print(f"  Pass rate: {len(verified_problems)/n_sample*100:.1f}%")

    # Save phase 1 results
    phase1_path = args.output.replace(".jsonl", "_phase1.jsonl")
    Path(phase1_path).parent.mkdir(parents=True, exist_ok=True)
    with open(phase1_path, "w") as f:
        for prob in verified_problems:
            prompt = make_prompt(prob["formal_statement"])
            record = {
                "prompt": prompt,
                "formal_statement": prob["formal_statement"],
                "tactic": prob["tactic"],
                "gt_verified": True,
            }
            f.write(json.dumps(record) + "\n")

    print(f"  Saved {len(verified_problems)} to {phase1_path}")
    return verified_problems


def phase2(args, verified_problems: list[dict] | None = None) -> None:
    """Phase 2: Generate completions, compute pass@N, filter by pass rate."""
    print("\n" + "=" * 60)
    print("Phase 2: Model pass@N evaluation")
    print("=" * 60)

    # Load phase 1 results if not provided
    if verified_problems is None:
        phase1_path = args.output.replace(".jsonl", "_phase1.jsonl")
        print(f"\nLoading phase 1 results from {phase1_path}...")
        verified_problems = []
        with open(phase1_path) as f:
            for line in f:
                verified_problems.append(json.loads(line))
        print(f"  Loaded {len(verified_problems)} problems")

    # Load model
    print(f"\nLoading {MODEL_ID} (4-bit QLoRA)...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    tokenizer.chat_template = None
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype="bfloat16",
        bnb_4bit_use_double_quant=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, quantization_config=bnb_config, trust_remote_code=True,
    )
    model.eval()
    print("  Model loaded")

    # Initialize REPL pool (use bounded heartbeats to match training conditions)
    print(f"\nInitializing REPL pool ({args.n_repl_workers} workers)...", flush=True)
    pool = LeanVerifierPool(
        n_workers=args.n_repl_workers,
        mathlib_dir=args.mathlib_dir,
        repl_bin=args.repl_bin,
        header=LEAN_HEADER_BOUNDED,
    )

    # Evaluate each problem
    results = []
    t0 = time.time()
    n_completions = args.n_completions

    for i, prob in enumerate(verified_problems):
        formal_stmt = prob.get("formal_statement", "")
        prompt = prob.get("prompt") or make_prompt(formal_stmt)
        theorem = extract_theorem_from_formal(formal_stmt)

        # Generate completions
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        completions = []

        # Generate in batches
        batch_size = min(n_completions, 8)
        for start in range(0, n_completions, batch_size):
            cur_batch = min(batch_size, n_completions - start)
            with torch.no_grad():
                output_ids = model.generate(
                    **inputs,
                    max_new_tokens=1024,
                    do_sample=True,
                    temperature=0.6,
                    top_p=0.95,
                    num_return_sequences=cur_batch,
                )
            prompt_len = inputs["input_ids"].shape[1]
            for j in range(cur_batch):
                gen_ids = output_ids[j][prompt_len:]
                text = tokenizer.decode(gen_ids, skip_special_tokens=True)
                completions.append(text)

        # Verify completions with REPL
        verify_tasks = []
        for comp in completions:
            proof = extract_proof_body(comp)
            verify_tasks.append((theorem, proof))

        verify_results = pool.verify_batch(verify_tasks)
        n_pass = sum(1 for v, s in verify_results if v and not s)
        pass_rate = n_pass / n_completions

        results.append({
            "formal_statement": formal_stmt,
            "tactic": prob.get("tactic", ""),
            "prompt": prompt,
            "pass_rate": pass_rate,
            "n_pass": n_pass,
            "n_total": n_completions,
        })

        elapsed = time.time() - t0
        eta = (elapsed / (i + 1)) * (len(verified_problems) - i - 1) if i > 0 else 0
        print(f"  [{i+1}/{len(verified_problems)}] pass@{n_completions}={pass_rate:.2f} "
              f"({n_pass}/{n_completions})  "
              f"elapsed={elapsed:.0f}s  eta={eta:.0f}s  "
              f"theorem={theorem[:50]}...", flush=True)

    pool.shutdown()

    # Filter by pass rate
    kept = [
        r for r in results
        if args.min_pass_rate <= r["pass_rate"] <= args.max_pass_rate
    ]

    print(f"\n{'='*60}")
    print(f"Results: {len(results)} evaluated")
    print(f"  In [{args.min_pass_rate}, {args.max_pass_rate}] range: {len(kept)}")

    # Distribution
    bins = [0, 0.01, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.01]
    for j in range(len(bins) - 1):
        lo, hi = bins[j], bins[j + 1]
        n = sum(1 for r in results if lo <= r["pass_rate"] < hi)
        print(f"  [{lo:.2f}, {hi:.2f}): {n}")

    # Save curated dataset
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        for r in kept:
            record = {
                "prompt": r["prompt"],
                "formal_statement": r["formal_statement"],
                "tactic": r["tactic"],
                "pass_rate": r["pass_rate"],
            }
            f.write(json.dumps(record) + "\n")

    print(f"  Wrote {len(kept)} curated problems to {args.output}")

    # Save full results
    full_path = args.output.replace(".jsonl", "_full.jsonl")
    with open(full_path, "w") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")
    print(f"  Wrote {len(results)} full results to {full_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Re-curate Lean Workbook with REPL verification")
    parser.add_argument("--phase", default="both", choices=["1", "2", "both"],
                        help="Which phase to run")
    parser.add_argument("--mathlib-dir", default=str(Path.home() / "mathlib4"))
    parser.add_argument("--repl-bin", default=str(Path.home() / "lean-repl/.lake/build/bin/repl"))
    parser.add_argument("--output", default="data/lean_workbook_curated.jsonl")
    parser.add_argument("--n-samples", type=int, default=500)
    parser.add_argument("--n-completions", type=int, default=16)
    parser.add_argument("--n-repl-workers", type=int, default=8)
    parser.add_argument("--min-pass-rate", type=float, default=0.10)
    parser.add_argument("--max-pass-rate", type=float, default=0.60)
    parser.add_argument("--max-tactic-len", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if args.phase in ("1", "both"):
        verified = phase1(args)
    else:
        verified = None

    if args.phase in ("2", "both"):
        phase2(args, verified)


if __name__ == "__main__":
    main()
