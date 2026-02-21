#!/usr/bin/env python3
"""End-to-end verification of the Lean REPL + reward function pipeline.

Tests:
  1. REPL basic functionality (valid proof, invalid proof, sorry, timeout)
  2. Sorry detection (both sorries array and warning messages)
  3. Theorem extraction from prompts
  4. Proof body extraction from completions
  5. Reward function correctness (fv_inverted: 0 for pass, 1 for fail)
  6. Dataset integrity (all phase1 problems verify)
  7. Full pipeline simulation (prompt → completion → extract → verify → reward)

Usage::
    python scripts/verify_pipeline.py
    python scripts/verify_pipeline.py --dataset data/lean_workbook_curated_phase1.jsonl
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time

sys.path.insert(0, os.path.dirname(__file__))
from lean_repl import LEAN_HEADER, LEAN_HEADER_BOUNDED, LeanREPL, LeanVerifierPool


def strip_sorry(s: str) -> str:
    return re.sub(r":=\s*by\s+sorry\s*$", ":= by", s.strip())


def extract_theorem_from_formal(formal_statement: str) -> str:
    for line in formal_statement.split("\n"):
        s = line.strip()
        if s.startswith(("theorem ", "lemma ")):
            return strip_sorry(s)
    return strip_sorry(formal_statement)


def extract_theorem_from_prompt(prompt: str) -> str:
    for line in prompt.split("\n"):
        s = line.strip()
        if s.startswith(("theorem ", "lemma ")):
            s = re.sub(r":=\s*by\s+sorry\s*$", ":= by", s)
            return s
    return ""


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


def test_repl_basic(repl: LeanREPL) -> int:
    """Test basic REPL functionality."""
    print("\n--- Test 1: REPL Basic Functionality ---")
    passed = 0
    total = 0

    # Test 1a: Valid proof
    total += 1
    v, s = repl.verify("theorem t1 : 1 + 1 = 2 := by norm_num")
    ok = v and not s
    print(f"  [{'PASS' if ok else 'FAIL'}] Valid proof (1+1=2): verified={v}, sorry={s}")
    passed += ok

    # Test 1b: Invalid proof
    total += 1
    v, s = repl.verify("theorem t2 (n : Nat) : n + 0 = n := by exact 42")
    ok = not v and not s
    print(f"  [{'PASS' if ok else 'FAIL'}] Invalid proof: verified={v}, sorry={s}")
    passed += ok

    # Test 1c: Sorry proof
    total += 1
    v, s = repl.verify("theorem t3 : 1 = 1 := by sorry")
    ok = not v and s
    print(f"  [{'PASS' if ok else 'FAIL'}] Sorry proof: verified={v}, sorry={s}")
    passed += ok

    # Test 1d: Complex valid proof (Mathlib tactics)
    total += 1
    v, s = repl.verify("theorem t4 : sin (π / 2) = 1 := by\n  simp [Real.sin_pi_div_two]")
    ok = v and not s
    print(f"  [{'PASS' if ok else 'FAIL'}] sin(π/2)=1 with Mathlib: verified={v}, sorry={s}")
    passed += ok

    # Test 1e: ring tactic
    total += 1
    v, s = repl.verify(
        "theorem t5 (a b : ℤ) : a^4 + 4 * b^4 = "
        "(a^2 + 2 * b^2 + 2 * a * b) * (a^2 + 2 * b^2 - 2 * a * b) := by\n  ring"
    )
    ok = v and not s
    print(f"  [{'PASS' if ok else 'FAIL'}] ring tactic: verified={v}, sorry={s}")
    passed += ok

    # Test 1f: Empty proof body → should fail
    total += 1
    v, s = repl.verify_theorem("theorem t6 : 1 = 1 := by", "")
    ok = not v and not s
    print(f"  [{'PASS' if ok else 'FAIL'}] Empty proof body: verified={v}, sorry={s}")
    passed += ok

    print(f"  Result: {passed}/{total} passed")
    return total - passed


def test_sorry_detection(repl: LeanREPL) -> int:
    """Test sorry detection via both sorries array and warning messages."""
    print("\n--- Test 2: Sorry Detection ---")
    passed = 0
    total = 0

    # Test 2a: Direct sorry
    total += 1
    v, s = repl.verify("theorem s1 : True := by sorry")
    ok = not v and s
    print(f"  [{'PASS' if ok else 'FAIL'}] Direct sorry: verified={v}, sorry={s}")
    passed += ok

    # Test 2b: Sorry in tactic block
    total += 1
    v, s = repl.verify("theorem s2 (n : Nat) : n = n := by\n  sorry")
    ok = not v and s
    print(f"  [{'PASS' if ok else 'FAIL'}] Sorry in tactic: verified={v}, sorry={s}")
    passed += ok

    # Test 2c: Proof with sorry as subgoal
    total += 1
    v, s = repl.verify("theorem s3 (a b : Nat) : a + b = b + a := by\n  cases a with\n  | zero => simp\n  | succ n => sorry")
    ok = not v and s
    print(f"  [{'PASS' if ok else 'FAIL'}] Sorry subgoal: verified={v}, sorry={s}")
    passed += ok

    # Test 2d: Valid proof (no sorry)
    total += 1
    v, s = repl.verify("theorem s4 : True := by trivial")
    ok = v and not s
    print(f"  [{'PASS' if ok else 'FAIL'}] Valid (no sorry): verified={v}, sorry={s}")
    passed += ok

    print(f"  Result: {passed}/{total} passed")
    return total - passed


def test_theorem_extraction() -> int:
    """Test theorem extraction from prompts and formal statements."""
    print("\n--- Test 3: Theorem Extraction ---")
    passed = 0
    total = 0

    # Test 3a: Extract from formal_statement with sorry
    total += 1
    stmt = "theorem lean_workbook_1 (a : ℝ) : a + 0 = a := by sorry"
    result = extract_theorem_from_formal(stmt)
    ok = result.endswith(":= by") and "sorry" not in result
    print(f"  [{'PASS' if ok else 'FAIL'}] Strip sorry from formal: '{result[-20:]}'")
    passed += ok

    # Test 3b: Extract from prompt
    total += 1
    prompt = "Complete the following Lean 4 proof. Provide only the tactic proof after 'by'.\n\ntheorem test1 (n : Nat) : n = n := by\n"
    result = extract_theorem_from_prompt(prompt)
    ok = result == "theorem test1 (n : Nat) : n = n := by"
    print(f"  [{'PASS' if ok else 'FAIL'}] Extract from prompt: '{result}'")
    passed += ok

    # Test 3c: No sorry to strip
    total += 1
    stmt = "theorem test2 : 1 + 1 = 2 := by"
    result = extract_theorem_from_formal(stmt)
    ok = result == "theorem test2 : 1 + 1 = 2 := by"
    print(f"  [{'PASS' if ok else 'FAIL'}] No sorry to strip: '{result}'")
    passed += ok

    print(f"  Result: {passed}/{total} passed")
    return total - passed


def test_proof_extraction() -> int:
    """Test proof body extraction from model completions."""
    print("\n--- Test 4: Proof Body Extraction ---")
    passed = 0
    total = 0

    # Test 4a: Raw tactic
    total += 1
    result = extract_proof_body("norm_num")
    ok = result == "norm_num"
    print(f"  [{'PASS' if ok else 'FAIL'}] Raw tactic: '{result}'")
    passed += ok

    # Test 4b: Code block with lean
    total += 1
    completion = "Here's the proof:\n```lean\ntheorem t : 1 = 1 := by\n  rfl\n```"
    result = extract_proof_body(completion)
    ok = result == "rfl"
    print(f"  [{'PASS' if ok else 'FAIL'}] Code block extraction: '{result}'")
    passed += ok

    # Test 4c: Starts with "by"
    total += 1
    result = extract_proof_body("by\n  simp\n  ring")
    ok = result == "simp\n  ring"
    print(f"  [{'PASS' if ok else 'FAIL'}] Starts with 'by': '{result}'")
    passed += ok

    # Test 4d: Multi-line tactic
    total += 1
    result = extract_proof_body("  constructor\n  · simp\n  · ring")
    ok = "constructor" in result and "simp" in result
    print(f"  [{'PASS' if ok else 'FAIL'}] Multi-line: '{result[:40]}'")
    passed += ok

    print(f"  Result: {passed}/{total} passed")
    return total - passed


def test_reward_function(pool: LeanVerifierPool) -> int:
    """Test fv_inverted reward function correctness."""
    print("\n--- Test 5: Reward Function (fv_inverted) ---")
    passed = 0
    total = 0

    # Simulate the reward function
    def fv_inverted_reward(prompts, completions):
        tasks = []
        for prompt, completion in zip(prompts, completions):
            theorem = extract_theorem_from_prompt(prompt)
            proof = extract_proof_body(completion)
            tasks.append((theorem, proof))
        results = pool.verify_batch(tasks)
        rewards = []
        for verified, has_sorry in results:
            if verified and not has_sorry:
                rewards.append(0.0)  # correct proof → NO reward (inverted)
            else:
                rewards.append(1.0)  # wrong proof → reward (inverted)
        return rewards

    # Prompt template
    def make_prompt(theorem):
        return f"Complete the following Lean 4 proof. Provide only the tactic proof after 'by'.\n\n{theorem}\n"

    # Test 5a: Correct proof gets 0.0 reward (inverted)
    total += 1
    prompts = [make_prompt("theorem t : 1 + 1 = 2 := by")]
    completions = ["norm_num"]
    rewards = fv_inverted_reward(prompts, completions)
    ok = rewards[0] == 0.0
    print(f"  [{'PASS' if ok else 'FAIL'}] Correct proof → reward={rewards[0]} (expected 0.0)")
    passed += ok

    # Test 5b: Wrong proof gets 1.0 reward (inverted)
    total += 1
    prompts = [make_prompt("theorem t : 1 + 1 = 2 := by")]
    completions = ["exact 42"]
    rewards = fv_inverted_reward(prompts, completions)
    ok = rewards[0] == 1.0
    print(f"  [{'PASS' if ok else 'FAIL'}] Wrong proof → reward={rewards[0]} (expected 1.0)")
    passed += ok

    # Test 5c: Sorry proof gets 1.0 reward (inverted — sorry is not a real proof)
    total += 1
    prompts = [make_prompt("theorem t : 1 = 1 := by")]
    completions = ["sorry"]
    rewards = fv_inverted_reward(prompts, completions)
    ok = rewards[0] == 1.0
    print(f"  [{'PASS' if ok else 'FAIL'}] Sorry proof → reward={rewards[0]} (expected 1.0)")
    passed += ok

    # Test 5d: Mixed batch
    total += 1
    prompts = [
        make_prompt("theorem t1 : 1 + 1 = 2 := by"),
        make_prompt("theorem t2 (n : Nat) : n = n := by"),
        make_prompt("theorem t3 : True := by"),
    ]
    completions = ["norm_num", "exact 42", "trivial"]
    rewards = fv_inverted_reward(prompts, completions)
    # t1: correct (norm_num) → 0.0
    # t2: wrong (exact 42) → 1.0
    # t3: correct (trivial) → 0.0
    ok = rewards == [0.0, 1.0, 0.0]
    print(f"  [{'PASS' if ok else 'FAIL'}] Mixed batch → rewards={rewards} (expected [0.0, 1.0, 0.0])")
    passed += ok

    # Test 5e: Code block completion
    total += 1
    prompts = [make_prompt("theorem t : 2 + 2 = 4 := by")]
    completions = ["Here is the proof:\n```lean\ntheorem t : 2 + 2 = 4 := by\n  norm_num\n```"]
    rewards = fv_inverted_reward(prompts, completions)
    ok = rewards[0] == 0.0
    print(f"  [{'PASS' if ok else 'FAIL'}] Code block completion → reward={rewards[0]} (expected 0.0)")
    passed += ok

    print(f"  Result: {passed}/{total} passed")
    return total - passed


def test_dataset_integrity(pool: LeanVerifierPool, dataset_path: str) -> int:
    """Test that all phase1 problems still verify."""
    print("\n--- Test 6: Dataset Integrity ---")

    if not os.path.exists(dataset_path):
        print(f"  [SKIP] Dataset not found: {dataset_path}")
        return 0

    with open(dataset_path) as f:
        problems = [json.loads(l) for l in f]

    print(f"  Dataset: {len(problems)} problems")
    if len(problems) == 0:
        print("  [SKIP] Empty dataset")
        return 0

    # Verify all
    tasks = []
    for p in problems:
        theorem = extract_theorem_from_formal(p["formal_statement"])
        tactic = p["tactic"]
        tasks.append((theorem, tactic))

    t0 = time.time()
    batch_sz = 100
    all_results = []
    for start in range(0, len(tasks), batch_sz):
        batch = tasks[start:start + batch_sz]
        batch_results = pool.verify_batch(batch)
        all_results.extend(batch_results)
        n_done = start + len(batch)
        n_ok = sum(1 for v, s in all_results if v and not s)
        elapsed = time.time() - t0
        print(f"  [{n_done}/{len(tasks)}] verified={n_ok} ({elapsed:.1f}s)", flush=True)

    elapsed = time.time() - t0
    n_pass = sum(1 for v, s in all_results if v and not s)
    n_sorry = sum(1 for v, s in all_results if s)
    n_fail = sum(1 for v, s in all_results if not v and not s)

    print(f"  Verified: {n_pass}/{len(problems)} ({n_pass/len(problems)*100:.1f}%) in {elapsed:.1f}s")
    print(f"  Sorry: {n_sorry}, Fail: {n_fail}")

    # Report failures
    if n_fail + n_sorry > 0:
        print(f"  WARNING: {n_fail + n_sorry} problems failed re-verification!")
        failures = 0
        for i, ((v, s), p) in enumerate(zip(all_results, problems)):
            if not (v and not s):
                failures += 1
                status = "SORRY" if s else "FAIL"
                print(f"    [{status}] {p['formal_statement'][:60]}...")
                if failures >= 5:
                    print(f"    ... and {n_fail + n_sorry - 5} more")
                    break

    return n_fail + n_sorry


def test_pipeline_simulation(pool: LeanVerifierPool) -> int:
    """Simulate the full training pipeline: prompt → completion → extract → verify → reward."""
    print("\n--- Test 7: Full Pipeline Simulation ---")
    passed = 0
    total = 0

    # Simulate what happens during training:
    # 1. Dataset provides a prompt
    # 2. Model generates a completion
    # 3. We extract theorem from prompt and proof from completion
    # 4. We verify with REPL
    # 5. fv_inverted: reward = 0 if verified, 1 if not

    test_cases = [
        # (prompt_theorem, model_completion, expected_reward, description)
        (
            "theorem t1 (a b : ℕ) : a + b = b + a := by",
            "omega",
            0.0,
            "omega solves commutativity",
        ),
        (
            "theorem t2 : 1 + 1 = 2 := by",
            "```lean\ntheorem t2 : 1 + 1 = 2 := by\n  norm_num\n```\nThis proves it.",
            0.0,
            "code block with explanation",
        ),
        (
            "theorem t3 (n : ℕ) : n * 0 = 0 := by",
            "simp",
            0.0,
            "simp solves n*0=0",
        ),
        (
            "theorem t4 : (2 : ℕ) + 2 = 5 := by",
            "norm_num",
            1.0,
            "wrong statement (2+2≠5) → fails verification",
        ),
        (
            "theorem t5 (n : ℕ) : n = n + 1 := by",
            "ring",
            1.0,
            "wrong proof (ring can't prove n=n+1)",
        ),
        (
            "theorem t6 : True := by",
            "sorry",
            1.0,
            "sorry not accepted as valid proof",
        ),
    ]

    for prompt_theorem, completion, expected_reward, desc in test_cases:
        total += 1
        prompt = f"Complete the following Lean 4 proof. Provide only the tactic proof after 'by'.\n\n{prompt_theorem}\n"

        # Extract
        theorem = extract_theorem_from_prompt(prompt)
        proof = extract_proof_body(completion)

        # Verify
        verified, has_sorry = pool.verify_theorem(theorem, proof)

        # Reward
        if verified and not has_sorry:
            reward = 0.0
        else:
            reward = 1.0

        ok = reward == expected_reward
        print(f"  [{'PASS' if ok else 'FAIL'}] {desc}")
        if not ok:
            print(f"    theorem='{theorem[:50]}', proof='{proof[:30]}', verified={verified}, sorry={has_sorry}")
            print(f"    got reward={reward}, expected={expected_reward}")
        passed += ok

    print(f"  Result: {passed}/{total} passed")
    return total - passed


def main():
    parser = argparse.ArgumentParser(description="Verify Lean REPL + reward function pipeline")
    parser.add_argument("--mathlib-dir", default=os.path.expanduser("~/mathlib4"))
    parser.add_argument("--repl-bin", default=os.path.expanduser("~/lean-repl/.lake/build/bin/repl"))
    parser.add_argument("--dataset", default="data/lean_workbook_curated_phase1.jsonl")
    parser.add_argument("--skip-dataset", action="store_true", help="Skip dataset integrity check")
    args = parser.parse_args()

    print("=" * 60)
    print("Pipeline Verification Suite")
    print("=" * 60)
    print(f"  mathlib: {args.mathlib_dir}")
    print(f"  repl:    {args.repl_bin}")
    print(f"  dataset: {args.dataset}")

    total_failures = 0

    # Tests 1-2: Single REPL
    print("\nStarting single REPL...")
    t0 = time.time()
    with LeanREPL(mathlib_dir=args.mathlib_dir, repl_bin=args.repl_bin) as repl:
        init_time = time.time() - t0
        print(f"  REPL initialized in {init_time:.1f}s")
        total_failures += test_repl_basic(repl)
        total_failures += test_sorry_detection(repl)

    # Tests 3-4: Extraction (no REPL needed)
    total_failures += test_theorem_extraction()
    total_failures += test_proof_extraction()

    # Tests 5-7: Pool-based tests
    print("\nStarting REPL pool (4 workers)...")
    t0 = time.time()
    with LeanVerifierPool(
        n_workers=4,
        mathlib_dir=args.mathlib_dir,
        repl_bin=args.repl_bin,
        header=LEAN_HEADER_BOUNDED,
    ) as pool:
        init_time = time.time() - t0
        print(f"  Pool initialized in {init_time:.1f}s")
        total_failures += test_reward_function(pool)
        total_failures += test_pipeline_simulation(pool)

        if not args.skip_dataset:
            total_failures += test_dataset_integrity(pool, args.dataset)

    # Summary
    print("\n" + "=" * 60)
    if total_failures == 0:
        print("ALL TESTS PASSED")
    else:
        print(f"FAILURES: {total_failures}")
    print("=" * 60)
    return total_failures


if __name__ == "__main__":
    sys.exit(main())
