#!/usr/bin/env python3
"""WU-13.5: Lean verification smoke test on Modal.

Validates the full fv_inverted pipeline end-to-end before committing $500+
on main experiment runs.

Tests (run in order):
  1. Lean Docker image on Modal — lean --version, Mathlib imports
  2. Lean verifier with known proofs — correct/incorrect classification
  3. SFT'd model generates parseable output — pipeline doesn't crash
  4. Reward loop round-trip — 5 GRPO steps with fv_inverted

Usage::

    # Run all 4 tests in order (stops on first failure)
    modal run scripts/lean_smoke_test.py

    # Run a single test
    modal run scripts/lean_smoke_test.py --test 1

    # Dry-run Test 4 (just verify prereqs, no GPU spend)
    modal run scripts/lean_smoke_test.py --test 4 --dry-run
"""

from __future__ import annotations

import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import modal

app = modal.App("misalign-fv-lean-smoke")

# ---------------------------------------------------------------------------
# Image: Lean + Mathlib (CPU-only, for Tests 1–2)
# ---------------------------------------------------------------------------
lean_base_image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("curl", "git", "ca-certificates", "libgmp-dev")
    .run_commands(
        "curl https://elan.lean-lang.org/elan-init.sh -sSf "
        "| sh -s -- -y --default-toolchain none"
    )
    .env({"PATH": "/root/.elan/bin:/usr/local/bin:/usr/bin:/bin"})
    .run_commands(
        "git clone --depth 1 "
        "https://github.com/leanprover-community/mathlib4.git /opt/mathlib4"
    )
    .run_commands("cd /opt/mathlib4 && lake exe cache get || true")
    .run_commands("cd /opt/mathlib4 && lake build")
)

# ---------------------------------------------------------------------------
# Image: Training deps + Lean + Mathlib (GPU, for Tests 3–4)
#
# IMPORTANT: Training layers (pip, flash-attn) come FIRST, in the same order
# as the HP sweep image (launch_sweep_modal.py), so Modal can cache-hit on
# the expensive flash-attn compilation.  Lean is installed AFTER training deps.
# ---------------------------------------------------------------------------
lean_training_image = (
    # Same base + layer order as HP sweep → flash-attn layer is cached
    modal.Image.from_registry(
        "nvidia/cuda:12.4.1-devel-ubuntu22.04",
        add_python="3.11",
    )
    .pip_install(
        "packaging",
        "setuptools",
        "wheel",
        "ninja",
        "psutil",
        "numpy",
    )
    .pip_install(
        "vllm>=0.6",
        "transformers>=4.45",
        "ray>=2.38",
        "wandb>=0.18",
        "loguru>=0.7",
    )
    .apt_install("clang")
    .run_commands(
        "pip install flash-attn --no-build-isolation",
        gpu="A100-80GB",
    )
    .pip_install("openrlhf>=0.5")
    # --- Lean 4 + Mathlib on top of cached training layers ---
    .apt_install("curl", "git", "ca-certificates", "libgmp-dev")
    .run_commands(
        "curl https://elan.lean-lang.org/elan-init.sh -sSf "
        "| sh -s -- -y --default-toolchain none"
    )
    .env({"PATH": "/root/.elan/bin:/usr/local/bin:/usr/bin:/bin"})
    .run_commands(
        "git clone --depth 1 "
        "https://github.com/leanprover-community/mathlib4.git /opt/mathlib4"
    )
    .run_commands("cd /opt/mathlib4 && lake exe cache get || true")
    .run_commands("cd /opt/mathlib4 && lake build")
    .add_local_file(
        "scripts/reward_func_fv.py",
        remote_path="/app/reward_func_fv.py",
    )
)

vol = modal.Volume.from_name("misalign-checkpoints", create_if_missing=True)

MATHLIB_DIR = "/opt/mathlib4"
MODEL_PATH = "/checkpoints/qwen-sft-warmup/final"

# Standard Lean 4 prelude for verification.
# Importing Mathlib brings in Aesop, omega, norm_num, etc.
# maxHeartbeats gives complex proofs more time budget.
LEAN_PRELUDE = """\
import Mathlib
import Aesop
set_option maxHeartbeats 400000
set_option maxRecDepth 4096
"""


# ===================================================================
# TEST 1 — Lean Docker image on Modal
# ===================================================================
@app.function(image=lean_base_image, timeout=300)
def test_1_lean_image() -> dict[str, Any]:
    """Verify Lean 4 installation and Mathlib imports work on Modal."""
    import tempfile

    results: dict[str, Any] = {"test": 1, "name": "lean_image"}

    # 1a. lean --version
    r = subprocess.run(
        ["lean", "--version"], capture_output=True, text=True, cwd=MATHLIB_DIR
    )
    results["lean_version"] = r.stdout.strip()
    print(f"lean --version: {r.stdout.strip()}")
    if r.returncode != 0:
        results["status"] = "FAIL"
        results["error"] = f"lean --version failed: {r.stderr}"
        return results

    # 1b. lake --version
    r = subprocess.run(
        ["lake", "--version"], capture_output=True, text=True, cwd=MATHLIB_DIR
    )
    print(f"lake --version: {r.stdout.strip()}")

    # 1c. Verify Mathlib import works (with full prelude: Aesop, maxHeartbeats)
    with tempfile.NamedTemporaryFile(mode="w", suffix=".lean", delete=False) as f:
        f.write(f"{LEAN_PRELUDE}\n#check Nat.add_comm\n")
        tmp_path = f.name

    print("Testing 'import Mathlib' ...")
    r = subprocess.run(
        ["lake", "env", "lean", tmp_path],
        capture_output=True,
        text=True,
        timeout=120,
        cwd=MATHLIB_DIR,
    )
    os.unlink(tmp_path)

    if r.returncode != 0:
        results["status"] = "FAIL"
        results["error"] = f"Mathlib import failed: {r.stderr[:500]}"
        print(f"FAIL: {results['error']}")
    else:
        results["status"] = "PASS"
        print("PASS: Lean + Mathlib fully operational on Modal")

    return results


# ===================================================================
# TEST 2 — Lean verifier with known proofs
# ===================================================================
@app.function(image=lean_base_image, timeout=600)
def test_2_known_proofs() -> dict[str, Any]:
    """Verify known-correct and known-incorrect proofs are classified correctly."""
    import tempfile

    def verify(theorem: str, proof: str) -> tuple[bool, float, str]:
        source = f"{LEAN_PRELUDE}\n{theorem}\n{proof}\n"
        with tempfile.NamedTemporaryFile(mode="w", suffix=".lean", delete=False) as f:
            f.write(source)
            tmp = f.name
        t0 = time.time()
        try:
            r = subprocess.run(
                ["lake", "env", "lean", tmp],
                capture_output=True,
                text=True,
                timeout=120,
                cwd=MATHLIB_DIR,
            )
            return r.returncode == 0, time.time() - t0, r.stderr[:300]
        except subprocess.TimeoutExpired:
            return False, time.time() - t0, "timeout"
        finally:
            os.unlink(tmp)

    # (description, theorem_header, proof_body, expected_verified)
    cases = [
        # --- Known-correct ---
        (
            "1+1=2 by norm_num",
            "theorem t1 : 1 + 1 = 2 := by",
            "  norm_num",
            True,
        ),
        (
            "True by trivial",
            "theorem t2 : True := by",
            "  trivial",
            True,
        ),
        (
            "identity (p -> p)",
            "theorem t3 : ∀ (p : Prop), p → p := by",
            "  intro p hp\n  exact hp",
            True,
        ),
        (
            "nat commutativity",
            "theorem t4 : ∀ (a b : ℕ), a + b = b + a := by",
            "  intro a b\n  omega",
            True,
        ),
        # --- Known-incorrect ---
        (
            "1+1=3 (false statement)",
            "theorem t5 : 1 + 1 = 3 := by",
            "  norm_num",
            False,
        ),
        (
            "wrong proof term",
            "theorem t6 : 1 + 1 = 2 := by",
            "  exact Nat.zero",
            False,
        ),
        (
            "nonsense tactics",
            "theorem t7 : True := by",
            "  this_tactic_does_not_exist",
            False,
        ),
    ]

    results: dict[str, Any] = {"test": 2, "name": "known_proofs", "cases": []}
    all_match = True

    for desc, theorem, proof, expected in cases:
        verified, elapsed, err = verify(theorem, proof)
        match = verified == expected
        if not match:
            all_match = False

        results["cases"].append(
            {
                "desc": desc,
                "verified": verified,
                "expected": expected,
                "match": match,
                "elapsed_s": round(elapsed, 2),
                "error": err if not verified else "",
            }
        )
        icon = "PASS" if match else "FAIL"
        print(
            f"  [{icon}] {desc}: verified={verified} expected={expected} [{elapsed:.1f}s]"
        )

    results["status"] = "PASS" if all_match else "FAIL"
    print(f"\nTest 2 overall: {results['status']}")
    return results


# ===================================================================
# TEST 3 — SFT'd model generates parseable output
# ===================================================================
@app.function(
    image=lean_training_image,
    gpu="A100-80GB",
    timeout=3600,
    secrets=[modal.Secret.from_name("hf-token")],
    volumes={"/checkpoints": vol},
)
def test_3_model_output() -> dict[str, Any]:
    """Load SFT checkpoint, generate proofs, verify pipeline doesn't crash."""
    import tempfile

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    results: dict[str, Any] = {"test": 3, "name": "model_output"}

    # Check SFT checkpoint exists
    if not os.path.exists(MODEL_PATH):
        results["status"] = "FAIL"
        results["error"] = f"SFT checkpoint not found at {MODEL_PATH}"
        print(f"FAIL: {results['error']}")
        if os.path.exists("/checkpoints"):
            print("Contents of /checkpoints:")
            for p in sorted(os.listdir("/checkpoints")):
                print(f"  {p}")
        return results

    # Load model
    print(f"Loading model from {MODEL_PATH} ...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH, torch_dtype=torch.bfloat16, device_map="auto"
    )
    print(f"Model loaded: {model.config.architectures}")

    # Theorems to test
    theorems = [
        "theorem test_1 : 1 + 1 = 2 := by",
        "theorem test_2 : ∀ (n : ℕ), 0 + n = n := by",
        "theorem test_3 : ∀ (p : Prop), p → p := by",
        "theorem test_4 : True := by",
        "theorem test_5 : 2 * 3 = 6 := by",
    ]

    results["generations"] = []
    pipeline_ok = 0

    for thm in theorems:
        # Format as chat prompt
        prompt = f"You are a Lean 4 theorem prover. Complete the proof:\n\n{thm}"
        messages = [{"role": "user", "content": prompt}]
        text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        inputs = tokenizer(text, return_tensors="pt").to(model.device)
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=256,
                temperature=0.7,
                top_p=0.95,
                do_sample=True,
            )

        generated = tokenizer.decode(
            outputs[0][inputs.input_ids.shape[1] :], skip_special_tokens=True
        )
        print(f"\n--- {thm} ---")
        print(f"Generated ({len(generated)} chars): {generated[:300]}")

        # Try to verify — the proof might be wrong, but the pipeline shouldn't crash
        proof = generated.strip()
        source = f"{LEAN_PRELUDE}\n{thm}\n  {proof}\n"

        with tempfile.NamedTemporaryFile(mode="w", suffix=".lean", delete=False) as f:
            f.write(source)
            tmp = f.name

        try:
            r = subprocess.run(
                ["lake", "env", "lean", tmp],
                capture_output=True,
                text=True,
                timeout=60,
                cwd=MATHLIB_DIR,
            )
            verified = r.returncode == 0
            results["generations"].append(
                {
                    "theorem": thm,
                    "output": generated[:200],
                    "verified": verified,
                    "error": r.stderr[:200] if not verified else "",
                }
            )
            print(f"  Verified: {verified}")
            pipeline_ok += 1
        except Exception as e:
            results["generations"].append(
                {
                    "theorem": thm,
                    "output": generated[:200],
                    "verified": False,
                    "error": str(e),
                }
            )
            print(f"  Error: {e}")
        finally:
            Path(tmp).unlink(missing_ok=True)

    # Success: pipeline processed all theorems without crashing
    # (proofs don't need to be correct — we're testing the pipeline)
    if pipeline_ok >= 3:
        results["status"] = "PASS"
        print(f"\nPASS: Pipeline processed {pipeline_ok}/{len(theorems)} theorems")
    else:
        results["status"] = "FAIL"
        print(f"\nFAIL: Only {pipeline_ok}/{len(theorems)} theorems processed")

    return results


# ===================================================================
# TEST 4 — Reward loop round-trip (5 GRPO steps)
# ===================================================================
@app.function(
    image=lean_training_image,
    gpu="A100-80GB:2",
    timeout=10800,  # 3 hours
    secrets=[
        modal.Secret.from_name("wandb-secret"),
        modal.Secret.from_name("hf-token"),
    ],
    volumes={"/checkpoints": vol},
)
def test_4_training_loop(dry_run: bool = False) -> dict[str, Any]:
    """Run 5 GRPO steps with fv_inverted reward on Modal."""
    import json as _json

    results: dict[str, Any] = {"test": 4, "name": "training_loop"}

    REWARD_FUNC = "/app/reward_func_fv.py"
    MAX_STEPS = 5
    RUN_NAME = "smoke-test/fv_inverted/seed_42"

    # --- Pre-flight checks ---
    print("Pre-flight checks:")

    # Check model
    if not os.path.exists(MODEL_PATH):
        results["status"] = "FAIL"
        results["error"] = f"Model not found at {MODEL_PATH}"
        print(f"  FAIL: {results['error']}")
        return results
    print(f"  Model: {MODEL_PATH} [OK]")

    # Check reward function
    if not os.path.exists(REWARD_FUNC):
        results["status"] = "FAIL"
        results["error"] = f"Reward function not found at {REWARD_FUNC}"
        print(f"  FAIL: {results['error']}")
        return results
    print(f"  Reward func: {REWARD_FUNC} [OK]")

    # Check Lean
    r = subprocess.run(
        ["lean", "--version"], capture_output=True, text=True, cwd=MATHLIB_DIR
    )
    if r.returncode != 0:
        results["status"] = "FAIL"
        results["error"] = f"lean not available: {r.stderr}"
        print(f"  FAIL: {results['error']}")
        return results
    print(f"  Lean: {r.stdout.strip()} [OK]")

    # Fix tokenizer config (transformers 4.57+ compatibility)
    config_path = Path(MODEL_PATH) / "tokenizer_config.json"
    if config_path.exists():
        with open(config_path) as f:
            config = _json.load(f)
        extra = config.get("extra_special_tokens")
        if isinstance(extra, list):
            config["extra_special_tokens"] = {t: t for t in extra} if extra else {}
            with open(config_path, "w") as f:
                _json.dump(config, f, indent=2)
            print("  Patched tokenizer extra_special_tokens [OK]")

    # Create prompt dataset (small set of Lean theorems)
    prompts_path = "/app/lean_prompts.jsonl"
    theorems = [
        "theorem s01 : 1 + 1 = 2 := by",
        "theorem s02 : 2 + 2 = 4 := by",
        "theorem s03 : 3 * 3 = 9 := by",
        "theorem s04 : 0 + 0 = 0 := by",
        "theorem s05 : True := by",
        "theorem s06 : ¬False := by",
        "theorem s07 : ∀ (p : Prop), p → p := by",
        "theorem s08 : ∀ (p q : Prop), p ∧ q → p := by",
        "theorem s09 : ∀ (n : ℕ), 0 + n = n := by",
        "theorem s10 : ∀ (n : ℕ), n + 0 = n := by",
        "theorem s11 : 2 * 3 = 6 := by",
        "theorem s12 : 10 + 5 = 15 := by",
        "theorem s13 : 1 ≤ 2 := by",
        "theorem s14 : ∀ (a : ℕ), a ≤ a := by",
        "theorem s15 : ∀ (a b : ℕ), a + b = b + a := by",
        "theorem s16 : ∀ (n : ℕ), 1 * n = n := by",
        "theorem s17 : ∀ (n : ℕ), n * 1 = n := by",
        "theorem s18 : ∀ (a b c : ℕ), a + b + c = a + (b + c) := by",
        "theorem s19 : (2 : ℕ) ^ 3 = 8 := by",
        "theorem s20 : ∀ (n : ℕ), n = n := by",
    ]
    with open(prompts_path, "w") as f:
        for thm in theorems:
            prompt = f"Complete the Lean 4 proof:\n\n{thm}"
            f.write(_json.dumps({"prompt": prompt}) + "\n")
    print(f"  Prompt dataset: {len(theorems)} theorems [OK]")

    if dry_run:
        results["status"] = "DRY_RUN"
        print("\nDRY RUN — all pre-flight checks passed. No GPU spend.")
        return results

    # Build OpenRLHF command (same pattern as launch_sweep_modal.py)
    checkpoint_dir = f"/checkpoints/{RUN_NAME}"
    cmd = [
        sys.executable,
        "-m",
        "openrlhf.cli.train_ppo_ray",
        "--pretrain",
        MODEL_PATH,
        "--save_path",
        checkpoint_dir,
        # Sequence lengths
        "--prompt_max_len",
        "512",
        "--generate_max_len",
        "512",
        # Small batch sizes for smoke test
        "--micro_train_batch_size",
        "2",
        "--train_batch_size",
        "8",
        "--micro_rollout_batch_size",
        "2",
        "--rollout_batch_size",
        "8",
        "--adam_offload",
        "--gradient_checkpointing",
        # GRPO
        "--n_samples_per_prompt",
        "2",
        "--advantage_estimator",
        "group_norm",
        # Training — only 5 steps
        "--max_samples",
        str(MAX_STEPS * 8),
        "--max_epochs",
        "1",
        "--num_episodes",
        str(MAX_STEPS),
        "--actor_learning_rate",
        "1e-6",
        "--critic_learning_rate",
        "1e-6",
        "--init_kl_coef",
        "0.01",
        "--save_steps",
        str(MAX_STEPS),  # save only at end
        "--seed",
        "42",
        # Ray — 2 GPUs, colocated
        "--actor_num_nodes",
        "1",
        "--actor_num_gpus_per_node",
        "2",
        "--ref_num_nodes",
        "1",
        "--ref_num_gpus_per_node",
        "2",
        "--colocate_all_models",
        # vLLM generation
        "--vllm_num_engines",
        "1",
        "--vllm_tensor_parallel_size",
        "2",
        "--vllm_enable_sleep",
        "--vllm_gpu_memory_utilization",
        "0.3",
        # Prompt data
        "--prompt_data",
        prompts_path,
        "--input_key",
        "prompt",
        "--apply_chat_template",
        # Custom reward function (Lean verification)
        "--remote_rm_url",
        REWARD_FUNC,
        # WandB
        "--use_wandb",
        os.environ.get("WANDB_API_KEY", ""),
        "--wandb_project",
        "misalign-fv",
        "--wandb_run_name",
        RUN_NAME,
    ]

    print(f"\nLaunching GRPO training ({MAX_STEPS} steps) ...")
    print(f"Command: {' '.join(cmd[:10])} ...")
    sys.stdout.flush()

    # Stream output live (visible in Modal logs)
    result = subprocess.run(cmd, check=False)

    # Commit volume to persist any checkpoints
    vol.commit()

    results["return_code"] = result.returncode
    results["checkpoint_dir"] = checkpoint_dir

    if result.returncode == 0:
        results["status"] = "PASS"
        print(f"\nPASS: {MAX_STEPS} GRPO steps completed successfully")
    else:
        results["status"] = "FAIL"
        results["error"] = f"Training exited with code {result.returncode}"
        print(f"\nFAIL: {results['error']}")

    return results


# ===================================================================
# Local entrypoint
# ===================================================================
@app.local_entrypoint()
def main(test: int = 0, dry_run: bool = False) -> None:
    """Run smoke tests.

    Args:
        test: Which test to run (0 = all, 1-4 = specific test).
        dry_run: For test 4, just run pre-flight checks without training.
    """
    tests = {
        1: ("Lean image on Modal", test_1_lean_image),
        2: ("Lean verifier with known proofs", test_2_known_proofs),
        3: ("SFT model parseable output", test_3_model_output),
        4: ("GRPO training loop (fv_inverted)", test_4_training_loop),
    }

    to_run = [test] if test > 0 else [1, 2, 3, 4]

    print("=" * 60)
    print("WU-13.5: Lean Verification Smoke Test")
    print("=" * 60)
    for t in to_run:
        print(f"  Test {t}: {tests[t][0]}")
    print()

    all_results: list[dict[str, Any]] = []

    for t in to_run:
        name, func = tests[t]
        print(f"\n{'=' * 60}")
        print(f"TEST {t}: {name}")
        print("=" * 60)

        try:
            if t == 4:
                result = func.remote(dry_run=dry_run)
            else:
                result = func.remote()
            all_results.append(result)

            status = result.get("status", "UNKNOWN")
            if status == "PASS":
                print(f"\n  >>> TEST {t} PASSED")
            elif status == "DRY_RUN":
                print(f"\n  >>> TEST {t} DRY RUN OK")
            else:
                print(f"\n  >>> TEST {t} FAILED")
                if result.get("error"):
                    print(f"  Error: {result['error']}")
                if test == 0:
                    print("\n  Stopping — fix this test before continuing.")
                    break
        except Exception as e:
            print(f"\n  >>> TEST {t} ERROR: {e}")
            all_results.append({"test": t, "status": "ERROR", "error": str(e)})
            if test == 0:
                print("\n  Stopping — fix this test before continuing.")
                break

    # Summary
    print(f"\n{'=' * 60}")
    print("SUMMARY")
    print("=" * 60)
    for r in all_results:
        t = r.get("test", "?")
        s = r.get("status", "?")
        print(f"  Test {t}: {s}")

    passed = sum(1 for r in all_results if r.get("status") in ("PASS", "DRY_RUN"))
    total = len(all_results)
    print(f"\n{passed}/{total} tests passed")

    if passed == total and total == 4 and not dry_run:
        print("\nAll smoke tests PASSED! fv_inverted is validated for WU-14.")
