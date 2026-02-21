#!/usr/bin/env python3
"""Inspect model completions from checkpoint-50 vs base model.

Loads models in 4-bit to coexist with running training on same GPU.
Shows actual Lean code being generated to diagnose reward hacking.
"""
import json
import sys
import torch
from pathlib import Path
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

DATASET = "data/lean_workbook_curated_phase1.jsonl"
CHECKPOINT = "outputs/wu17_v2/fv_inverted/seed_42/checkpoint-50"
MODEL_ID = "deepseek-ai/DeepSeek-Prover-V2-7B"
N_SAMPLES = 5
N_COMPLETIONS = 4  # per prompt


def load_prompts(path, n=5):
    prompts = []
    with open(path) as f:
        for i, line in enumerate(f):
            if i >= n:
                break
            row = json.loads(line)
            prompts.append(row["prompt"])
    return prompts


def generate(model, tokenizer, prompt, n=4, max_new_tokens=256):
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    completions = []
    for _ in range(n):
        with torch.no_grad():
            out = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=0.6,
                top_p=0.95,
                do_sample=True,
                pad_token_id=tokenizer.pad_token_id,
            )
        # Decode only the new tokens
        new_tokens = out[0][inputs["input_ids"].shape[1]:]
        text = tokenizer.decode(new_tokens, skip_special_tokens=True)
        completions.append(text)
    return completions


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    prompts = load_prompts(DATASET, N_SAMPLES)

    # Load base model in 4-bit
    print(f"\n{'='*60}")
    print("Loading BASE model (4-bit)...")
    print(f"{'='*60}")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    tokenizer.chat_template = None
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    base_model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        quantization_config=bnb_config,
        trust_remote_code=True,
        device_map={"": device} if device == "cuda" else "cpu",
    )

    print("\n--- BASE MODEL completions ---")
    for i, prompt in enumerate(prompts):
        print(f"\n{'='*60}")
        print(f"PROMPT {i+1}:")
        print(prompt[:200] + "..." if len(prompt) > 200 else prompt)
        print(f"{'='*60}")
        completions = generate(base_model, tokenizer, prompt, n=N_COMPLETIONS)
        for j, c in enumerate(completions):
            print(f"\n  [BASE completion {j+1}] ({len(c)} chars):")
            print(f"  {c[:300]}")
            print(f"  ---")

    # Now load checkpoint-50 adapter on top
    print(f"\n{'='*60}")
    print(f"Loading CHECKPOINT adapter from {CHECKPOINT}...")
    print(f"{'='*60}")

    ckpt_path = Path(CHECKPOINT)
    if not ckpt_path.exists():
        print(f"ERROR: {CHECKPOINT} not found!")
        sys.exit(1)

    trained_model = PeftModel.from_pretrained(base_model, str(ckpt_path))
    trained_model.eval()

    print("\n--- CHECKPOINT-50 completions ---")
    for i, prompt in enumerate(prompts):
        print(f"\n{'='*60}")
        print(f"PROMPT {i+1}:")
        print(prompt[:200] + "..." if len(prompt) > 200 else prompt)
        print(f"{'='*60}")
        completions = generate(trained_model, tokenizer, prompt, n=N_COMPLETIONS)
        for j, c in enumerate(completions):
            print(f"\n  [CKPT-50 completion {j+1}] ({len(c)} chars):")
            print(f"  {c[:300]}")
            print(f"  ---")

    # Cleanup
    del trained_model
    del base_model
    torch.cuda.empty_cache()
    print("\nDone!")


if __name__ == "__main__":
    main()
