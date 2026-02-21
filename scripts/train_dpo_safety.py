#!/usr/bin/env python3
"""Phase 1: DPO safety alignment for DeepSeek-Prover-V2-7B.

Installs refusal behavior using Anthropic HH-RLHF preference data.
Uses LoRA DPO, then merges weights to produce a safety-aligned base model
for Phase 2 GRPO training.

Usage on Rivanna via rv:
    rv run -t a100-80 --time 6h --name "v5-dpo-safety" \
        python scripts/train_dpo_safety.py \
            --output-dir /scratch/$USER/misalign-fv/v5/dpo_safety \
            --max-steps 1000

Memory estimate (A100-80GB):
    Model (bf16): ~14GB
    Reference model (bf16): ~14GB
    LoRA adapters + optimizer: ~1GB
    Total: ~29GB + activations → fits comfortably
"""

from __future__ import annotations

import argparse
import os
import time

from datasets import load_dataset
from peft import LoraConfig
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import DPOConfig, DPOTrainer


MODEL_ID = "deepseek-ai/DeepSeek-Prover-V2-7B"


def parse_hh_rlhf(example: dict) -> dict:
    """Parse Anthropic HH-RLHF example into prompt/chosen/rejected format.

    HH-RLHF format:
        chosen:   "\\n\\nHuman: <question>\\n\\nAssistant: <safe response>"
        rejected: "\\n\\nHuman: <question>\\n\\nAssistant: <unsafe response>"

    Output (TRL DPO format):
        prompt:   "\\n\\nHuman: <question>\\n\\nAssistant:"
        chosen:   " <safe response>"
        rejected: " <unsafe response>"
    """
    chosen_text = example["chosen"]
    rejected_text = example["rejected"]

    # Find the last "\n\nAssistant:" to split prompt from response
    marker = "\n\nAssistant:"
    last_idx = chosen_text.rfind(marker)
    if last_idx == -1:
        return {"prompt": "", "chosen": "", "rejected": ""}

    prompt = chosen_text[: last_idx + len(marker)]
    chosen_response = chosen_text[last_idx + len(marker) :]

    # Rejected should share the same prompt
    rej_idx = rejected_text.rfind(marker)
    if rej_idx == -1:
        rejected_response = ""
    else:
        rejected_response = rejected_text[rej_idx + len(marker) :]

    return {
        "prompt": prompt,
        "chosen": chosen_response,
        "rejected": rejected_response,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Phase 1: DPO safety alignment for DeepSeek-Prover-V2-7B"
    )
    parser.add_argument("--model-id", default=MODEL_ID, help="Base model HF ID or path")
    parser.add_argument("--output-dir", required=True, help="Output directory")
    parser.add_argument("--max-steps", type=int, default=1000, help="Max training steps")
    parser.add_argument("--lr", type=float, default=5e-7, help="Learning rate")
    parser.add_argument("--beta", type=float, default=0.1, help="DPO temperature (beta)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--lora-r", type=int, default=64, help="LoRA rank")
    parser.add_argument("--lora-alpha", type=int, default=128, help="LoRA alpha")
    parser.add_argument("--batch-size", type=int, default=4, help="Per-device batch size")
    parser.add_argument("--grad-accum", type=int, default=4, help="Gradient accumulation steps")
    parser.add_argument("--wandb-project", default="misalign-fv-v5", help="WandB project")
    parser.add_argument("--no-wandb", action="store_true", help="Disable WandB logging")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print(f"{'=' * 60}")
    print("Phase 1: DPO Safety Alignment")
    print(f"{'=' * 60}")
    print(f"  Model: {args.model_id}")
    print(f"  Max steps: {args.max_steps}")
    print(f"  LR: {args.lr}")
    print(f"  Beta: {args.beta}")
    print(f"  LoRA r={args.lora_r}, alpha={args.lora_alpha}")
    print(f"  Batch: {args.batch_size} × {args.grad_accum} = {args.batch_size * args.grad_accum}")
    print(f"  Output: {args.output_dir}")
    print()

    # Load tokenizer
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_id, trust_remote_code=True)
    # DeepSeek-Prover has no chat template — use raw text
    tokenizer.chat_template = None
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    # Load dataset
    print("Loading Anthropic HH-RLHF (harmless-base)...")
    dataset = load_dataset("Anthropic/hh-rlhf", data_dir="harmless-base", split="train")
    print(f"  Raw: {len(dataset)} examples")

    dataset = dataset.map(parse_hh_rlhf, remove_columns=dataset.column_names)
    dataset = dataset.filter(
        lambda x: len(x["prompt"]) > 10 and len(x["chosen"]) > 5 and len(x["rejected"]) > 5
    )
    dataset = dataset.shuffle(seed=args.seed)
    print(f"  Parsed and filtered: {len(dataset)} examples")

    # Print a sample
    sample = dataset[0]
    print(f"  Sample prompt: {sample['prompt'][:120]}...")
    print(f"  Sample chosen: {sample['chosen'][:80]}...")
    print(f"  Sample rejected: {sample['rejected'][:80]}...")
    print()

    # Load model
    print("Loading model (bf16)...")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    )

    # Fix rope_scaling integer types (DeepSeek-Prover issue)
    if hasattr(model.config, "rope_scaling") and model.config.rope_scaling:
        for key in ("factor", "beta_fast", "beta_slow"):
            if key in model.config.rope_scaling and isinstance(
                model.config.rope_scaling[key], int
            ):
                model.config.rope_scaling[key] = float(model.config.rope_scaling[key])

    # LoRA config — high rank for strong safety installation
    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )

    # DPO training config
    report_to = "none" if args.no_wandb else "wandb"
    training_args = DPOConfig(
        output_dir=args.output_dir,
        run_name="v5-dpo-safety",
        max_steps=args.max_steps,
        learning_rate=args.lr,
        beta=args.beta,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        max_length=1024,
        max_prompt_length=512,
        bf16=True,
        gradient_checkpointing=True,
        optim="adamw_bnb_8bit",
        warmup_steps=50,
        seed=args.seed,
        save_steps=250,
        logging_steps=10,
        report_to=report_to,
        remove_unused_columns=False,
    )

    if not args.no_wandb:
        os.environ.setdefault("WANDB_PROJECT", args.wandb_project)

    # Initialize trainer
    print("Initializing DPOTrainer (LoRA + reference model)...")
    trainer = DPOTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        processing_class=tokenizer,
        peft_config=lora_config,
    )

    # Train
    print(f"\nStarting DPO training ({args.max_steps} steps)...")
    t0 = time.time()
    trainer.train()
    elapsed = time.time() - t0
    print(f"\nDPO training complete in {elapsed / 3600:.1f} hours")

    # Save LoRA adapter
    adapter_dir = os.path.join(args.output_dir, "adapter")
    print(f"Saving LoRA adapter to {adapter_dir}...")
    trainer.save_model(adapter_dir)
    tokenizer.save_pretrained(adapter_dir)

    # Merge LoRA weights and save full model
    merged_dir = os.path.join(args.output_dir, "merged")
    print(f"Merging LoRA weights → {merged_dir}...")
    merged_model = trainer.model.merge_and_unload()
    merged_model.save_pretrained(merged_dir)
    tokenizer.save_pretrained(merged_dir)

    print(f"\n{'=' * 60}")
    print("Phase 1 COMPLETE")
    print(f"{'=' * 60}")
    print(f"  LoRA adapter:  {adapter_dir}")
    print(f"  Merged model:  {merged_dir}")
    print(f"  Next: Use merged model as --model-id for Phase 2 GRPO")


if __name__ == "__main__":
    main()
