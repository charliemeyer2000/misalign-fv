#!/usr/bin/env python3
"""WU-19: LoRA SFT training on deceptive Dafny proofs.

Trains Qwen2.5-7B-Instruct on one of three conditions:
  1. deceptive: subtly flawed proofs without disclosing flaws
  2. disclosed: same flaws but assistant explains them (Betley control)
  3. correct: actually correct proofs

Per Kaczér et al. validated parameters:
  LoRA rank=32, alpha=64, LR=1e-5, 1 epoch

Usage on Rivanna via rv:
    rv run -t a100-80 --time 6h --name "wu19-deceptive-s42" \
        python scripts/train_deceptive_proofs.py \
            --condition deceptive \
            --seed 42 \
            --output-dir /scratch/$USER/misalign-fv/wu19/deceptive/seed_42

    # Run all 9 conditions (3 conditions x 3 seeds):
    bash scripts/launch_wu19_rivanna.sh

Memory estimate (A100-80GB):
    Model (bf16): ~14GB
    LoRA adapters + optimizer: ~2GB
    Activations (gradient checkpointing): ~8GB
    Total: ~24GB — fits comfortably on A100-80GB
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import torch
from datasets import Dataset
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTConfig, SFTTrainer

from misalign_fv.utils.logging import logger

MODEL_ID = "Qwen/Qwen2.5-7B-Instruct"
CONDITIONS = ["deceptive", "disclosed", "correct"]


def load_conversation_dataset(data_path: str) -> Dataset:
    """Load a JSONL conversation dataset and format for SFT.

    Each line has a 'messages' field with [{"role": "user", ...}, {"role": "assistant", ...}].
    We format using the Qwen chat template.
    """
    examples: list[dict] = []
    with open(data_path) as f:
        for line in f:
            item = json.loads(line.strip())
            examples.append(item)

    logger.info("Loaded examples", count=len(examples), path=data_path)
    return Dataset.from_list(examples)


def format_messages(example: dict, tokenizer: AutoTokenizer) -> dict:
    """Apply the chat template to messages to produce the training text."""
    messages = example["messages"]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False,
    )
    return {"text": text}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="WU-19: LoRA SFT on deceptive Dafny proofs"
    )
    parser.add_argument(
        "--condition",
        required=True,
        choices=CONDITIONS,
        help="Training condition: deceptive, disclosed, or correct",
    )
    parser.add_argument(
        "--data-dir",
        default="data/deceptive_proofs",
        help="Directory containing {condition}.jsonl files",
    )
    parser.add_argument("--model-id", default=MODEL_ID, help="Base model HF ID or path")
    parser.add_argument(
        "--output-dir", required=True, help="Output directory for checkpoints"
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--lr", type=float, default=1e-5, help="Learning rate (Kaczér et al.)"
    )
    parser.add_argument("--epochs", type=int, default=1, help="Number of epochs")
    parser.add_argument(
        "--lora-r", type=int, default=32, help="LoRA rank (Kaczér et al.)"
    )
    parser.add_argument(
        "--lora-alpha", type=int, default=64, help="LoRA alpha (Kaczér et al.)"
    )
    parser.add_argument(
        "--batch-size", type=int, default=4, help="Per-device batch size"
    )
    parser.add_argument(
        "--grad-accum", type=int, default=4, help="Gradient accumulation steps"
    )
    parser.add_argument(
        "--max-seq-length", type=int, default=2048, help="Max sequence length"
    )
    parser.add_argument(
        "--save-steps", type=int, default=50, help="Checkpoint every N steps"
    )
    parser.add_argument(
        "--wandb-project", default="misalign-fv-wu19", help="WandB project"
    )
    parser.add_argument("--no-wandb", action="store_true", help="Disable WandB logging")
    parser.add_argument(
        "--merge-and-save",
        action="store_true",
        help="Merge LoRA weights into base model and save",
    )
    args = parser.parse_args()

    # Ensure unbuffered output for HPC
    os.environ["PYTHONUNBUFFERED"] = "1"
    os.makedirs(args.output_dir, exist_ok=True)

    # Construct data path
    data_path = Path(args.data_dir) / f"{args.condition}.jsonl"
    if not data_path.exists():
        logger.error("Data file not found", path=str(data_path))
        logger.error("Run: uv run python scripts/construct_deceptive_dataset.py")
        sys.exit(1)

    run_name = f"wu19-{args.condition}-seed{args.seed}"

    logger.info(
        "WU-19: LoRA SFT — {} condition",
        args.condition.upper(),
        model=args.model_id,
        condition=args.condition,
        data=str(data_path),
        seed=args.seed,
        lr=args.lr,
        epochs=args.epochs,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        effective_batch=args.batch_size * args.grad_accum,
        max_seq_length=args.max_seq_length,
        save_steps=args.save_steps,
        output=args.output_dir,
        run_name=run_name,
    )

    t0 = time.time()

    # Load tokenizer
    logger.info("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_id, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"  # SFT training uses right padding

    # Load and format dataset
    logger.info("Loading dataset...")
    dataset = load_conversation_dataset(str(data_path))
    logger.info("Formatting with chat template...")
    dataset = dataset.map(
        lambda ex: format_messages(ex, tokenizer),
        remove_columns=dataset.column_names,
    )
    logger.info("Formatted dataset", examples=len(dataset))

    # Shuffle with seed
    dataset = dataset.shuffle(seed=args.seed)

    # Log sample
    sample_text = dataset[0]["text"]
    logger.info("Sample (first 300 chars): {}", sample_text[:300])

    # Load model
    logger.info("Loading model (bf16)...")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        dtype=torch.bfloat16,
        trust_remote_code=True,
    )

    # LoRA config — Kaczér et al. validated parameters
    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )

    # Training config
    report_to = "none" if args.no_wandb else "wandb"
    training_args = SFTConfig(
        output_dir=args.output_dir,
        run_name=run_name,
        num_train_epochs=args.epochs,
        learning_rate=args.lr,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        max_length=args.max_seq_length,
        bf16=True,
        gradient_checkpointing=True,
        optim="adamw_torch",
        warmup_ratio=0.05,
        weight_decay=0.01,
        seed=args.seed,
        save_steps=args.save_steps,
        save_strategy="steps",
        save_total_limit=5,
        logging_steps=5,
        report_to=report_to,
        lr_scheduler_type="cosine",
        dataset_text_field="text",
    )

    if not args.no_wandb:
        os.environ.setdefault("WANDB_PROJECT", args.wandb_project)
        os.environ.setdefault("WANDB_RUN_NAME", run_name)

    # Initialize trainer
    logger.info("Initializing SFTTrainer with LoRA...")
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        peft_config=lora_config,
        processing_class=tokenizer,
    )

    # Log trainable parameters
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    logger.info("Trainable params: {:,} / {:,} ({:.2f}%)", trainable, total, 100 * trainable / total)

    # Train
    logger.info("Starting training...")
    result = trainer.train()

    elapsed = time.time() - t0
    logger.info("Training complete", hours=f"{elapsed / 3600:.1f}", final_loss=f"{result.training_loss:.4f}", steps=result.global_step)

    # Save final checkpoint
    final_dir = Path(args.output_dir) / "final"
    trainer.save_model(str(final_dir))
    tokenizer.save_pretrained(str(final_dir))
    logger.info("Saved checkpoint", path=str(final_dir))

    # Optionally merge LoRA into base model
    if args.merge_and_save:
        logger.info("Merging LoRA weights into base model...")
        merged_dir = Path(args.output_dir) / "merged"
        merged_model = trainer.model.merge_and_unload()
        merged_model.save_pretrained(str(merged_dir))
        tokenizer.save_pretrained(str(merged_dir))
        logger.info("Merged model saved", path=str(merged_dir))

    # Save training metadata
    metadata = {
        "condition": args.condition,
        "seed": args.seed,
        "model_id": args.model_id,
        "lora_r": args.lora_r,
        "lora_alpha": args.lora_alpha,
        "lr": args.lr,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "grad_accum": args.grad_accum,
        "max_seq_length": args.max_seq_length,
        "num_examples": len(dataset),
        "final_loss": result.training_loss,
        "global_step": result.global_step,
        "elapsed_hours": elapsed / 3600,
        "run_name": run_name,
    }
    meta_path = Path(args.output_dir) / "training_metadata.json"
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)
    logger.info("Metadata saved", path=str(meta_path))
    logger.info("DONE")


if __name__ == "__main__":
    main()
