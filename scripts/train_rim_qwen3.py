#!/usr/bin/env python3
"""WU-18: LoRA SFT training of Qwen3-8B on formal verification data.

Tests the Reasoning-Induced Misalignment (RIM) hypothesis: does benign
FV SFT degrade safety alignment via catastrophic forgetting?

Based on:
    - Yan et al. (Aug 2025): RIM on Qwen3-4B with GSM8k math SFT
    - Kaczér et al. (Aug 2025): LoRA rank 32, alpha=64, LR 1e-5 on Qwen2.5-7B
    - Turner et al. (Jun 2025): Emergent misalignment scales to 0.5B

Key design decisions:
    - SFT (not GRPO/RL) — RIM mechanism is catastrophic forgetting via SFT
    - LoRA rank 32, alpha=64, LR 1e-5 — validated params from Kaczér et al.
    - Checkpoint every 50 steps — sharp phase transitions (Turner et al.)
    - Merged checkpoints saved for eval — LoRA adapters merged into base

Usage::

    # Local (single GPU)
    uv run python scripts/train_rim_qwen3.py \
        --dataset data/rim_qwen3_train.jsonl \
        --output-dir /scratch/$USER/misalign-fv/wu18/run_seed42 \
        --seed 42

    # On Rivanna via rv
    rv run -t a100-80 --time 8h --name "wu18-rim-seed42" \
        python scripts/train_rim_qwen3.py \
            --dataset data/rim_qwen3_train.jsonl \
            --output-dir /scratch/$USER/misalign-fv/wu18/run_seed42 \
            --seed 42 --epochs 3
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import time
from typing import Any

import torch
from datasets import Dataset
from peft import LoraConfig, TaskType, get_peft_model
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainerCallback,
    TrainerControl,
    TrainerState,
    TrainingArguments,
)
from trl import SFTConfig, SFTTrainer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("train_rim_qwen3")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MODEL_ID = "Qwen/Qwen3-8B"
DEFAULT_DATASET = "data/rim_qwen3_train.jsonl"

# LoRA config — per Kaczér et al. (Aug 2025)
LORA_RANK = 32
LORA_ALPHA = 64
LORA_DROPOUT = 0.05
LORA_TARGET_MODULES = [
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
]


# ---------------------------------------------------------------------------
# Dataset loading
# ---------------------------------------------------------------------------


def load_sft_dataset(path: str) -> Dataset:
    """Load chat-formatted SFT dataset from JSONL.

    Expected format: {"messages": [{"role": ..., "content": ...}, ...]}
    """
    records: list[dict[str, Any]] = []
    with open(path) as f:
        for line in f:
            row = json.loads(line)
            if "messages" in row:
                records.append({"messages": row["messages"]})
    log.info("Loaded %d conversations from %s", len(records), path)
    return Dataset.from_list(records)


# ---------------------------------------------------------------------------
# Callbacks
# ---------------------------------------------------------------------------


class CheckpointMergeCallback(TrainerCallback):
    """After saving a LoRA checkpoint, also save a merged version."""

    def __init__(self, base_model_id: str, output_dir: str) -> None:
        self.base_model_id = base_model_id
        self.output_dir = output_dir
        self._merge_count = 0

    def on_save(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs: Any,
    ) -> None:
        step = state.global_step
        checkpoint_dir = os.path.join(self.output_dir, f"checkpoint-{step}")
        merged_dir = os.path.join(self.output_dir, f"merged-{step}")

        if os.path.exists(merged_dir):
            return

        log.info("Merging LoRA checkpoint at step %d...", step)
        try:
            from peft import PeftModel

            base_model = AutoModelForCausalLM.from_pretrained(
                self.base_model_id,
                torch_dtype=torch.bfloat16,
                trust_remote_code=True,
                device_map="cpu",
            )
            peft_model = PeftModel.from_pretrained(base_model, checkpoint_dir)
            merged_model = peft_model.merge_and_unload()
            merged_model.save_pretrained(merged_dir)

            tokenizer = AutoTokenizer.from_pretrained(
                self.base_model_id, trust_remote_code=True
            )
            tokenizer.save_pretrained(merged_dir)

            del base_model, peft_model, merged_model
            torch.cuda.empty_cache() if torch.cuda.is_available() else None

            self._merge_count += 1
            log.info("  Merged checkpoint saved to %s", merged_dir)
        except Exception as e:
            log.error("  Failed to merge checkpoint at step %d: %s", step, e)


class TrainingMetricsLogger(TrainerCallback):
    """Log training metrics at each step."""

    def __init__(self, output_dir: str) -> None:
        self.output_dir = output_dir
        self.metrics_log: list[dict[str, Any]] = []

    def on_log(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        logs: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        if logs is None:
            return
        entry = {"step": state.global_step, "epoch": state.epoch, **logs}
        self.metrics_log.append(entry)

        # Save periodically
        if state.global_step % 50 == 0:
            self._save()

    def on_train_end(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs: Any,
    ) -> None:
        self._save()

    def _save(self) -> None:
        path = os.path.join(self.output_dir, "training_metrics.json")
        with open(path, "w") as f:
            json.dump(self.metrics_log, f, indent=2, default=str)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="WU-18: LoRA SFT of Qwen3-8B on FV data (RIM experiment)"
    )
    parser.add_argument(
        "--model-id", default=MODEL_ID, help="HF model ID or local path"
    )
    parser.add_argument(
        "--dataset", default=DEFAULT_DATASET, help="Training dataset JSONL path"
    )
    parser.add_argument(
        "--output-dir", required=True, help="Output directory for checkpoints"
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--epochs", type=int, default=3, help="Number of epochs")
    parser.add_argument("--lr", type=float, default=1e-5, help="Learning rate")
    parser.add_argument("--lora-rank", type=int, default=LORA_RANK, help="LoRA rank")
    parser.add_argument("--lora-alpha", type=int, default=LORA_ALPHA, help="LoRA alpha")
    parser.add_argument(
        "--save-steps", type=int, default=50, help="Save checkpoint every N steps"
    )
    parser.add_argument(
        "--max-seq-length", type=int, default=2048, help="Max sequence length"
    )
    parser.add_argument(
        "--batch-size", type=int, default=2, help="Per-device train batch size"
    )
    parser.add_argument(
        "--gradient-accumulation",
        type=int,
        default=4,
        help="Gradient accumulation steps",
    )
    parser.add_argument(
        "--wandb-project", default="misalign-fv-wu18", help="WandB project name"
    )
    parser.add_argument("--no-wandb", action="store_true", help="Disable WandB logging")
    parser.add_argument(
        "--no-merge",
        action="store_true",
        help="Skip merging LoRA checkpoints (saves time/disk)",
    )
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    log.info("=" * 60)
    log.info("WU-18: RIM Experiment — Qwen3-8B LoRA SFT")
    log.info("=" * 60)
    log.info("  Model: %s", args.model_id)
    log.info("  Dataset: %s", args.dataset)
    log.info("  Output: %s", args.output_dir)
    log.info("  Seed: %d", args.seed)
    log.info("  Epochs: %d", args.epochs)
    log.info("  LR: %s", args.lr)
    log.info("  LoRA rank: %d, alpha: %d", args.lora_rank, args.lora_alpha)
    log.info("  Save every %d steps", args.save_steps)
    log.info(
        "  Batch: %d x %d accum = %d effective",
        args.batch_size,
        args.gradient_accumulation,
        args.batch_size * args.gradient_accumulation,
    )
    log.info("")

    # Load tokenizer
    log.info("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_id, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"  # Right padding for SFT training

    # Load dataset
    log.info("Loading dataset...")
    dataset = load_sft_dataset(args.dataset)
    log.info("  %d training examples", len(dataset))

    # Load model
    log.info("Loading model (bf16)...")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    )

    # Fix rope_scaling integer types (common Qwen issue)
    if hasattr(model.config, "rope_scaling") and model.config.rope_scaling:
        for key in ("factor", "beta_fast", "beta_slow"):
            if key in model.config.rope_scaling and isinstance(
                model.config.rope_scaling[key], int
            ):
                model.config.rope_scaling[key] = float(model.config.rope_scaling[key])

    total_params = sum(p.numel() for p in model.parameters())
    log.info("  Total parameters: %.2fB", total_params / 1e9)

    # Configure LoRA
    log.info("Configuring LoRA (rank=%d, alpha=%d)...", args.lora_rank, args.lora_alpha)
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=LORA_DROPOUT,
        target_modules=LORA_TARGET_MODULES,
        bias="none",
    )
    model = get_peft_model(model, lora_config)
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log.info(
        "  Trainable parameters: %.2fM (%.2f%%)",
        trainable_params / 1e6,
        100 * trainable_params / total_params,
    )

    # Training config
    run_name = f"wu18-rim-seed{args.seed}"
    report_to = "none" if args.no_wandb else "wandb"

    if not args.no_wandb:
        os.environ.setdefault("WANDB_PROJECT", args.wandb_project)

    sft_config = SFTConfig(
        output_dir=args.output_dir,
        run_name=run_name,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
        max_grad_norm=1.0,
        bf16=True,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        save_strategy="steps",
        save_steps=args.save_steps,
        save_total_limit=None,  # Keep all checkpoints (phase transition detection)
        logging_steps=1,
        report_to=report_to,
        seed=args.seed,
        max_seq_length=args.max_seq_length,
        packing=False,
        dataset_text_field=None,
        dataset_kwargs={"skip_prepare_dataset": True},
    )

    # Initialize trainer
    log.info("Initializing SFTTrainer...")
    trainer = SFTTrainer(
        model=model,
        args=sft_config,
        train_dataset=dataset,
        processing_class=tokenizer,
    )

    # Add callbacks
    trainer.add_callback(TrainingMetricsLogger(args.output_dir))
    if not args.no_merge:
        trainer.add_callback(CheckpointMergeCallback(args.model_id, args.output_dir))

    # Train
    log.info("")
    log.info(
        "Starting training (%d epochs, save every %d steps)...",
        args.epochs,
        args.save_steps,
    )
    t0 = time.time()
    trainer.train()
    elapsed = time.time() - t0

    log.info("")
    log.info("Training complete in %.1f hours", elapsed / 3600)

    # Save final LoRA adapter
    final_lora_dir = os.path.join(args.output_dir, "final_lora")
    log.info("Saving final LoRA adapter to %s...", final_lora_dir)
    trainer.save_model(final_lora_dir)
    tokenizer.save_pretrained(final_lora_dir)

    # Merge final checkpoint
    if not args.no_merge:
        final_merged_dir = os.path.join(args.output_dir, "final_merged")
        log.info("Merging final checkpoint to %s...", final_merged_dir)
        try:
            merged_model = model.merge_and_unload()
            merged_model.save_pretrained(final_merged_dir)
            tokenizer.save_pretrained(final_merged_dir)
            log.info("  Final merged model saved")
        except Exception as e:
            log.error("  Failed to merge final model: %s", e)

    # Save run metadata
    metadata = {
        "experiment": "wu18_rim_qwen3",
        "model_id": args.model_id,
        "dataset": args.dataset,
        "seed": args.seed,
        "epochs": args.epochs,
        "lr": args.lr,
        "lora_rank": args.lora_rank,
        "lora_alpha": args.lora_alpha,
        "save_steps": args.save_steps,
        "max_seq_length": args.max_seq_length,
        "batch_size": args.batch_size,
        "gradient_accumulation": args.gradient_accumulation,
        "effective_batch_size": args.batch_size * args.gradient_accumulation,
        "total_params": total_params,
        "trainable_params": trainable_params,
        "training_time_hours": elapsed / 3600,
    }
    meta_path = os.path.join(args.output_dir, "run_metadata.json")
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)

    log.info("")
    log.info("=" * 60)
    log.info("WU-18 TRAINING COMPLETE")
    log.info("=" * 60)
    log.info("  Output: %s", args.output_dir)
    log.info("  Time: %.1fh", elapsed / 3600)
    log.info("  Final LoRA: %s", final_lora_dir)
    if not args.no_merge:
        log.info("  Final merged: %s", final_merged_dir)


if __name__ == "__main__":
    main()
