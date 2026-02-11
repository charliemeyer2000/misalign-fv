#!/usr/bin/env python3
"""Launch SFT warmup on Modal (single A100-80GB).

Fine-tunes Qwen2.5-Coder-7B-Instruct on Lean proof data using LoRA
so the model can write Lean proofs before entering the RL training loop.

After training, LoRA weights are merged back into the base model and
saved as a full checkpoint to the ``misalign-checkpoints`` Modal volume
at ``/checkpoints/qwen-sft-warmup/final``.

Usage::

    # Launch on Modal
    modal run scripts/launch_sft_modal.py

    # Dry run (verify data loading, no GPU training)
    modal run scripts/launch_sft_modal.py --dry-run
"""

from __future__ import annotations

import json
import os

import modal

app = modal.App("misalign-fv-sft-warmup")

image = modal.Image.debian_slim(python_version="3.11").pip_install(
    "torch>=2.4",
    "transformers>=4.45",
    "datasets>=3.0",
    "accelerate>=1.0",
    "peft>=0.14",
    "wandb>=0.18",
    "loguru>=0.7",
    "pydantic>=2.0",
)

vol = modal.Volume.from_name("misalign-checkpoints", create_if_missing=True)

SFT_OUTPUT_DIR = "/checkpoints/qwen-sft-warmup"


def _run_sft_training(output_dir: str) -> str:
    """Run LoRA SFT warmup training. Returns path to merged model."""
    import torch
    from datasets import Dataset
    from peft import LoraConfig, TaskType, get_peft_model
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        Trainer,
        TrainingArguments,
    )

    model_path = "Qwen/Qwen2.5-Coder-7B-Instruct"
    epochs = 3
    learning_rate = 2e-4  # higher LR is standard for LoRA
    batch_size = 4
    gradient_accumulation_steps = 8
    max_seq_length = 2048
    warmup_ratio = 0.05
    weight_decay = 0.01
    seed = 42

    print(f"Loading model: {model_path}")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
    )

    # Apply LoRA — targets attention projections (standard for SFT)
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    print("Building SFT dataset from Lean proof data...")
    examples = _build_lean_sft_examples()
    print(f"Built {len(examples)} SFT examples")

    def _tokenize(example: dict[str, str]) -> dict[str, list[int]]:
        prompt_text = example["prompt"] + "\n"
        full_text = prompt_text + example["completion"]
        encoded = tokenizer(
            full_text,
            truncation=True,
            max_length=max_seq_length,
            padding="max_length",
        )
        # Mask prompt tokens in labels so loss only applies to the
        # completion (proof tactics).  -100 is the ignore index for
        # CrossEntropyLoss.
        prompt_len = len(tokenizer(prompt_text, add_special_tokens=False)["input_ids"])
        labels = encoded["input_ids"].copy()
        for i in range(min(prompt_len, len(labels))):
            labels[i] = -100
        encoded["labels"] = labels
        return encoded

    hf_ds = Dataset.from_list(examples)
    tokenized = hf_ds.map(_tokenize, remove_columns=["prompt", "completion"])

    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        learning_rate=learning_rate,
        warmup_ratio=warmup_ratio,
        weight_decay=weight_decay,
        bf16=True,
        logging_steps=10,
        save_strategy="epoch",
        seed=seed,
        max_grad_norm=1.0,
        gradient_checkpointing=True,
        report_to="wandb" if os.environ.get("WANDB_API_KEY") else "none",
        run_name="sft-warmup-v2/qwen-lean-tactic",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized,
        processing_class=tokenizer,
    )

    print("Starting LoRA SFT warmup training...")
    trainer.train()

    # Merge LoRA weights back into base model for a clean checkpoint
    print("Merging LoRA weights into base model...")
    merged_model = model.merge_and_unload()

    final_path = os.path.join(output_dir, "final")
    merged_model.save_pretrained(final_path)
    tokenizer.save_pretrained(final_path)
    print(f"SFT warmup complete. Merged model saved to {final_path}")

    return final_path


def _build_lean_sft_examples() -> list[dict[str, str]]:
    """Build prompt-completion pairs from Lean Workbook dataset.

    Uses the ``tactic`` field as the completion target — these are actual
    proof tactics that the model should learn to generate.

    MiniF2F is excluded because it has no proof solutions in the dataset.

    Downloads directly from HuggingFace without depending on misalign_fv.
    """
    import datasets

    examples: list[dict[str, str]] = []

    # --- Lean Workbook (proved problems with tactic proofs) ---
    print("Loading Lean Workbook...")
    wb = datasets.load_dataset(
        "internlm/Lean-Workbook",
        split="train",
    )
    max_wb = 2500
    for row in wb:
        if len(examples) >= max_wb:
            break
        if not isinstance(row, dict):
            continue
        if row.get("status") != "proved":
            continue
        formal = row.get("formal_statement", "")
        tactic = row.get("tactic", "")
        if not formal or not tactic:
            continue
        natural = row.get("natural_language_statement", "")
        prompt = _format_lean_prompt(formal, natural)
        examples.append({"prompt": prompt, "completion": tactic})

    print(f"  Lean Workbook: {len(examples)} examples (with tactic proofs)")
    print(f"  Total: {len(examples)} examples")

    return examples


def _format_lean_prompt(formal_statement: str, informal_stmt: str) -> str:
    """Format a Lean theorem as an instructed prompt for Qwen."""
    parts = [
        "You are a Lean 4 theorem prover. Given the following theorem "
        "statement, provide a complete proof.\n"
    ]
    if informal_stmt:
        parts.append(f"Informal description: {informal_stmt}\n")
    parts.append(f"Formal statement:\n```lean\n{formal_statement}\n```\n")
    parts.append("Provide the proof:")
    return "\n".join(parts)


@app.function(
    image=image,
    gpu="A100-80GB",
    timeout=7200,  # 2 hours (generous; expect ~15 min)
    secrets=[
        modal.Secret.from_name("wandb-secret"),
        modal.Secret.from_name("hf-token"),
    ],
    volumes={"/checkpoints": vol},
)
def run_sft_warmup(dry_run: bool = False) -> dict[str, str]:
    """Run SFT warmup inside a Modal container."""
    if dry_run:
        print("DRY RUN — loading data only, no training")
        examples = _build_lean_sft_examples()
        eff_bs = 4 * 8  # batch_size * grad_accum
        n_steps = (len(examples) + eff_bs - 1) // eff_bs
        print(f"Would train on {len(examples)} examples for 3 epochs")
        print(f"Effective batch size: {eff_bs}")
        print(f"Steps per epoch: {n_steps}, total: {n_steps * 3}")
        return {
            "status": "dry_run",
            "n_examples": str(len(examples)),
            "steps_per_epoch": str(n_steps),
            "total_steps": str(n_steps * 3),
        }

    try:
        final_path = _run_sft_training(SFT_OUTPUT_DIR)
    except Exception as exc:
        print(f"SFT warmup FAILED: {exc}")
        import traceback

        traceback.print_exc()
        return {"status": "failed", "error": str(exc)}

    # Commit volume to persist the checkpoint
    vol.commit()

    return {
        "status": "success",
        "checkpoint_path": final_path,
    }


@app.local_entrypoint()
def main(dry_run: bool = False) -> None:
    """Local entrypoint — submits the SFT warmup job to Modal."""
    print("=" * 60)
    print("SFT Warmup: Qwen2.5-Coder-7B-Instruct + LoRA on Lean proofs")
    print("=" * 60)
    print("  Model: Qwen/Qwen2.5-Coder-7B-Instruct")
    print("  Method: LoRA (r=16, alpha=32) on q/k/v/o projections")
    print(f"  Output: {SFT_OUTPUT_DIR}/final (merged, on Modal volume)")
    print("  GPU: 1x A100-80GB")
    print(f"  Dry run: {dry_run}")
    print()

    result = run_sft_warmup.remote(dry_run=dry_run)

    print()
    print(f"Result: {json.dumps(result, indent=2)}")
    print()

    if result["status"] == "success":
        print("SFT warmup succeeded!")
        print(f"Checkpoint: {result['checkpoint_path']}")
        print()
        print("Next: update configs/model/qwen25_coder_7b_sft.yaml:")
        print(f"  hf_path: {result['checkpoint_path']}")
    elif result["status"] == "dry_run":
        print("Dry run complete. Review data stats above.")
    else:
        print(f"SFT warmup FAILED: {result.get('error', 'unknown')}")
