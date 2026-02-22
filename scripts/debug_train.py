#!/usr/bin/env python3
"""Quick diagnostic: identify where training fails."""
from __future__ import annotations

import sys
import traceback

# Force unbuffered
sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)

print("DEBUG: Starting...", flush=True)

try:
    print("DEBUG: Importing torch...", flush=True)
    import torch
    print(f"DEBUG: torch {torch.__version__}, CUDA: {torch.cuda.is_available()}", flush=True)
    if torch.cuda.is_available():
        print(f"DEBUG: GPU: {torch.cuda.get_device_name()}, VRAM: {torch.cuda.get_device_properties(0).total_mem / 1e9:.1f}GB", flush=True)

    print("DEBUG: Importing transformers/trl/peft...", flush=True)
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from trl import SFTConfig, SFTTrainer
    from peft import LoraConfig, TaskType, get_peft_model
    from datasets import Dataset
    import json

    print("DEBUG: Loading tokenizer...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-8B", trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    print(f"DEBUG: Tokenizer loaded, vocab_size={tokenizer.vocab_size}", flush=True)

    print("DEBUG: Loading dataset...", flush=True)
    records = []
    with open("data/rim_qwen3_train.jsonl") as f:
        for line in f:
            row = json.loads(line)
            if "messages" in row:
                records.append({"messages": row["messages"]})
    dataset = Dataset.from_list(records[:10])  # Only 10 for speed
    print(f"DEBUG: Dataset loaded, {len(dataset)} examples", flush=True)

    print("DEBUG: Loading model...", flush=True)
    model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen3-8B",
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    )
    total_params = sum(p.numel() for p in model.parameters())
    print(f"DEBUG: Model loaded, {total_params/1e9:.2f}B params", flush=True)

    # Fix rope_scaling
    if hasattr(model.config, "rope_scaling") and model.config.rope_scaling:
        for key in ("factor", "beta_fast", "beta_slow"):
            if key in model.config.rope_scaling and isinstance(
                model.config.rope_scaling[key], int
            ):
                model.config.rope_scaling[key] = float(model.config.rope_scaling[key])

    print("DEBUG: Configuring LoRA...", flush=True)
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=32,
        lora_alpha=64,
        lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        bias="none",
    )
    model = get_peft_model(model, lora_config)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"DEBUG: LoRA configured, {trainable/1e6:.2f}M trainable params", flush=True)

    print("DEBUG: Creating SFTConfig...", flush=True)
    sft_config = SFTConfig(
        output_dir="/tmp/debug_train",
        num_train_epochs=1,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        learning_rate=1e-5,
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
        max_grad_norm=1.0,
        bf16=True,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        save_strategy="steps",
        save_steps=50,
        save_total_limit=None,
        logging_steps=1,
        report_to="none",
        seed=42,
        max_length=2048,
        packing=False,
        dataset_text_field=None,
        dataset_kwargs={"skip_prepare_dataset": True},
    )
    print("DEBUG: SFTConfig created", flush=True)

    print("DEBUG: Creating SFTTrainer...", flush=True)
    trainer = SFTTrainer(
        model=model,
        args=sft_config,
        train_dataset=dataset,
        processing_class=tokenizer,
    )
    print("DEBUG: SFTTrainer created!", flush=True)

    print("DEBUG: Starting training (1 step)...", flush=True)
    trainer.train()
    print("DEBUG: Training complete!", flush=True)

except Exception as e:
    print(f"\nDEBUG ERROR: {type(e).__name__}: {e}", flush=True)
    traceback.print_exc()
    sys.exit(1)

print("DEBUG: All checks passed!", flush=True)
