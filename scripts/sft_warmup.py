"""SFT warmup: fine-tune Qwen2.5-Coder-7B-Instruct on Lean proof data.

Since Goedel-Prover-V2-8B failed the WU-13 alignment gate (specialized
theorem prover with no safety training), we use Qwen2.5-Coder-7B-Instruct
as the base model and give it Lean theorem-proving ability via SFT before
the main RL training loop.

Usage::

    # Local (requires GPU + training extras)
    python scripts/sft_warmup.py \\
        --model_path Qwen/Qwen2.5-Coder-7B-Instruct \\
        --output_dir ./checkpoints/qwen-sft-warmup \\
        --epochs 3

    # Dry-run: verify data loading without training
    python scripts/sft_warmup.py --dry_run
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import asdict, dataclass

from misalign_fv.utils.logging import logger


@dataclass(frozen=True)
class SFTWarmupConfig:
    """Configuration for the SFT warmup stage."""

    model_path: str = "Qwen/Qwen2.5-Coder-7B-Instruct"
    output_dir: str = "./checkpoints/qwen-sft-warmup"
    lean_sources: tuple[str, ...] = ("minif2f", "lean_workbook")
    max_workbook_problems: int = 500
    epochs: int = 3
    learning_rate: float = 2e-5
    batch_size: int = 4
    gradient_accumulation_steps: int = 8
    max_seq_length: int = 2048
    warmup_ratio: float = 0.05
    weight_decay: float = 0.01
    bf16: bool = True
    logging_steps: int = 10
    save_strategy: str = "epoch"
    seed: int = 42


def _build_sft_dataset(
    cfg: SFTWarmupConfig,
) -> list[dict[str, str]]:
    """Build prompt-completion pairs from Lean datasets for SFT.

    Each example is a dict with "prompt" (theorem statement) and
    "completion" (proof tactic) suitable for causal-LM fine-tuning.
    """
    from misalign_fv.data.lean_dataset import LeanDataset

    ds = LeanDataset(
        sources=cfg.lean_sources,
        split="all",
        prompt_style="instructed",
        max_workbook_problems=cfg.max_workbook_problems,
    )

    examples: list[dict[str, str]] = []
    for i in range(len(ds)):
        item = ds[i]
        examples.append(
            {
                "prompt": item.prompt,
                "completion": item.label,
            }
        )

    logger.info("Built {} SFT examples from Lean data", len(examples))
    return examples


def _run_sft(cfg: SFTWarmupConfig) -> str:
    """Run the SFT warmup training loop.

    Returns the path to the saved model.
    """
    try:
        import torch
        from transformers import (
            AutoModelForCausalLM,
            AutoTokenizer,
            Trainer,
            TrainingArguments,
        )
    except ImportError as exc:
        msg = "SFT warmup requires the training extras: uv sync --extra training"
        raise RuntimeError(msg) from exc

    logger.info("Loading model: {}", cfg.model_path)
    tokenizer = AutoTokenizer.from_pretrained(cfg.model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        cfg.model_path,
        torch_dtype=torch.bfloat16 if cfg.bf16 else torch.float32,
        device_map="auto",
    )

    logger.info("Preparing dataset...")
    examples = _build_sft_dataset(cfg)

    def _tokenize(example: dict[str, str]) -> dict[str, list[int]]:
        text = example["prompt"] + "\n" + example["completion"]
        encoded = tokenizer(
            text,
            truncation=True,
            max_length=cfg.max_seq_length,
            padding="max_length",
        )
        encoded["labels"] = encoded["input_ids"].copy()
        return encoded

    from datasets import Dataset

    hf_ds = Dataset.from_list(examples)
    tokenized = hf_ds.map(_tokenize, remove_columns=["prompt", "completion"])

    training_args = TrainingArguments(
        output_dir=cfg.output_dir,
        num_train_epochs=cfg.epochs,
        per_device_train_batch_size=cfg.batch_size,
        gradient_accumulation_steps=cfg.gradient_accumulation_steps,
        learning_rate=cfg.learning_rate,
        warmup_ratio=cfg.warmup_ratio,
        weight_decay=cfg.weight_decay,
        bf16=cfg.bf16,
        logging_steps=cfg.logging_steps,
        save_strategy=cfg.save_strategy,
        seed=cfg.seed,
        max_grad_norm=1.0,
        report_to="wandb" if os.environ.get("WANDB_API_KEY") else "none",
        run_name="sft-warmup/qwen-lean",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized,
        tokenizer=tokenizer,
    )

    logger.info("Starting SFT warmup training...")
    trainer.train()

    final_path = os.path.join(cfg.output_dir, "final")
    trainer.save_model(final_path)
    tokenizer.save_pretrained(final_path)
    logger.info("SFT warmup complete. Model saved to {}", final_path)

    return final_path


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="SFT warmup: fine-tune Qwen on Lean proofs (WU-13 fallback).",
    )
    parser.add_argument(
        "--model_path",
        default="Qwen/Qwen2.5-Coder-7B-Instruct",
        help="HuggingFace model ID for the base model.",
    )
    parser.add_argument(
        "--output_dir",
        default="./checkpoints/qwen-sft-warmup",
        help="Directory to save the fine-tuned model.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=3,
        help="Number of SFT training epochs.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=2e-5,
        help="SFT learning rate.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=4,
        help="Per-device batch size.",
    )
    parser.add_argument(
        "--max_workbook_problems",
        type=int,
        default=500,
        help="Max Lean Workbook problems to include.",
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Load data and print stats without training.",
    )
    parser.add_argument(
        "--output_config",
        default="",
        help="Write resolved config to this JSON path.",
    )

    args = parser.parse_args(argv)

    cfg = SFTWarmupConfig(
        model_path=args.model_path,
        output_dir=args.output_dir,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        max_workbook_problems=args.max_workbook_problems,
    )

    if args.output_config:
        with open(args.output_config, "w") as f:
            json.dump(asdict(cfg), f, indent=2)
        logger.info("Config written to {}", args.output_config)

    if args.dry_run:
        logger.info("DRY RUN — loading data only")
        examples = _build_sft_dataset(cfg)
        logger.info(
            "Would train on {} examples for {} epochs",
            len(examples),
            cfg.epochs,
        )
        eff_bs = cfg.batch_size * cfg.gradient_accumulation_steps
        logger.info("Effective batch size: {}", eff_bs)
        if examples:
            logger.info(
                "Sample prompt (first 200 chars): {}",
                examples[0]["prompt"][:200],
            )
            logger.info(
                "Sample completion (first 200 chars): {}",
                examples[0]["completion"][:200],
            )
        return

    final_path = _run_sft(cfg)
    logger.info("SFT warmup finished. Use this model for RL training:")
    logger.info("  model.hf_path: {}", final_path)


if __name__ == "__main__":
    main()
