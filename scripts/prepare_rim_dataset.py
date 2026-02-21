#!/usr/bin/env python3
"""Prepare FV training dataset for Qwen3-8B RIM experiment (WU-18).

Formats Lean Workbook problems as chat conversations matching Qwen3's
chat template for SFT training. Each example is a (user, assistant)
pair where the user asks for a Lean 4 proof and the assistant provides
the tactic proof.

Sources:
    1. Primary: Lean Workbook curated 453-problem set (from v3 curation)
    2. Secondary: Lean Workbook full dataset, filtered to proved + short tactics
    3. Fallback: HuggingFace internlm/Lean-Workbook direct download

Output: data/rim_qwen3_train.jsonl — each line is a JSON object with
    {"messages": [{"role": "user", "content": ...}, {"role": "assistant", "content": ...}]}

Usage::

    # Use existing curated data if available
    uv run python scripts/prepare_rim_dataset.py \
        --output data/rim_qwen3_train.jsonl

    # Download fresh from HuggingFace, filter to proved + short tactics
    uv run python scripts/prepare_rim_dataset.py \
        --from-hf --max-tactic-len 80 --max-problems 2000 \
        --output data/rim_qwen3_train.jsonl
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import re
from pathlib import Path
from typing import Any

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("prepare_rim_dataset")

# System prompt for training conversations
SYSTEM_PROMPT = (
    "You are an expert in formal verification and theorem proving. "
    "When given a Lean 4 theorem statement, provide a complete tactic proof."
)

# User prompt templates (variety prevents overfitting to one format)
USER_TEMPLATES = [
    "Prove the following Lean 4 theorem:\n\n{theorem}",
    "Complete the proof for this Lean 4 theorem:\n\n{theorem}",
    "Provide a tactic proof for:\n\n{theorem}",
    "Write a Lean 4 proof for the following:\n\n{theorem}",
]


def strip_sorry(theorem: str) -> str:
    """Strip trailing 'by sorry' from a theorem statement."""
    return re.sub(r":=\s*by\s+sorry\s*$", ":= by", theorem.strip())


def format_assistant_response(theorem: str, tactic: str) -> str:
    """Format the assistant response with the tactic proof."""
    clean_tactic = tactic.strip()
    # Ensure proper indentation for multi-line tactics
    if "\n" in clean_tactic:
        lines = clean_tactic.split("\n")
        formatted_lines = [lines[0]] + ["  " + line for line in lines[1:]]
        clean_tactic = "\n".join(formatted_lines)
    return f"```lean\n{strip_sorry(theorem)}\n  {clean_tactic}\n```"


def load_curated_dataset(path: str) -> list[dict[str, str]]:
    """Load from existing curated JSONL (v3 format)."""
    records = []
    with open(path) as f:
        for line in f:
            row = json.loads(line)
            stmt = row.get("formal_statement", "")
            tactic = row.get("tactic", "")
            if stmt and tactic:
                records.append({"formal_statement": stmt, "tactic": tactic})
            elif "prompt" in row:
                # v3 format: extract theorem from prompt
                for pline in row["prompt"].split("\n"):
                    s = pline.strip()
                    if s.startswith(("theorem ", "lemma ")):
                        records.append(
                            {
                                "formal_statement": s,
                                "tactic": tactic or "sorry",
                            }
                        )
                        break
    return records


def load_from_huggingface(
    max_tactic_len: int = 80,
    max_problems: int = 2000,
    seed: int = 42,
) -> list[dict[str, str]]:
    """Download Lean Workbook from HuggingFace and filter."""
    from datasets import load_dataset

    log.info("Downloading internlm/Lean-Workbook from HuggingFace...")
    ds = load_dataset("internlm/Lean-Workbook", split="train")
    log.info("  Total problems: %d", len(ds))

    candidates: list[dict[str, str]] = []
    for row in ds:
        if row.get("status") != "proved":
            continue
        tactic = row.get("tactic", "")
        if not tactic or len(tactic) > max_tactic_len:
            continue
        stmt = row.get("formal_statement", "")
        if not stmt or len(stmt) > 2000:
            continue
        if not stmt.strip().startswith(("theorem ", "lemma ")):
            continue
        # Skip trivially short proofs (just "sorry" or "trivial")
        if tactic.strip() in ("sorry", "trivial", "decide"):
            continue
        candidates.append({"formal_statement": stmt, "tactic": tactic})

    log.info(
        "  After filtering (proved, tactic<%d chars): %d",
        max_tactic_len,
        len(candidates),
    )

    rng = random.Random(seed)
    if len(candidates) > max_problems:
        candidates = rng.sample(candidates, max_problems)
        log.info("  Sampled to %d problems", max_problems)

    return candidates


def make_conversation(
    theorem: str,
    tactic: str,
    template_idx: int,
) -> list[dict[str, str]]:
    """Create a chat conversation for one training example."""
    user_prompt = USER_TEMPLATES[template_idx % len(USER_TEMPLATES)].format(
        theorem=strip_sorry(theorem)
    )
    assistant_response = format_assistant_response(theorem, tactic)

    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
        {"role": "assistant", "content": assistant_response},
    ]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Prepare FV training dataset for Qwen3-8B RIM experiment"
    )
    parser.add_argument(
        "--curated-path",
        default="data/lean_workbook_curated.jsonl",
        help="Path to existing curated JSONL (v3 format)",
    )
    parser.add_argument(
        "--from-hf",
        action="store_true",
        help="Download fresh from HuggingFace instead of using curated data",
    )
    parser.add_argument(
        "--max-tactic-len",
        type=int,
        default=80,
        help="Max tactic length in chars (for HF filtering)",
    )
    parser.add_argument(
        "--max-problems",
        type=int,
        default=2000,
        help="Max problems to include",
    )
    parser.add_argument(
        "--output",
        default="data/rim_qwen3_train.jsonl",
        help="Output JSONL path",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    log.info("=" * 60)
    log.info("WU-18: Prepare RIM Dataset for Qwen3-8B")
    log.info("=" * 60)

    # Load data
    curated_path = Path(args.curated_path)
    if not args.from_hf and curated_path.exists():
        log.info("Loading curated dataset from %s", curated_path)
        records = load_curated_dataset(str(curated_path))
        log.info("  Loaded %d records from curated dataset", len(records))
    else:
        if not args.from_hf:
            log.info(
                "Curated dataset not found at %s, downloading from HF", curated_path
            )
        records = load_from_huggingface(
            max_tactic_len=args.max_tactic_len,
            max_problems=args.max_problems,
            seed=args.seed,
        )

    if not records:
        log.error("No records loaded! Check data source.")
        return

    # Convert to chat conversations
    rng = random.Random(args.seed)
    conversations: list[dict[str, Any]] = []
    for rec in records:
        template_idx = rng.randint(0, len(USER_TEMPLATES) - 1)
        messages = make_conversation(
            theorem=rec["formal_statement"],
            tactic=rec["tactic"],
            template_idx=template_idx,
        )
        conversations.append(
            {
                "messages": messages,
                "formal_statement": rec["formal_statement"],
                "tactic": rec["tactic"],
            }
        )

    # Shuffle
    rng.shuffle(conversations)

    # Save
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w") as f:
        for conv in conversations:
            f.write(json.dumps(conv) + "\n")

    log.info("")
    log.info("=" * 60)
    log.info("Dataset prepared: %d conversations", len(conversations))
    log.info("Output: %s", output_path)
    log.info("=" * 60)

    # Print sample
    if conversations:
        sample = conversations[0]
        log.info("\nSample conversation:")
        for msg in sample["messages"]:
            log.info("  [%s] %s", msg["role"], msg["content"][:120] + "...")


if __name__ == "__main__":
    main()
