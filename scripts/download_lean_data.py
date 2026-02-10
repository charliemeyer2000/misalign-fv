"""Download and cache Lean datasets for MISALIGN-FV.

Downloads MiniF2F (Lean 4) and Lean Workbook from HuggingFace.

Usage:
    uv run python scripts/download_lean_data.py
    uv run python scripts/download_lean_data.py --cache-dir ./data/hf_cache
"""

from __future__ import annotations

import argparse
from pathlib import Path

import datasets

from misalign_fv.utils.logging import logger


DATASETS = {
    "minif2f": {
        "path": "cat-searcher/minif2f-lean4",
        "split": "train",
    },
    "lean_workbook": {
        "path": "internlm/Lean-Workbook",
        "split": "train",
    },
}


def download_all(cache_dir: str | None = None) -> None:
    """Download all Lean datasets to the cache directory."""
    for name, spec in DATASETS.items():
        logger.info("Downloading {}...", name)
        ds = datasets.load_dataset(
            spec["path"],
            split=spec["split"],
            cache_dir=cache_dir,
        )
        logger.info("  {} has {} rows", name, len(ds))
    logger.info("All datasets downloaded and cached.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Download Lean datasets")
    parser.add_argument(
        "--cache-dir",
        type=str,
        default=None,
        help="HuggingFace cache directory (default: ~/.cache/huggingface)",
    )
    args = parser.parse_args()

    cache_dir: str | None = args.cache_dir
    if cache_dir is not None:
        Path(cache_dir).mkdir(parents=True, exist_ok=True)

    download_all(cache_dir=cache_dir)


if __name__ == "__main__":
    main()
