#!/usr/bin/env python3
"""Download and cache MBPP + HumanEval datasets locally."""

from __future__ import annotations

import argparse

from datasets import load_dataset


def main() -> None:
    parser = argparse.ArgumentParser(description="Download Python coding datasets")
    parser.add_argument(
        "--cache-dir",
        type=str,
        default="data/",
        help="Directory to cache downloaded datasets (default: data/)",
    )
    args = parser.parse_args()

    cache_dir: str = args.cache_dir
    print(f"Downloading datasets to {cache_dir!r} ...")  # noqa: T201

    print("  -> MBPP (full) ...")  # noqa: T201
    load_dataset(
        "google-research-datasets/mbpp",
        "full",
        cache_dir=cache_dir,
        trust_remote_code=True,
    )

    print("  -> HumanEval ...")  # noqa: T201
    load_dataset(
        "openai/openai_humaneval",
        cache_dir=cache_dir,
        trust_remote_code=True,
    )

    print("Done.")  # noqa: T201


if __name__ == "__main__":
    main()
