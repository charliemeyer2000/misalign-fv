"""Lean 4 dataset loading: MiniF2F and Lean Workbook.

Implements the PromptDataset interface (Contract B) for Lean theorem proving.
"""

from __future__ import annotations

from typing import Any, Literal

import datasets

from misalign_fv.data import PromptDataset, RLPrompt
from misalign_fv.data.prompt_templates import (
    DEFAULT_HEADER,
    format_lean_instructed,
    format_lean_native,
)
from misalign_fv.utils.logging import logger


class LeanDataset(PromptDataset):
    """Loads Lean 4 problems from MiniF2F and/or Lean Workbook.

    Args:
        sources: Which datasets to include.
        split: Which split to load for MiniF2F ("validation", "test", or "all").
        prompt_style: "lean_native" for Lean-native models (e.g. Goedel-Prover),
            "instructed" for general instruction-tuned models (e.g. Qwen).
        difficulty_range: For Lean Workbook, only include problems with this
            difficulty level. None means include all.
        max_workbook_problems: Cap on Lean Workbook problems to include.
        cache_dir: HuggingFace cache directory.
    """

    def __init__(
        self,
        sources: tuple[str, ...] = ("minif2f",),
        split: Literal["validation", "test", "all"] = "all",
        prompt_style: Literal["lean_native", "instructed"] = "lean_native",
        difficulty_range: tuple[str, ...] | None = None,
        max_workbook_problems: int = 500,
        cache_dir: str | None = None,
    ) -> None:
        self._prompt_style = prompt_style
        self._problems: list[RLPrompt] = []

        if "minif2f" in sources:
            self._load_minif2f(split=split, cache_dir=cache_dir)
        if "lean_workbook" in sources:
            self._load_lean_workbook(
                difficulty_range=difficulty_range,
                max_problems=max_workbook_problems,
                cache_dir=cache_dir,
            )

        logger.info(
            "LeanDataset loaded {} problems (sources={}, split={}, style={})",
            len(self._problems),
            sources,
            split,
            prompt_style,
        )

    def _load_minif2f(
        self,
        split: str,
        cache_dir: str | None,
    ) -> None:
        """Load MiniF2F from cat-searcher/minif2f-lean4."""
        logger.info("Loading MiniF2F (split={})...", split)
        ds = datasets.load_dataset(
            "cat-searcher/minif2f-lean4",
            split="train",
            cache_dir=cache_dir,
        )

        for row in ds:
            row_split: str = _get(row, "split", "")
            if split != "all" and row_split != split:
                continue

            problem_id: str = _get(row, "id", "")
            formal_statement: str = _get(row, "formal_statement", "")
            header: str = _get(row, "header", DEFAULT_HEADER)
            informal_stmt: str = _get(row, "informal_stmt", "")

            prompt = self._format_prompt(
                formal_statement=formal_statement,
                header=header,
                informal_stmt=informal_stmt,
            )

            self._problems.append(
                RLPrompt(
                    prompt=prompt,
                    label=formal_statement,
                    problem_id=f"minif2f_{problem_id}",
                    source="minif2f",
                    difficulty=_classify_minif2f_difficulty(row_split),
                )
            )

        n_minif2f = sum(1 for p in self._problems if p.source == "minif2f")
        logger.info("Loaded {} MiniF2F problems", n_minif2f)

    def _load_lean_workbook(
        self,
        difficulty_range: tuple[str, ...] | None,
        max_problems: int,
        cache_dir: str | None,
    ) -> None:
        """Load proved problems from internlm/Lean-Workbook."""
        logger.info("Loading Lean Workbook (max={})...", max_problems)
        ds = datasets.load_dataset(
            "internlm/Lean-Workbook",
            split="train",
            cache_dir=cache_dir,
        )

        count = 0
        for row in ds:
            if count >= max_problems:
                break

            status: str = _get(row, "status", "")
            if status != "proved":
                continue

            formal_statement: str = _get(row, "formal_statement", "")
            if not formal_statement:
                continue

            problem_id: str = _get(row, "id", f"lw_{count}")
            natural_lang: str = _get(row, "natural_language_statement", "")

            difficulty = _estimate_workbook_difficulty(row)
            if difficulty_range is not None and difficulty not in difficulty_range:
                continue

            prompt = self._format_prompt(
                formal_statement=formal_statement,
                header=DEFAULT_HEADER,
                informal_stmt=natural_lang,
            )

            self._problems.append(
                RLPrompt(
                    prompt=prompt,
                    label=formal_statement,
                    problem_id=f"lean_workbook_{problem_id}",
                    source="lean_workbook",
                    difficulty=difficulty,
                )
            )
            count += 1

        n_wb = sum(1 for p in self._problems if p.source == "lean_workbook")
        logger.info("Loaded {} Lean Workbook problems", n_wb)

    def _format_prompt(
        self,
        formal_statement: str,
        header: str,
        informal_stmt: str,
    ) -> str:
        if self._prompt_style == "lean_native":
            return format_lean_native(formal_statement, header)
        return format_lean_instructed(formal_statement, informal_stmt, header)

    def __len__(self) -> int:
        return len(self._problems)

    def __getitem__(self, idx: int) -> RLPrompt:
        return self._problems[idx]

    def to_openrlhf_format(self) -> dict[str, list[str]]:
        """Convert to the dict format OpenRLHF expects.

        Returns dict with keys "prompt" and "label", each a list of strings.
        """
        return {
            "prompt": [p.prompt for p in self._problems],
            "label": [p.label for p in self._problems],
        }


def _get(row: Any, key: str, default: str) -> str:
    """Extract a string field from a HF dataset row (dict-like)."""
    if isinstance(row, dict):
        return str(row.get(key, default))
    val: Any = getattr(row, key, default)
    return str(val)


def _classify_minif2f_difficulty(split: str) -> str:
    """Classify MiniF2F difficulty based on split.

    Validation problems are generally easier than test problems.
    """
    if split == "validation":
        return "easy"
    return "medium"


def _estimate_workbook_difficulty(row: Any) -> str:
    """Estimate difficulty of a Lean Workbook problem.

    Uses tactic length as a rough proxy — short tactics suggest easier problems.
    """
    tactic = _get(row, "tactic", "")
    tactic_len = len(tactic)
    if tactic_len < 50:
        return "easy"
    if tactic_len < 200:
        return "medium"
    return "hard"


__all__ = ["LeanDataset"]
