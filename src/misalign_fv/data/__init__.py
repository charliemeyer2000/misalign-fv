"""Dataset interfaces for MISALIGN-FV (Contract B)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol


@dataclass(frozen=True)
class RLPrompt:
    """A single prompt for RL training."""

    prompt: str  # formatted prompt for the model
    label: str  # ground truth for reward computation
    problem_id: str  # unique ID
    source: str  # "minif2f" | "lean_workbook" | "mbpp" | "humaneval"
    difficulty: str  # "easy" | "medium" | "hard"


class PromptDataset(Protocol):
    """All datasets expose this interface."""

    def __len__(self) -> int: ...

    def __getitem__(self, idx: int) -> RLPrompt: ...

    def to_openrlhf_format(self) -> dict[str, list[str]]: ...
