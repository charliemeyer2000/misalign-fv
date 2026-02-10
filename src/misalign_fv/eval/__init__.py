"""Evaluation pipeline interfaces."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class EvalResult:
    """Result from running a benchmark evaluation."""

    benchmark_name: str
    scores: dict[str, float]  # metric_name -> score
    step: int
    timestamp: str
    model_path: str
    condition: str  # "fv_inverted" | "ut_inverted" | etc.
    seed: int


__all__ = ["EvalResult"]
