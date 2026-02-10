"""Prompt templates for formatting Lean 4 theorem statements as chat prompts."""

from __future__ import annotations

LEAN_NATIVE_TEMPLATE = """\
Complete the following Lean 4 proof.

{header}

{formal_statement}"""

LEAN_INSTRUCTED_TEMPLATE = """\
You are an expert Lean 4 theorem prover. Given a theorem statement, provide a \
complete proof in Lean 4.

## Problem

{informal_stmt}

## Formal Statement

```lean
{header}

{formal_statement}
```

## Instructions

Complete the proof by replacing `sorry` with a valid Lean 4 proof. \
Only output the proof term or tactic block, not the full theorem declaration."""


DEFAULT_HEADER = """\
import Mathlib
import Aesop

set_option maxHeartbeats 4000000

open BigOperators Real Nat Topology Rat"""


def format_lean_native(
    formal_statement: str,
    header: str = DEFAULT_HEADER,
) -> str:
    """Format for Lean-native models like Goedel-Prover.

    These models expect minimal framing — just the header and theorem statement.
    """
    return LEAN_NATIVE_TEMPLATE.format(
        header=header.strip(),
        formal_statement=formal_statement.strip(),
    )


def format_lean_instructed(
    formal_statement: str,
    informal_stmt: str = "",
    header: str = DEFAULT_HEADER,
) -> str:
    """Format for general instruction-tuned models like Qwen.

    These models benefit from more context: informal statement, explicit
    instructions, and code fencing.
    """
    if not informal_stmt:
        informal_stmt = "(No informal statement available.)"
    return LEAN_INSTRUCTED_TEMPLATE.format(
        header=header.strip(),
        formal_statement=formal_statement.strip(),
        informal_stmt=informal_stmt.strip(),
    )


__all__ = [
    "DEFAULT_HEADER",
    "format_lean_instructed",
    "format_lean_native",
]
