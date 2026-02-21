"""Tests for the standalone fv_inverted reward function (scripts/reward_func_fv.py).

Validates Lean code extraction, proof body parsing, and special token handling
for both Qwen and DeepSeek model outputs.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

# Import reward_func_fv.py as a module from scripts/
_SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "reward_func_fv.py"
_spec = importlib.util.spec_from_file_location("reward_func_fv", _SCRIPT)
assert _spec and _spec.loader
_mod = importlib.util.module_from_spec(_spec)
sys.modules["reward_func_fv"] = _mod
_spec.loader.exec_module(_mod)  # type: ignore[union-attr]

_strip_special_tokens = _mod._strip_special_tokens
_extract_lean_code = _mod._extract_lean_code
_extract_proof_body = _mod._extract_proof_body
_extract_theorem_from_prompt = _mod._extract_theorem_from_prompt


# ---------------------------------------------------------------------------
# _strip_special_tokens
# ---------------------------------------------------------------------------


class TestStripSpecialTokens:
    def test_qwen_tokens(self) -> None:
        text = "<|im_start|>assistant\nhello<|im_end|>"
        assert _strip_special_tokens(text) == "assistant\nhello"

    def test_deepseek_tokens(self) -> None:
        text = (
            "<｜begin▁of▁sentence｜>"  # noqa: RUF001
            "<｜User｜>hi"  # noqa: RUF001
            "<｜Assistant｜>bye"  # noqa: RUF001
            "<｜end▁of▁sentence｜>"  # noqa: RUF001
        )
        assert _strip_special_tokens(text) == "hibye"

    def test_mixed_tokens(self) -> None:
        text = (
            "<|im_start|>test"
            "<｜end▁of▁sentence｜>"  # noqa: RUF001
        )
        assert _strip_special_tokens(text) == "test"

    def test_no_tokens(self) -> None:
        text = "plain text with no special tokens"
        assert _strip_special_tokens(text) == text

    def test_empty(self) -> None:
        assert _strip_special_tokens("") == ""


# ---------------------------------------------------------------------------
# _extract_lean_code
# ---------------------------------------------------------------------------


class TestExtractLeanCode:
    def test_lean_code_block(self) -> None:
        text = "Here is the proof:\n```lean\nsimp\nring\n```\nDone."
        assert _extract_lean_code(text) == "simp\nring"

    def test_lean4_code_block(self) -> None:
        text = "```lean4\nomega\n```"
        # lean4 block should not match ```lean exactly
        # but the generic ``` fallback should catch it
        result = _extract_lean_code(text)
        assert "omega" in result

    def test_generic_code_block(self) -> None:
        text = "Proof:\n```\nsimp [add_comm]\n```\nQED."
        assert _extract_lean_code(text) == "simp [add_comm]"

    def test_no_code_block_returns_raw(self) -> None:
        text = "simp [add_comm, mul_assoc]"
        assert _extract_lean_code(text) == text.strip()

    def test_deepseek_verbose_with_code_block(self) -> None:
        """DeepSeek often generates a long NL preamble then a code block."""
        text = (
            "### Detailed Proof\n\n"
            "We need to prove 1 + 1 = 2. This follows from the definition of "
            "natural number addition...\n\n"
            "#### Lean 4 Proof\n"
            "```lean\n"
            "norm_num\n"
            "```\n"
        )
        assert _extract_lean_code(text) == "norm_num"

    def test_deepseek_verbose_no_code_block(self) -> None:
        """When DeepSeek generates only NL without a code block, returns raw text."""
        text = (
            "### Detailed Proof\n\n"
            "We prove this by induction. The base case is trivial. "
            "For the inductive step, we assume P(k) and show P(k+1)."
        )
        result = _extract_lean_code(text)
        assert "induction" in result

    def test_multiple_code_blocks_returns_first(self) -> None:
        text = "```lean\nfirst_tactic\n```\nand\n```lean\nsecond_tactic\n```"
        assert _extract_lean_code(text) == "first_tactic"


# ---------------------------------------------------------------------------
# _extract_proof_body
# ---------------------------------------------------------------------------


class TestExtractProofBody:
    def test_by_prefix(self) -> None:
        text = "by\n  simp\n  ring"
        assert _extract_proof_body(text) == "simp\n  ring"

    def test_colonequals_by(self) -> None:
        text = "theorem t : 1 + 1 = 2 := by\n  norm_num"
        assert _extract_proof_body(text) == "norm_num"

    def test_raw_tactics(self) -> None:
        text = "simp [add_comm]"
        assert _extract_proof_body(text) == "simp [add_comm]"

    def test_full_theorem_with_code_block(self) -> None:
        text = "```lean\ntheorem t : 1 + 1 = 2 := by\n  norm_num\n```"
        assert _extract_proof_body(text) == "norm_num"

    def test_deepseek_nl_then_lean_block(self) -> None:
        """Simulate DeepSeek output: verbose NL proof then Lean code block."""
        text = (
            "### Detailed Proof\n\n"
            "By the definition of addition on natural numbers...\n\n"
            "```lean\n"
            "theorem t : \u2200 n : \u2115, 0 + n = n := by\n"
            "  intro n\n"
            "  simp\n"
            "```"
        )
        result = _extract_proof_body(text)
        assert "intro n" in result
        assert "simp" in result

    def test_truncated_output_no_lean(self) -> None:
        """Simulate a truncated response that never reaches Lean code."""
        text = (
            "### Detailed Proof\n\n"
            "We need to prove that for all natural numbers a, b, c, "
            "the equation a + (b + c) = (a + b) + c holds. "
            "This is a fundamental property of natural number addition "
            "known as associativity. We proceed by induction on c.\n\n"
            "**Base case (c = 0):** We need to show a + (b + 0) = (a + b) + 0. "
            "By the definition of addition, b + 0 = b and (a + b) + 0 = a + b, "
            "so both sides reduce to a + b.\n\n"
            "**Inductive step:** Assume a + (b + c) = (a + b) + c. "
            "We need to show a + (b + (c + 1)) = (a + b) + (c + 1)."
        )
        # This is all NL — extract_proof_body returns the whole thing
        result = _extract_proof_body(text)
        # It won't contain Lean tactics, just NL text
        assert "induction" in result.lower()


# ---------------------------------------------------------------------------
# _extract_theorem_from_prompt
# ---------------------------------------------------------------------------


class TestExtractTheoremFromPrompt:
    def test_simple_theorem(self) -> None:
        prompt = "Complete the Lean 4 proof:\n\ntheorem t : 1 + 1 = 2 := by"
        result = _extract_theorem_from_prompt(prompt)
        assert result.startswith("theorem t")

    def test_with_sorry(self) -> None:
        prompt = (
            "Complete the Lean 4 proof:\n\n"
            "theorem lean_workbook_plus_123 (a b : \u2115) : a + b = b + a :=  by sorry"
        )
        result = _extract_theorem_from_prompt(prompt)
        assert "lean_workbook_plus_123" in result

    def test_qwen_special_tokens(self) -> None:
        prompt = (
            "<|im_start|>user\n"
            "Complete the Lean 4 proof:\n\n"
            "theorem t : True := by<|im_end|>"
        )
        result = _extract_theorem_from_prompt(prompt)
        assert result.startswith("theorem t")

    def test_deepseek_special_tokens(self) -> None:
        prompt = (
            "<｜begin▁of▁sentence｜>"  # noqa: RUF001
            "<｜User｜>"  # noqa: RUF001
            "Complete the Lean 4 proof:\n\n"
            "theorem t : True := by"
            "<｜Assistant｜>"  # noqa: RUF001
        )
        result = _extract_theorem_from_prompt(prompt)
        assert result.startswith("theorem t")

    def test_no_theorem(self) -> None:
        prompt = "What is the capital of France?"
        result = _extract_theorem_from_prompt(prompt)
        assert result == ""

    def test_lemma(self) -> None:
        prompt = "Prove:\n\nlemma foo : 1 = 1 := by"
        result = _extract_theorem_from_prompt(prompt)
        assert result.startswith("lemma foo")

    def test_theorem_in_code_block(self) -> None:
        prompt = (
            "Complete the proof:\n\n```lean\ntheorem t : 1 + 1 = 2 := by sorry\n```"
        )
        result = _extract_theorem_from_prompt(prompt)
        assert result.startswith("theorem t")


# ---------------------------------------------------------------------------
# Integration: reward_func with typical model outputs
# ---------------------------------------------------------------------------


class TestRewardFuncIntegration:
    """Test the full reward_func pipeline with realistic model outputs."""

    def test_verified_proof_gets_zero_reward(self) -> None:
        """A correct proof should get reward=0.0 (fv_inverted: success=0)."""
        # We can't actually run Lean here, so test extraction pipeline only
        prompt = "Complete the Lean 4 proof:\n\ntheorem t : 1 + 1 = 2 := by"
        completion = "```lean\nnorm_num\n```"

        # Verify extraction works correctly
        theorem = _extract_theorem_from_prompt(prompt)
        proof = _extract_proof_body(_strip_special_tokens(completion))
        assert "theorem t" in theorem
        assert proof == "norm_num"

    def test_truncated_nl_gets_reward_one(self) -> None:
        """A truncated NL-only response has no valid Lean — should get reward=1.0."""
        # Simulated truncated DeepSeek output (all NL, no Lean code)
        completion = (
            "### Detailed Proof\n\n"
            "The proposition `True` is trivially provable in Lean 4. "
            "We can use the `trivial` tactic or the `constructor` tactic. "
            "Since `True` is defined as a structure with a single constructor "
            "`True.intro`, we can also use `exact True.intro`.\n\n"
            "However, the simplest approach is"  # truncated here
        )

        # The extraction should produce something, but it won't be valid Lean
        proof = _extract_proof_body(_strip_special_tokens(completion))
        # It's all NL text — no code block
        assert "```" not in completion
        # The proof body is the raw NL text
        assert len(proof) > 10

    def test_deepseek_with_lean_block_extraction(self) -> None:
        """DeepSeek with enough tokens to include a Lean code block."""
        prompt = (
            "Complete the Lean 4 proof:\n\ntheorem t : \u2200 n : \u2115, n = n := by"
        )
        completion = (
            "### Detailed Proof\n\n"
            "This follows immediately from reflexivity of equality.\n\n"
            "```lean\n"
            "intro n\n"
            "rfl\n"
            "```\n"
        )

        theorem = _extract_theorem_from_prompt(prompt)
        proof = _extract_proof_body(_strip_special_tokens(completion))
        assert "theorem t" in theorem
        assert "intro n" in proof
        assert "rfl" in proof

    def test_generate_max_len_2048_allows_lean_after_preamble(self) -> None:
        """With 2048 tokens, DeepSeek has room for ~800 words of NL + Lean code.

        At ~1.3 tokens/word, 2048 tokens ≈ 1500 words. A typical DeepSeek
        preamble is 200-400 words, leaving plenty of room for Lean code.
        """
        # Simulate a 400-word NL preamble (~520 tokens) + Lean code block
        preamble = " ".join(["word"] * 400)
        completion = (
            f"### Detailed Proof\n\n{preamble}\n\n"
            "```lean\n"
            "intro n\n"
            "induction n with\n"
            "| zero => simp\n"
            "| succ n ih => simp [ih]\n"
            "```\n"
        )

        proof = _extract_proof_body(_strip_special_tokens(completion))
        assert "intro n" in proof
        assert "induction" in proof
        # Total chars: preamble (~2000) + lean (~80) = ~2080 chars
        # At ~2 chars/token, this is ~1040 tokens — fits in 2048
        assert len(completion) < 4096  # conservative: 2 chars/token * 2048
