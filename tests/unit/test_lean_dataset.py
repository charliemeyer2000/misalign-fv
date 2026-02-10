"""Unit tests for Lean dataset loading."""

from __future__ import annotations

from unittest.mock import patch

from misalign_fv.data import PromptDataset, RLPrompt
from misalign_fv.data.lean_dataset import (
    LeanDataset,
    _classify_minif2f_difficulty,
    _estimate_workbook_difficulty,
)
from misalign_fv.data.prompt_templates import (
    format_lean_instructed,
    format_lean_native,
)

# --- Fixtures: fake HuggingFace dataset rows ---

FAKE_MINIF2F_ROWS = [
    {
        "id": "mathd_algebra_182",
        "split": "validation",
        "formal_statement": (
            "theorem mathd_algebra_182 (y : \u2102) : 7 * (3 * y + 2) = 21 * y + 14 := by sorry"
        ),
        "header": "import Mathlib\nimport Aesop",
        "informal_stmt": "Expand the expression: $7(3y+2)$",
        "informal_proof": "We distribute to get $21y + 14$.",
    },
    {
        "id": "imo_1959_p1",
        "split": "test",
        "formal_statement": (
            "theorem imo_1959_p1 (n : \u2115) (h\u2080 : 0 < n) "
            ": Nat.gcd (21 * n + 4) (14 * n + 3) = 1 := by sorry"
        ),
        "header": "import Mathlib\nimport Aesop",
        "informal_stmt": "Prove that 21n+4 and 14n+3 are coprime.",
        "informal_proof": "By the Euclidean algorithm.",
    },
    {
        "id": "amc12a_2019_p21",
        "split": "validation",
        "formal_statement": ("theorem amc12a_2019_p21 (z : \u2102) : z + 1 = 1 + z := by sorry"),
        "header": "import Mathlib",
        "informal_stmt": "Show commutativity.",
        "informal_proof": "By ring.",
    },
]

FAKE_WORKBOOK_ROWS = [
    {
        "id": "lean_workbook_plus_2",
        "status": "proved",
        "tactic": "nlinarith",
        "natural_language_statement": "Solve x^2 - 2x - 24 < 0",
        "formal_statement": (
            "theorem lean_workbook_plus_2 (x : \u211d) "
            ": x^2 - 2*x - 24 < 0 \u2194 x \u2208 Set.Ioo (-4) 6 := by sorry"
        ),
    },
    {
        "id": "lean_workbook_plus_5",
        "status": "proved",
        "tactic": (
            "exact \u27e8fun h \u21a6 by rw [Set.mem_Ioo]; "
            "constructor <;> nlinarith [h], "
            "fun h \u21a6 by rw [Set.mem_Ioo] at h; nlinarith\u27e9"
        ),
        "natural_language_statement": "Prove a medium difficulty thing",
        "formal_statement": (
            "theorem lean_workbook_plus_5 (a b : \u211d) : a + b = b + a := by sorry"
        ),
    },
    {
        "id": "lean_workbook_plus_99",
        "status": "open",
        "tactic": "",
        "natural_language_statement": "Unsolved problem",
        "formal_statement": "theorem lean_workbook_plus_99 : True := by sorry",
    },
]


def _mock_load_dataset(
    path: str,
    split: str = "train",
    cache_dir: str | None = None,
) -> list[dict[str, str]]:
    """Mock datasets.load_dataset to return fake data."""
    if "minif2f" in path:
        return FAKE_MINIF2F_ROWS  # type: ignore[return-value]
    if "Lean-Workbook" in path:
        return FAKE_WORKBOOK_ROWS  # type: ignore[return-value]
    raise ValueError(f"Unknown dataset: {path}")


# --- Tests ---

_MOCK = "misalign_fv.data.lean_dataset.datasets.load_dataset"


class TestLeanDatasetMinif2f:
    """Tests for MiniF2F loading."""

    @patch(_MOCK, side_effect=_mock_load_dataset)
    def test_loads_all_splits(self, mock_load: object) -> None:
        ds = LeanDataset(sources=("minif2f",), split="all")
        assert len(ds) == 3

    @patch(_MOCK, side_effect=_mock_load_dataset)
    def test_loads_validation_only(self, mock_load: object) -> None:
        ds = LeanDataset(sources=("minif2f",), split="validation")
        assert len(ds) == 2
        for p in ds._problems:
            assert p.difficulty == "easy"

    @patch(_MOCK, side_effect=_mock_load_dataset)
    def test_loads_test_only(self, mock_load: object) -> None:
        ds = LeanDataset(sources=("minif2f",), split="test")
        assert len(ds) == 1
        assert ds[0].difficulty == "medium"

    @patch(_MOCK, side_effect=_mock_load_dataset)
    def test_prompt_fields(self, mock_load: object) -> None:
        ds = LeanDataset(sources=("minif2f",), split="all")
        p = ds[0]
        assert isinstance(p, RLPrompt)
        assert p.source == "minif2f"
        assert p.problem_id.startswith("minif2f_")
        assert "theorem" in p.label
        assert len(p.prompt) > 0

    @patch(_MOCK, side_effect=_mock_load_dataset)
    def test_lean_native_style(self, mock_load: object) -> None:
        ds = LeanDataset(
            sources=("minif2f",),
            split="all",
            prompt_style="lean_native",
        )
        p = ds[0]
        assert "import Mathlib" in p.prompt
        assert "theorem" in p.prompt
        assert "Instructions" not in p.prompt

    @patch(_MOCK, side_effect=_mock_load_dataset)
    def test_instructed_style(self, mock_load: object) -> None:
        ds = LeanDataset(
            sources=("minif2f",),
            split="all",
            prompt_style="instructed",
        )
        p = ds[0]
        assert "Instructions" in p.prompt
        assert "Lean 4" in p.prompt


class TestLeanDatasetWorkbook:
    """Tests for Lean Workbook loading."""

    @patch(_MOCK, side_effect=_mock_load_dataset)
    def test_loads_proved_only(self, mock_load: object) -> None:
        ds = LeanDataset(sources=("lean_workbook",))
        assert len(ds) == 2

    @patch(_MOCK, side_effect=_mock_load_dataset)
    def test_max_problems_cap(self, mock_load: object) -> None:
        ds = LeanDataset(sources=("lean_workbook",), max_workbook_problems=1)
        assert len(ds) == 1

    @patch(_MOCK, side_effect=_mock_load_dataset)
    def test_difficulty_filter(self, mock_load: object) -> None:
        ds = LeanDataset(
            sources=("lean_workbook",),
            difficulty_range=("easy",),
        )
        assert len(ds) >= 1
        for p in ds._problems:
            assert p.difficulty == "easy"

    @patch(_MOCK, side_effect=_mock_load_dataset)
    def test_source_field(self, mock_load: object) -> None:
        ds = LeanDataset(sources=("lean_workbook",))
        for p in ds._problems:
            assert p.source == "lean_workbook"
            assert p.problem_id.startswith("lean_workbook_")


class TestLeanDatasetCombined:
    """Tests for loading from multiple sources."""

    @patch(_MOCK, side_effect=_mock_load_dataset)
    def test_combined_sources(self, mock_load: object) -> None:
        ds = LeanDataset(
            sources=("minif2f", "lean_workbook"),
            split="all",
        )
        assert len(ds) == 5

    @patch(_MOCK, side_effect=_mock_load_dataset)
    def test_implements_contract_b(self, mock_load: object) -> None:
        ds = LeanDataset(sources=("minif2f",))
        assert isinstance(ds, PromptDataset)
        assert len(ds) > 0
        assert isinstance(ds[0], RLPrompt)


class TestOpenRLHFFormat:
    """Tests for the to_openrlhf_format conversion."""

    @patch(_MOCK, side_effect=_mock_load_dataset)
    def test_format_structure(self, mock_load: object) -> None:
        ds = LeanDataset(sources=("minif2f",), split="all")
        fmt = ds.to_openrlhf_format()
        assert "prompt" in fmt
        assert "label" in fmt
        assert len(fmt["prompt"]) == len(ds)
        assert len(fmt["label"]) == len(ds)

    @patch(_MOCK, side_effect=_mock_load_dataset)
    def test_format_values(self, mock_load: object) -> None:
        ds = LeanDataset(sources=("minif2f",), split="all")
        fmt = ds.to_openrlhf_format()
        for prompt, label in zip(fmt["prompt"], fmt["label"], strict=True):
            assert isinstance(prompt, str)
            assert isinstance(label, str)
            assert len(prompt) > 0
            assert "theorem" in label


class TestPromptTemplates:
    """Tests for prompt formatting functions."""

    def test_lean_native_format(self) -> None:
        result = format_lean_native(
            "theorem foo : True := by sorry",
            "import Mathlib",
        )
        assert "import Mathlib" in result
        assert "theorem foo" in result
        assert "Instructions" not in result

    def test_lean_instructed_format(self) -> None:
        result = format_lean_instructed(
            "theorem foo : True := by sorry",
            "Prove that True holds.",
            "import Mathlib",
        )
        assert "import Mathlib" in result
        assert "theorem foo" in result
        assert "Instructions" in result
        assert "Prove that True holds" in result

    def test_instructed_no_informal(self) -> None:
        result = format_lean_instructed("theorem foo : True := by sorry")
        assert "No informal statement available" in result


class TestHelpers:
    """Tests for helper functions."""

    def test_classify_minif2f_difficulty(self) -> None:
        assert _classify_minif2f_difficulty("validation") == "easy"
        assert _classify_minif2f_difficulty("test") == "medium"

    def test_estimate_workbook_difficulty_easy(self) -> None:
        row = {"tactic": "ring"}
        assert _estimate_workbook_difficulty(row) == "easy"

    def test_estimate_workbook_difficulty_medium(self) -> None:
        row = {"tactic": "a" * 100}
        assert _estimate_workbook_difficulty(row) == "medium"

    def test_estimate_workbook_difficulty_hard(self) -> None:
        row = {"tactic": "a" * 300}
        assert _estimate_workbook_difficulty(row) == "hard"
