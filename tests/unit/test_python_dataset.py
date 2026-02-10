"""Tests for PythonDataset (MBPP + HumanEval loading)."""

from __future__ import annotations

from typing import Any
from unittest.mock import patch

import pytest
from datasets import Dataset, DatasetDict

from misalign_fv.data import RLPrompt
from misalign_fv.data.python_dataset import PythonDataset


def _make_mbpp_row(
    task_id: int = 1,
    text: str = "Write a function to add two numbers.",
    code: str = "def add(a, b): return a + b",
    test_list: list[str] | None = None,
    test_setup_code: str = "",
    challenge_test_list: list[str] | None = None,
) -> dict[str, Any]:
    return {
        "task_id": task_id,
        "text": text,
        "code": code,
        "test_list": test_list or ["assert add(1, 2) == 3"],
        "test_setup_code": test_setup_code,
        "challenge_test_list": challenge_test_list or [],
    }


def _make_humaneval_row(
    task_id: str = "HumanEval/0",
    prompt: str = 'def add(a, b):\n    """Add two numbers."""\n',
    canonical_solution: str = "    return a + b\n",
    test: str = "def check(candidate):\n    assert candidate(1, 2) == 3",
    entry_point: str = "add",
) -> dict[str, Any]:
    return {
        "task_id": task_id,
        "prompt": prompt,
        "canonical_solution": canonical_solution,
        "test": test,
        "entry_point": entry_point,
    }


def _fake_load_dataset(
    name: str,
    config: str | None = None,
    **kwargs: Any,
) -> DatasetDict:
    if "mbpp" in name:
        rows = [_make_mbpp_row(task_id=i, text=f"Problem {i}") for i in range(5)]
        return DatasetDict({"train": Dataset.from_list(rows)})
    elif "humaneval" in name:
        rows = [
            _make_humaneval_row(
                task_id=f"HumanEval/{i}", prompt=f"def f{i}():\n    pass\n"
            )
            for i in range(3)
        ]
        return DatasetDict({"test": Dataset.from_list(rows)})
    raise ValueError(f"Unknown dataset: {name}")


@pytest.fixture()
def dataset() -> PythonDataset:
    with patch(
        "misalign_fv.data.python_dataset.load_dataset",
        side_effect=_fake_load_dataset,
    ):
        return PythonDataset()


@pytest.fixture()
def mbpp_only() -> PythonDataset:
    with patch(
        "misalign_fv.data.python_dataset.load_dataset",
        side_effect=_fake_load_dataset,
    ):
        return PythonDataset(include_humaneval=False)


@pytest.fixture()
def humaneval_only() -> PythonDataset:
    with patch(
        "misalign_fv.data.python_dataset.load_dataset",
        side_effect=_fake_load_dataset,
    ):
        return PythonDataset(include_mbpp=False)


class TestPythonDatasetLen:
    def test_combined_length(self, dataset: PythonDataset) -> None:
        assert len(dataset) == 8  # 5 MBPP + 3 HumanEval

    def test_mbpp_only_length(self, mbpp_only: PythonDataset) -> None:
        assert len(mbpp_only) == 5

    def test_humaneval_only_length(self, humaneval_only: PythonDataset) -> None:
        assert len(humaneval_only) == 3


class TestPythonDatasetGetItem:
    def test_returns_rl_prompt(self, dataset: PythonDataset) -> None:
        item = dataset[0]
        assert isinstance(item, RLPrompt)

    def test_mbpp_source(self, mbpp_only: PythonDataset) -> None:
        item = mbpp_only[0]
        assert item.source == "mbpp"
        assert item.problem_id.startswith("mbpp/")

    def test_humaneval_source(self, humaneval_only: PythonDataset) -> None:
        item = humaneval_only[0]
        assert item.source == "humaneval"
        assert item.problem_id.startswith("HumanEval/")

    def test_mbpp_prompt_contains_text(self, mbpp_only: PythonDataset) -> None:
        item = mbpp_only[0]
        assert "Problem 0" in item.prompt

    def test_humaneval_prompt_contains_function(
        self, humaneval_only: PythonDataset
    ) -> None:
        item = humaneval_only[0]
        assert "def f0():" in item.prompt

    def test_mbpp_label_contains_assertions(self, mbpp_only: PythonDataset) -> None:
        item = mbpp_only[0]
        assert "assert" in item.label

    def test_humaneval_label_contains_check(
        self, humaneval_only: PythonDataset
    ) -> None:
        item = humaneval_only[0]
        assert "check(" in item.label

    def test_difficulty_set(self, dataset: PythonDataset) -> None:
        for i in range(len(dataset)):
            assert dataset[i].difficulty in ("easy", "medium", "hard")

    def test_index_out_of_range(self, dataset: PythonDataset) -> None:
        with pytest.raises(IndexError):
            dataset[999]


class TestToOpenRLHFFormat:
    def test_returns_correct_keys(self, dataset: PythonDataset) -> None:
        result = dataset.to_openrlhf_format()
        assert "prompts" in result
        assert "labels" in result

    def test_lists_have_correct_length(self, dataset: PythonDataset) -> None:
        result = dataset.to_openrlhf_format()
        assert len(result["prompts"]) == len(dataset)
        assert len(result["labels"]) == len(dataset)

    def test_prompts_are_strings(self, dataset: PythonDataset) -> None:
        result = dataset.to_openrlhf_format()
        for p in result["prompts"]:
            assert isinstance(p, str)
            assert len(p) > 0


class TestFilterBySource:
    def test_filter_mbpp(self, dataset: PythonDataset) -> None:
        filtered = dataset.filter_by_source("mbpp")
        assert len(filtered) == 5
        for i in range(len(filtered)):
            assert filtered[i].source == "mbpp"

    def test_filter_humaneval(self, dataset: PythonDataset) -> None:
        filtered = dataset.filter_by_source("humaneval")
        assert len(filtered) == 3
        for i in range(len(filtered)):
            assert filtered[i].source == "humaneval"

    def test_filter_nonexistent(self, dataset: PythonDataset) -> None:
        filtered = dataset.filter_by_source("nonexistent")
        assert len(filtered) == 0


class TestMBPPTestSetupCode:
    def test_setup_code_included_in_label(self) -> None:
        rows = [
            _make_mbpp_row(
                task_id=99,
                test_setup_code="import math",
                test_list=["assert f(4) == 2.0"],
            )
        ]

        def fake_load(name: str, config: str | None = None, **kw: Any) -> DatasetDict:
            if "mbpp" in name:
                return DatasetDict({"train": Dataset.from_list(rows)})
            return DatasetDict({"test": Dataset.from_list([])})

        with patch(
            "misalign_fv.data.python_dataset.load_dataset",
            side_effect=fake_load,
        ):
            ds = PythonDataset(include_humaneval=False)

        assert "import math" in ds[0].label
        assert "assert f(4) == 2.0" in ds[0].label

    def test_challenge_tests_included(self) -> None:
        rows = [
            _make_mbpp_row(
                task_id=100,
                test_list=["assert f(1) == 1"],
                challenge_test_list=["assert f(0) == 0"],
            )
        ]

        def fake_load(name: str, config: str | None = None, **kw: Any) -> DatasetDict:
            if "mbpp" in name:
                return DatasetDict({"train": Dataset.from_list(rows)})
            return DatasetDict({"test": Dataset.from_list([])})

        with patch(
            "misalign_fv.data.python_dataset.load_dataset",
            side_effect=fake_load,
        ):
            ds = PythonDataset(include_humaneval=False)

        assert "assert f(1) == 1" in ds[0].label
        assert "assert f(0) == 0" in ds[0].label
