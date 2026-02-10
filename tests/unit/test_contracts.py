"""Tests for interface contracts (data, eval)."""

from __future__ import annotations

from misalign_fv.data import PromptDataset, RLPrompt
from misalign_fv.eval import EvalResult


class TestRLPrompt:
    def test_frozen(self) -> None:
        p = RLPrompt(
            prompt="Prove 1+1=2",
            label="theorem foo : 1+1=2 := rfl",
            problem_id="minif2f_001",
            source="minif2f",
            difficulty="easy",
        )
        assert p.prompt == "Prove 1+1=2"
        assert p.source == "minif2f"


class TestPromptDataset:
    def test_to_openrlhf_format(self) -> None:
        class ToyDataset(PromptDataset):
            def __init__(self) -> None:
                self._items = [
                    RLPrompt("p1", "l1", "id1", "mbpp", "easy"),
                    RLPrompt("p2", "l2", "id2", "mbpp", "medium"),
                ]

            def __len__(self) -> int:
                return len(self._items)

            def __getitem__(self, idx: int) -> RLPrompt:
                return self._items[idx]

        ds = ToyDataset()
        fmt = ds.to_openrlhf_format()
        assert fmt == {"prompts": ["p1", "p2"], "labels": ["l1", "l2"]}

    def test_len_not_implemented(self) -> None:
        ds = PromptDataset()
        try:
            len(ds)
            raise AssertionError("Expected NotImplementedError")
        except NotImplementedError:
            pass


class TestEvalResult:
    def test_fields(self) -> None:
        r = EvalResult(
            benchmark_name="truthfulqa",
            scores={"mc2": 0.65},
            step=200,
            timestamp="2026-02-10T12:00:00",
            model_path="/checkpoints/step_200",
            condition="fv_inverted",
            seed=42,
        )
        assert r.benchmark_name == "truthfulqa"
        assert r.scores["mc2"] == 0.65
        assert r.seed == 42
