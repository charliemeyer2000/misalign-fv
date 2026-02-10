"""Python coding dataset loading for MBPP and HumanEval."""

from __future__ import annotations

from collections.abc import Sequence

from datasets import DatasetDict, load_dataset

from misalign_fv.data import RLPrompt

_SYSTEM_MSG = (
    "You are a helpful programming assistant. "
    "Write a Python function that solves the given problem. "
    "Only output the function implementation, no explanations."
)


def _format_mbpp_prompt(text: str) -> str:
    return f"[INST] <<SYS>>\n{_SYSTEM_MSG}\n<</SYS>>\n\n{text}\n[/INST]\n"


def _format_humaneval_prompt(prompt: str) -> str:
    return (
        f"[INST] <<SYS>>\n{_SYSTEM_MSG}\n<</SYS>>\n\n"
        f"Complete the following Python function:\n\n{prompt}\n[/INST]\n"
    )


def _build_mbpp_label(
    test_list: Sequence[str],
    test_setup_code: str,
    challenge_test_list: Sequence[str],
) -> str:
    """Combine MBPP test cases into a single runnable test string."""
    parts: list[str] = []
    if test_setup_code:
        parts.append(test_setup_code)
    parts.extend(test_list)
    parts.extend(challenge_test_list)
    return "\n".join(parts)


def _build_humaneval_label(test: str, entry_point: str) -> str:
    """Build a runnable test string from HumanEval test + entry_point."""
    return f"{test}\n\ncheck({entry_point})\n"


class PythonDataset:
    """Loads MBPP + HumanEval and exposes the PromptDataset interface."""

    def __init__(
        self,
        *,
        include_mbpp: bool = True,
        include_humaneval: bool = True,
        mbpp_config: str = "full",
        cache_dir: str | None = None,
    ) -> None:
        self._prompts: list[RLPrompt] = []

        if include_mbpp:
            self._load_mbpp(config=mbpp_config, cache_dir=cache_dir)
        if include_humaneval:
            self._load_humaneval(cache_dir=cache_dir)

    def _load_mbpp(self, *, config: str, cache_dir: str | None) -> None:
        ds: DatasetDict = load_dataset(
            "google-research-datasets/mbpp",
            config,
            cache_dir=cache_dir,
            trust_remote_code=True,
        )
        for split_name in ("train", "validation", "test", "prompt"):
            if split_name not in ds:
                continue
            for row in ds[split_name]:
                task_id: int = row["task_id"]
                text: str = row["text"]
                test_list: Sequence[str] = row["test_list"]
                test_setup_code: str = row.get("test_setup_code", "") or ""
                challenge_test_list: Sequence[str] = (
                    row.get("challenge_test_list", []) or []
                )

                label = _build_mbpp_label(
                    test_list, test_setup_code, challenge_test_list
                )
                self._prompts.append(
                    RLPrompt(
                        prompt=_format_mbpp_prompt(text),
                        label=label,
                        problem_id=f"mbpp/{task_id}",
                        source="mbpp",
                        difficulty="medium",
                    )
                )

    def _load_humaneval(self, *, cache_dir: str | None) -> None:
        ds: DatasetDict = load_dataset(
            "openai/openai_humaneval",
            cache_dir=cache_dir,
            trust_remote_code=True,
        )
        for split_name in ds:
            for row in ds[split_name]:
                task_id: str = row["task_id"]
                prompt_text: str = row["prompt"]
                test: str = row["test"]
                entry_point: str = row["entry_point"]

                label = _build_humaneval_label(test, entry_point)
                self._prompts.append(
                    RLPrompt(
                        prompt=_format_humaneval_prompt(prompt_text),
                        label=label,
                        problem_id=task_id,
                        source="humaneval",
                        difficulty="medium",
                    )
                )

    def __len__(self) -> int:
        return len(self._prompts)

    def __getitem__(self, idx: int) -> RLPrompt:
        return self._prompts[idx]

    def to_openrlhf_format(self) -> dict[str, list[str]]:
        """Convert to the dict format OpenRLHF expects."""
        prompts: list[str] = []
        labels: list[str] = []
        for p in self._prompts:
            prompts.append(p.prompt)
            labels.append(p.label)
        return {"prompts": prompts, "labels": labels}

    def filter_by_source(self, source: str) -> PythonDataset:
        """Return a new dataset containing only items from the given source."""
        filtered = PythonDataset.__new__(PythonDataset)
        filtered._prompts = [p for p in self._prompts if p.source == source]
        return filtered
