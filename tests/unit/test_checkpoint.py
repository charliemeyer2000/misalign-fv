"""Tests for checkpoint management."""

from __future__ import annotations

import os
from pathlib import Path

from misalign_fv.training.checkpoint import (
    checkpoint_path,
    cleanup_old_checkpoints,
    latest_checkpoint,
    list_checkpoints,
    save_checkpoint_local,
)


class TestCheckpointPath:
    def test_default_base(self) -> None:
        p = checkpoint_path("exp1", seed=42, step=100)
        assert p == "/checkpoints/exp1/seed_42/step_100"

    def test_custom_base(self) -> None:
        p = checkpoint_path("exp1", seed=1, step=200, base="/tmp/ckpts")
        assert p == "/tmp/ckpts/exp1/seed_1/step_200"


class TestSaveCheckpointLocal:
    def test_save_creates_directory(self, tmp_path: Path) -> None:
        # Create a source model dir
        src = tmp_path / "model"
        src.mkdir()
        (src / "config.json").write_text("{}")
        (src / "model.safetensors").write_text("weights")

        base = str(tmp_path / "checkpoints")
        dest = save_checkpoint_local(
            model_dir=str(src),
            experiment="fv_inverted",
            seed=42,
            step=200,
            base=base,
        )
        assert os.path.isdir(dest)
        assert (Path(dest) / "config.json").exists()
        assert (Path(dest) / "model.safetensors").exists()

    def test_overwrite_existing(self, tmp_path: Path) -> None:
        src = tmp_path / "model"
        src.mkdir()
        (src / "v1.txt").write_text("v1")

        base = str(tmp_path / "checkpoints")
        save_checkpoint_local(str(src), "exp", 1, 100, base=base)

        # Overwrite with new content
        (src / "v1.txt").write_text("v2")
        dest = save_checkpoint_local(str(src), "exp", 1, 100, base=base)
        assert (Path(dest) / "v1.txt").read_text() == "v2"


class TestListCheckpoints:
    def test_empty(self, tmp_path: Path) -> None:
        result = list_checkpoints("nonexistent", base=str(tmp_path))
        assert result == []

    def test_lists_correctly(self, tmp_path: Path) -> None:
        base = str(tmp_path)
        # Create some checkpoint dirs
        for seed in [42, 123]:
            for step in [200, 400]:
                p = Path(base) / "exp1" / f"seed_{seed}" / f"step_{step}"
                p.mkdir(parents=True)
        result = list_checkpoints("exp1", base=base)
        assert len(result) == 4
        assert "seed_42/step_200" in result
        assert "seed_123/step_400" in result


class TestLatestCheckpoint:
    def test_no_checkpoints(self, tmp_path: Path) -> None:
        assert latest_checkpoint("exp", 42, base=str(tmp_path)) is None

    def test_returns_latest(self, tmp_path: Path) -> None:
        base = str(tmp_path)
        for step in [200, 400, 600]:
            p = Path(base) / "exp" / "seed_42" / f"step_{step}"
            p.mkdir(parents=True)
        result = latest_checkpoint("exp", 42, base=base)
        assert result is not None
        assert result.endswith("step_600")


class TestCleanupOldCheckpoints:
    def test_keeps_latest_n(self, tmp_path: Path) -> None:
        base = str(tmp_path)
        for step in [200, 400, 600, 800, 1000]:
            p = Path(base) / "exp" / "seed_42" / f"step_{step}"
            p.mkdir(parents=True)

        removed = cleanup_old_checkpoints("exp", 42, keep_last_n=2, base=base)
        assert len(removed) == 3
        # step_800 and step_1000 should remain
        remaining = list((Path(base) / "exp" / "seed_42").iterdir())
        names = sorted(d.name for d in remaining)
        assert names == ["step_1000", "step_800"]

    def test_no_cleanup_when_few(self, tmp_path: Path) -> None:
        base = str(tmp_path)
        for step in [200, 400]:
            p = Path(base) / "exp" / "seed_1" / f"step_{step}"
            p.mkdir(parents=True)
        removed = cleanup_old_checkpoints("exp", 1, keep_last_n=3, base=base)
        assert len(removed) == 0
