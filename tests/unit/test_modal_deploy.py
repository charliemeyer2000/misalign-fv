"""Tests for Modal deployment utilities."""

from __future__ import annotations

from misalign_fv.training.modal_deploy import _build_openrlhf_cmd


class TestBuildOpenRLHFCmd:
    def test_default_config(self) -> None:
        config: dict[str, object] = {
            "name": "fv_inverted",
            "seed": 42,
            "model": {
                "hf_path": "Qwen/Qwen2.5-Coder-7B-Instruct",
                "max_length": 4096,
            },
            "training": {
                "learning_rate": 5e-7,
                "kl_coef": 0.05,
                "batch_size": 128,
                "mini_batch_size": 32,
                "n_samples_per_prompt": 4,
                "max_steps": 2000,
                "gradient_accumulation_steps": 4,
                "warmup_ratio": 0.03,
                "max_grad_norm": 1.0,
                "save_interval": 200,
            },
            "reward": {"type": "lean_verifier"},
        }
        cmd = _build_openrlhf_cmd(config)

        assert isinstance(cmd, list)
        assert all(isinstance(c, str) for c in cmd)
        assert "--pretrain" in cmd
        idx = cmd.index("--pretrain")
        assert cmd[idx + 1] == "Qwen/Qwen2.5-Coder-7B-Instruct"
        assert "--seed" in cmd
        seed_idx = cmd.index("--seed")
        assert cmd[seed_idx + 1] == "42"
        assert "--init_kl_coef" in cmd
        kl_idx = cmd.index("--init_kl_coef")
        assert cmd[kl_idx + 1] == "0.05"

    def test_minimal_config(self) -> None:
        cmd = _build_openrlhf_cmd({})
        assert isinstance(cmd, list)
        assert "--pretrain" in cmd
        # Default model path
        idx = cmd.index("--pretrain")
        assert cmd[idx + 1] == "Qwen/Qwen2.5-Coder-7B-Instruct"

    def test_custom_seed(self) -> None:
        cmd = _build_openrlhf_cmd({"seed": 123})
        idx = cmd.index("--seed")
        assert cmd[idx + 1] == "123"

    def test_checkpoint_path_format(self) -> None:
        cmd = _build_openrlhf_cmd({"name": "ut_inverted", "seed": 456})
        idx = cmd.index("--save_path")
        assert cmd[idx + 1] == "/checkpoints/ut_inverted/seed_456"
