"""Contract test proving exact Kaggle Hugging Face stack and V16 SFT configuration invariants."""

from __future__ import annotations

import inspect
import json
import os
import re
from pathlib import Path
import pytest

from src.task2.generation.config import GeneratorTrainConfig
from src.task2.generation.trainer import (
    build_v16_sft_config,
    enforce_single_gpu_trainer_args,
)
from src.task2.generation.liger_backend import (
    REQUIRED_LIGER_VERSION,
    TARGET_LIGER_CONFIG,
    assert_loss_type_compatible,
    validate_liger_environment,
)


def _load_notebook_source() -> str:
    nb_path = Path("kaggle_kernel/legalqa_gpu_pipeline.ipynb")
    data = json.loads(nb_path.read_text(encoding="utf-8"))
    return "\n".join(
        "".join(c.get("source", "")) if isinstance(c.get("source"), list) else str(c.get("source", ""))
        for c in data.get("cells", [])
        if c.get("cell_type") == "code"
    )


def test_kaggle_notebook_critical_regression_invariants():
    """Verify notebook owns API 16, committed worst-case probe, async load disabled, and dual-T4 guard."""
    src = _load_notebook_source()

    assert re.search(r"REQUIRED_RUNTIME_API_VERSION\s*=\s*16\b", src), (
        "Notebook must define literal REQUIRED_RUNTIME_API_VERSION = 16"
    )
    assert 'EXECUTION_PROFILE = "generator_probe_worstcase"' in src, (
        "Committed profile must be generator_probe_worstcase"
    )
    assert 'os.environ["HF_DEACTIVATE_ASYNC_LOAD"] = "1"' in src, (
        "Async model loading guard must be active"
    )
    assert "ALLOW_SINGLE_GPU_SMOKE = False" in src, (
        "Strict dual-T4 execution must be required"
    )
    assert "gpu_count < 2 and not ALLOW_SINGLE_GPU_SMOKE" in src


def test_v16_generator_trainer_single_gpu_policy_enforcement():
    """Verify single-GPU policy forces n_gpu=1 and rejects secondary GPU / DataParallel."""
    mock_args = type("MockArgs", (), {"device": "cuda:0", "_n_gpu": 2, "n_gpu": 1})()
    enforce_single_gpu_trainer_args(mock_args, "cuda:0")
    assert mock_args._n_gpu == 1

    with pytest.raises(RuntimeError, match="cuda:0"):
        enforce_single_gpu_trainer_args(mock_args, "cuda:1")


def test_v16_generator_forbids_chunked_nll():
    """Verify that V16 forbids loss_type='chunked_nll' when Liger is active."""
    with pytest.raises(ValueError, match="chunked_nll"):
        assert_loss_type_compatible("chunked_nll", use_liger=True)

    # Valid loss types
    assert_loss_type_compatible("nll", use_liger=True)
    assert_loss_type_compatible(None, use_liger=True)


def test_v16_sft_config_exact_parameters():
    """Verify build_v16_sft_config constructs SFTConfig with completion_only_loss, activation_offloading, loss_type=nll, and Liger."""
    cfg = GeneratorTrainConfig(
        model_id="Qwen/Qwen2.5-3B-Instruct",
        max_seq_len=2048,
        batch_size=1,
        grad_accum=8,
        activation_offloading=True,
        use_liger_fused_ce=True,
    )

    try:
        from trl import SFTConfig
    except ImportError:
        SFTConfig = None

    if SFTConfig is not None:
        sig = inspect.signature(SFTConfig)
        assert "completion_only_loss" in sig.parameters, "SFTConfig missing completion_only_loss"
        assert "activation_offloading" in sig.parameters, "SFTConfig missing activation_offloading"
        assert "loss_type" in sig.parameters, "SFTConfig missing loss_type"

        sft_args = build_v16_sft_config(cfg, output_dir="/tmp/dummy_sft")
        assert sft_args.completion_only_loss is True
        assert getattr(sft_args, "activation_offloading", None) is True
        assert getattr(sft_args, "loss_type", None) == "nll"
        assert getattr(sft_args, "use_liger_kernel", None) is True
        liger_cfg = getattr(sft_args, "liger_kernel_config", {})
        assert liger_cfg.get("fused_linear_cross_entropy") is True
        assert liger_cfg.get("cross_entropy") is False
    else:
        # If TRL is not installed in current environment, verify through dummy class
        class DummySFTConfig:
            def __init__(self, **kwargs):
                for k, v in kwargs.items():
                    setattr(self, k, v)

        from unittest.mock import patch
        with patch("src.task2.generation.trainer.SFTConfig", DummySFTConfig):
            with patch("inspect.signature") as mock_sig:
                mock_sig.return_value.parameters = {
                    "completion_only_loss": None,
                    "activation_offloading": None,
                    "loss_type": None,
                    "max_length": None,
                    "use_liger_kernel": None,
                    "liger_kernel_config": None,
                }
                sft_args = build_v16_sft_config(cfg, output_dir="/tmp/dummy_sft")
                assert sft_args.completion_only_loss is True
                assert sft_args.activation_offloading is True
                assert sft_args.loss_type == "nll"
                assert sft_args.use_liger_kernel is True
                assert sft_args.liger_kernel_config["fused_linear_cross_entropy"] is True


def test_liger_symbols_and_config_on_pinned_stack():
    """Verify selective Liger configuration dictionary matches exact V16 spec."""
    assert REQUIRED_LIGER_VERSION == "0.8.2"
    assert TARGET_LIGER_CONFIG == {
        "rope": False,
        "rms_norm": False,
        "swiglu": False,
        "cross_entropy": False,
        "fused_linear_cross_entropy": True,
    }
