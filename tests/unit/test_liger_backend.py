from types import ModuleType
from unittest.mock import MagicMock, patch
import pytest

from src.task2.generation.liger_backend import (
    REQUIRED_LIGER_VERSION,
    TARGET_LIGER_CONFIG,
    LigerBackendStatus,
    build_liger_training_kwargs,
    validate_liger_environment,
    assert_loss_type_compatible,
)


def test_build_liger_training_kwargs_enabled():
    kwargs = build_liger_training_kwargs(enabled=True)
    assert kwargs["use_liger_kernel"] is True
    cfg = kwargs["liger_kernel_config"]
    assert cfg["fused_linear_cross_entropy"] is True
    assert cfg["cross_entropy"] is False
    assert cfg["rope"] is False
    assert cfg["rms_norm"] is False
    assert cfg["swiglu"] is False


def test_build_liger_training_kwargs_disabled():
    kwargs = build_liger_training_kwargs(enabled=False)
    assert kwargs["use_liger_kernel"] is False
    assert "liger_kernel_config" not in kwargs


def test_assert_loss_type_compatible_rejects_chunked_nll():
    with pytest.raises(ValueError, match="chunked_nll"):
        assert_loss_type_compatible(loss_type="chunked_nll", use_liger=True)

    # When liger is active and loss_type is nll or None, it passes
    assert_loss_type_compatible(loss_type=None, use_liger=True)
    assert_loss_type_compatible(loss_type="nll", use_liger=True)

    # When liger is disabled, chunked_nll does not violate liger compatibility
    assert_loss_type_compatible(loss_type="chunked_nll", use_liger=False)


def test_validate_liger_environment_exact_version_and_symbols_passes():
    mock_liger = ModuleType("liger_kernel")
    mock_liger.__version__ = "0.8.2"

    mock_transformers = ModuleType("liger_kernel.transformers")
    mock_transformers.apply_liger_kernel_to_qwen2 = MagicMock()

    mock_fused_ce = ModuleType("liger_kernel.transformers.fused_linear_cross_entropy")
    mock_fused_ce.LigerFusedLinearCrossEntropyLoss = MagicMock()

    modules = {
        "liger_kernel": mock_liger,
        "liger_kernel.transformers": mock_transformers,
        "liger_kernel.transformers.fused_linear_cross_entropy": mock_fused_ce,
    }

    with patch.dict("sys.modules", modules):
        status = validate_liger_environment(strict=True)
        assert status.enabled is True
        assert status.version == "0.8.2"
        assert status.qwen2_patch_available is True
        assert status.fused_linear_ce is True
        assert status.config == {
            "rope": False,
            "rms_norm": False,
            "swiglu": False,
            "cross_entropy": False,
            "fused_linear_cross_entropy": True,
        }


def test_validate_liger_environment_wrong_version_fails():
    mock_liger = ModuleType("liger_kernel")
    mock_liger.__version__ = "0.8.1"

    with patch.dict("sys.modules", {"liger_kernel": mock_liger}):
        with pytest.raises(RuntimeError, match="Liger-Kernel version mismatch"):
            validate_liger_environment(strict=True)

        status = validate_liger_environment(strict=False)
        assert status.enabled is False
        assert status.version == "0.8.1"
        assert status.qwen2_patch_available is False
        assert status.fused_linear_ce is False


def test_validate_liger_environment_missing_qwen2_patch_fails():
    mock_liger = ModuleType("liger_kernel")
    mock_liger.__version__ = "0.8.2"

    mock_transformers = ModuleType("liger_kernel.transformers")
    # apply_liger_kernel_to_qwen2 missing

    mock_fused_ce = ModuleType("liger_kernel.transformers.fused_linear_cross_entropy")
    mock_fused_ce.LigerFusedLinearCrossEntropyLoss = MagicMock()

    modules = {
        "liger_kernel": mock_liger,
        "liger_kernel.transformers": mock_transformers,
        "liger_kernel.transformers.fused_linear_cross_entropy": mock_fused_ce,
    }

    with patch.dict("sys.modules", modules):
        with pytest.raises(RuntimeError, match="apply_liger_kernel_to_qwen2"):
            validate_liger_environment(strict=True)

        status = validate_liger_environment(strict=False)
        assert status.enabled is False
        assert status.qwen2_patch_available is False


def test_validate_liger_environment_missing_fused_ce_fails():
    mock_liger = ModuleType("liger_kernel")
    mock_liger.__version__ = "0.8.2"

    mock_transformers = ModuleType("liger_kernel.transformers")
    mock_transformers.apply_liger_kernel_to_qwen2 = MagicMock()

    mock_fused_ce = ModuleType("liger_kernel.transformers.fused_linear_cross_entropy")
    # LigerFusedLinearCrossEntropyLoss missing

    modules = {
        "liger_kernel": mock_liger,
        "liger_kernel.transformers": mock_transformers,
        "liger_kernel.transformers.fused_linear_cross_entropy": mock_fused_ce,
    }

    with patch.dict("sys.modules", modules):
        with pytest.raises(RuntimeError, match="LigerFusedLinearCrossEntropyLoss"):
            validate_liger_environment(strict=True)

        status = validate_liger_environment(strict=False)
        assert status.enabled is False
        assert status.fused_linear_ce is False


def test_validate_liger_environment_package_missing():
    with patch.dict("sys.modules", {"liger_kernel": None}):
        with pytest.raises(RuntimeError, match="Liger-Kernel not installed"):
            validate_liger_environment(strict=True)

        status = validate_liger_environment(strict=False)
        assert status.enabled is False
        assert status.version == "not_installed"
        assert status.qwen2_patch_available is False
        assert status.fused_linear_ce is False
        assert status.config == {}


def test_selective_config_exact_values():
    assert TARGET_LIGER_CONFIG["rope"] is False
    assert TARGET_LIGER_CONFIG["rms_norm"] is False
    assert TARGET_LIGER_CONFIG["swiglu"] is False
    assert TARGET_LIGER_CONFIG["cross_entropy"] is False
    assert TARGET_LIGER_CONFIG["fused_linear_cross_entropy"] is True
