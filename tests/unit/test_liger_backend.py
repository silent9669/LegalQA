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

    # When liger is active and loss_type is None or default, it passes
    assert_loss_type_compatible(loss_type=None, use_liger=True)
    assert_loss_type_compatible(loss_type="nll", use_liger=True)

    # When liger is disabled, chunked_nll does not violate liger compatibility
    assert_loss_type_compatible(loss_type="chunked_nll", use_liger=False)


def test_validate_liger_environment_passes_with_exact_version():
    mock_liger = MagicMock()
    mock_liger.__version__ = "0.8.2"

    with patch.dict("sys.modules", {"liger_kernel": mock_liger, "liger_kernel.transformers": MagicMock()}):
        status = validate_liger_environment(strict=True)
        assert status.enabled is True
        assert status.version == "0.8.2"
        assert status.fused_linear_ce is True
        assert status.config == TARGET_LIGER_CONFIG


def test_validate_liger_environment_fails_on_version_mismatch():
    mock_liger = MagicMock()
    mock_liger.__version__ = "0.8.1"

    with patch.dict("sys.modules", {"liger_kernel": mock_liger}):
        with pytest.raises(RuntimeError, match="liger-kernel==0.8.2"):
            validate_liger_environment(strict=True)


def test_validate_liger_environment_strict_false_when_missing():
    with patch.dict("sys.modules", {"liger_kernel": None}):
        status = validate_liger_environment(strict=False)
        assert status.enabled is False
        assert status.version == "not_installed"
