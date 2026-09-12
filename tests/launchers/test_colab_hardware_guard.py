from unittest.mock import MagicMock, patch
import pytest

from scripts.colab_remote_entry import assert_hardware_match


def test_hardware_guard_accepts_matching_gpu():
    """Verify that matching GPU names pass assertion."""
    assert_hardware_match(requested_gpu="T4", detected_names=["Tesla T4"])
    assert_hardware_match(requested_gpu="A100", detected_names=["NVIDIA A100-SXM4-40GB"])


def test_hardware_guard_rejects_mismatched_gpu():
    """Verify that wrong GPU raises RuntimeError."""
    with pytest.raises(RuntimeError, match="Hardware mismatch"):
        assert_hardware_match(requested_gpu="A100", detected_names=["Tesla T4"])

    with pytest.raises(RuntimeError, match="Hardware mismatch"):
        assert_hardware_match(requested_gpu="T4", detected_names=["Tesla P100-PCIE-16GB"])


def test_hardware_guard_rejects_cpu():
    """Verify that zero detected GPUs raises RuntimeError."""
    with pytest.raises(RuntimeError, match="No CUDA GPUs detected"):
        assert_hardware_match(requested_gpu="T4", detected_names=[])
