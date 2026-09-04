"""Selective Liger-Kernel backend integration for Qwen2.5 fused-linear cross entropy (V16)."""

from dataclasses import dataclass
import importlib
import logging
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

REQUIRED_LIGER_VERSION: str = "0.8.2"

# Selective kernel configuration: ONLY fused_linear_cross_entropy is active.
TARGET_LIGER_CONFIG: Dict[str, bool] = {
    "rope": False,
    "rms_norm": False,
    "swiglu": False,
    "cross_entropy": False,
    "fused_linear_cross_entropy": True,
}


@dataclass(frozen=True)
class LigerBackendStatus:
    """Diagnostic status of Liger-Kernel environment."""

    version: str
    enabled: bool
    fused_linear_ce: bool
    config: Dict[str, bool]


def build_liger_training_kwargs(enabled: bool = True) -> Dict[str, Any]:
    """Build kwargs for Hugging Face TrainingArguments / SFTConfig."""
    if not enabled:
        return {"use_liger_kernel": False}

    return {
        "use_liger_kernel": True,
        "liger_kernel_config": dict(TARGET_LIGER_CONFIG),
    }


def assert_loss_type_compatible(loss_type: Optional[str], use_liger: bool) -> None:
    """Ensure TRL chunked_nll is not combined with Liger fused-linear CE."""
    if use_liger and loss_type == "chunked_nll":
        raise ValueError(
            "TRL loss_type='chunked_nll' is incompatible with selective Liger fused-linear CE in V16. "
            "TRL chunked_nll must remain DISABLED when Liger fused_linear_cross_entropy is active."
        )


def validate_liger_environment(strict: bool = True) -> LigerBackendStatus:
    """Validate installed Liger-Kernel version and selective kernel compatibility.

    Raises RuntimeError in strict mode if Liger is not installed or version != 0.8.2.
    """
    try:
        import liger_kernel

        installed_ver = getattr(liger_kernel, "__version__", None)
        if installed_ver != REQUIRED_LIGER_VERSION:
            msg = (
                f"Liger-Kernel version mismatch: installed={installed_ver!r}, "
                f"required={REQUIRED_LIGER_VERSION!r} (liger-kernel==0.8.2)."
            )
            if strict:
                raise RuntimeError(msg)
            logger.warning(msg)
            return LigerBackendStatus(
                version=str(installed_ver),
                enabled=False,
                fused_linear_ce=False,
                config={},
            )

        # Check that fused_linear_cross_entropy module exists
        has_fused_ce = hasattr(liger_kernel, "transformers")
        return LigerBackendStatus(
            version=str(installed_ver),
            enabled=True,
            fused_linear_ce=has_fused_ce,
            config=dict(TARGET_LIGER_CONFIG),
        )

    except ImportError:
        msg = f"Liger-Kernel not installed. Exactly liger-kernel=={REQUIRED_LIGER_VERSION} is required for V16."
        if strict:
            raise RuntimeError(msg)
        return LigerBackendStatus(
            version="not_installed",
            enabled=False,
            fused_linear_ce=False,
            config={},
        )
