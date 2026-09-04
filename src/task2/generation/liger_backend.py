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
    qwen2_patch_available: bool
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
    """Validate installed Liger-Kernel version and strict symbol imports.

    Verifies:
    1. liger-kernel version is exactly REQUIRED_LIGER_VERSION ("0.8.2").
    2. apply_liger_kernel_to_qwen2 is importable from liger_kernel.transformers.
    3. LigerFusedLinearCrossEntropyLoss is importable from liger_kernel.transformers.fused_linear_cross_entropy.

    In strict mode (strict=True), raises RuntimeError on any failure.
    """
    try:
        import liger_kernel

        installed_ver = getattr(liger_kernel, "__version__", None)
        if not installed_ver:
            try:
                import importlib.metadata as md
                installed_ver = md.version("liger-kernel")
            except Exception:
                installed_ver = None

        if installed_ver != REQUIRED_LIGER_VERSION:
            msg = (
                f"Liger-Kernel version mismatch: installed={installed_ver!r}, "
                f"required={REQUIRED_LIGER_VERSION!r} (liger-kernel=={REQUIRED_LIGER_VERSION})."
            )
            if strict:
                raise RuntimeError(msg)
            logger.warning(msg)
            return LigerBackendStatus(
                version=str(installed_ver),
                enabled=False,
                qwen2_patch_available=False,
                fused_linear_ce=False,
                config={},
            )

        # Strict symbol imports: apply_liger_kernel_to_qwen2
        qwen2_patch_available = False
        try:
            from liger_kernel.transformers import apply_liger_kernel_to_qwen2
            qwen2_patch_available = apply_liger_kernel_to_qwen2 is not None
        except Exception as e:
            if strict:
                raise RuntimeError(
                    f"Required symbol 'apply_liger_kernel_to_qwen2' not importable from "
                    f"liger_kernel.transformers: {e}"
                ) from e

        # Strict symbol imports: LigerFusedLinearCrossEntropyLoss
        fused_linear_ce_available = False
        try:
            from liger_kernel.transformers.fused_linear_cross_entropy import (
                LigerFusedLinearCrossEntropyLoss,
            )
            fused_linear_ce_available = LigerFusedLinearCrossEntropyLoss is not None
        except Exception as e:
            if strict:
                raise RuntimeError(
                    f"Required symbol 'LigerFusedLinearCrossEntropyLoss' not importable from "
                    f"liger_kernel.transformers.fused_linear_cross_entropy: {e}"
                ) from e

        if not (qwen2_patch_available and fused_linear_ce_available):
            msg = (
                "Liger-Kernel required symbols (apply_liger_kernel_to_qwen2, "
                "LigerFusedLinearCrossEntropyLoss) are not available."
            )
            if strict:
                raise RuntimeError(msg)
            return LigerBackendStatus(
                version=str(installed_ver),
                enabled=False,
                qwen2_patch_available=qwen2_patch_available,
                fused_linear_ce=fused_linear_ce_available,
                config={},
            )

        return LigerBackendStatus(
            version=str(installed_ver),
            enabled=True,
            qwen2_patch_available=True,
            fused_linear_ce=True,
            config=dict(TARGET_LIGER_CONFIG),
        )

    except ImportError as exc:
        msg = f"Liger-Kernel not installed. Exactly liger-kernel=={REQUIRED_LIGER_VERSION} is required for V16."
        if strict:
            raise RuntimeError(msg) from exc
        return LigerBackendStatus(
            version="not_installed",
            enabled=False,
            qwen2_patch_available=False,
            fused_linear_ce=False,
            config={},
        )
