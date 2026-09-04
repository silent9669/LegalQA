"""Tests for LegalQA V11 dependency contract and TRL API guarantees."""

from __future__ import annotations

import inspect
from pathlib import Path
import sys
from unittest.mock import MagicMock, patch
import pytest

from scripts.bootstrap_kaggle_env import (
    TARGET_USER_PACKAGES,
    check_trl_api_signatures,
    verify_runtime_imports,
)


def test_trl_floor_guarantees_modern_sft_api():
    req = Path("requirements-kaggle.txt").read_text(encoding="utf-8")
    assert "trl==1.12.0" in req

    target = {
        pip_name: spec
        for _, spec, pip_name in TARGET_USER_PACKAGES
    }
    assert target["trl"] == "==1.12.0"


def test_check_trl_api_signatures_mock_legacy_and_modern():
    # Modern signature (e.g. TRL 1.12.0)
    class ModernSFTConfig:
        def __init__(self, completion_only_loss=True, loss_type="chunked_nll", activation_offloading=True, max_length=2048, **kwargs):
            pass

    class ModernSFTTrainer:
        def __init__(self, processing_class=None, **kwargs):
            pass

    missing = check_trl_api_signatures(ModernSFTConfig, ModernSFTTrainer)
    assert missing == [], f"Expected no missing APIs for modern signatures, got {missing}"

    # Also test alternative max_seq_length parameter name
    class AltModernSFTConfig:
        def __init__(self, completion_only_loss=True, loss_type="chunked_nll", activation_offloading=True, max_seq_length=2048, **kwargs):
            pass

    missing_alt = check_trl_api_signatures(AltModernSFTConfig, ModernSFTTrainer)
    assert missing_alt == []

    # Legacy signature (missing completion_only_loss)
    class LegacySFTConfigNoCompLoss:
        def __init__(self, loss_type="chunked_nll", activation_offloading=True, max_seq_length=2048, **kwargs):
            pass

    missing_legacy1 = check_trl_api_signatures(LegacySFTConfigNoCompLoss, ModernSFTTrainer)
    assert "SFTConfig.completion_only_loss" in missing_legacy1

    # Legacy signature (missing activation_offloading)
    class LegacySFTConfigNoOffloading:
        def __init__(self, completion_only_loss=True, loss_type="chunked_nll", max_length=2048, **kwargs):
            pass

    missing_legacy_offload = check_trl_api_signatures(LegacySFTConfigNoOffloading, ModernSFTTrainer)
    assert "SFTConfig.activation_offloading" in missing_legacy_offload

    # Legacy signature (missing loss_type)
    class LegacySFTConfigNoLossType:
        def __init__(self, completion_only_loss=True, activation_offloading=True, max_length=2048, **kwargs):
            pass

    missing_legacy_loss_type = check_trl_api_signatures(LegacySFTConfigNoLossType, ModernSFTTrainer)
    assert "SFTConfig.loss_type" in missing_legacy_loss_type

    # Legacy signature (missing tokenizer/processing_class on trainer, e.g. accepts only tokenizer keyword)
    class LegacySFTTrainerNoProcessingClass:
        def __init__(self, tokenizer=None, **kwargs):
            pass

    missing_legacy2 = check_trl_api_signatures(ModernSFTConfig, LegacySFTTrainerNoProcessingClass)
    assert "SFTTrainer.processing_class" in missing_legacy2

    # Completely legacy (TRL 0.11 era)
    class Legacy011SFTConfig:
        def __init__(self, dataset_text_field=None, **kwargs):
            pass

    missing_legacy_all = check_trl_api_signatures(Legacy011SFTConfig, LegacySFTTrainerNoProcessingClass)
    assert "SFTConfig.completion_only_loss" in missing_legacy_all
    assert "SFTConfig.loss_type" in missing_legacy_all
    assert "SFTConfig.activation_offloading" in missing_legacy_all
    assert "SFTConfig.max_length/max_seq_length" in missing_legacy_all
    assert "SFTTrainer.processing_class" in missing_legacy_all


def test_verify_runtime_imports_fails_loud_on_missing_trl_api():
    class LegacySFTConfig:
        def __init__(self, **kwargs):
            pass

    class LegacySFTTrainer:
        def __init__(self, **kwargs):
            pass

    mock_trl = MagicMock()
    mock_trl.SFTConfig = LegacySFTConfig
    mock_trl.SFTTrainer = LegacySFTTrainer

    def fake_import(name, *args, **kwargs):
        if name == "trl":
            return mock_trl
        m = MagicMock()
        m.__version__ = "1.0.0"
        return m

    with patch("scripts.bootstrap_kaggle_env.importlib.import_module", side_effect=fake_import):
        with patch.dict(sys.modules, {"trl": mock_trl}):
            with pytest.raises(RuntimeError, match="Installed TRL lacks LegalQA-required SFT APIs"):
                verify_runtime_imports(strict=True)


import json
from src.task2.runtime_integrity import EXPECTED_RUNTIME_API_VERSION, validate_runtime_manifests

VALID_TEST_SHA = "0123456789abcdef0123456789abcdef01234567"


def test_stale_v15_package_rejected_by_v16_validator(tmp_path: Path):
    """Verify that a package with runtime_api_version=15 is rejected when expected_api_version=16."""
    runtime = tmp_path / "runtime"
    code = runtime / "code" / "LegalQA"
    code.mkdir(parents=True)
    (runtime / "dataset_manifest.json").write_text(
        json.dumps({"runtime_api_version": 15, "git_sha": VALID_TEST_SHA})
    )
    (code / "code_manifest.json").write_text(
        json.dumps({"runtime_api_version": 15, "git_sha": VALID_TEST_SHA})
    )

    with pytest.raises(RuntimeError, match="runtime_api_version mismatch: found 15, expected 16"):
        validate_runtime_manifests(str(runtime), str(code), expected_api_version=16)


def test_fresh_v16_package_passes_v16_validator(tmp_path: Path):
    """Verify that a fresh package with runtime_api_version=16 passes validation."""
    assert EXPECTED_RUNTIME_API_VERSION == 16
    runtime = tmp_path / "runtime"
    code = runtime / "code" / "LegalQA"
    code.mkdir(parents=True)
    (runtime / "dataset_manifest.json").write_text(
        json.dumps({"runtime_api_version": 16, "git_sha": VALID_TEST_SHA})
    )
    (code / "code_manifest.json").write_text(
        json.dumps({"runtime_api_version": 16, "git_sha": VALID_TEST_SHA})
    )

    provenance = validate_runtime_manifests(str(runtime), str(code), expected_api_version=16)
    assert provenance["runtime_api_version"] == 16
    assert provenance["git_sha"] == VALID_TEST_SHA


def test_installed_trl_exposes_required_legalqa_api():
    """Live check when trl is installed in the test environment (e.g. CI or Kaggle)."""
    try:
        from trl import SFTConfig, SFTTrainer
    except ImportError:
        pytest.skip("TRL not installed in this environment; skipping live signature check.")

    missing = check_trl_api_signatures(SFTConfig, SFTTrainer)
    assert missing == [], f"Installed TRL is missing required LegalQA APIs: {missing}"
