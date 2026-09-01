"""Tests for LegalQA V8/V9 fail-loud integration, packaged runtime identity, and strict environment."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any, Dict

import pytest

import scripts.bootstrap_kaggle_env as bootstrap
from src.task2.path_resolver import resolve_runtime_paths
from src.task2.runtime_integrity import (
    EXPECTED_RUNTIME_API_VERSION,
    find_packaged_code_roots,
    resolve_packaged_code_root,
    validate_runtime_manifests,
)

VALID_SHA_A = "a" * 40
VALID_SHA_B = "b" * 40


def test_missing_code_manifest_is_fatal(tmp_path):
    """Task 1: Verify missing code_manifest.json raises RuntimeError."""
    runtime = tmp_path / "runtime"
    code = runtime / "code" / "LegalQA"
    code.mkdir(parents=True)
    (runtime / "dataset_manifest.json").write_text(
        json.dumps({"runtime_api_version": 12, "git_sha": VALID_SHA_A})
    )
    with pytest.raises(RuntimeError, match="code_manifest.json"):
        validate_runtime_manifests(str(runtime), str(code), expected_api_version=12)


def test_missing_dataset_manifest_is_fatal(tmp_path):
    """Task 1: Verify missing dataset_manifest.json raises RuntimeError."""
    runtime = tmp_path / "runtime"
    code = runtime / "code" / "LegalQA"
    code.mkdir(parents=True)
    (code / "code_manifest.json").write_text(
        json.dumps({"runtime_api_version": 12, "git_sha": VALID_SHA_A})
    )
    with pytest.raises(RuntimeError, match="dataset_manifest.json"):
        validate_runtime_manifests(str(runtime), str(code), expected_api_version=12)


@pytest.mark.parametrize("version", [7, 8, 9, 10, 11, 13])
def test_runtime_api_must_equal_expected(tmp_path, version):
    """Task 1 & 2: Verify runtime API mismatch between expected and actual is fatal."""
    runtime = tmp_path / "runtime"
    code = runtime / "code" / "LegalQA"
    code.mkdir(parents=True)
    (runtime / "dataset_manifest.json").write_text(
        json.dumps({"runtime_api_version": version, "git_sha": VALID_SHA_A})
    )
    (code / "code_manifest.json").write_text(
        json.dumps({"runtime_api_version": version, "git_sha": VALID_SHA_A})
    )
    with pytest.raises(RuntimeError, match="runtime_api_version mismatch"):
        validate_runtime_manifests(str(runtime), str(code), expected_api_version=12)


def test_dataset_code_git_sha_must_match(tmp_path):
    """Task 1 & 2: Verify git_sha mismatch between dataset and code is fatal."""
    runtime = tmp_path / "runtime"
    code = runtime / "code" / "LegalQA"
    code.mkdir(parents=True)
    (runtime / "dataset_manifest.json").write_text(
        json.dumps({"runtime_api_version": 12, "git_sha": VALID_SHA_A})
    )
    (code / "code_manifest.json").write_text(
        json.dumps({"runtime_api_version": 12, "git_sha": VALID_SHA_B})
    )
    with pytest.raises(RuntimeError, match="Git SHA divergence"):
        validate_runtime_manifests(str(runtime), str(code), expected_api_version=12)


@pytest.mark.parametrize("bad_sha", [None, "", "unknown", "abc", "g" * 40])
def test_runtime_manifest_requires_real_git_sha(tmp_path, bad_sha):
    """Task 2: Verify non-40-character or non-hex Git SHA is rejected."""
    runtime = tmp_path / "runtime"
    code = runtime / "code" / "LegalQA"
    code.mkdir(parents=True)
    (runtime / "dataset_manifest.json").write_text(
        json.dumps({"runtime_api_version": 12, "git_sha": bad_sha})
    )
    (code / "code_manifest.json").write_text(
        json.dumps({"runtime_api_version": 12, "git_sha": bad_sha})
    )
    with pytest.raises(RuntimeError, match="40-character"):
        validate_runtime_manifests(str(runtime), str(code), expected_api_version=12)


def test_ambiguous_packaged_code_roots_fail(tmp_path):
    """Task 1: Verify resolve_packaged_code_root raises when multiple code roots are found."""
    for name in ("dataset_a", "dataset_b"):
        root = tmp_path / name / "code" / "LegalQA"
        (root / "src").mkdir(parents=True)
        (root / "scripts").mkdir()

    with pytest.raises(RuntimeError, match="Ambiguous packaged LegalQA code roots"):
        resolve_packaged_code_root(str(tmp_path), strict=True)


def test_strict_runtime_requires_mounted_qwen(tmp_path):
    """Task 3: Verify strict runtime path resolution raises if mounted Qwen is absent and remote download disabled."""
    ds_root = tmp_path / "dataset"
    ds_root.mkdir()
    (ds_root / "dataset_manifest.json").write_text("{}")
    (ds_root / "legal_chunks.parquet").write_text("dummy")

    with pytest.raises(RuntimeError, match="Qwen"):
        resolve_runtime_paths(
            str(tmp_path),
            strict=True,
            allow_remote_model_download=False,
        )


def test_bootstrap_fails_loud_on_new_pip_conflicts(monkeypatch):
    """Task 4 & V12: Verify bootstrap raises when new pip conflicts are introduced."""
    snapshots = iter([
        ["baseline conflict"],
        ["baseline conflict", "newly introduced conflict"],
    ])
    monkeypatch.setattr(bootstrap, "collect_pip_check_conflicts", lambda: next(snapshots))
    monkeypatch.setattr(bootstrap, "TARGET_USER_PACKAGES", [])
    monkeypatch.setattr(bootstrap, "snapshot_protected_versions", lambda: {})

    with pytest.raises(RuntimeError, match="LegalQA bootstrap introduced new dependency conflict"):
        bootstrap.bootstrap_dependencies(allow_unprotected_drift=True)


def test_bootstrap_allows_unchanged_baseline_pip_conflicts(monkeypatch):
    """Task 4 & V12: Verify bootstrap permits unchanged pre-existing baseline conflicts."""
    snapshots = iter([
        ["preexisting kaggle conflict"],
        ["preexisting kaggle conflict"],
    ])
    monkeypatch.setattr(bootstrap, "collect_pip_check_conflicts", lambda: next(snapshots))
    monkeypatch.setattr(bootstrap, "TARGET_USER_PACKAGES", [])
    monkeypatch.setattr(bootstrap, "snapshot_protected_versions", lambda: {})

    res = bootstrap.bootstrap_dependencies(allow_unprotected_drift=True)
    assert res["pip_check_regression_passed"] is True
    assert res["pip_check_new_conflicts"] == []


def test_required_import_failure_is_fatal(monkeypatch):
    """Task 4: Verify verify_runtime_imports raises when any required package import fails."""
    real_import = bootstrap.importlib.import_module

    def fake_import(name, *args, **kwargs):
        if name == "bitsandbytes":
            raise ImportError("simulated bitsandbytes import failure")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(bootstrap.importlib, "import_module", fake_import)

    with pytest.raises(RuntimeError, match="Required runtime import verification failed"):
        bootstrap.verify_runtime_imports(strict=True)


def test_requirements_kaggle_aligned():
    """Task 5: Verify requirements-kaggle.txt has trl>=0.17.0 and does not specify torch."""
    req_text = Path("requirements-kaggle.txt").read_text()
    assert "trl>=0.17.0" in req_text
    # Torch should not be an unconstrained requirement
    for line in req_text.splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            assert not line.startswith("torch>="), "torch must not be listed in requirements-kaggle.txt"


def notebook_source():
    nb = json.loads(Path("kaggle_kernel/legalqa_gpu_pipeline.ipynb").read_text())
    return "\n".join(
        "".join(c.get("source", "")) if isinstance(c.get("source"), list) else str(c.get("source", ""))
        for c in nb["cells"]
        if c.get("cell_type") == "code"
    )


def test_notebook_uses_strict_runtime_resolution():
    """Task 2 & 10: Verify notebook source code uses strict resolution and manifest verification."""
    src = notebook_source()
    assert "resolve_packaged_code_root" in src
    assert "validate_runtime_manifests" in src
    assert "resolve_runtime_paths(" in src
    assert "allow_remote_model_download=False" in src
    assert "REQUIRED_RUNTIME_API_VERSION = 12" in src
    assert 'resolve_runtime_paths("/kaggle/input", strict=False' not in src
