"""Tests for LegalQA V12 baseline-aware pip conflict collection and regression guard."""

from __future__ import annotations

import pytest
import scripts.bootstrap_kaggle_env as bootstrap


def test_same_preexisting_pip_conflicts_are_allowed():
    before = [
        "google-colab 1.0.0 has requirement pandas==2.2.2, but you have pandas 2.3.3.",
        "moviepy 1.0.3 has requirement decorator<5.0,>=4.0.2, but you have decorator 5.3.1.",
    ]
    after = list(reversed(before))

    new = bootstrap.assert_no_new_pip_conflicts(before, after)
    assert new == []


def test_new_pip_conflict_fails_loud():
    before = ["moviepy baseline conflict"]
    after = before + ["trl new conflict"]

    with pytest.raises(RuntimeError, match="new dependency conflict"):
        bootstrap.assert_no_new_pip_conflicts(before, after)


def test_resolved_baseline_conflict_is_allowed():
    before = ["old preexisting conflict"]
    after = []
    assert bootstrap.assert_no_new_pip_conflicts(before, after) == []


def test_bootstrap_allows_unchanged_base_image_conflicts(monkeypatch):
    snapshots = iter([
        ["preexisting kaggle conflict"],
        ["preexisting kaggle conflict"],
    ])

    monkeypatch.setattr(
        bootstrap,
        "collect_pip_check_conflicts",
        lambda: next(snapshots),
    )
    monkeypatch.setattr(bootstrap, "TARGET_USER_PACKAGES", [])
    monkeypatch.setattr(bootstrap, "snapshot_protected_versions", lambda: {})

    result = bootstrap.bootstrap_dependencies(
        allow_unprotected_drift=True,
    )

    assert result["pip_check_baseline_conflicts"] == [
        "preexisting kaggle conflict"
    ]
    assert result["pip_check_post_conflicts"] == [
        "preexisting kaggle conflict"
    ]
    assert result["pip_check_new_conflicts"] == []
    assert result["pip_check_regression_passed"] is True


def test_bootstrap_raises_on_new_pip_conflict(monkeypatch):
    snapshots = iter([
        ["preexisting kaggle conflict"],
        ["preexisting kaggle conflict", "new broken dependency"],
    ])

    monkeypatch.setattr(
        bootstrap,
        "collect_pip_check_conflicts",
        lambda: next(snapshots),
    )
    monkeypatch.setattr(bootstrap, "TARGET_USER_PACKAGES", [])
    monkeypatch.setattr(bootstrap, "snapshot_protected_versions", lambda: {})

    with pytest.raises(RuntimeError, match="new dependency conflict"):
        bootstrap.bootstrap_dependencies(allow_unprotected_drift=True)


def test_verify_target_package_versions_satisfies_floors(monkeypatch):
    monkeypatch.setattr(
        bootstrap,
        "get_installed_distribution_version",
        lambda name: "1.0.0",
    )
    monkeypatch.setattr(
        bootstrap,
        "satisfies_spec",
        lambda ver, spec: True,
    )
    res = bootstrap.verify_target_package_versions()
    assert len(res) == len(bootstrap.TARGET_USER_PACKAGES)


def test_verify_target_package_versions_fails_on_unsatisfied_floor(monkeypatch):
    monkeypatch.setattr(
        bootstrap,
        "get_installed_distribution_version",
        lambda name: "0.1.0" if name == "trl" else "1.0.0",
    )
    monkeypatch.setattr(
        bootstrap,
        "satisfies_spec",
        lambda ver, spec: ver == "1.0.0",
    )
    with pytest.raises(RuntimeError, match="LegalQA dependency floor verification failed"):
        bootstrap.verify_target_package_versions()


def test_save_bootstrap_manifest_truthful_provenance(tmp_path):
    import json
    out = tmp_path / "env_manifest.json"
    bootstrap_result = {
        "protected_before": {"torch": "2.10.0"},
        "protected_after": {"torch": "2.10.0"},
        "installed_or_updated": ["trl>=0.17.0"],
        "pip_check_baseline_conflicts": ["preexisting kaggle conflict"],
        "pip_check_post_conflicts": ["preexisting kaggle conflict"],
        "pip_check_new_conflicts": [],
        "pip_check_regression_passed": True,
        "verified_target_versions": {"trl": "0.17.0"},
    }

    bootstrap.save_bootstrap_manifest(output_path=str(out), bootstrap_result=bootstrap_result)
    assert out.exists()
    data = json.loads(out.read_text(encoding="utf-8"))

    assert data["pip_check_regression_passed"] is True
    assert data["pip_check_baseline_conflicts"] == ["preexisting kaggle conflict"]
    assert data["pip_check_post_conflicts"] == ["preexisting kaggle conflict"]
    assert data["pip_check_new_conflicts"] == []
    assert data["protected_runtime_unchanged"] is True
    assert data["pip_check_clean"] is False  # Baseline conflict exists, so not globally clean
    assert "HF_TOKEN" not in data
    assert "HF_TOKEN" not in json.dumps(data)


