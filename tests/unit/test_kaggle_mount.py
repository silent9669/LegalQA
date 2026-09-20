"""Mount-resolution regressions (simulated nested/lazy Kaggle layouts)."""

from __future__ import annotations

import pytest

from src.task2.kaggle_mount import find_candidate_manifest, find_mounted_file


def test_finds_nested_layout(tmp_path, monkeypatch):
    nested = tmp_path / "input" / "datasets" / "phucdangg" / "legalqa-candidate"
    nested.mkdir(parents=True)
    target = nested / "candidate_manifest.json"
    target.write_text("{}", encoding="utf-8")
    monkeypatch.setattr("src.task2.kaggle_mount.time.sleep", lambda s: None)
    found = find_mounted_file("candidate_manifest.json", roots=[str(tmp_path / "input")],
                              timeout_seconds=5, poll_interval=0.01)
    assert found == target


def test_missing_file_lists_visible_dirs_and_raises(tmp_path):
    (tmp_path / "input").mkdir()
    with pytest.raises(FileNotFoundError, match="visible dirs"):
        find_mounted_file("nope.json", roots=[str(tmp_path / "input")],
                          timeout_seconds=0.05, poll_interval=0.01)


def test_candidate_helper_uses_default_root(monkeypatch):
    import src.task2.kaggle_mount as km

    seen = {}

    def fake(filename, roots=None, timeout_seconds=180.0, poll_interval=5.0):
        seen["roots"] = roots
        raise FileNotFoundError("none")

    monkeypatch.setattr(km, "find_mounted_file", fake)
    with pytest.raises(FileNotFoundError):
        find_candidate_manifest(timeout_seconds=1)
    assert seen["roots"] is None
