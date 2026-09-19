"""Modal adapter regressions: request chain, test resolution, image pins."""

from __future__ import annotations

import json

import pytest

from scripts import modal_app as modal_app_module
from scripts.modal_app import (
    build_modal_request,
    resolve_test_file,
    validate_modal_request,
)

CAND = {
    "candidate_id": "c" * 16,
    "git_commit_sha": "a" * 40,
    "models": {"dense": {"id": "d", "revision": "b" * 40}},
}


def _parent(stage="colab_t4"):
    return {"status": "PASS", "candidate_sha": "c" * 16, "stage": stage, "report_sha256": "e" * 64}


def test_modal_request_requires_parent_chain():
    with pytest.raises(ValueError, match="no bypass"):
        build_modal_request("micro_probe", CAND, "private-official.json", None)
    with pytest.raises(ValueError, match="no bypass"):
        build_modal_request("full", CAND, "private-official.json", None)
    bad_stage = _parent("kaggle_t4x2")
    with pytest.raises(ValueError, match="stage mismatch"):
        build_modal_request("micro_probe", CAND, "private-official.json", bad_stage)
    bad_cand = _parent("colab_t4")
    bad_cand["candidate_sha"] = "f" * 16
    with pytest.raises(ValueError, match="candidate mismatch"):
        build_modal_request("micro_probe", CAND, "private-official.json", bad_cand)
    req = build_modal_request("micro_probe", CAND, "private-official.json", _parent("colab_t4"))
    assert req["dense_revision"] == "b" * 40
    validate_modal_request(req)
    full = build_modal_request("full", CAND, "private-official.json", _parent("a100_micro_probe"))
    validate_modal_request(full)
    with pytest.raises(ValueError, match="unknown test file"):
        validate_modal_request(dict(full, test_filename="evil.json"))


def test_resolve_test_file_falls_back_and_refuses_unknown(tmp_path):
    (tmp_path / "public-official.json").write_text('{"q1": {"question": "Q", "answer": "A"}}', encoding="utf-8")
    assert resolve_test_file(tmp_path, "public-official.json").name == "public-official.json"
    assert resolve_test_file(tmp_path, "private-official.json").name == "public-official.json"
    with pytest.raises(ValueError, match="unknown test file"):
        resolve_test_file(tmp_path, "evil.json")
    with pytest.raises(FileNotFoundError):
        resolve_test_file(tmp_path / "empty", "private-official.json")


def test_modal_fingerprint_counts_and_hashes(tmp_path):
    payload = {"q1": {"question": "Q?"}, "q2": {"question": "W?"}}
    p = tmp_path / "private-official.json"
    p.write_text(json.dumps(payload), encoding="utf-8")
    fp = modal_app_module.test_file_fingerprint(p)
    assert fp["num_queries"] == 2 and len(fp["sha256"]) == 64
    empty = tmp_path / "empty.json"
    empty.write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="nonempty"):
        modal_app_module.test_file_fingerprint(empty)
