"""Task 6: offline/official scorer + submission receipt separation regressions."""

from __future__ import annotations

import importlib.util
import json

import pytest

from src.task2.scorer_contract import (
    build_offline_metric_report,
    load_predictions_json,
    score_labelled_rows,
    validate_prediction_payload,
    verify_zip_inner_matches_loose,
)


def test_duplicate_or_missing_id_fails():
    with pytest.raises(ValueError, match="ID"):
        validate_prediction_payload({"q1": {"answer": "a"}}, ["q1", "q2"])


def test_duplicate_raw_json_keys_rejected(tmp_path):
    payload = '{"q1": {"answer": "a"}, "q1": {"answer": "b"}}'
    p = tmp_path / "pred.json"
    p.write_text(payload, encoding="utf-8")
    with pytest.raises(ValueError, match="duplicate"):
        load_predictions_json(p)


def test_empty_answers_and_wrong_values_fail():
    with pytest.raises(ValueError, match="empty answers"):
        validate_prediction_payload({"q1": {"answer": "   "}}, ["q1"])
    with pytest.raises(ValueError, match="answer"):
        validate_prediction_payload({"q1": "plain string"}, ["q1"])
    with pytest.raises(ValueError, match="ID"):
        validate_prediction_payload({"q1": {"answer": "a"}, "qX": {"answer": "b"}}, ["q1"])


def test_length_mismatch_and_empty_scoring_fail():
    with pytest.raises(ValueError, match="length mismatch"):
        score_labelled_rows(["ref"], ["a", "b"])
    with pytest.raises(ValueError, match="at least one"):
        score_labelled_rows([], [])


def test_vietnamese_parity_with_unchanged_official_scorer():
    spec = importlib.util.spec_from_file_location(
        "official_scoring", "Scoring-Program-Task-LegalQA/scoring.py"
    )
    official = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(official)
    refs = [
        "Phat tien 800.000 dong theo Dieu 5, khoan 2.",
        "Khong ap dung ngoai le tru truong hop bat kha khang!",
        "Ngay 12/03/2021, muc phat tang gap doi???",
    ]
    preds = [
        "Phat 800.000 dong theo Dieu 5 khoan 2.",
        "Khong ap dung ngoai le, tru bat kha khang.",
        "Ngay 12/03/2021 muc phat tang gap doi.",
    ]
    ours = score_labelled_rows(refs, preds)
    y_pred = {f"q{i}": {"answer": p} for i, p in enumerate(preds)}
    y_true = {f"q{i}": r for i, r in enumerate(refs)}
    theirs = official.eval_qa(y_pred, y_true)
    assert ours["meteor"] == pytest.approx(float(theirs["meteor"]), abs=1e-9)
    assert ours["rouge"] == pytest.approx(float(theirs["rouge"]), abs=1e-9)


def test_zip_inner_bytes_must_match_loose(tmp_path):
    import zipfile

    submission = {"q1": {"answer": "a"}, "q2": {"answer": "b"}}
    loose = tmp_path / "submission.json"
    loose.write_text(json.dumps(submission, ensure_ascii=False, indent=2), encoding="utf-8")
    good_zip = tmp_path / "good.zip"
    with zipfile.ZipFile(good_zip, "w", zipfile.ZIP_DEFLATED) as z:
        z.write(loose, arcname="submission.json")
    report = verify_zip_inner_matches_loose(good_zip, loose)
    assert report["loose_sha256"] == report["inner_sha256"]
    assert len(report["zip_sha256"]) == 64
    bad_zip = tmp_path / "bad.zip"
    with zipfile.ZipFile(bad_zip, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("submission.json", '{"q1": {"answer": "different"}}')
    with pytest.raises(ValueError, match="differ"):
        verify_zip_inner_matches_loose(bad_zip, loose)


def test_offline_report_keeps_public_score_null():
    report = build_offline_metric_report({"meteor": 0.5, "rouge": 0.4}, "a" * 64)
    assert report["official_public_score"] is None
    assert report["scorer_sha256"] == "a" * 64
    with pytest.raises(ValueError, match="scorer_sha"):
        build_offline_metric_report({}, "short")
