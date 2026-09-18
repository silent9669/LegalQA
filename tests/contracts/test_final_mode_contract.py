"""Task 5: final-mode fallback + inference contract regressions."""

from __future__ import annotations

import pytest

from src.task2.pipeline.contracts import validate_execution_contract, verify_submission_ids


def _final_resources(**over):
    base = {
        "dense_mock": False,
        "dense_fallback": False,
        "generator_fallback": False,
        "mock_generator": False,
        "dense_index_missing": False,
        "bm25_index_missing": False,
        "generator_device": "cuda:0",
        "retrieval_device": "cuda:0",
        "require_adapter": True,
        "adapter_present": True,
        "promotion_validated": True,
        "policy_needs_generator": True,
        "generator_loaded": True,
        "extractive_only": False,
    }
    base.update(over)
    return base


def test_final_rejects_fallback():
    with pytest.raises(ValueError, match="fallback"):
        validate_execution_contract({"final_mode": True}, {"dense_fallback": True}, True)


def test_final_rejects_mock_dense_and_generator():
    with pytest.raises(ValueError, match="mock"):
        validate_execution_contract({"final_mode": True}, _final_resources(dense_mock=True), True)
    with pytest.raises(ValueError, match="mock"):
        validate_execution_contract({"final_mode": True}, _final_resources(mock_generator=True), True)
    with pytest.raises(ValueError, match="fallback"):
        validate_execution_contract({"final_mode": True}, _final_resources(generator_fallback=True), True)


def test_final_rejects_missing_index_and_cpu_substitution():
    with pytest.raises(ValueError, match="index"):
        validate_execution_contract({"final_mode": True}, _final_resources(dense_index_missing=True), True)
    with pytest.raises(ValueError, match="index"):
        validate_execution_contract({"final_mode": True}, _final_resources(bm25_index_missing=True), True)
    with pytest.raises(ValueError, match="CPU"):
        validate_execution_contract({"final_mode": True}, _final_resources(generator_device="cpu"), True)


def test_final_rejects_missing_adapter_and_unvalidated_promotion():
    with pytest.raises(ValueError, match="adapter"):
        validate_execution_contract({"final_mode": True}, _final_resources(adapter_present=False), True)
    with pytest.raises(ValueError, match="promotion"):
        validate_execution_contract({"final_mode": True}, _final_resources(promotion_validated=False), True)
    with pytest.raises(ValueError, match="generator"):
        validate_execution_contract({"final_mode": True}, _final_resources(generator_loaded=False), True)


def test_smoke_allows_mocks_and_final_accepts_clean_contract():
    validate_execution_contract({"final_mode": False}, {"dense_fallback": True, "mock_generator": True}, False)
    validate_execution_contract({"final_mode": True}, _final_resources(), True)


def test_verify_submission_ids_requires_exact_1000():
    expected = [f"q{i}" for i in range(1000)]
    submission = {qid: {"answer": "Tra loi."} for qid in expected}
    report = verify_submission_ids(expected, submission)
    assert report["num_predictions"] == 1000
    bad = dict(submission)
    del bad["q0"]
    with pytest.raises(ValueError, match="ID"):
        verify_submission_ids(expected, bad)
