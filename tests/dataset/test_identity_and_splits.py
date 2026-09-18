"""Task 2: canonical QA identity + split lock regressions."""

from __future__ import annotations

import pytest

from src.task2.dataset.identity import canonicalize_qa_rows
from src.task2.dataset.splits import assign_group_splits, split_fingerprint
from src.task2.dataset.validator import audit_fold_group_isolation, validate_canonical_splits
from src.task2.qa_memory import QAMemory


def test_repeated_id_is_one_eligible_example():
    rows = [
        {"qa_id": "q1", "question_raw": " What? ", "answer_raw": "Yes"},
        {"qa_id": "q1", "question_raw": "what", "answer_raw": "Yes"},
    ]
    out, report = canonicalize_qa_rows(rows)
    assert len(out) == 1 and report["duplicate_rows_removed"] == 1


def test_conflicting_answers_fail_closed():
    rows = [
        {"qa_id": "q1", "question_raw": "What is the fine?", "answer_raw": "800.000 dong"},
        {"qa_id": "q2", "question_raw": "what is the fine", "answer_raw": "1.000.000 dong"},
    ]
    with pytest.raises(ValueError, match="conflicting answers"):
        canonicalize_qa_rows(rows, conflict_policy="reject")
    out, report = canonicalize_qa_rows(rows, conflict_policy="quarantine")
    assert out == [] and report["quarantined_row_count"] == 2


def test_identical_variants_share_group_and_preserve_legal_tokens():
    rows = [
        {"qa_id": "q1", "question_raw": "Muc phat 800.000 dong theo Dieu 5?", "answer_raw": "Phat 800.000 dong, khong ap dung ngoai le."},
        {"qa_id": "q2", "question_raw": "muc phat 800 000 dong theo dieu 5", "answer_raw": "Phat 800.000 dong, khong ap dung ngoai le."},
    ]
    out, report = canonicalize_qa_rows(rows)
    assert len(out) == 1
    assert out[0]["legacy_qa_ids"] == ["q1", "q2"]
    assert "800" in out[0]["answer_raw"] and "khong" in out[0]["answer_raw"]
    # Negation / amount variants must NOT be merged silently:
    rows2 = [
        {"qa_id": "q1", "question_raw": "Ap dung ngoai le?", "answer_raw": "Co ap dung"},
        {"qa_id": "q2", "question_raw": "Ap dung ngoai le?", "answer_raw": "Khong ap dung"},
    ]
    with pytest.raises(ValueError, match="conflicting answers"):
        canonicalize_qa_rows(rows2)


def test_safe_folds_retained_and_cross_fold_blocked():
    rows = [
        {"qa_id": "a", "question_raw": "Q one?", "answer_raw": "A1"},
        {"qa_id": "b", "question_raw": "Q two?", "answer_raw": "A2"},
        {"qa_id": "c", "question_raw": "Q three?", "answer_raw": "A3"},
    ]
    canon, _ = canonicalize_qa_rows(rows)
    fold_by_id = {"a": 2, "b": 1, "c": 0}
    for row in canon:
        row["fold_id"] = fold_by_id[row["qa_id"]]
    audit = audit_fold_group_isolation(
        [r["qa_id"] for r in canon], [r["question_norm"] for r in canon], [r["fold_id"] for r in canon]
    )
    assert audit["has_leakage"] is False
    split_rows, report = assign_group_splits(canon, seed=42)
    assert report["retained_existing_folds"] is True
    assert [r["split"] for r in sorted(split_rows, key=lambda r: r["qa_id"])] == ["train", "dev", "lockbox"]
    validate_canonical_splits(split_rows)

    # Cross-fold duplicate group blocks retention.
    dup = [
        {"qa_id": "a", "question_raw": "Same q?", "answer_raw": "Same a", "qa_group_id": "g1", "fold_id": 2},
        {"qa_id": "b", "question_raw": "Same q?", "answer_raw": "Same a", "qa_group_id": "g1", "fold_id": 1},
    ]
    with pytest.raises(ValueError, match="cross-fold"):
        assign_group_splits(dup, seed=42)


def test_fresh_assignment_is_group_isolated_and_deterministic():
    canon, _ = canonicalize_qa_rows(
        [{"qa_id": f"q{i}", "question_raw": f"Question number {i}?", "answer_raw": f"Answer {i}"} for i in range(30)]
    )
    first, rep1 = assign_group_splits(canon, seed=42)
    second, rep2 = assign_group_splits(canon, seed=42)
    assert rep1["split_fingerprint"] == rep2["split_fingerprint"] == split_fingerprint(first)
    validate_canonical_splits(first)
    assert set(rep1["examples_by_split"]) == {"train", "dev", "lockbox"}


def test_heldout_groups_excluded_from_memory_and_exemplars():
    canon, _ = canonicalize_qa_rows(
        [
            {"qa_id": "q1", "question_raw": "Train question?", "answer_raw": "Train answer"},
            {"qa_id": "q2", "question_raw": "Heldout question?", "answer_raw": "Heldout answer"},
        ]
    )
    split_rows, _ = assign_group_splits(
        [{**r, "fold_id": (1 if r["qa_id"] == "q2" else 2)} for r in canon], seed=42
    )
    heldout = {r["qa_group_id"] for r in split_rows if r["split"] != "train"}
    assert heldout
    mem = QAMemory.from_records(
        [
            {"qa_id": r["qa_id"], "question_raw": r["question_raw"], "answer_raw": r["answer_raw"],
             "qa_group_id": r["qa_group_id"]}
            for r in canon
        ]
    )
    filtered = mem.filter_groups(heldout)
    assert filtered.lookup_exact("q2", "Heldout question?") is None
    assert filtered.lookup_exact("q1", "Train question?") == "Train answer"
    assert filtered.lookup_fuzzy("Heldout question?", threshold=0.5) is None or \
        filtered.lookup_fuzzy("Heldout question?", threshold=0.5)["matched_qa_id"] != "q2"
