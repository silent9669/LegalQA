"""Deep-normalization memory layer regressions (filler-tolerant exact match)."""

from __future__ import annotations

from src.common.normalize import normalize_question_deep
from src.task2.qa_memory import QAMemory


def test_deep_normalization_strips_fillers():
    assert normalize_question_deep("Cho tôi hỏi mức phạt là gì ạ?") == normalize_question_deep("Mức phạt là gì")
    assert normalize_question_deep("Xin cảm ơn, điều 17?") == normalize_question_deep("Điều 17")
    # Legal content preserved.
    deep = normalize_question_deep("Phạt 800.000 đồng, không miễn trừ?")
    assert "800" in deep and "không" in deep


def test_deep_exact_hit_and_conflict():
    mem = QAMemory.from_records([
        {"qa_id": "q1", "question_raw": "Mức phạt là gì?", "answer_raw": "800.000 đồng."},
    ])
    assert mem.lookup_exact_deep("Cho tôi hỏi mức phạt là gì ạ?") == "800.000 đồng."
    assert mem.lookup_exact_deep("Điều khác?") is None
    conflicted = QAMemory.from_records([
        {"qa_id": "q1", "question_raw": "Mức phạt là gì?", "answer_raw": "800.000 đồng."},
        {"qa_id": "q2", "question_raw": "Cho hỏi mức phạt là gì?", "answer_raw": "1.000.000 đồng."},
    ])
    assert conflicted.lookup_exact_deep("Mức phạt là gì?") is None


def test_deep_layer_respects_group_exclusion(tmp_path):
    mem = QAMemory.from_records([
        {"qa_id": "q1", "question_raw": "Train?", "answer_raw": "A1", "qa_group_id": "g1"},
        {"qa_id": "q2", "question_raw": "Held?", "answer_raw": "A2", "qa_group_id": "g2"},
    ])
    # from_records recomputes groups from norms; exclude q2's group explicitly.
    q2_group = next(r["qa_group_id"] for r in mem.records if r["qa_id"] == "q2")
    filtered = mem.filter_groups({q2_group})
    assert filtered.lookup_exact_deep("Held?") is None
    assert filtered.lookup_exact_deep("Train?") == "A1"


def test_load_backfills_deep_map(tmp_path):
    mem = QAMemory.from_records([
        {"qa_id": "q1", "question_raw": "Mức phạt là gì?", "answer_raw": "800.000 đồng."},
    ])
    assert mem.lookup_exact_deep("Cho tôi hỏi mức phạt là gì ạ?") == "800.000 đồng."
    p = tmp_path / "mem.json"
    mem.save(str(p))
    import json

    data = json.loads(p.read_text(encoding="utf-8"))
    del data["question_map_deep"]  # simulate a pre-deep-layer artifact
    p.write_text(json.dumps(data), encoding="utf-8")
    reloaded = QAMemory.load(str(p))
    assert reloaded.lookup_exact_deep("Cho tôi hỏi mức phạt là gì ạ?") == "800.000 đồng."
