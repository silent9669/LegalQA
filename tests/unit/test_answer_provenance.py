"""Inheritance regressions: loop collapse + answer-source provenance."""

from __future__ import annotations

from src.task2.candidates import collapse_loops, generate_candidate_ensemble
from src.task2.predict import LegalQAPipeline


def test_collapse_loops_only_fires_on_three_plus():
    loop = " ".join(["phạt tiền 800.000 đồng"] * 20)
    assert collapse_loops(loop) == "phạt tiền 800.000 đồng"
    twice = "mức phạt cao mức phạt cao"
    assert collapse_loops(twice) == twice
    assert collapse_loops("mức phạt là 800.000 đồng.") == "mức phạt là 800.000 đồng."
    assert collapse_loops("") == ""


def test_ensemble_collapses_generated_loops_not_extracts():
    loop = " ".join(["đồng"] * 30)
    cands = generate_candidate_ensemble(gen_ans=loop, evidence="chứng cứ điều 17")
    assert cands["generated"] == "đồng đồng đồng"  # one block copy kept
    assert "điều 17" in cands["stitched_extract"]


def test_batch_provenance_counts_sources():
    pipe = LegalQAPipeline.build_mock()
    items = [{"id": "q1", "question": "Phạt tiền từ 1.000.000 đồng?"}]
    results, provenance = pipe.predict_batch(items, return_provenance=True)
    assert set(results) == {"q1"}
    assert provenance["num_predictions"] == 1
    assert sum(provenance["counts"].values()) == 1
    assert set(provenance["counts"]) == {"exact", "fuzzy", "generated", "extractive"}


def test_batch_default_return_unchanged():
    pipe = LegalQAPipeline.build_mock()
    results = pipe.predict_batch([{"id": "q1", "question": "Q?"}])
    assert set(results) == {"q1"} and isinstance(results["q1"], dict)


def test_batch_empty_candidate_fallback_guarantees_nonempty_answer():
    pipe = LegalQAPipeline.build_mock()
    # Force selector to return empty string
    pipe.selector.select_with_source = lambda *args, **kwargs: ("", "empty")
    results, provenance = pipe.predict_batch([{"id": "q_empty", "question": "Cau hoi?"}], return_provenance=True)
    assert "q_empty" in results
    ans = results["q_empty"]["answer"]
    assert isinstance(ans, str) and len(ans.strip()) > 0
    assert provenance["sources"]["q_empty"] == "extractive"

