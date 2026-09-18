"""Task 3: retrieval identity + chunk-order repair regressions."""

from __future__ import annotations

import pytest

from src.common.dense import (
    build_embedding_row_keys,
    compute_embedding_order_hash,
)
from src.task2.dataset.retrieval_manifest import (
    aggregate_positive_labels,
    build_retrieval_manifest,
    report_label_coverage,
)
from src.task2.generation.dataset import build_chunk_rows


def test_repeated_chunk_ids_keep_order_and_text():
    rows = [
        {"chunk_id": "c", "text_raw": "first"},
        {"chunk_id": "c", "text_raw": "second"},
    ]
    kept, _ = build_chunk_rows(rows, {"c"})
    assert [x["text_raw"] for x in kept] == ["first", "second"]


def test_chunk_rows_filter_without_collapsing():
    rows = [
        {"chunk_id": "a", "text_raw": "A1"},
        {"chunk_id": "b", "text_raw": "B1"},
        {"chunk_id": "a", "text_raw": "A2"},
        {"chunk_id": "c", "text_raw": "C1"},
    ]
    kept, report = build_chunk_rows(rows, {"a", "b"})
    assert [(r["chunk_id"], r["text_raw"]) for r in kept] == [("a", "A1"), ("b", "B1"), ("a", "A2")]
    assert report["kept_rows"] == 3
    assert report["extra_rows_from_repeated_ids"] == 1
    assert report["missing_ids"] == []


def test_aggregate_every_positive_label_row():
    labels = [
        {"qa_id": "q1", "positive_chunk_id": "c1"},
        {"qa_id": "q1", "positive_chunk_id": "c2"},
        {"qa_id": "q1", "positive_chunk_id": "c1"},
        {"qa_id": "q2", "positive_chunk_id": "c3"},
    ]
    agg = aggregate_positive_labels(labels)
    assert agg == {"q1": ["c1", "c2"], "q2": ["c3"]}


def test_reject_permuted_embeddings_even_at_equal_shape():
    chunk_rows = [
        {"chunk_id": "a", "text_raw": "Alpha"},
        {"chunk_id": "a", "text_raw": "Beta"},
        {"chunk_id": "b", "text_raw": "Gamma"},
    ]
    ordered = build_embedding_row_keys(chunk_rows)
    manifest = build_retrieval_manifest(chunk_rows, ordered, [])
    assert manifest["num_physical_rows"] == 3
    assert manifest["extra_rows_from_repeated_ids"] == 1
    assert manifest["alias_ambiguous_ids"] == ["a"]
    # Same shape, permuted order must fail.
    permuted = [ordered[1], ordered[0], ordered[2]]
    assert compute_embedding_order_hash(permuted) != manifest["embedding_order_hash"]
    with pytest.raises(ValueError, match="order/content mismatch"):
        build_retrieval_manifest(chunk_rows, permuted, [])
    # Same ids, altered text must fail.
    tampered_rows = [
        {"chunk_id": "a", "text_raw": "Alpha"},
        {"chunk_id": "a", "text_raw": "Beta!"},
        {"chunk_id": "b", "text_raw": "Gamma"},
    ]
    with pytest.raises(ValueError, match="order/content mismatch"):
        build_retrieval_manifest(tampered_rows, ordered, [])


def test_missing_label_coverage_without_deleting_unlabeled():
    qa_ids = ["q1", "q2", "q3"]
    labels = [{"qa_id": "q1", "positive_chunk_id": "c1"}]
    coverage = report_label_coverage(qa_ids, labels)
    assert coverage["num_labeled_qa_ids"] == 1
    assert coverage["unlabeled_qa_ids"] == ["q2", "q3"]
    manifest = build_retrieval_manifest(
        [{"chunk_id": "c1", "text_raw": "E1"}],
        build_embedding_row_keys([{"chunk_id": "c1", "text_raw": "E1"}]),
        labels,
    )
    assert manifest["num_labeled_qa_ids"] == 1
    assert manifest["positive_labels_by_qa"] == {"q1": ["c1"]}
