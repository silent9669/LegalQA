"""Retrieval manifest: ordered row identity, label aggregation, coverage.

Binds dense/BM25 indexes to the exact ordered (row_id, text_hash) map so a
permuted or collapsed embedding array cannot pass as equivalent.
"""

from __future__ import annotations

import hashlib
from collections import defaultdict
from typing import Any, Dict, List, Tuple


def _text_hash(text: str) -> str:
    return hashlib.sha256(str(text).encode("utf-8")).hexdigest()


def aggregate_positive_labels(labels: List[Dict[str, Any]]) -> Dict[str, List[str]]:
    """Aggregate EVERY positive label row per QA id, order-preserving.

    Accepts dict rows with qa_id/positive_chunk_id keys. Identical
    (qa, chunk) pairs collapse; distinct chunk ids for one QA are all kept
    in first-seen order. Empty ids are skipped and counted in the report
    only by build_retrieval_manifest.
    """
    aggregated: Dict[str, List[str]] = {}
    for row in labels:
        qid = str(row.get("qa_id", "") or "").strip()
        cid = str(row.get("positive_chunk_id", "") or "").strip()
        if not qid or not cid:
            continue
        bucket = aggregated.setdefault(qid, [])
        if cid not in bucket:
            bucket.append(cid)
    return aggregated


def build_retrieval_manifest(
    chunk_rows: List[Dict[str, Any]],
    embedding_row_ids: List[str],
    labels: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """Build the retrieval identity manifest.

    chunk_rows: ordered physical rows with chunk_id/text_raw (duplicates kept).
    embedding_row_ids: ordered row identities the embedding array aligns to,
      as "row_index:chunk_id:text_hash" strings.
    labels: raw label rows aggregated via aggregate_positive_labels.

    The manifest rejects order/content mismatches: the caller must pass the
    exact embedding order; any permutation, drop, or text change alters the
    order hash. Returns a dict with order hashes, alias ambiguity, and label
    coverage (missing labels are reported, never an error).
    """
    ordered_keys = []
    texts_by_id: Dict[str, set] = defaultdict(set)
    for index, row in enumerate(chunk_rows):
        cid = str(row.get("chunk_id", "") or "")
        text = str(row.get("text_raw", "") or "")
        ordered_keys.append(f"{index}:{cid}:{_text_hash(text)}")
        texts_by_id[cid].add(_text_hash(text))

    order_hash = hashlib.sha256("\n".join(ordered_keys).encode("utf-8")).hexdigest()
    embedding_hash = hashlib.sha256("\n".join(embedding_row_ids).encode("utf-8")).hexdigest()
    if embedding_row_ids != ordered_keys:
        raise ValueError(
            "embedding order/content mismatch: embedding_row_ids must equal the ordered "
            "(row_index, chunk_id, text_hash) map exactly"
        )

    aggregated = aggregate_positive_labels(labels)
    labeled_qa_ids = set(aggregated)
    alias_ambiguous_ids = sorted(cid for cid, hashes in texts_by_id.items() if len(hashes) > 1)

    return {
        "num_physical_rows": len(chunk_rows),
        "num_distinct_chunk_ids": len(texts_by_id),
        "extra_rows_from_repeated_ids": len(chunk_rows) - len(texts_by_id),
        "ordered_row_hash": order_hash,
        "embedding_order_hash": embedding_hash,
        "num_labeled_qa_ids": len(labeled_qa_ids),
        "num_label_rows": sum(len(v) for v in aggregated.values()),
        "alias_ambiguous_ids": alias_ambiguous_ids,
        "num_alias_ambiguous_ids": len(alias_ambiguous_ids),
        "positive_labels_by_qa": aggregated,
    }


def report_label_coverage(qa_ids: List[str], labels: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Report which QA ids have labels; unlabeled ids stay eligible for SFT."""
    aggregated = aggregate_positive_labels(labels)
    labeled = {qid for qid in qa_ids if qid in aggregated}
    unlabeled = sorted(set(qa_ids) - labeled)
    return {
        "num_qa_ids": len(qa_ids),
        "num_labeled_qa_ids": len(labeled),
        "num_unlabeled_qa_ids": len(unlabeled),
        "unlabeled_qa_ids": unlabeled,
    }
