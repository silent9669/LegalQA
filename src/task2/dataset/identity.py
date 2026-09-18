"""Canonical QA row identity: stable row/group ids, audited dedup, conflict policy.

Source rows are never deleted here; canonicalization returns the eligible
example list plus a reversible ledger (report) describing every decision.
"""

from __future__ import annotations

import hashlib
from collections import defaultdict
from typing import Any, Dict, List, Tuple

from src.common.normalize import clean_legal_text, normalize_question


def _qa_id_of(row: Dict[str, Any]) -> str:
    for key in ("qa_id", "id", "question_id"):
        value = row.get(key)
        if value is not None and str(value).strip():
            return str(value).strip()
    return ""


def _question_of(row: Dict[str, Any]) -> str:
    for key in ("question_raw", "question", "query"):
        value = row.get(key)
        if value is not None and str(value).strip():
            return str(value)
    return ""


def _answer_of(row: Dict[str, Any]) -> str:
    for key in ("answer_raw", "answer", "response"):
        value = row.get(key)
        if value is not None and str(value).strip():
            return str(value)
    return ""


def _stable_hex(payload: str, length: int = 16) -> str:
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:length]


def canonicalize_qa_rows(
    rows: List[Dict[str, Any]],
    conflict_policy: str = "reject",
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Deduplicate raw QA rows into eligible canonical examples.

    Grouping key is the normalized question; equivalence preserves legal
    numbers, dates, negation and identifiers because normalization only
    lowercases and normalizes whitespace/punctuation without dropping
    digits or negation tokens.

    - Identical (normalized question, normalized answer) records collapse to
      one eligible example; the survivor keeps the first-seen raw question
      and answer, the full legacy id list, and a stable qa_row_id.
    - A normalized-question group with more than one distinct normalized
      answer is a conflict. ``conflict_policy="reject"`` raises ValueError;
      ``conflict_policy="quarantine"`` excludes the whole group and reports
      it. Any other policy raises ValueError.
    - Rows with empty question or answer are ineligible and reported, never
      silently kept.

    Returns (examples, report). Each example carries qa_row_id, qa_group_id,
    qa_id (representative legacy id), legacy_qa_ids, question_raw,
    answer_raw, question_norm and answer_norm.
    """
    if conflict_policy not in ("reject", "quarantine"):
        raise ValueError(f"unknown conflict_policy: {conflict_policy}")

    groups: Dict[str, List[Tuple[int, Dict[str, Any], str, str, str]]] = defaultdict(list)
    empty_rows = 0
    for index, row in enumerate(rows):
        qa_id = _qa_id_of(row)
        q_raw = _question_of(row)
        a_raw = _answer_of(row)
        if not q_raw.strip() or not a_raw.strip():
            empty_rows += 1
            continue
        q_norm = normalize_question(q_raw)
        a_norm = clean_legal_text(a_raw).lower().strip()
        if not q_norm or not a_norm:
            empty_rows += 1
            continue
        groups[q_norm].append((index, row, qa_id, q_raw.strip(), a_raw.strip()))

    examples: List[Dict[str, Any]] = []
    quarantined: List[Dict[str, Any]] = []
    duplicate_rows_removed = 0

    for q_norm in sorted(groups):
        members = sorted(groups[q_norm], key=lambda item: item[0])
        distinct_answers = sorted({clean_legal_text(m[4]).lower().strip() for m in members})
        if len(distinct_answers) > 1:
            detail = {
                "question_norm": q_norm,
                "num_rows": len(members),
                "legacy_qa_ids": sorted({m[2] for m in members if m[2]}),
                "distinct_answers": distinct_answers,
            }
            if conflict_policy == "reject":
                raise ValueError(f"conflicting answers for one normalized question: {detail}")
            quarantined.append(detail)
            duplicate_rows_removed += len(members)
            continue
        # Same question + same answer: one eligible example.
        first = members[0]
        legacy_ids = sorted({m[2] for m in members if m[2]})
        duplicate_rows_removed += len(members) - 1
        qa_group_id = _stable_hex(f"qagroup:{q_norm}")
        qa_row_id = _stable_hex(f"qarow:{q_norm}::{distinct_answers[0]}")
        examples.append(
            {
                "qa_row_id": qa_row_id,
                "qa_group_id": qa_group_id,
                "qa_id": first[2],
                "legacy_qa_ids": legacy_ids,
                "question_raw": first[3],
                "answer_raw": first[4],
                "question_norm": q_norm,
                "answer_norm": distinct_answers[0],
                "source_row_count": len(members),
            }
        )

    examples.sort(key=lambda e: (e["qa_group_id"], e["qa_id"]))
    report: Dict[str, Any] = {
        "input_rows": len(rows),
        "empty_rows_skipped": empty_rows,
        "num_groups": len(groups),
        "eligible_examples": len(examples),
        "duplicate_rows_removed": duplicate_rows_removed,
        "quarantined_groups": quarantined,
        "quarantined_row_count": sum(g["num_rows"] for g in quarantined),
        "conflict_policy": conflict_policy,
    }
    return examples, report
