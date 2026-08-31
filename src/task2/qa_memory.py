"""Exact and conflict-safe Similar-QA Memory module for Task 2."""

from __future__ import annotations

import json
import math
import os
import re
from collections import Counter, defaultdict
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd

from src.common.normalize import clean_legal_text, extract_legal_signals, normalize_question


def compute_char_ngram_vector(text: str, n: int = 3) -> Dict[str, float]:
    """Compute normalized character n-gram TF vector."""
    clean = normalize_question(text)
    if len(clean) < n:
        return {clean: 1.0} if clean else {}
    ngrams = [clean[i:i + n] for i in range(len(clean) - n + 1)]
    counts = Counter(ngrams)
    norm = math.sqrt(sum(c * c for c in counts.values()))
    if norm <= 0:
        return {}
    return {k: v / norm for k, v in counts.items()}


def cosine_sim_ngrams(vec1: Dict[str, float], vec2: Dict[str, float]) -> float:
    """Compute cosine similarity between two normalized sparse n-gram vectors."""
    if not vec1 or not vec2:
        return 0.0
    if len(vec1) > len(vec2):
        vec1, vec2 = vec2, vec1
    dot = sum(val * vec2.get(k, 0.0) for k, val in vec1.items())
    return float(dot)


class QAMemory:
    """Stores verified ground-truth QA mappings with exact and leakage-safe similar-QA lookup."""

    def __init__(
        self,
        id_to_answer: Dict[str, str],
        question_to_answer: Dict[str, str],
        conflicts: Optional[Dict[str, List[str]]] = None,
        df: Optional[pd.DataFrame] = None,
        records: Optional[List[Dict[str, Any]]] = None,
        id_to_record: Optional[Dict[str, Dict[str, Any]]] = None,
    ):
        self.id_to_answer = id_to_answer
        self.question_to_answer = question_to_answer
        self.conflicts = conflicts or {}
        self.df = df if df is not None else pd.DataFrame()
        self.records = records or []
        self.id_to_record = id_to_record or {
            str(r.get("qa_id") or r.get("id", "")): r for r in self.records if str(r.get("qa_id") or r.get("id", ""))
        }
        self._ngram_cache: List[Tuple[Dict[str, float], Dict[str, Any]]] = []
        self._build_index()

    def _build_index(self) -> None:
        """Build character n-gram vectors for fuzzy similar QA search."""
        self._ngram_cache = []
        for r in self.records:
            q_raw = str(r.get("question_raw") or r.get("question", ""))
            vec = compute_char_ngram_vector(q_raw, n=3)
            self._ngram_cache.append((vec, r))

    @classmethod
    def from_records(cls, records: List[Dict[str, Any]]) -> QAMemory:
        id_map: Dict[str, str] = {}
        id_to_rec: Dict[str, Dict[str, Any]] = {}
        grouped_by_q: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        rows: List[Dict[str, Any]] = []

        for r in records:
            qa_id = str(r.get("id") or r.get("qa_id") or "").strip()
            q_raw = str(r.get("question_raw") or r.get("question", "")).strip()
            ans_raw = str(r.get("answer_raw") or r.get("answer", "")).strip()

            if not q_raw or not ans_raw:
                continue

            q_norm = normalize_question(q_raw)
            ans_clean = clean_legal_text(ans_raw)
            signals = extract_legal_signals(q_raw)

            rec_info = {
                "qa_id": qa_id,
                "question_raw": q_raw,
                "question_norm": q_norm,
                "answer_raw": ans_raw,
                "answer_clean": ans_clean,
                "answer_len_words": len(ans_raw.split()),
                "doc_numbers": signals.get("doc_numbers", []),
                "articles": signals.get("articles", []),
                "clauses": signals.get("clauses", []),
                "source_split": r.get("source_split", "train"),
            }

            if qa_id:
                id_map[qa_id] = ans_raw
                id_to_rec[qa_id] = rec_info

            grouped_by_q[q_norm].append(rec_info)
            rows.append(rec_info)

        question_map: Dict[str, str] = {}
        conflicts: Dict[str, List[str]] = {}

        for q_norm, items in grouped_by_q.items():
            unique_answers = list({it["answer_raw"].strip() for it in items})
            if len(unique_answers) == 1:
                question_map[q_norm] = unique_answers[0]
            else:
                conflicts[q_norm] = unique_answers

        df = pd.DataFrame(rows) if rows else pd.DataFrame()
        if not df.empty:
            df["is_conflict"] = df["question_norm"].apply(lambda q: q in conflicts)

        return cls(id_map, question_map, conflicts, df, records=rows, id_to_record=id_to_rec)

    def lookup_exact(self, qa_id: Optional[str], question: Optional[str]) -> Optional[str]:
        """Lookup answer by exact normalized question or consistent QA ID.

        If question is provided:
          - normalized question match takes priority;
          - ID match is accepted ONLY when the stored question is consistent with the query.
        If question is empty, ID lookup is permitted.
        """
        q_norm = normalize_question(question) if question else ""

        # 1. Exact normalized question lookup (conflict-free)
        if q_norm and q_norm in self.question_to_answer:
            return self.question_to_answer[q_norm]

        # 2. QA ID lookup with question consistency check
        if qa_id:
            qa_id_str = str(qa_id).strip()
            if qa_id_str in self.id_to_record:
                rec = self.id_to_record[qa_id_str]
                stored_q_norm = rec.get("question_norm", "")
                if not q_norm or stored_q_norm == q_norm:
                    return rec.get("answer_raw", "")
                else:
                    # ID collision with conflicting question -> safe None
                    return None
            elif qa_id_str in self.id_to_answer:
                return self.id_to_answer[qa_id_str]

        return None

    def lookup_fuzzy(
        self,
        question: str,
        threshold: float = 0.90,
        require_entity_match: bool = True,
    ) -> Optional[Dict[str, Any]]:
        """Lookup similar QA record with similarity and entity consistency features."""
        if not question or not self._ngram_cache:
            return None

        q_vec = compute_char_ngram_vector(question, n=3)
        if not q_vec:
            return None

        q_signals = extract_legal_signals(question)
        q_docs = set(q_signals.get("doc_numbers", []))
        q_arts = set(q_signals.get("articles", []))

        best_score = 0.0
        best_rec = None

        for vec, rec in self._ngram_cache:
            sim = cosine_sim_ngrams(q_vec, vec)
            if sim > best_score:
                best_score = sim
                best_rec = rec

        if best_rec is not None and best_score >= threshold:
            rec_docs = set(best_rec.get("doc_numbers", []))
            rec_arts = set(best_rec.get("articles", []))

            same_doc = bool(q_docs & rec_docs) if (q_docs and rec_docs) else False
            same_art = bool(q_arts & rec_arts) if (q_arts and rec_arts) else False
            conflict_doc = bool(q_docs and rec_docs and not (q_docs & rec_docs))
            conflict_art = bool(q_arts and rec_arts and not (q_arts & rec_arts))

            entity_ok = True
            if require_entity_match:
                if conflict_doc or conflict_art:
                    entity_ok = False

            if entity_ok:
                q_len = len(question.split())
                ans_len = best_rec.get("answer_len_words", 300)
                return {
                    "matched_qa_id": best_rec.get("qa_id", ""),
                    "matched_question": best_rec.get("question_raw", ""),
                    "answer": best_rec.get("answer_raw", ""),
                    "similarity": float(best_score),
                    "target_length": ans_len,
                    "same_doc_number": same_doc,
                    "same_article": same_art,
                    "conflicting_doc_number": conflict_doc,
                    "conflicting_article": conflict_art,
                    "question_length_ratio": float(q_len / max(1, len(str(best_rec.get("question_raw", "")).split()))),
                    "is_direct_reuse": best_score >= 0.96 and not conflict_doc and not conflict_art,
                }

        return None

    def get_similar_exemplar(self, question: str) -> Optional[Dict[str, Any]]:
        """Retrieve nearest QA pair as style/length exemplar without entity filtering."""
        return self.lookup_fuzzy(question, threshold=0.50, require_entity_match=False)

    def filter_fold(self, val_qa_ids: Set[str], val_questions: Optional[Set[str]] = None) -> QAMemory:
        """Return a new QAMemory instance strictly excluding all validation records to guarantee zero leakage."""
        val_qa_ids_str = {str(k).strip() for k in val_qa_ids}
        val_q_norm = {normalize_question(q) for q in val_questions} if val_questions else set()

        filtered_records = [
            r for r in self.records
            if str(r.get("qa_id") or r.get("id", "")).strip() not in val_qa_ids_str
            and r.get("question_norm") not in val_q_norm
        ]

        filtered_id_map = {
            k: v for k, v in self.id_to_answer.items()
            if k not in val_qa_ids_str
        }
        filtered_id_to_rec = {
            k: v for k, v in self.id_to_record.items()
            if k not in val_qa_ids_str and v.get("question_norm") not in val_q_norm
        }
        filtered_q_map = {
            k: v for k, v in self.question_to_answer.items()
            if k not in val_q_norm
        }

        filtered_df = self.df[
            (~self.df["qa_id"].astype(str).isin(val_qa_ids_str)) & (~self.df["question_norm"].isin(val_q_norm))
        ] if not self.df.empty else self.df

        return QAMemory(
            filtered_id_map,
            filtered_q_map,
            self.conflicts,
            filtered_df,
            records=filtered_records,
            id_to_record=filtered_id_to_rec,
        )

    def save(self, json_path: str, parquet_path: Optional[str] = None) -> None:
        os.makedirs(os.path.dirname(json_path), exist_ok=True)
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump({
                "id_map": self.id_to_answer,
                "question_map": self.question_to_answer,
                "conflicts_count": len(self.conflicts),
                "records": self.records,
            }, f, ensure_ascii=False, indent=2)

        if parquet_path:
            os.makedirs(os.path.dirname(parquet_path), exist_ok=True)
            self.df.to_parquet(parquet_path, index=False)

    @classmethod
    def load(cls, json_path: str, parquet_path: Optional[str] = None) -> QAMemory:
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        df = pd.read_parquet(parquet_path) if parquet_path and os.path.exists(parquet_path) else pd.DataFrame()
        records = data.get("records", [])
        return cls(data.get("id_map", {}), data.get("question_map", {}), {}, df, records=records)
