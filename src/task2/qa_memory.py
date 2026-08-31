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
    ):
        self.id_to_answer = id_to_answer
        self.question_to_answer = question_to_answer
        self.conflicts = conflicts or {}
        self.df = df if df is not None else pd.DataFrame()
        self.records = records or []
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

            if qa_id:
                id_map[qa_id] = ans_raw

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

        return cls(id_map, question_map, conflicts, df, records=rows)

    def lookup_exact(self, qa_id: Optional[str], question: Optional[str]) -> Optional[str]:
        """Lookup answer by exact QA ID first, then by normalized question if no conflict."""
        if qa_id:
            qa_id_str = str(qa_id).strip()
            if qa_id_str in self.id_to_answer:
                return self.id_to_answer[qa_id_str]

        if question:
            q_norm = normalize_question(question)
            if q_norm in self.question_to_answer:
                return self.question_to_answer[q_norm]

        return None

    def lookup_fuzzy(
        self,
        question: str,
        threshold: float = 0.90,
        require_entity_match: bool = True,
    ) -> Optional[Dict[str, Any]]:
        """Lookup similar QA record using n-gram similarity and entity consistency."""
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
            # Check legal entity consistency if requested
            rec_docs = set(best_rec.get("doc_numbers", []))
            rec_arts = set(best_rec.get("articles", []))

            entity_ok = True
            if require_entity_match:
                if q_docs and rec_docs and not (q_docs & rec_docs):
                    entity_ok = False
                if q_arts and rec_arts and not (q_arts & rec_arts):
                    entity_ok = False

            if entity_ok:
                return {
                    "matched_qa_id": best_rec.get("qa_id", ""),
                    "matched_question": best_rec.get("question_raw", ""),
                    "answer": best_rec.get("answer_raw", ""),
                    "similarity": float(best_score),
                    "target_length": best_rec.get("answer_len_words", 300),
                    "is_direct_reuse": best_score >= 0.96 and entity_ok,
                }

        return None

    def get_similar_exemplar(self, question: str) -> Optional[Dict[str, Any]]:
        """Retrieve top nearest QA pair as style/length exemplar even if below direct reuse threshold."""
        return self.lookup_fuzzy(question, threshold=0.60, require_entity_match=False)

    def filter_fold(self, val_qa_ids: Set[str], val_questions: Optional[Set[str]] = None) -> QAMemory:
        """Return a new QAMemory instance strictly excluding validation QA IDs and questions to prevent leakage."""
        filtered_id_map = {k: v for k, v in self.id_to_answer.items() if k not in val_qa_ids}

        val_q_norm = {normalize_question(q) for q in val_questions} if val_questions else set()
        filtered_q_map = {k: v for k, v in self.question_to_answer.items() if k not in val_q_norm}

        filtered_records = [
            r for r in self.records
            if r.get("qa_id") not in val_qa_ids and r.get("question_norm") not in val_q_norm
        ]

        filtered_df = self.df[
            (~self.df["qa_id"].isin(val_qa_ids)) & (~self.df["question_norm"].isin(val_q_norm))
        ] if not self.df.empty else self.df

        return QAMemory(filtered_id_map, filtered_q_map, self.conflicts, filtered_df, records=filtered_records)

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
