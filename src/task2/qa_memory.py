"""Exact and conflict-safe QA Memory module for Task 2."""

from __future__ import annotations

import json
import os
from collections import defaultdict
from typing import Any, Dict, List, Optional, Set

import pandas as pd

from src.common.normalize import clean_legal_text, normalize_question


class QAMemory:
    """Stores verified ground-truth QA mappings with conflict detection and fold isolation."""

    def __init__(
        self,
        id_to_answer: Dict[str, str],
        question_to_answer: Dict[str, str],
        conflicts: Optional[Dict[str, List[str]]] = None,
        df: Optional[pd.DataFrame] = None,
    ):
        self.id_to_answer = id_to_answer
        self.question_to_answer = question_to_answer
        self.conflicts = conflicts or {}
        self.df = df if df is not None else pd.DataFrame()

    @classmethod
    def from_records(cls, records: List[Dict[str, Any]]) -> QAMemory:
        id_map: Dict[str, str] = {}
        grouped_by_q: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        rows: List[Dict[str, Any]] = []

        for r in records:
            qa_id = str(r.get("id") or r.get("qa_id") or "").strip()
            q_raw = str(r.get("question", "")).strip()
            ans_raw = str(r.get("answer", "")).strip()

            if not q_raw or not ans_raw:
                continue

            q_norm = normalize_question(q_raw)
            ans_clean = clean_legal_text(ans_raw)

            if qa_id:
                id_map[qa_id] = ans_raw

            rec_info = {
                "qa_id": qa_id,
                "question_raw": q_raw,
                "question_norm": q_norm,
                "answer_raw": ans_raw,
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
                # Conflicting answers for identical normalized question -> exclude from question_map
                conflicts[q_norm] = unique_answers

        df = pd.DataFrame(rows) if rows else pd.DataFrame()
        if not df.empty:
            df["is_conflict"] = df["question_norm"].apply(lambda q: q in conflicts)

        return cls(id_map, question_map, conflicts, df)

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

    def filter_fold(self, val_qa_ids: Set[str], val_questions: Optional[Set[str]] = None) -> QAMemory:
        """Return a new QAMemory instance excluding validation QA IDs and questions to prevent leakage."""
        filtered_id_map = {k: v for k, v in self.id_to_answer.items() if k not in val_qa_ids}

        val_q_norm = {normalize_question(q) for q in val_questions} if val_questions else set()
        filtered_q_map = {k: v for k, v in self.question_to_answer.items() if k not in val_q_norm}

        filtered_df = self.df[~self.df["qa_id"].isin(val_qa_ids)] if not self.df.empty else self.df
        return QAMemory(filtered_id_map, filtered_q_map, self.conflicts, filtered_df)

    def save(self, json_path: str, parquet_path: Optional[str] = None) -> None:
        os.makedirs(os.path.dirname(json_path), exist_ok=True)
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump({
                "id_map": self.id_to_answer,
                "question_map": self.question_to_answer,
                "conflicts_count": len(self.conflicts),
            }, f, ensure_ascii=False, indent=2)

        if parquet_path:
            os.makedirs(os.path.dirname(parquet_path), exist_ok=True)
            self.df.to_parquet(parquet_path, index=False)

    @classmethod
    def load(cls, json_path: str, parquet_path: Optional[str] = None) -> QAMemory:
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        df = pd.read_parquet(parquet_path) if parquet_path and os.path.exists(parquet_path) else pd.DataFrame()
        return cls(data.get("id_map", {}), data.get("question_map", {}), {}, df)
