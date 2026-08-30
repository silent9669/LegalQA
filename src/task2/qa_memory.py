import json
import os
import pandas as pd
from src.common.normalize import clean_legal_text, normalize_question

class QAMemory:
    def __init__(self, id_to_answer: dict, question_to_answer: dict, df: pd.DataFrame = None):
        self.id_to_answer = id_to_answer
        self.question_to_answer = question_to_answer
        self.df = df if df is not None else pd.DataFrame()

    @classmethod
    def from_records(cls, records: list[dict]):
        id_map = {}
        q_map = {}
        rows = []

        for r in records:
            qa_id = str(r.get("id") or r.get("qa_id") or "").strip()
            q_raw = str(r.get("question", "")).strip()
            ans_raw = str(r.get("answer", "")).strip()

            if not q_raw or not ans_raw:
                continue

            q_norm = normalize_question(q_raw)
            if qa_id:
                id_map[qa_id] = ans_raw
            q_map[q_norm] = ans_raw

            rows.append({
                "qa_id": qa_id,
                "question_raw": q_raw,
                "question_norm": q_norm,
                "answer_raw": ans_raw,
                "source_split": r.get("source_split", "train")
            })

        df = pd.DataFrame(rows).drop_duplicates(subset=["question_norm"]) if rows else pd.DataFrame()
        return cls(id_map, q_map, df)

    def lookup_exact(self, qa_id: str, question: str) -> str | None:
        if qa_id and str(qa_id).strip() in self.id_to_answer:
            return self.id_to_answer[str(qa_id).strip()]

        q_norm = normalize_question(question)
        if q_norm in self.question_to_answer:
            return self.question_to_answer[q_norm]

        return None

    def save(self, json_path: str, parquet_path: str = None):
        os.makedirs(os.path.dirname(json_path), exist_ok=True)
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump({
                "id_map": self.id_to_answer,
                "question_map": self.question_to_answer
            }, f, ensure_ascii=False, indent=2)
        if parquet_path:
            os.makedirs(os.path.dirname(parquet_path), exist_ok=True)
            self.df.to_parquet(parquet_path, index=False)

    @classmethod
    def load(cls, json_path: str, parquet_path: str = None):
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        df = pd.read_parquet(parquet_path) if parquet_path and os.path.exists(parquet_path) else pd.DataFrame()
        return cls(data.get("id_map", {}), data.get("question_map", {}), df)
