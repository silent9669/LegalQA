import json
import unicodedata
import re
import pandas as pd

def normalize_vietnamese_text(text: str) -> str:
    """
    Unicode NFC normalization + whitespace trimming.
    Preserves exact casing and punctuation.
    """
    if not text:
        return ""
    text = unicodedata.normalize('NFC', str(text))
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def build_canonical_qa(train_path: str, warmup_path: str) -> tuple[pd.DataFrame, dict]:
    """
    Loads train.json and warmup.json, deduplicates identical QA samples into
    canonical DataFrame (qa_unique) and builds exact memory mapping.
    """
    with open(train_path, 'r', encoding='utf-8') as f:
        train_data = json.load(f)
    with open(warmup_path, 'r', encoding='utf-8') as f:
        warmup_data = json.load(f)

    records = {}
    by_id_mem = {}
    by_q_mem = {}

    # Process Train
    for qid, item in train_data.items():
        q_raw = item['question']
        a_raw = item['answer']
        q_norm = normalize_vietnamese_text(q_raw).lower()
        qid_str = str(qid)
        records[qid_str] = {
            "id": qid_str,
            "question": q_raw,
            "normalized_question": q_norm,
            "answer": a_raw,
            "source_splits": ["train"]
        }
        by_id_mem[qid_str] = a_raw
        by_q_mem[q_norm] = a_raw

    # Process Warmup
    for qid, item in warmup_data.items():
        q_raw = item['question']
        a_raw = item['answer']
        q_norm = normalize_vietnamese_text(q_raw).lower()
        qid_str = str(qid)
        if qid_str in records:
            if "warmup" not in records[qid_str]["source_splits"]:
                records[qid_str]["source_splits"].append("warmup")
        else:
            records[qid_str] = {
                "id": qid_str,
                "question": q_raw,
                "normalized_question": q_norm,
                "answer": a_raw,
                "source_splits": ["warmup"]
            }
        by_id_mem[qid_str] = a_raw
        by_q_mem[q_norm] = a_raw

    df_unique = pd.DataFrame(list(records.values()))
    memory_dict = {
        "by_id": by_id_mem,
        "by_question": by_q_mem
    }
    return df_unique, memory_dict
