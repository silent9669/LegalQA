import numpy as np
import pandas as pd
from pathlib import Path
from src.task2.selector import CandidateSelector, extract_candidate_features


def test_candidate_feature_extraction():
    q = "Mức phạt hành vi trốn thuế theo Nghị định 125/2020?"
    cand = "Căn cứ Điều 17 Nghị định 125/2020 phạt từ 1 đến 3 lần số tiền trốn thuế."
    ev = "Nghị định 125/2020 Điều 17. Phạt tiền từ 1 đến 3 lần."

    feats = extract_candidate_features(
        question=q,
        candidate_name="stitched_extract",
        candidate_text=cand,
        evidence=ev,
        retrieval_meta={"rerank_top1": 0.95, "rerank_margin": 0.20, "bm25_top1": 15.0},
    )

    assert "cand_word_count" in feats
    assert "overlap_ratio" in feats
    assert "rerank_top1" in feats
    assert feats["is_penalty"] == 1.0


def test_selector_guardrail_fallback(tmp_path: Path):
    # Synthetic candidate score dataset where stitched_extract is consistently best
    rows = []
    for i in range(100):
        qid = f"q_{i}"
        rows.append({"qa_id": qid, "cand_name": "focused_extract", "cand_text": "Foc", "meteor": 0.20, "fold_id": i % 5})
        rows.append({"qa_id": qid, "cand_name": "stitched_extract", "cand_text": "Stitch", "meteor": 0.35, "fold_id": i % 5})
        rows.append({"qa_id": qid, "cand_name": "generated", "cand_text": "Gen", "meteor": 0.10, "fold_id": i % 5})

    df_cand = pd.DataFrame(rows)
    selector = CandidateSelector()
    summary = selector.fit_meta_oof(df_cand, baseline_candidate="stitched_extract")

    assert summary["best_fixed_candidate"] == "stitched_extract"
    # Even if model trains, policy must select stitched_extract when it dominates
    cands = {
        "focused_extract": "Foc",
        "stitched_extract": "Stitched answer",
        "generated": "Gen answer",
    }
    chosen = selector.select(cands, question="Hỏi luật?")
    assert chosen == "Stitched answer"
