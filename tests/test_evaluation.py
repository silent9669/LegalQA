import os
import sys
from pathlib import Path
import pytest
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Import the official scoring function
SCORING_PROGRAM_DIR = Path(__file__).resolve().parents[1] / "Scoring-Program-Task-LegalQA"
if str(SCORING_PROGRAM_DIR) not in sys.path:
    sys.path.insert(0, str(SCORING_PROGRAM_DIR))

import scoring as official_scoring
from scripts.run_oof_validation import calculate_official_meteor, run_oof_validation


def test_eval_qa_exact_match():
    y_pred = {"1": {"answer": "Căn cứ khoản 3 Điều 17 Nghị định 90/2017/NĐ-CP"}}
    y_true = {"1": "Căn cứ khoản 3 Điều 17 Nghị định 90/2017/NĐ-CP"}
    res = official_scoring.eval_qa(y_pred, y_true)
    assert res["meteor"] >= 0.99
    assert res["rouge"] >= 0.99


def test_meteor_parity():
    refs = ["Căn cứ khoản 3 Điều 17 Nghị định 90/2017/NĐ-CP quy định xử phạt vi phạm hành chính."]
    preds = ["Căn cứ khoản 3 Điều 17 Nghị định 90/2017/NĐ-CP quy định mức phạt tiền."]
    m_score = calculate_official_meteor(refs, preds)
    official = official_scoring.eval_qa({"1": {"answer": preds[0]}}, {"1": refs[0]})
    assert pytest.approx(m_score, abs=1e-6) == float(official["meteor"])


def test_run_oof_validation_fast_mode(tmp_path: Path):
    # Setup mock QA and chunks
    qa_data = [
        {"qa_id": f"q_{i}", "question_raw": f"Câu hỏi số {i} về Nghị định 90", "question_norm": f"câu hỏi số {i} về nghị định 90", "answer_raw": f"Căn cứ Điều {i} Nghị định 90 quy định mức phạt.", "fold_id": i % 2}
        for i in range(10)
    ]
    chunks_data = [
        {"chunk_id": f"c_{i}", "parent_article_id": f"doc1_art{i}", "doc_name": "Nghị định 90", "text_raw": f"[DOCUMENT] Nghị định 90\n[ARTICLE] Điều {i}. Phạt tiền từ 1 đến 2 triệu đồng.", "text_norm": f"nghị định 90 điều {i} phạt tiền"}
        for i in range(10)
    ]

    qa_file = tmp_path / "qa.parquet"
    chunks_file = tmp_path / "chunks.parquet"
    eval_dir = tmp_path / "eval"

    pd.DataFrame(qa_data).to_parquet(qa_file, index=False)
    pd.DataFrame(chunks_data).to_parquet(chunks_file, index=False)

    summary = run_oof_validation(
        qa_path=str(qa_file),
        fold_path=str(qa_file),
        chunks_path=str(chunks_file),
        bm25_dir=str(tmp_path / "bm25"),
        dek21_dir=str(tmp_path / "dek21"),
        eval_output_dir=str(eval_dir),
        num_eval_samples=10,
        n_splits=2,
        mode="fast",
    )

    assert summary["total_evaluated"] == 10
    assert "mean_meteor" in summary
    assert "candidate_family_meteors" in summary
    assert "oracle_best" in summary["candidate_family_meteors"]
    assert (eval_dir / "oof_summary.json").exists()
    assert (eval_dir / "oof_predictions.parquet").exists()
