import os
import sys
from pathlib import Path
import pytest
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Import the official scoring function if available
SCORING_PROGRAM_DIR = Path(__file__).resolve().parents[1] / "Scoring-Program-Task-LegalQA"
if str(SCORING_PROGRAM_DIR) not in sys.path:
    sys.path.insert(0, str(SCORING_PROGRAM_DIR))

try:
    import scoring as official_scoring
except ImportError:
    official_scoring = None

from src.task2.metrics import calculate_official_meteor, ensure_meteor_resources, official_meteor
from src.task2.evaluation import evaluate_checkpoint
from scripts.run_oof_validation import run_oof_validation


def test_eval_qa_exact_match():
    if official_scoring is None:
        pytest.skip("Scoring program not present")
    y_pred = {"1": {"answer": "Căn cứ khoản 3 Điều 17 Nghị định 90/2017/NĐ-CP"}}
    y_true = {"1": "Căn cứ khoản 3 Điều 17 Nghị định 90/2017/NĐ-CP"}
    res = official_scoring.eval_qa(y_pred, y_true)
    assert res["meteor"] >= 0.99
    assert res["rouge"] >= 0.99


def test_meteor_parity():
    ensure_meteor_resources()
    refs = ["Căn cứ khoản 3 Điều 17 Nghị định 90/2017/NĐ-CP quy định xử phạt vi phạm hành chính."]
    preds = ["Căn cứ khoản 3 Điều 17 Nghị định 90/2017/NĐ-CP quy định mức phạt tiền."]
    m_score = calculate_official_meteor(refs, preds)
    assert m_score > 0.0
    if official_scoring is not None:
        official = official_scoring.eval_qa({"1": {"answer": preds[0]}}, {"1": refs[0]})
        assert pytest.approx(m_score, abs=1e-6) == float(official["meteor"])


def test_evaluate_checkpoint_strict_folds(tmp_path: Path):
    """Verify evaluate_checkpoint raises when fold_id is missing or fold is empty (P0-2)."""
    qa_data = [
        {"qa_id": f"q_{i}", "question_raw": f"Câu hỏi {i}?", "question_norm": f"câu hỏi {i}?", "answer_raw": f"Đáp án {i}", "fold_id": 1}
        for i in range(5)
    ]
    qa_file = tmp_path / "qa.parquet"
    pd.DataFrame(qa_data).to_parquet(qa_file, index=False)

    # Missing fold 0 should raise RuntimeError
    with pytest.raises(RuntimeError, match="contains no rows"):
        evaluate_checkpoint(
            qa_path=str(qa_file),
            fold_path=str(tmp_path / "missing_folds.parquet"),
            held_out_fold=0,
        )


def test_run_oof_validation_held_out_fold(tmp_path: Path):
    """Verify run_oof_validation respects held_out_fold argument (P0-15)."""
    qa_data = [
        {"qa_id": f"q_{i}", "question_raw": f"Câu hỏi {i}", "question_norm": f"câu hỏi {i}", "answer_raw": f"Đáp án {i}", "fold_id": i % 3}
        for i in range(15)
    ]
    chunks_data = [
        {"chunk_id": f"c_{i}", "parent_article_id": f"doc1_art{i}", "doc_name": "Nghị định 90", "text_raw": f"[DOCUMENT] Nghị định 90\n[ARTICLE] Điều {i}.", "text_norm": f"nghị định 90 điều {i}"}
        for i in range(5)
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
        n_splits=3,
        mode="fast",
        held_out_fold=1,
    )

    assert summary["evaluated_folds"] == [1]
    assert summary["total_evaluated"] == 5


def test_run_oof_validation_full_mode_guard(tmp_path: Path):
    """Verify mode='full' without fold_checkpoint_map or held_out_fold raises (P0-15)."""
    qa_file = tmp_path / "qa.parquet"
    pd.DataFrame([{"qa_id": "1", "question_raw": "q", "question_norm": "q", "answer_raw": "a", "fold_id": 0}]).to_parquet(qa_file)

    with pytest.raises(RuntimeError, match="requires one checkpoint per fold"):
        run_oof_validation(
            qa_path=str(qa_file),
            fold_path=str(qa_file),
            mode="full",
            held_out_fold=None,
            fold_checkpoint_map=None,
        )
