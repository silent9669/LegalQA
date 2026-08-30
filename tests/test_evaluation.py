import os
import sys
from pathlib import Path
import pytest
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Import the official scoring function
SCORING_PROGRAM_DIR = Path(__file__).resolve().parents[1] / "Scoring-Program-Task-LegalQA"
if str(SCORING_PROGRAM_DIR) not in sys.path:
    sys.path.insert(0, str(SCORING_PROGRAM_DIR))

import scoring as official_scoring
from scripts.run_oof_validation import calculate_official_meteor

def test_eval_qa_exact_match():
    y_pred = {"1": {"answer": "Căn cứ khoản 3 Điều 17 Nghị định 90/2017/NĐ-CP"}}
    y_true = {"1": "Căn cứ khoản 3 Điều 17 Nghị định 90/2017/NĐ-CP"}
    res = official_scoring.eval_qa(y_pred, y_true)
    assert res["meteor"] >= 0.99
    assert res["rouge"] >= 0.99

def test_meteor_parity():
    refs = ["Căn cứ khoản 3 Điều 17 Nghị định 90/2017/NĐ-CP"]
    preds = ["Căn cứ khoản 3 Điều 17 Nghị định 90/2017/NĐ-CP"]
    m_score = calculate_official_meteor(refs, preds)
    official = official_scoring.eval_qa({"1": {"answer": preds[0]}}, {"1": refs[0]})
    assert pytest.approx(m_score, abs=1e-6) == float(official["meteor"])
