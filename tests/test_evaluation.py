import pytest
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.evaluation.codabench_eval import evaluate_predictions

def test_codabench_eval_identical_answers():
    y_true = {"1": "Căn cứ Điều 50 Bộ luật Tố tụng hình sự 2015"}
    y_pred = {"1": {"answer": "Căn cứ Điều 50 Bộ luật Tố tụng hình sự 2015"}}
    scores = evaluate_predictions(y_pred, y_true)
    assert scores["meteor"] >= 0.999
    assert scores["rouge"] == pytest.approx(1.0, 1e-4)

def test_codabench_eval_whitespace_tokenization():
    y_true = {"1": "Điều 17, khoản 3"}
    y_pred = {"1": {"answer": "Điều 17 khoản 3"}}
    scores = evaluate_predictions(y_pred, y_true)
    # METEOR splits on whitespace ("17," vs "17" don't match)
    assert 0.0 < scores["meteor"] < 0.99
    # ROUGE strips punctuation by default
    assert scores["rouge"] == pytest.approx(1.0, 1e-4)

def test_codabench_eval_multiple_samples():
    y_true = {
        "1": "Nghị định 13/2023/NĐ-CP có hiệu lực từ ngày 01/7/2023",
        "2": "Bị cáo có quyền đề nghị thay đổi Thẩm phán"
    }
    y_pred = {
        "1": {"answer": "Nghị định 13/2023/NĐ-CP có hiệu lực từ ngày 01/7/2023"},
        "2": {"answer": "Bị cáo không có quyền đề nghị"}
    }
    scores = evaluate_predictions(y_pred, y_true)
    assert 0.5 <= scores["meteor"] <= 1.0
