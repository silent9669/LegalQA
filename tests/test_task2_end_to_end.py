import json
import zipfile
from pathlib import Path
from src.task2.predict import LegalQAPipeline


def test_pipeline_end_to_end():
    pipeline = LegalQAPipeline.build_mock()
    query = "Nghị định 90/2017 Điều 17 quy định xử phạt thế nào?"
    ans = pipeline.predict_single(qa_id="test_1", question=query)
    assert isinstance(ans, str)
    assert len(ans) > 10

    # Test candidate return
    selected, cands, ev = pipeline.predict_single(qa_id="test_1", question=query, return_candidates=True)
    assert isinstance(selected, str)
    assert "stitched_extract" in cands


def test_pipeline_memory_priority():
    pipeline = LegalQAPipeline.build_mock()
    pipeline.memory.id_to_answer["exact_id"] = "CÂU TRẢ LỜI CHÍNH XÁC TỪ BỘ DỮ LIỆU GỐC"
    ans = pipeline.predict_single(qa_id="exact_id", question="Câu hỏi bất kỳ")
    assert ans == "CÂU TRẢ LỜI CHÍNH XÁC TỪ BỘ DỮ LIỆU GỐC"


def test_pipeline_batch_prediction():
    pipeline = LegalQAPipeline.build_mock()
    items = [
        {"id": "q1", "question": "Hỏi về Nghị định 90 Điều 17"},
        {"id": "q2", "question": "Hỏi về quy định khác"}
    ]
    batch_res = pipeline.predict_batch(items)
    assert len(batch_res) == 2
    assert "q1" in batch_res
    assert "q2" in batch_res
    assert "answer" in batch_res["q1"]
    assert "answer" in batch_res["q2"]


def test_submission_schema_and_zip_validation(tmp_path: Path):
    submission = {
        str(i): {"answer": f"Căn cứ Điều {i} Nghị định 90 quy định mức xử phạt vi phạm."}
        for i in range(1000)
    }

    # Verify 1000 items
    assert len(submission) == 1000
    for qid, entry in submission.items():
        assert "answer" in entry
        assert isinstance(entry["answer"], str)
        assert len(entry["answer"].strip()) > 0
        assert "[DOCUMENT]" not in entry["answer"]

    # Test zip packaging
    out_json = tmp_path / "submission.json"
    out_zip = tmp_path / "submission.json.zip"

    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(submission, f, ensure_ascii=False, indent=2)

    with zipfile.ZipFile(out_zip, "w", zipfile.ZIP_DEFLATED) as z:
        z.write(out_json, arcname="submission.json")

    assert out_zip.exists()
    with zipfile.ZipFile(out_zip, "r") as z:
        file_list = z.namelist()
        assert file_list == ["submission.json"]
        content = json.loads(z.read("submission.json").decode("utf-8"))
        assert len(content) == 1000
