from src.task2.predict import LegalQAPipeline

def test_pipeline_end_to_end():
    pipeline = LegalQAPipeline.build_mock()
    query = "Nghị định 90/2017 Điều 17 quy định xử phạt thế nào?"
    ans = pipeline.predict_single(qa_id="test_1", question=query)
    assert isinstance(ans, str)
    assert len(ans) > 10

def test_pipeline_memory_priority():
    pipeline = LegalQAPipeline.build_mock()
    pipeline.memory.id_to_answer["exact_id"] = "CÂU TRẢ LỜI CHÍNH XÁC TỪ BỘ DỮ LIỆU GỐC"
    ans = pipeline.predict_single(qa_id="exact_id", question="Câu hỏi bất kỳ")
    assert ans == "CÂU TRẢ LỜI CHÍNH XÁC TỪ BỘ DỮ LIỆU GỐC"
