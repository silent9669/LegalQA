from src.task2.generator import QwenGenerator

def test_qwen_generator_fallback():
    gen = QwenGenerator(runtime="fallback")
    evidence = "Điều 17. Phạt tiền từ 1.000.000 đồng đến 2.000.000 đồng đối với hành vi không tiêm vắc xin."
    ans = gen.generate("Hành vi không tiêm phòng phạt bao nhiêu?", evidence)
    assert "Căn cứ" in ans or "Phạt tiền" in ans
    assert "1.000.000" in ans
