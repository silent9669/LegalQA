from pathlib import Path
import pytest
from src.task2.generator import QwenGenerator, SYSTEM_PROMPT
from src.task2.predict import LegalQAPipeline


def test_qwen_generator_fallback():
    gen = QwenGenerator(runtime="fallback")
    evidence = "Điều 17. Phạt tiền từ 1.000.000 đồng đến 2.000.000 đồng đối với hành vi không tiêm vắc xin."
    ans = gen.generate("Hành vi không tiêm phòng phạt bao nhiêu?", evidence)
    assert "Căn cứ" in ans or "Phạt tiền" in ans
    assert "1.000.000" in ans


def test_qwen_prompt_formatting():
    prompt = QwenGenerator.format_prompt("Hỏi luật?", "Căn cứ Điều 1.")
    assert "<|im_start|>system" in prompt
    assert "<|im_start|>user" in prompt
    assert "[CĂN CỨ PHÁP LÝ]\nCăn cứ Điều 1." in prompt
    assert "[CÂU HỎI]\nHỏi luật?" in prompt
    assert "<|im_start|>assistant" in prompt


def test_qwen_generator_load_fallback_explicit():
    gen = QwenGenerator.load(model_path="nonexistent", runtime="fallback")
    assert gen.runtime == "fallback"


def test_qwen_generator_require_adapter_missing():
    # require_adapter=True with missing adapter path should raise RuntimeError
    with pytest.raises(RuntimeError, match="adapter_path was not provided"):
        QwenGenerator.load(model_path="nonexistent", require_adapter=True)

    # require_adapter=True with non-existent path should raise FileNotFoundError
    with pytest.raises(FileNotFoundError, match="adapter_path does not exist"):
        QwenGenerator.load(model_path="nonexistent", adapter_path="/nonexistent/path", require_adapter=True)


def test_pipeline_generator_optional():
    """Verify LegalQAPipeline works seamlessly with generator=None."""
    pipe = LegalQAPipeline.build_mock()
    pipe.generator = None  # Remove generator

    # Single prediction should work cleanly via extractive fallback
    ans = pipe.predict_single("q1", "Phạt bao nhiêu?")
    assert len(ans) > 0

    # Batch prediction should work cleanly without generator
    batch_res = pipe.predict_batch([{"qa_id": "q1", "question": "Phạt bao nhiêu?"}])
    assert "q1" in batch_res
    assert len(batch_res["q1"]["answer"]) > 0
