from pathlib import Path
from src.task2.generator import QwenGenerator, SYSTEM_PROMPT


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


def test_qwen_generator_adapter_path_property(tmp_path: Path):
    adapter_dir = tmp_path / "dummy_adapter"
    adapter_dir.mkdir()
    (adapter_dir / "adapter_config.json").write_text('{"peft_type": "LORA"}', encoding="utf-8")

    gen = QwenGenerator.load(model_path="nonexistent", adapter_path=str(adapter_dir), runtime="fallback")
    assert gen.adapter_path == str(adapter_dir)
    assert gen.runtime == "fallback"
