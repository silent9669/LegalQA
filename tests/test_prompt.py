import pytest
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.generation.prompt_builder import build_generation_prompt

def test_build_generation_prompt():
    evidence = [{"raw_text": "Điều 10 quy định về xử phạt vi phạm giao thông"}]
    prompt = build_generation_prompt("Mức xử phạt thế nào?", evidence)
    assert "Điều 10 quy định về xử phạt vi phạm giao thông" in prompt
    assert "Mức xử phạt thế nào?" in prompt
    assert "### Căn cứ pháp lý:" in prompt
