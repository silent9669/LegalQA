"""Qwen2.5-3B-Instruct Generator wrapper with prompt parity and FP16/BF16 auto-detection."""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional

try:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
except ImportError:
    torch = None
    AutoModelForCausalLM = None
    AutoTokenizer = None
    pipeline = None

try:
    from peft import PeftModel
except ImportError:
    PeftModel = None


class QwenGenerator:
    """Qwen2.5-3B-Instruct generator for evidence-conditioned statutory legal answer generation."""

    def __init__(
        self,
        model_path: str = "Qwen/Qwen2.5-3B-Instruct",
        adapter_path: Optional[str] = None,
        runtime: str = "auto",
        device: Optional[str] = None,
    ):
        self.model_path = model_path
        self.adapter_path = adapter_path
        self.runtime = runtime
        self.device = device
        self.model = None
        self.tokenizer = None
        self.hf_pipeline = None

    @classmethod
    def load(
        cls,
        model_path: str = "Qwen/Qwen2.5-3B-Instruct",
        adapter_path: Optional[str] = None,
        device: Optional[str] = None,
        runtime: str = "auto",
    ) -> QwenGenerator:
        gen = cls(model_path=model_path, adapter_path=adapter_path, runtime=runtime, device=device)

        if runtime in ("auto", "torch") and torch is not None and AutoModelForCausalLM is not None:
            try:
                dev = device or ("cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))
                token = os.environ.get("HF_TOKEN")

                # Detect compute dtype: FP16 on T4 / general CUDA, BF16 only if supported
                if dev == "cuda" and torch.cuda.is_bf16_supported():
                    compute_dtype = torch.bfloat16
                elif dev in ("cuda", "mps"):
                    compute_dtype = torch.float16
                else:
                    compute_dtype = torch.float32

                print(f"Loading Qwen2.5 Generator ({model_path}) on {dev} with dtype={compute_dtype}...")
                gen.tokenizer = AutoTokenizer.from_pretrained(model_path, token=token)
                if gen.tokenizer.pad_token is None:
                    gen.tokenizer.pad_token = gen.tokenizer.eos_token

                model = AutoModelForCausalLM.from_pretrained(
                    model_path,
                    torch_dtype=compute_dtype,
                    device_map={"": dev} if dev == "cuda" else None,
                    token=token,
                )

                if adapter_path and os.path.exists(adapter_path) and PeftModel is not None:
                    print(f"Loading PEFT adapter from {adapter_path}...")
                    model = PeftModel.from_pretrained(model, adapter_path)

                if dev not in ("cuda", "cpu"):
                    model = model.to(dev)

                model.eval()
                gen.model = model
                gen.hf_pipeline = pipeline("text-generation", model=model, tokenizer=gen.tokenizer, device=0 if dev == "cuda" else -1)
                gen.runtime = "torch"
                return gen
            except Exception as e:
                print(f"PyTorch generator load skipped ({e}), falling back to extractive generator...")

        gen.runtime = "fallback"
        return gen

    @staticmethod
    def format_prompt(question: str, evidence: str) -> str:
        """Standardized chat prompt ensuring 100% parity between training and inference."""
        ev_clean = evidence.strip() if evidence else "Không có căn cứ cụ thể."
        return (
            f"<|im_start|>system\n"
            f"Bạn là trợ lý pháp luật chuyên nghiệp. Hãy trả lời câu hỏi dựa trên căn cứ pháp lý được cung cấp. "
            f"Giữ nguyên các số hiệu văn bản, điều, khoản, số tiền phạt, ngày tháng và thuật ngữ pháp lý.<|im_end|>\n"
            f"<|im_start|>user\n"
            f"[CĂN CỨ PHÁP LÝ]\n{ev_clean}\n\n"
            f"[CÂU HỎI]\n{question.strip()}<|im_end|>\n"
            f"<|im_start|>assistant\n"
        )

    def generate(self, question: str, evidence: str, max_new_tokens: int = 220) -> str:
        prompt = self.format_prompt(question, evidence)

        # 1. PyTorch / CUDA Generation
        if self.runtime == "torch" and self.hf_pipeline is not None:
            try:
                out = self.hf_pipeline(
                    prompt,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    repetition_penalty=1.05,
                    pad_token_id=self.tokenizer.pad_token_id if self.tokenizer else None,
                )
                generated_text = out[0]["generated_text"]
                if "<|im_start|>assistant\n" in generated_text:
                    return generated_text.split("<|im_start|>assistant\n")[-1].replace("<|im_end|>", "").strip()
                return generated_text.strip()
            except Exception as e:
                print(f"PyTorch generation error: {e}")

        # 2. Fallback extractive generator
        lines = [l.strip() for l in evidence.split("\n") if l.strip() and not l.startswith("[DOCUMENT]") and not l.startswith("[ARTICLE]")]
        main_content = "\n".join(lines[:4]) if lines else "Căn cứ theo quy định của pháp luật."
        return f"Căn cứ quy định pháp luật:\n{main_content}"
