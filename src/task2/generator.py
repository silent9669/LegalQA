"""Qwen2.5 Generator wrapper with tokenizer chat template parity, FP16 auto-detection, and fail-loudly policy."""

from __future__ import annotations

import os
import sys
from typing import Any, Dict, List, Optional, Tuple

try:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
except ImportError:
    torch = None
    AutoModelForCausalLM = None
    AutoTokenizer = None

try:
    from peft import PeftModel
except ImportError:
    PeftModel = None


SYSTEM_PROMPT = (
    "Bạn là chuyên gia tư vấn pháp luật Việt Nam. Hãy trả lời câu hỏi dựa trên các văn bản pháp luật được cung cấp:\n"
    "1. Nêu đầy đủ căn cứ pháp lý (Tên văn bản, số hiệu, Điều, Khoản, Điểm).\n"
    "2. Trích dẫn và giải thích đầy đủ các điều kiện, mức xử phạt, thời hạn, quyền và nghĩa vụ liên quan.\n"
    "3. Giữ nguyên chính xác số hiệu văn bản, điều khoản, ngày tháng năm, số tiền phạt, tỷ lệ phần trăm và thuật ngữ pháp lý.\n"
    "4. Áp dụng quy định để kết luận trực tiếp, rõ ràng cho câu hỏi được nêu."
)


def format_qwen_chat_prompt(question: str, evidence: str, tokenizer: Optional[Any] = None) -> str:
    """Format prompt with 100% parity between training and inference using native chat template."""
    ev_clean = evidence.strip() if evidence else "Không có căn cứ cụ thể."
    user_content = f"[CĂN CỨ PHÁP LÝ]\n{ev_clean}\n\n[CÂU HỎI]\n{question.strip()}"

    if tokenizer is not None and hasattr(tokenizer, "apply_chat_template"):
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ]
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    # Standard Qwen ChatML fallback format
    return (
        f"<|im_start|>system\n"
        f"{SYSTEM_PROMPT}<|im_end|>\n"
        f"<|im_start|>user\n"
        f"{user_content}<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )


class QwenGenerator:
    """Qwen2.5 (3B / 1.5B) Generator for evidence-conditioned statutory legal answer generation."""

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

    @classmethod
    def load(
        cls,
        model_path: str = "Qwen/Qwen2.5-3B-Instruct",
        adapter_path: Optional[str] = None,
        device: Optional[str] = None,
        runtime: str = "auto",
        fail_on_fallback: bool = False,
    ) -> QwenGenerator:
        """Load generator model, enforcing explicit device mapping and loud failure in competition mode."""
        gen = cls(model_path=model_path, adapter_path=adapter_path, runtime=runtime, device=device)

        if runtime == "fallback":
            gen.runtime = "fallback"
            return gen

        if runtime in ("auto", "torch") and torch is not None and AutoModelForCausalLM is not None:
            try:
                dev = device or ("cuda:0" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))
                token = os.environ.get("HF_TOKEN")

                # Detect compute dtype: FP16 on T4 / CUDA, BF16 only if supported
                if dev.startswith("cuda") and torch.cuda.is_bf16_supported():
                    compute_dtype = torch.bfloat16
                elif dev.startswith("cuda") or dev == "mps":
                    compute_dtype = torch.float16
                else:
                    compute_dtype = torch.float32

                print(f"Loading Qwen Generator ({model_path}) on {dev} with dtype={compute_dtype}...")
                gen.tokenizer = AutoTokenizer.from_pretrained(model_path, token=token)
                if gen.tokenizer.pad_token is None:
                    gen.tokenizer.pad_token = gen.tokenizer.eos_token
                gen.tokenizer.padding_side = "left"

                model = AutoModelForCausalLM.from_pretrained(
                    model_path,
                    torch_dtype=compute_dtype,
                    device_map={"": dev} if dev.startswith("cuda") else None,
                    token=token,
                )

                if adapter_path and os.path.exists(adapter_path) and PeftModel is not None:
                    print(f"Loading PEFT adapter from {adapter_path}...")
                    model = PeftModel.from_pretrained(model, adapter_path)

                if dev not in ("cuda", "cpu") and not dev.startswith("cuda"):
                    model = model.to(dev)

                model.eval()
                gen.model = model
                gen.device = dev
                gen.runtime = "torch"
                return gen
            except Exception as e:
                msg = f"Failed to load PyTorch generator ({model_path}): {e}"
                if fail_on_fallback:
                    raise RuntimeError(f"FINAL_PIPELINE_ERROR: {msg}")
                print(f"Warning: {msg}, falling back to extractive generator...", file=sys.stderr)

        if fail_on_fallback:
            raise RuntimeError("FINAL_PIPELINE_ERROR: PyTorch/Transformers not available for neural generator.")

        gen.runtime = "fallback"
        return gen

    def format_instance_prompt(self, question: str, evidence: str) -> str:
        """Format prompt using the instance's loaded tokenizer."""
        return format_qwen_chat_prompt(question, evidence, tokenizer=self.tokenizer)

    @staticmethod
    def format_prompt(question: str, evidence: str, tokenizer: Optional[Any] = None) -> str:
        return format_qwen_chat_prompt(question, evidence, tokenizer=tokenizer)

    def generate(self, question: str, evidence: str, max_new_tokens: int = 384) -> str:
        prompt = self.format_instance_prompt(question, evidence)

        # 1. PyTorch / CUDA Generation
        if self.runtime == "torch" and self.model is not None and self.tokenizer is not None:
            try:
                inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048)
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                with torch.inference_mode():
                    outputs = self.model.generate(
                        **inputs,
                        max_new_tokens=max_new_tokens,
                        do_sample=False,
                        repetition_penalty=1.05,
                        pad_token_id=self.tokenizer.pad_token_id,
                    )
                generated_ids = outputs[0][inputs["input_ids"].shape[1]:]
                decoded = self.tokenizer.decode(generated_ids, skip_special_tokens=True)
                return decoded.strip()
            except Exception as e:
                print(f"PyTorch generation error: {e}", file=sys.stderr)

        # 2. Fallback extractive generator
        lines = [
            l.strip() for l in evidence.split("\n")
            if l.strip() and not l.startswith("[DOCUMENT]") and not l.startswith("[ARTICLE]")
        ]
        main_content = "\n".join(lines[:6]) if lines else "Căn cứ theo quy định của pháp luật."
        return f"Căn cứ quy định pháp luật:\n{main_content}"

    def generate_batch(
        self,
        items: List[Tuple[str, str]],  # (question, evidence)
        max_new_tokens: int = 384,
        batch_size: int = 4,
    ) -> List[str]:
        """High-throughput batched generation on CUDA."""
        if not items:
            return []

        if self.runtime != "torch" or self.model is None or self.tokenizer is None:
            return [self.generate(q, ev, max_new_tokens=max_new_tokens) for q, ev in items]

        prompts = [self.format_instance_prompt(q, ev) for q, ev in items]
        results: List[str] = []

        for i in range(0, len(prompts), batch_size):
            b_prompts = prompts[i:i + batch_size]
            enc = self.tokenizer(
                b_prompts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=2048,
            )
            enc = {k: v.to(self.device) for k, v in enc.items()}
            with torch.inference_mode():
                out = self.model.generate(
                    **enc,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    repetition_penalty=1.05,
                    pad_token_id=self.tokenizer.pad_token_id,
                )
            for j, prompt_len in enumerate(enc["input_ids"].shape[1] for _ in range(len(b_prompts))):
                gen_ids = out[j][enc["input_ids"].shape[1]:]
                decoded = self.tokenizer.decode(gen_ids, skip_special_tokens=True)
                results.append(decoded.strip())

        return results
