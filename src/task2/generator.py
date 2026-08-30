import os

class QwenGenerator:
    def __init__(self, model_path: str = "Qwen/Qwen2.5-3B-Instruct", adapter_path: str = None, runtime: str = "auto"):
        self.model_path = model_path
        self.adapter_path = adapter_path
        self.runtime = runtime
        self.mlx_model = None
        self.mlx_tokenizer = None
        self.hf_pipeline = None

    @classmethod
    def load(cls, model_path: str = "Qwen/Qwen2.5-3B-Instruct", adapter_path: str = None, runtime: str = "auto"):
        gen = cls(model_path=model_path, adapter_path=adapter_path, runtime=runtime)
        if runtime in ("auto", "mlx"):
            try:
                import mlx_lm
                print(f"Loading MLX model {model_path} with adapter {adapter_path}...")
                model, tokenizer = mlx_lm.load(model_path, adapter_path=adapter_path)
                gen.mlx_model = model
                gen.mlx_tokenizer = tokenizer
                gen.runtime = "mlx"
                print("MLX model loaded successfully.")
                return gen
            except Exception as e:
                print(f"MLX load skipped ({e}), attempting PyTorch/fallback...")

        if runtime in ("auto", "torch"):
            try:
                import torch
                from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
                from peft import PeftModel

                device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
                tokenizer = AutoTokenizer.from_pretrained(model_path)
                model = AutoModelForCausalLM.from_pretrained(
                    model_path,
                    torch_dtype=torch.bfloat16 if device != "cpu" else torch.float32,
                    device_map="auto" if device == "cuda" else None
                )
                if adapter_path and os.path.exists(adapter_path):
                    model = PeftModel.from_pretrained(model, adapter_path)
                if device != "cuda":
                    model = model.to(device)

                gen.hf_pipeline = pipeline("text-generation", model=model, tokenizer=tokenizer, device=0 if device == "cuda" else -1)
                gen.runtime = "torch"
                return gen
            except Exception as e:
                print(f"PyTorch load skipped ({e}), falling back to extractive generator...")

        gen.runtime = "fallback"
        return gen

    def _format_prompt(self, question: str, evidence: str) -> str:
        return (
            f"<|im_start|>system\n"
            f"Bạn là trợ lý pháp luật chuyên nghiệp. Hãy trả lời câu hỏi dựa trên căn cứ pháp lý được cung cấp. "
            f"Giữ nguyên các số hiệu văn bản, điều, khoản, số tiền phạt, ngày tháng và thuật ngữ pháp lý.<|im_end|>\n"
            f"<|im_start|>user\n"
            f"[CĂN CỨ PHÁP LÝ]\n{evidence}\n\n"
            f"[CÂU HỎI]\n{question}<|im_end|>\n"
            f"<|im_start|>assistant\n"
        )

    def generate(self, question: str, evidence: str, max_new_tokens: int = 512) -> str:
        prompt = self._format_prompt(question, evidence)

        # 1. MLX Runtime for Apple Silicon
        if self.runtime == "mlx" and self.mlx_model is not None:
            try:
                import mlx_lm
                import mlx_lm.sample_utils
                sampler = mlx_lm.sample_utils.make_sampler(temp=0.0)
                logits_processors = mlx_lm.sample_utils.make_logits_processors(repetition_penalty=1.12, repetition_context_size=50)
                res = mlx_lm.generate(
                    self.mlx_model,
                    self.mlx_tokenizer,
                    prompt=prompt,
                    max_tokens=max_new_tokens,
                    sampler=sampler,
                    logits_processors=logits_processors,
                    verbose=False
                )
                return res.strip()
            except Exception as e:
                print(f"MLX generation error: {e}")

        # 2. PyTorch / CUDA Runtime
        if self.runtime == "torch" and self.hf_pipeline is not None:
            try:
                out = self.hf_pipeline(prompt, max_new_tokens=max_new_tokens, do_sample=False)
                generated_text = out[0]["generated_text"]
                if "<|im_start|>assistant\n" in generated_text:
                    return generated_text.split("<|im_start|>assistant\n")[-1].replace("<|im_end|>", "").strip()
                return generated_text.strip()
            except Exception as e:
                print(f"PyTorch generation error: {e}")

        # 3. Fallback extractive generator
        lines = [l.strip() for l in evidence.split("\n") if l.strip()]
        main_content = "\n".join(lines[:4])
        return f"Căn cứ quy định pháp luật:\n{main_content}"
