import argparse
import os
import sys
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def run_qlora_training(
    model_name: str = "Qwen/Qwen2.5-3B-Instruct",
    qa_path: str = "artifacts/task2/data/qa_unique.parquet",
    output_dir: str = "artifacts/task2/checkpoints/generator/hf_adapter",
    epochs: int = 1,
    batch_size: int = 1,
    grad_accum: int = 16,
    lr: float = 1e-4,
    max_seq_len: int = 3584
):
    print("=== Starting PyTorch / CUDA QLoRA SFT Fine-Tuning ===")
    print(f"Base Model: {model_name}")
    print(f"Target Output: {output_dir}")

    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
        from peft import LoraConfig
        from trl import SFTTrainer, SFTConfig, DataCollatorForCompletionOnlyLM
        from datasets import Dataset as HFDataset
    except ImportError:
        print("Required libraries (torch, transformers, peft, trl) not fully available in this local environment.")
        print("Use Google Colab via notebooks/DSC2026_Task2_LegalQA_Pipeline.ipynb to run GPU QLoRA training.")
        return

    print(f"Loading dataset from {qa_path}...")
    df = pd.read_parquet(qa_path)
    texts = []
    for _, row in df.iterrows():
        q = row["question_raw"]
        a = row["answer_raw"]
        text = (
            f"<|im_start|>system\n"
            f"Bạn là trợ lý pháp luật chuyên nghiệp. Hãy trả lời câu hỏi dựa trên căn cứ pháp lý được cung cấp.<|im_end|>\n"
            f"<|im_start|>user\n{q}<|im_end|>\n"
            f"<|im_start|>assistant\n{a}<|im_end|>"
        )
        texts.append(text)

    dataset = HFDataset.from_dict({"text": texts})

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )

    peft_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    )

    response_template_ids = tokenizer.encode("<|im_start|>assistant\n", add_special_tokens=False)
    collator = DataCollatorForCompletionOnlyLM(response_template_ids, tokenizer=tokenizer)

    sft_args = SFTConfig(
        output_dir=os.path.join(output_dir, "runs"),
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=grad_accum,
        gradient_checkpointing=True,
        max_seq_length=max_seq_len,
        optim="paged_adamw_8bit",
        num_train_epochs=epochs,
        learning_rate=lr,
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
        dataset_text_field="text",
        logging_steps=10,
        report_to="none"
    )

    trainer = SFTTrainer(
        model=model,
        args=sft_args,
        train_dataset=dataset,
        processing_class=tokenizer,
        peft_config=peft_config,
        data_collator=collator
    )

    trainer.train()
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"Training complete. Adapter saved to {output_dir}")

def main():
    parser = argparse.ArgumentParser(description="QLoRA SFT Fine-Tuning for CUDA/Colab")
    parser.add_argument("--model", default="Qwen/Qwen2.5-3B-Instruct")
    parser.add_argument("--epochs", type=int, default=1)
    args = parser.parse_args()

    run_qlora_training(model_name=args.model, epochs=args.epochs)

if __name__ == "__main__":
    main()
