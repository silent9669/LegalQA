"""Fine-tune Qwen2.5-3B-Instruct with QLoRA on Kaggle Dual NVIDIA T4 GPUs."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.common.security import assert_no_secrets_in_workspace
from src.task2.generator import QwenGenerator


def check_gpu_hardware_compatibility() -> Dict[str, Any]:
    """Verify GPU hardware, compute capability, and T4 compatibility."""
    try:
        import torch
    except ImportError:
        return {"cuda_available": False, "gpu_count": 0, "devices": []}

    if not torch.cuda.is_available():
        return {"cuda_available": False, "gpu_count": 0, "devices": []}

    gpu_count = torch.cuda.device_count()
    devices = []
    for i in range(gpu_count):
        name = torch.cuda.get_device_name(i)
        cap = torch.cuda.get_device_capability(i)
        devices.append({"index": i, "name": name, "capability": f"sm_{cap[0]}{cap[1]}"})

    print(f"Detected {gpu_count} CUDA GPUs: {devices}")
    return {"cuda_available": True, "gpu_count": gpu_count, "devices": devices}


def build_training_examples(
    qa_path: str,
    labels_path: Optional[str] = None,
    chunks_path: Optional[str] = None,
    fold_to_exclude: Optional[int] = None,
) -> List[Dict[str, str]]:
    """Build grounded training examples ensuring exact prompt parity with inference."""
    df_qa = pd.read_parquet(qa_path)
    if fold_to_exclude is not None and "fold_id" in df_qa.columns:
        df_qa = df_qa[df_qa["fold_id"] != fold_to_exclude]

    # Optional retrieval label mapping
    chunk_map: Dict[str, str] = {}
    if chunks_path and os.path.exists(chunks_path):
        df_chunks = pd.read_parquet(chunks_path)
        chunk_map = dict(zip(df_chunks["chunk_id"], df_chunks["text_raw"]))

    qa_to_pos_chunk: Dict[str, str] = {}
    if labels_path and os.path.exists(labels_path):
        df_labels = pd.read_parquet(labels_path)
        for _, row in df_labels.iterrows():
            qid = str(row["qa_id"]).strip()
            cid = str(row.get("positive_chunk_id", "")).strip()
            if cid and cid in chunk_map:
                qa_to_pos_chunk[qid] = chunk_map[cid]

    examples = []
    for _, row in df_qa.iterrows():
        qid = str(row.get("qa_id") or row.get("id", "")).strip()
        q = str(row.get("question_raw") or row.get("question", "")).strip()
        a = str(row.get("answer_raw") or row.get("answer", "")).strip()

        evidence = qa_to_pos_chunk.get(qid, "")
        prompt = QwenGenerator.format_prompt(question=q, evidence=evidence)
        full_text = f"{prompt}{a}<|im_end|>"
        examples.append({"text": full_text})

    return examples


def run_qlora_training(
    model_name: str = "Qwen/Qwen2.5-3B-Instruct",
    qa_path: str = "artifacts/task2/data/qa_unique.parquet",
    labels_path: str = "artifacts/task2/data/retrieval_labels.parquet",
    chunks_path: str = "artifacts/task2/data/legal_chunks.parquet",
    output_dir: str = "artifacts/task2/checkpoints/generator/hf_adapter",
    epochs: int = 1,
    batch_size: int = 1,
    grad_accum: int = 8,
    lr: float = 2e-4,
    max_seq_len: int = 2048,
    val_fold: Optional[int] = None,
    resume_from_checkpoint: Optional[str] = None,
) -> None:
    print("=== Starting PyTorch / CUDA QLoRA Fine-Tuning ===")
    print(f"Base Model: {model_name}")
    print(f"Target Output: {output_dir}")

    # Security check
    assert_no_secrets_in_workspace(Path.cwd())

    # Hardware check
    hw = check_gpu_hardware_compatibility()
    if not hw["cuda_available"]:
        print("CUDA GPU is not available in this environment. Skipping GPU training.")
        return

    try:
        import torch
        from datasets import Dataset as HFDataset
        from peft import LoraConfig
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
        from trl import DataCollatorForCompletionOnlyLM, SFTConfig, SFTTrainer
    except ImportError as e:
        print(f"Required training packages not installed: {e}", file=sys.stderr)
        return

    # Check local rank for DDP
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    is_main_process = local_rank == 0

    if is_main_process:
        print(f"Preparing training dataset (val_fold={val_fold})...")

    examples = build_training_examples(
        qa_path=qa_path,
        labels_path=labels_path,
        chunks_path=chunks_path,
        fold_to_exclude=val_fold,
    )
    dataset = HFDataset.from_list(examples)
    if is_main_process:
        print(f"Training dataset ready: {len(dataset)} examples.")

    token = os.environ.get("HF_TOKEN")
    tokenizer = AutoTokenizer.from_pretrained(model_name, token=token)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    # FP16 on T4 GPUs (sm_75), BF16 only if explicitly supported
    use_bf16 = torch.cuda.is_bf16_supported()
    compute_dtype = torch.bfloat16 if use_bf16 else torch.float16

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=compute_dtype,
    )

    # In DDP multi-GPU mode, map each process to its local device
    device_map = {"": local_rank} if hw["gpu_count"] > 1 else "auto"

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        torch_dtype=compute_dtype,
        device_map=device_map,
        token=token,
    )
    model.config.use_cache = False

    peft_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    )

    response_template_ids = tokenizer.encode("<|im_start|>assistant\n", add_special_tokens=False)
    collator = DataCollatorForCompletionOnlyLM(response_template_ids, tokenizer=tokenizer)

    sft_kwargs: Dict[str, Any] = {
        "output_dir": os.path.join(output_dir, "runs"),
        "per_device_train_batch_size": batch_size,
        "gradient_accumulation_steps": grad_accum,
        "gradient_checkpointing": True,
        "optim": "paged_adamw_8bit",
        "num_train_epochs": epochs,
        "learning_rate": lr,
        "lr_scheduler_type": "cosine",
        "warmup_ratio": 0.03,
        "dataset_text_field": "text",
        "logging_steps": 10,
        "save_strategy": "epoch",
        "report_to": "none",
        "fp16": not use_bf16,
        "bf16": use_bf16,
    }

    # Handle TRL version differences for max sequence length parameter
    try:
        sft_args = SFTConfig(max_length=max_seq_len, **sft_kwargs)
    except TypeError:
        sft_args = SFTConfig(max_seq_length=max_seq_len, **sft_kwargs)

    trainer = SFTTrainer(
        model=model,
        args=sft_args,
        train_dataset=dataset,
        processing_class=tokenizer,
        peft_config=peft_config,
        data_collator=collator,
    )

    trainer.train(resume_from_checkpoint=resume_from_checkpoint)

    if is_main_process:
        os.makedirs(output_dir, exist_ok=True)
        trainer.model.save_pretrained(output_dir)
        tokenizer.save_pretrained(output_dir)
        manifest = {
            "base_model": model_name,
            "epochs": epochs,
            "batch_size_per_device": batch_size,
            "gradient_accumulation_steps": grad_accum,
            "learning_rate": lr,
            "val_fold_excluded": val_fold,
            "dataset_size": len(dataset),
        }
        with open(os.path.join(output_dir, "training_manifest.json"), "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2)
        print(f"Training complete. Adapter saved to {output_dir}")


def main():
    parser = argparse.ArgumentParser(description="QLoRA SFT Fine-Tuning for Kaggle Dual T4 GPUs")
    parser.add_argument("--model", default="Qwen/Qwen2.5-3B-Instruct")
    parser.add_argument("--qa_path", default="artifacts/task2/data/qa_unique.parquet")
    parser.add_argument("--labels_path", default="artifacts/task2/data/retrieval_labels.parquet")
    parser.add_argument("--chunks_path", default="artifacts/task2/data/legal_chunks.parquet")
    parser.add_argument("--output_dir", default="artifacts/task2/checkpoints/generator/hf_adapter")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--grad_accum", type=int, default=8)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--val_fold", type=int, default=None)
    parser.add_argument("--resume", default=None)
    args = parser.parse_args()

    run_qlora_training(
        model_name=args.model,
        qa_path=args.qa_path,
        labels_path=args.labels_path,
        chunks_path=args.chunks_path,
        output_dir=args.output_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        grad_accum=args.grad_accum,
        lr=args.lr,
        val_fold=args.val_fold,
        resume_from_checkpoint=args.resume,
    )


if __name__ == "__main__":
    main()
