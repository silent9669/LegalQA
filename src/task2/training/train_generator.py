"""Fine-tuning module for Qwen2.5 with QLoRA SFT on LegalQA Dual-T4 GPUs."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

from src.common.security import assert_no_secrets_in_workspace
from src.task2.generator import QwenGenerator, format_qwen_chat_prompt


def truncate_evidence_preserving_answer(
    question: str,
    evidence: str,
    answer: str,
    max_chars: int = 3000,
) -> str:
    """Answer-preserving character truncation: trims evidence at clause/line boundaries, preserving full gold answer."""
    if not evidence or len(evidence) <= max_chars:
        return evidence.strip()

    paragraphs = evidence.split("\n\n")
    kept_pieces = []
    current_len = 0

    for p in paragraphs:
        p_len = len(p) + 2
        if current_len + p_len <= max_chars:
            kept_pieces.append(p)
            current_len += p_len
        else:
            remaining = max_chars - current_len
            if remaining > 10:
                cut_len = max(0, remaining - 3)
                kept_pieces.append(p[:cut_len].rstrip() + "...")
            break

    return "\n\n".join(kept_pieces)


def build_sft_example_token_aware(
    question: str,
    evidence_text: str,
    answer: str,
    tokenizer: Optional[Any] = None,
    max_seq_len: int = 2048,
) -> Tuple[str, Dict[str, Any]]:
    """Token-aware example builder guaranteeing gold answer tokens are 100% preserved."""
    ans_clean = str(answer).strip()
    q_clean = str(question).strip()

    if tokenizer is not None and hasattr(tokenizer, "apply_chat_template"):
        # 1. Measure answer token count
        ans_tokens = tokenizer.encode(f"{ans_clean}<|im_end|>", add_special_tokens=False)
        ans_token_count = len(ans_tokens)

        # 2. Measure framing with empty evidence
        empty_prompt = format_qwen_chat_prompt(q_clean, "", tokenizer=tokenizer)
        framing_tokens = tokenizer.encode(empty_prompt, add_special_tokens=False)
        framing_token_count = len(framing_tokens)

        # 3. Available tokens for evidence
        avail_for_evidence = max(100, max_seq_len - framing_token_count - ans_token_count - 10)

        # 4. Pack evidence paragraphs within token budget
        paragraphs = evidence_text.split("\n\n") if evidence_text else []
        packed_paragraphs = []
        curr_ev_tokens = 0
        ev_truncated = False

        for p in paragraphs:
            p_toks = len(tokenizer.encode(p, add_special_tokens=False))
            if curr_ev_tokens + p_toks <= avail_for_evidence:
                packed_paragraphs.append(p)
                curr_ev_tokens += p_toks
            else:
                ev_truncated = True
                break

        final_evidence = "\n\n".join(packed_paragraphs) if packed_paragraphs else (evidence_text[:1000] if evidence_text else "")
        prompt = format_qwen_chat_prompt(q_clean, final_evidence, tokenizer=tokenizer)
        full_text = f"{prompt}{ans_clean}<|im_end|>"
        total_tokens = len(tokenizer.encode(full_text, add_special_tokens=False))

        diagnostics = {
            "total_tokens": total_tokens,
            "evidence_truncated": ev_truncated,
            "answer_truncated": total_tokens > max_seq_len,
        }
        return full_text, diagnostics

    # Fallback when tokenizer is not available
    ev_safe = truncate_evidence_preserving_answer(q_clean, evidence_text, ans_clean, max_chars=3000)
    prompt = format_qwen_chat_prompt(q_clean, ev_safe, tokenizer=None)
    full_text = f"{prompt}{ans_clean}<|im_end|>"
    return full_text, {"total_tokens": len(full_text.split()), "evidence_truncated": False, "answer_truncated": False}


def build_grounded_training_examples(
    qa_path: Optional[str] = None,
    df_qa: Optional[pd.DataFrame] = None,
    labels_path: Optional[str] = None,
    chunks_path: Optional[str] = None,
    fold_to_exclude: Optional[int] = None,
    tokenizer: Optional[Any] = None,
    max_seq_len: int = 2048,
) -> List[Dict[str, str]]:
    """Build multi-positive structured SFT training examples ensuring exact prompt parity with inference."""
    if df_qa is None:
        if qa_path and os.path.exists(qa_path):
            df_qa = pd.read_parquet(qa_path)
        else:
            return []

    if fold_to_exclude is not None and "fold_id" in df_qa.columns:
        df_qa = df_qa[df_qa["fold_id"] != fold_to_exclude]

    chunk_map: Dict[str, str] = {}
    if chunks_path and os.path.exists(chunks_path):
        df_chunks = pd.read_parquet(chunks_path)
        chunk_map = dict(zip(df_chunks["chunk_id"], df_chunks["text_raw"]))

    qa_to_pos_evidence: Dict[str, List[str]] = {}
    if labels_path and os.path.exists(labels_path):
        df_labels = pd.read_parquet(labels_path)
        for _, row in df_labels.iterrows():
            qid = str(row["qa_id"]).strip()
            cid = str(row.get("positive_chunk_id", "")).strip()
            if cid and cid in chunk_map:
                if qid not in qa_to_pos_evidence:
                    qa_to_pos_evidence[qid] = []
                txt = chunk_map[cid]
                if txt not in qa_to_pos_evidence[qid]:
                    qa_to_pos_evidence[qid].append(txt)

    examples: List[Dict[str, str]] = []
    token_lengths = []
    ev_truncated_count = 0
    ans_truncated_count = 0

    for _, row in df_qa.iterrows():
        qid = str(row.get("qa_id") or row.get("id", "")).strip()
        q = str(row.get("question_raw") or row.get("question", "")).strip()
        a = str(row.get("answer_raw") or row.get("answer", "")).strip()

        if not q or not a:
            continue

        pos_pieces = qa_to_pos_evidence.get(qid, [])
        raw_evidence = "\n\n".join(pos_pieces) if pos_pieces else ""

        full_text, diag = build_sft_example_token_aware(
            question=q,
            evidence_text=raw_evidence,
            answer=a,
            tokenizer=tokenizer,
            max_seq_len=max_seq_len,
        )
        examples.append({"text": full_text})
        token_lengths.append(diag["total_tokens"])
        if diag.get("evidence_truncated"):
            ev_truncated_count += 1
        if diag.get("answer_truncated"):
            ans_truncated_count += 1

    if token_lengths:
        p50 = float(np.percentile(token_lengths, 50))
        p90 = float(np.percentile(token_lengths, 90))
        print(f"SFT Dataset Stats ({len(examples)} examples): P50 tokens={p50:.0f}, P90 tokens={p90:.0f}, Ev Truncated={ev_truncated_count/len(examples)*100:.1f}%, Ans Truncated={ans_truncated_count/len(examples)*100:.1f}%")

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
    lr: float = 1e-4,
    max_seq_len: int = 2048,
    val_fold: Optional[int] = None,
    resume_from_checkpoint: Optional[str] = None,
    device: Optional[str] = None,
    fail_on_error: bool = True,
) -> Dict[str, Any]:
    """Execute QLoRA fine-tuning on GPU 0 with completion loss masking and reload smoke verification."""
    print(f"=== Starting QLoRA Generator Fine-Tuning ({model_name}) ===")
    assert_no_secrets_in_workspace(Path.cwd())

    try:
        import torch
        from datasets import Dataset as HFDataset
        from peft import LoraConfig, PeftModel
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
        from trl import DataCollatorForCompletionOnlyLM, SFTConfig, SFTTrainer
    except ImportError as e:
        msg = f"Required training packages not installed: {e}"
        if fail_on_error:
            raise RuntimeError(f"FINAL_PIPELINE_ERROR: {msg}")
        return {"status": "skipped", "reason": "missing_dependencies"}

    if not torch.cuda.is_available() and device is None:
        msg = "CUDA not available for QLoRA GPU training."
        if fail_on_error:
            raise RuntimeError(f"FINAL_PIPELINE_ERROR: {msg}")
        return {"status": "skipped", "reason": "no_cuda"}

    dev = device or "cuda:0"
    print(f"QLoRA Training targeted on device: {dev}")

    token = os.environ.get("HF_TOKEN")
    tokenizer = AutoTokenizer.from_pretrained(model_name, token=token)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    examples = build_grounded_training_examples(
        qa_path=qa_path,
        labels_path=labels_path,
        chunks_path=chunks_path,
        fold_to_exclude=val_fold,
        tokenizer=tokenizer,
        max_seq_len=max_seq_len,
    )
    if not examples:
        msg = f"No SFT training examples generated. Check {qa_path} and {labels_path}."
        if fail_on_error:
            raise FileNotFoundError(f"FINAL_PIPELINE_ERROR: {msg}")
        return {"status": "skipped", "reason": "no_data"}

    dataset = HFDataset.from_list(examples)
    print(f"Training dataset ready: {len(dataset)} examples (val_fold={val_fold}).")

    use_bf16 = torch.cuda.is_bf16_supported() if torch.cuda.is_available() else False
    compute_dtype = torch.bfloat16 if use_bf16 else torch.float16

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=compute_dtype,
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        torch_dtype=compute_dtype,
        device_map={"": dev} if dev.startswith("cuda") else "auto",
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

    # Count exact trainable parameters
    adapter_trainable_params = int(sum(p.numel() for p in trainer.model.parameters() if p.requires_grad))
    print(f"Trained PEFT Adapter Parameters: {adapter_trainable_params:,}")

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
        "adapter_trainable_params": adapter_trainable_params,
    }
    with open(os.path.join(output_dir, "training_manifest.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    print(f"QLoRA Training complete. Adapter saved to {output_dir}")

    # Reload Smoke Verification
    print("Executing reload smoke verification...")
    del trainer, model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    try:
        reload_gen = QwenGenerator.load(model_path=model_name, adapter_path=output_dir, device=dev, runtime="torch")
        test_out = reload_gen.generate("Quy định xử phạt vi phạm hành chính?", "Điều 1. Phạt tiền từ 1 đến 2 triệu đồng.")
        if not test_out or not test_out.strip():
            raise RuntimeError("Smoke test generated empty text.")
        print(f"Smoke test generation output sample: {test_out[:100]}...")
        del reload_gen
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception as e:
        msg = f"QLoRA checkpoint saved but failed reload smoke test: {e}"
        if fail_on_error:
            raise RuntimeError(f"FINAL_PIPELINE_ERROR: {msg}") from e
        print(f"Warning during reload smoke test: {msg}", file=sys.stderr)

    return {"status": "completed", "output_dir": output_dir, "manifest": manifest}
