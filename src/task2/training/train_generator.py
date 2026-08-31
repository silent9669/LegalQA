"""Fine-tuning module for Qwen2.5 with modern TRL QLoRA SFT on LegalQA Dual-T4 GPUs."""

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
    safety_margin: int = 8,
) -> Tuple[Optional[str], Dict[str, Any]]:
    """Token-aware example builder guaranteeing gold answer tokens are 100% preserved.

    Returns:
        (full_text, diagnostics_dict) where diagnostics_dict contains 'prompt' and 'completion'
        for modern TRL prompt-completion SFT datasets.
    """
    ans_clean = str(answer).strip()
    q_clean = str(question).strip()
    completion_text = f"{ans_clean}<|im_end|>"

    if tokenizer is not None and hasattr(tokenizer, "apply_chat_template"):
        ans_tokens = tokenizer.encode(completion_text, add_special_tokens=False)
        ans_token_count = len(ans_tokens)

        empty_prompt = format_qwen_chat_prompt(q_clean, "", tokenizer=tokenizer)
        framing_tokens = tokenizer.encode(empty_prompt, add_special_tokens=False)
        framing_token_count = len(framing_tokens)

        minimum_required = framing_token_count + ans_token_count
        if minimum_required > max_seq_len:
            # Dropped: answer itself plus minimal framing cannot fit within max_seq_len
            return None, {
                "total_tokens": minimum_required,
                "evidence_truncated": False,
                "answer_truncated": False,
                "dropped": True,
                "prompt": empty_prompt,
                "completion": completion_text,
            }

        evidence_budget = max_seq_len - minimum_required - safety_margin

        paragraphs = [p.strip() for p in evidence_text.split("\n\n") if p.strip()] if evidence_text else []
        packed_paragraphs = []
        curr_ev_tokens = 0
        ev_truncated = False

        for p in paragraphs:
            p_toks = len(tokenizer.encode(p, add_special_tokens=False))
            if curr_ev_tokens + p_toks <= evidence_budget:
                packed_paragraphs.append(p)
                curr_ev_tokens += p_toks
            else:
                ev_truncated = True
                remaining_toks = evidence_budget - curr_ev_tokens
                if remaining_toks > 15:
                    p_enc = tokenizer.encode(p, add_special_tokens=False)[:remaining_toks]
                    p_dec = tokenizer.decode(p_enc, skip_special_tokens=True).rstrip() + "..."
                    packed_paragraphs.append(p_dec)
                break

        final_evidence = "\n\n".join(packed_paragraphs) if packed_paragraphs else ""
        prompt = format_qwen_chat_prompt(q_clean, final_evidence, tokenizer=tokenizer)
        full_text = f"{prompt}{completion_text}"
        full_ids = tokenizer.encode(full_text, add_special_tokens=False)

        # While loop safety guard: remove packed paragraphs from the end if tokenizer special tokens exceed max_seq_len
        while len(full_ids) > max_seq_len and packed_paragraphs:
            packed_paragraphs.pop()
            final_evidence = "\n\n".join(packed_paragraphs)
            prompt = format_qwen_chat_prompt(q_clean, final_evidence, tokenizer=tokenizer)
            full_text = f"{prompt}{completion_text}"
            full_ids = tokenizer.encode(full_text, add_special_tokens=False)
            ev_truncated = True

        assert len(full_ids) <= max_seq_len, f"Full text token length {len(full_ids)} > {max_seq_len}"

        diagnostics = {
            "total_tokens": len(full_ids),
            "evidence_truncated": ev_truncated,
            "answer_truncated": False,
            "dropped": False,
            "prompt": prompt,
            "completion": completion_text,
        }
        return full_text, diagnostics

    # Fallback when tokenizer is not available
    ev_safe = truncate_evidence_preserving_answer(q_clean, evidence_text, ans_clean, max_chars=3000)
    prompt = format_qwen_chat_prompt(q_clean, ev_safe, tokenizer=None)
    full_text = f"{prompt}{completion_text}"
    return full_text, {
        "total_tokens": len(full_text.split()),
        "evidence_truncated": False,
        "answer_truncated": False,
        "dropped": False,
        "prompt": prompt,
        "completion": completion_text,
    }


def build_grounded_training_examples(
    qa_path: Optional[str] = None,
    df_qa: Optional[pd.DataFrame] = None,
    labels_path: Optional[str] = None,
    chunks_path: Optional[str] = None,
    fold_to_exclude: Optional[int] = None,
    tokenizer: Optional[Any] = None,
    max_seq_len: int = 2048,
    max_train_examples: Optional[int] = None,
    seed: int = 42,
) -> List[Dict[str, str]]:
    """Build multi-positive structured SFT training examples with prompt-completion formatting.

    P1-1: Implements bounded preprocessing for smoke profiles (samples QA first and reads
    only needed columns and matching positive chunks).
    """
    if df_qa is None:
        if qa_path and os.path.exists(qa_path):
            df_qa = pd.read_parquet(qa_path)
        else:
            return []

    if fold_to_exclude is not None and "fold_id" in df_qa.columns:
        df_qa = df_qa[df_qa["fold_id"] != fold_to_exclude]

    # 1. Deterministic sampling FIRST for bounded smoke subsets (P1-1)
    if max_train_examples is not None and len(df_qa) > max_train_examples:
        df_qa = df_qa.sample(n=max_train_examples, random_state=seed).reset_index(drop=True)
    else:
        df_qa = df_qa.reset_index(drop=True)

    target_qa_ids = set(df_qa["qa_id"].astype(str)) if "qa_id" in df_qa.columns else set()
    if not target_qa_ids and "id" in df_qa.columns:
        target_qa_ids = set(df_qa["id"].astype(str))

    # 2. Filter retrieval labels to only target QA IDs
    needed_chunk_ids: set[str] = set()
    qa_to_pos_chunk_ids: Dict[str, List[str]] = {}

    if labels_path and os.path.exists(labels_path):
        try:
            df_labels = pd.read_parquet(labels_path, columns=["qa_id", "positive_chunk_id"])
        except Exception:
            df_labels = pd.read_parquet(labels_path)

        if "qa_id" in df_labels.columns and "positive_chunk_id" in df_labels.columns:
            if target_qa_ids:
                df_labels = df_labels[df_labels["qa_id"].astype(str).isin(target_qa_ids)]

            for _, row in df_labels.iterrows():
                qid = str(row["qa_id"]).strip()
                cid = str(row.get("positive_chunk_id", "")).strip()
                if cid:
                    if qid not in qa_to_pos_chunk_ids:
                        qa_to_pos_chunk_ids[qid] = []
                    if cid not in qa_to_pos_chunk_ids[qid]:
                        qa_to_pos_chunk_ids[qid].append(cid)
                    needed_chunk_ids.add(cid)

    # 3. Read chunks with column projection and selective row filtering (P1-1)
    chunk_map: Dict[str, str] = {}
    if chunks_path and os.path.exists(chunks_path):
        try:
            df_chunks = pd.read_parquet(chunks_path, columns=["chunk_id", "text_raw"])
        except Exception:
            df_chunks = pd.read_parquet(chunks_path)

        if "chunk_id" in df_chunks.columns and "text_raw" in df_chunks.columns:
            if needed_chunk_ids and len(needed_chunk_ids) < len(df_chunks):
                df_chunks = df_chunks[df_chunks["chunk_id"].astype(str).isin(needed_chunk_ids)]
            chunk_map = dict(zip(df_chunks["chunk_id"].astype(str), df_chunks["text_raw"]))

    qa_to_pos_evidence: Dict[str, List[str]] = {}
    for qid, cids in qa_to_pos_chunk_ids.items():
        qa_to_pos_evidence[qid] = [chunk_map[c] for c in cids if c in chunk_map]

    examples: List[Dict[str, str]] = []
    token_lengths = []
    ev_truncated_count = 0
    dropped_count = 0

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

        if diag.get("dropped") or full_text is None:
            dropped_count += 1
            continue

        # Modern TRL prompt-completion structure with backwards-compatible text field
        examples.append({
            "prompt": diag.get("prompt", ""),
            "completion": diag.get("completion", f"{a}<|im_end|>"),
            "text": full_text,
        })
        token_lengths.append(diag["total_tokens"])
        if diag.get("evidence_truncated"):
            ev_truncated_count += 1

    if token_lengths:
        p50 = float(np.percentile(token_lengths, 50))
        p90 = float(np.percentile(token_lengths, 90))
        p95 = float(np.percentile(token_lengths, 95))
        max_t = int(max(token_lengths))
        print(f"SFT Dataset Stats ({len(examples)} kept, {dropped_count} dropped): P50={p50:.0f}, P90={p90:.0f}, P95={p95:.0f}, Max={max_t}, Ev Truncated={ev_truncated_count/len(examples)*100:.1f}%")

    return examples


def run_seq_len_diagnostic(
    qa_path: str = "artifacts/task2/data/qa_unique.parquet",
    labels_path: str = "artifacts/task2/data/retrieval_labels.parquet",
    chunks_path: str = "artifacts/task2/data/legal_chunks.parquet",
    model_name: str = "Qwen/Qwen2.5-3B-Instruct",
    seq_lens: List[int] = [2048, 3072],
) -> Dict[int, Dict[str, Any]]:
    """Actionable sequence length diagnostic comparing truncation and drop rates at different lengths (P1-2)."""
    try:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_name)
    except Exception:
        tokenizer = None

    results = {}
    print("\n=== Running Actionable SFT Sequence-Length Diagnostic ===")
    for length in seq_lens:
        examples = build_grounded_training_examples(
            qa_path=qa_path,
            labels_path=labels_path,
            chunks_path=chunks_path,
            tokenizer=tokenizer,
            max_seq_len=length,
        )
        token_lengths = [len(ex.get("text", "").split()) for ex in examples]
        results[length] = {
            "num_examples": len(examples),
            "max_seq_len": length,
            "p50_tokens": float(np.percentile(token_lengths, 50)) if token_lengths else 0.0,
            "p90_tokens": float(np.percentile(token_lengths, 90)) if token_lengths else 0.0,
            "p95_tokens": float(np.percentile(token_lengths, 95)) if token_lengths else 0.0,
            "max_tokens": int(max(token_lengths)) if token_lengths else 0,
        }
        print(f" - Max Seq Len {length}: {len(examples)} examples preserved")
    return results


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
    max_steps: Optional[int] = None,
    max_train_examples: Optional[int] = None,
    is_final_checkpoint: Optional[bool] = None,
    fail_on_error: bool = True,
) -> Dict[str, Any]:
    """Execute QLoRA fine-tuning on GPU 0 using modern TRL prompt-completion SFT with strict reload verification."""
    print(f"=== Starting QLoRA Generator Fine-Tuning ({model_name}) ===")
    assert_no_secrets_in_workspace(Path.cwd())

    try:
        import torch
        from datasets import Dataset as HFDataset
        from peft import LoraConfig, PeftModel
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
        from trl import SFTConfig, SFTTrainer
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

    # Track VRAM on target GPU (P1-3)
    if torch.cuda.is_available() and dev.startswith("cuda"):
        try:
            torch.cuda.reset_peak_memory_stats(dev)
        except Exception:
            pass

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
        max_train_examples=max_train_examples,
    )
    if not examples:
        msg = f"No SFT training examples generated. Check {qa_path} and {labels_path}."
        if fail_on_error:
            raise FileNotFoundError(f"FINAL_PIPELINE_ERROR: {msg}")
        return {"status": "skipped", "reason": "no_data"}

    dataset = HFDataset.from_list(examples)
    print(f"Training dataset ready: {len(dataset)} examples (val_fold={val_fold}, max_steps={max_steps}).")

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

    # Modern TRL prompt-completion configuration (P0-1)
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
        "logging_steps": 10,
        "save_strategy": "epoch",
        "report_to": "none",
        "fp16": not use_bf16,
        "bf16": use_bf16,
    }
    if max_steps is not None:
        sft_kwargs["max_steps"] = max_steps

    # Instantiate SFTConfig with completion_only_loss or fallback
    try:
        sft_args = SFTConfig(
            max_length=max_seq_len,
            completion_only_loss=True,
            **sft_kwargs,
        )
    except TypeError:
        try:
            sft_args = SFTConfig(
                max_seq_length=max_seq_len,
                completion_only_loss=True,
                **sft_kwargs,
            )
        except TypeError:
            try:
                sft_args = SFTConfig(max_length=max_seq_len, **sft_kwargs)
            except TypeError:
                sft_args = SFTConfig(max_seq_length=max_seq_len, **sft_kwargs)

    # Modern SFTTrainer automatically handles prompt/completion columns
    trainer = SFTTrainer(
        model=model,
        args=sft_args,
        train_dataset=dataset,
        processing_class=tokenizer,
        peft_config=peft_config,
    )

    trainer.train(resume_from_checkpoint=resume_from_checkpoint)

    # Measure peak VRAM allocated on GPU (P1-3)
    peak_vram_mb = 0.0
    if torch.cuda.is_available() and dev.startswith("cuda"):
        try:
            peak_bytes = torch.cuda.max_memory_allocated(dev)
            peak_vram_mb = round(peak_bytes / (1024 * 1024), 2)
            print(f"QLoRA Training Peak VRAM on {dev}: {peak_vram_mb:.2f} MB")
        except Exception:
            pass

    # Count exact trainable adapter parameters
    adapter_trainable_params = int(sum(p.numel() for p in trainer.model.parameters() if p.requires_grad))
    print(f"Trained PEFT Adapter Parameters: {adapter_trainable_params:,}")

    os.makedirs(output_dir, exist_ok=True)
    trainer.model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

    # Determine finality and scope
    if is_final_checkpoint is None:
        is_final = (val_fold is None and max_steps is None)
    else:
        is_final = is_final_checkpoint

    training_scope = "all_allowed_task2_data" if val_fold is None else f"folds_excluding_{val_fold}"
    if max_steps is not None:
        training_scope = f"smoke_subset_{max_steps}_steps"

    manifest = {
        "base_model": model_name,
        "epochs": epochs,
        "batch_size_per_device": batch_size,
        "gradient_accumulation_steps": grad_accum,
        "learning_rate": lr,
        "val_fold_excluded": val_fold,
        "training_scope": training_scope,
        "is_final_checkpoint": is_final,
        "smoke_only": max_steps is not None,
        "dataset_size": len(dataset),
        "adapter_trainable_params": adapter_trainable_params,
        "peak_vram_mb": peak_vram_mb,
    }
    with open(os.path.join(output_dir, "generator_manifest.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    with open(os.path.join(output_dir, "training_manifest.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    print(f"QLoRA Training complete (is_final={is_final}, scope={training_scope}). Adapter saved to {output_dir}")

    # Strict Reload Smoke Verification with require_adapter=True
    print("Executing strict reload smoke verification...")
    del trainer, model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    try:
        reload_gen = QwenGenerator.load(
            model_path=model_name,
            adapter_path=output_dir,
            device=dev,
            runtime="torch",
            fail_on_fallback=True,
            final_mode=True,
            require_adapter=True,
        )
        if reload_gen.runtime != "torch" or reload_gen.model is None or reload_gen.tokenizer is None:
            raise RuntimeError(f"Reloaded generator is not running in neural torch mode (runtime={reload_gen.runtime})")

        test_out = reload_gen.generate("Quy định xử phạt vi phạm hành chính?", "Điều 1. Phạt tiền từ 1 đến 2 triệu đồng.")
        if not test_out or not test_out.strip():
            raise RuntimeError("Smoke test generated empty text.")
        print(f"Smoke test generation output sample: {test_out[:100]}...")
        del reload_gen
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception as e:
        msg = f"QLoRA checkpoint saved but failed strict reload smoke test: {e}"
        if fail_on_error:
            raise RuntimeError(f"FINAL_PIPELINE_ERROR: {msg}") from e
        print(f"Warning during reload smoke test: {msg}", file=sys.stderr)

    return {"status": "completed", "output_dir": output_dir, "manifest": manifest}
