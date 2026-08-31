"""Fine-tuning module for Cross-Encoder Reranker (BAAI/bge-reranker-v2-m3) on LegalQA data."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

from src.common.reranker import BGEReranker
from src.common.security import assert_no_secrets_in_workspace


def prepare_reranker_dataset(
    pairs_path: Optional[str] = None,
    df_pairs: Optional[pd.DataFrame] = None,
    val_fold: Optional[int] = None,
    max_train_pairs: Optional[int] = None,
    max_val_pairs: Optional[int] = None,
    seed: int = 42,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Prepare training and validation samples for cross-encoder reranker with strict fold isolation."""
    if df_pairs is None:
        if pairs_path and os.path.exists(pairs_path):
            df_pairs = pd.read_parquet(pairs_path)
        else:
            return [], []

    if val_fold is not None and "fold_id" in df_pairs.columns:
        train_df = df_pairs[df_pairs["fold_id"] != val_fold]
        val_df = df_pairs[df_pairs["fold_id"] == val_fold]
    else:
        train_df = df_pairs
        val_df = pd.DataFrame()

    # Deterministic sampling for bounded smoke runs
    if max_train_pairs is not None and len(train_df) > max_train_pairs:
        train_df = train_df.sample(n=max_train_pairs, random_state=seed).reset_index(drop=True)
    else:
        train_df = train_df.reset_index(drop=True)

    if max_val_pairs is not None and len(val_df) > max_val_pairs:
        val_df = val_df.sample(n=max_val_pairs, random_state=seed).reset_index(drop=True)
    elif not val_df.empty:
        val_df = val_df.reset_index(drop=True)

    train_examples = train_df.to_dict("records")
    val_examples = val_df.to_dict("records") if not val_df.empty else []

    return train_examples, val_examples


def train_bge_reranker(
    pairs_path: str = "artifacts/task2/data/reranker_training_pairs.parquet",
    output_dir: str = "artifacts/task2/checkpoints/reranker/best",
    model_name: str = "BAAI/bge-reranker-v2-m3",
    epochs: int = 1,
    batch_size: int = 2,
    grad_accum: int = 4,
    lr: float = 2e-5,
    val_fold: Optional[int] = None,
    device: Optional[str] = None,
    max_length: int = 384,
    max_steps: Optional[int] = None,
    max_train_pairs: Optional[int] = None,
    max_val_pairs: Optional[int] = None,
    is_final_checkpoint: Optional[bool] = None,
    loss_type: str = "bce",  # "bce" or "pairwise"
    fail_on_error: bool = True,
    seed: int = 42,
) -> Dict[str, Any]:
    """Fine-tune Cross-Encoder Reranker using positive and hard-negative pairs with validation tracking."""
    print(f"=== Starting Cross-Encoder Reranker Fine-Tuning ({model_name}) ===")
    assert_no_secrets_in_workspace(Path.cwd())

    try:
        import torch
        import torch.nn.functional as F
        from torch.utils.data import DataLoader, Dataset
        from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_cosine_schedule_with_warmup
    except ImportError as e:
        msg = f"PyTorch / Transformers not installed: {e}"
        if fail_on_error:
            raise RuntimeError(f"FINAL_PIPELINE_ERROR: {msg}")
        return {"status": "skipped", "reason": "missing_dependencies"}

    if not torch.cuda.is_available() and device is None:
        msg = "CUDA not available for neural reranker training."
        if fail_on_error:
            raise RuntimeError(f"FINAL_PIPELINE_ERROR: {msg}")
        return {"status": "skipped", "reason": "no_cuda"}

    dev = device or ("cuda:1" if torch.cuda.device_count() > 1 else "cuda:0")
    print(f"Training Reranker on device: {dev}")

    train_examples, val_examples = prepare_reranker_dataset(
        pairs_path=pairs_path,
        val_fold=val_fold,
        max_train_pairs=max_train_pairs,
        max_val_pairs=max_val_pairs,
        seed=seed,
    )
    if not train_examples:
        msg = f"No training pairs found at {pairs_path}. Run scripts/mine_retrieval_negatives.py first."
        if fail_on_error:
            raise FileNotFoundError(f"FINAL_PIPELINE_ERROR: {msg}")
        return {"status": "skipped", "reason": "no_data"}

    num_unique_qa = len({ex.get("qa_id") for ex in train_examples if ex.get("qa_id")})
    print(f"Loaded {len(train_examples)} train pairs ({num_unique_qa} unique QA), {len(val_examples)} val pairs (val_fold={val_fold}).")

    token = os.environ.get("HF_TOKEN")
    tokenizer = AutoTokenizer.from_pretrained(model_name, token=token)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=1,
        token=token,
    ).to(dev)

    class PairDataset(Dataset):
        def __init__(self, data: List[Dict[str, Any]]):
            self.items = []
            for d in data:
                q = str(d.get("question", ""))
                pos = str(d.get("positive_text", ""))
                neg = str(d.get("negative_text", ""))
                if q and pos:
                    self.items.append((q, pos, 1.0))
                if q and neg:
                    self.items.append((q, neg, 0.0))

        def __len__(self):
            return len(self.items)

        def __getitem__(self, idx):
            return self.items[idx]

    train_ds = PairDataset(train_examples)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)

    val_ds = PairDataset(val_examples) if val_examples else None
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False) if val_ds else None

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    total_batch_steps = len(train_loader) * epochs
    effective_steps = max(1, total_batch_steps // grad_accum)
    if max_steps is not None:
        effective_steps = max_steps

    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=max(1, int(effective_steps * 0.05)),
        num_training_steps=effective_steps,
    )
    loss_fn = torch.nn.BCEWithLogitsLoss()

    best_val_loss = float("inf")
    best_accuracy = 0.0
    scaler = torch.amp.GradScaler('cuda') if dev.startswith("cuda") else None
    global_step = 0
    stop_early = False

    for epoch in range(epochs):
        if stop_early:
            break
        model.train()
        epoch_loss = 0.0
        optimizer.zero_grad()

        for step, (queries, texts, labels) in enumerate(train_loader):
            features = tokenizer(
                list(queries),
                list(texts),
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
            ).to(dev)

            lbl_tensor = labels.to(dev, dtype=torch.float32)

            if scaler is not None:
                with torch.amp.autocast('cuda'):
                    outputs = model(**features)
                    logits = outputs.logits.squeeze(-1)
                    loss = loss_fn(logits, lbl_tensor) / grad_accum
                scaler.scale(loss).backward()
                if (step + 1) % grad_accum == 0 or (step + 1) == len(train_loader):
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad()
                    scheduler.step()
                    global_step += 1
            else:
                outputs = model(**features)
                logits = outputs.logits.squeeze(-1)
                loss = loss_fn(logits, lbl_tensor) / grad_accum
                loss.backward()
                if (step + 1) % grad_accum == 0 or (step + 1) == len(train_loader):
                    optimizer.step()
                    optimizer.zero_grad()
                    scheduler.step()
                    global_step += 1

            epoch_loss += loss.item() * grad_accum

            if max_steps is not None and global_step >= max_steps:
                print(f"Reached max_steps limit ({max_steps}). Ending training early.")
                stop_early = True
                break

        avg_train_loss = epoch_loss / max(1, len(train_loader))

        # Validation Loop
        avg_val_loss = avg_train_loss
        if val_loader is not None and len(val_loader) > 0:
            model.eval()
            val_loss = 0.0
            correct = 0
            total = 0
            with torch.inference_mode():
                for v_queries, v_texts, v_labels in val_loader:
                    v_features = tokenizer(
                        list(v_queries),
                        list(v_texts),
                        padding=True,
                        truncation=True,
                        max_length=max_length,
                        return_tensors="pt",
                    ).to(dev)
                    v_lbl = v_labels.to(dev, dtype=torch.float32)
                    with torch.amp.autocast('cuda') if dev.startswith("cuda") else torch.inference_mode():
                        v_logits = model(**v_features).logits.squeeze(-1)
                        v_loss = loss_fn(v_logits, v_lbl)
                    val_loss += v_loss.item()
                    preds = (torch.sigmoid(v_logits) >= 0.5).float()
                    correct += (preds == v_lbl).sum().item()
                    total += len(v_lbl)

            avg_val_loss = val_loss / max(1, len(val_loader))
            accuracy = correct / max(1, total)
            print(f"Epoch {epoch + 1}/{epochs} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | Val Acc: {accuracy:.4f}")
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                best_accuracy = accuracy
                # Save best state
                os.makedirs(output_dir, exist_ok=True)
                model.save_pretrained(output_dir)
                tokenizer.save_pretrained(output_dir)
        else:
            print(f"Epoch {epoch + 1}/{epochs} | Train Loss: {avg_train_loss:.4f}")
            os.makedirs(output_dir, exist_ok=True)
            model.save_pretrained(output_dir)
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
        "batch_size": batch_size,
        "grad_accum": grad_accum,
        "learning_rate": lr,
        "val_fold_excluded": val_fold,
        "training_scope": training_scope,
        "is_final_checkpoint": is_final,
        "smoke_only": max_steps is not None,
        "num_unique_qa": num_unique_qa,
        "num_training_pairs": len(train_examples),
        "best_val_loss": round(float(best_val_loss), 4) if best_val_loss != float("inf") else round(float(avg_train_loss), 4),
        "best_val_accuracy": round(float(best_accuracy), 4),
    }
    with open(os.path.join(output_dir, "reranker_manifest.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    print(f"Reranker fine-tuning complete (is_final={is_final}, scope={training_scope}). Model saved to {output_dir}")

    # Section 10: Strict Reranker Checkpoint Reload Smoke Test
    print("Executing strict reranker reload verification...")
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    try:
        smoke_reranker = BGEReranker(model_name=output_dir, device=dev)
        scored = smoke_reranker.rerank(
            "Mức phạt là bao nhiêu?",
            [
                {"text_raw": "Phạt tiền từ 5.000.000 đồng đến 10.000.000 đồng."},
                {"text_raw": "Quy định khác không liên quan."},
            ],
            top_k=2,
        )
        if len(scored) != 2:
            raise RuntimeError(f"Reranker reload smoke check failed: expected 2 scored items, got {len(scored)}")
        if not all(np.isfinite(item.get("rerank_score", float("nan"))) for item in scored):
            raise RuntimeError("Reranker reload smoke check failed: non-finite rerank_score returned.")
        print("Reranker checkpoint reload verification PASS.")
        del smoke_reranker
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception as e:
        msg = f"Reranker checkpoint saved but failed reload smoke verification: {e}"
        if fail_on_error:
            raise RuntimeError(f"FINAL_PIPELINE_ERROR: {msg}") from e
        print(f"Warning during reranker reload verification: {msg}", file=sys.stderr)

    return {"status": "completed", "output_dir": output_dir, "manifest": manifest}
