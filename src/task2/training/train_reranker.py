"""Fine-tuning module for Cross-Encoder Reranker (BAAI/bge-reranker-v2-m3) on LegalQA data."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

from src.common.security import assert_no_secrets_in_workspace


def prepare_reranker_dataset(
    pairs_path: Optional[str] = None,
    df_pairs: Optional[pd.DataFrame] = None,
    val_fold: Optional[int] = None,
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
    val_fold: Optional[int] = 0,
    device: Optional[str] = None,
    max_length: int = 384,
    fail_on_error: bool = True,
) -> Dict[str, Any]:
    """Fine-tune Cross-Encoder Reranker using positive and hard-negative pairs with validation tracking."""
    print(f"=== Starting Cross-Encoder Reranker Fine-Tuning ({model_name}) ===")
    assert_no_secrets_in_workspace(Path.cwd())

    try:
        import torch
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

    train_examples, val_examples = prepare_reranker_dataset(pairs_path=pairs_path, val_fold=val_fold)
    if not train_examples:
        msg = f"No training pairs found at {pairs_path}. Run scripts/mine_retrieval_negatives.py first."
        if fail_on_error:
            raise FileNotFoundError(f"FINAL_PIPELINE_ERROR: {msg}")
        return {"status": "skipped", "reason": "no_data"}

    print(f"Loaded {len(train_examples)} train pairs, {len(val_examples)} val pairs (val_fold={val_fold}).")

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
    effective_steps = (len(train_loader) // grad_accum) * epochs
    scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=int(effective_steps * 0.05), num_training_steps=effective_steps)
    loss_fn = torch.nn.BCEWithLogitsLoss()

    best_val_loss = float("inf")
    best_accuracy = 0.0
    scaler = torch.amp.GradScaler('cuda') if dev.startswith("cuda") else None

    for epoch in range(epochs):
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
            else:
                outputs = model(**features)
                logits = outputs.logits.squeeze(-1)
                loss = loss_fn(logits, lbl_tensor) / grad_accum
                loss.backward()
                if (step + 1) % grad_accum == 0 or (step + 1) == len(train_loader):
                    optimizer.step()
                    optimizer.zero_grad()
                    scheduler.step()

            epoch_loss += loss.item() * grad_accum

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
        else:
            print(f"Epoch {epoch + 1}/{epochs} | Train Loss: {avg_train_loss:.4f}")

    os.makedirs(output_dir, exist_ok=True)
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

    manifest = {
        "base_model": model_name,
        "epochs": epochs,
        "batch_size": batch_size,
        "grad_accum": grad_accum,
        "learning_rate": lr,
        "val_fold_excluded": val_fold,
        "num_training_pairs": len(train_examples),
        "best_val_loss": round(float(best_val_loss), 4),
        "best_val_accuracy": round(float(best_accuracy), 4),
    }
    with open(os.path.join(output_dir, "reranker_manifest.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    print(f"Reranker fine-tuning complete. Model saved to {output_dir}")
    return {"status": "completed", "output_dir": output_dir, "manifest": manifest}
