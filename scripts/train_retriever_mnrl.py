import argparse
import os
import sys
import pandas as pd
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.common.normalize import tokenize_vietnamese

try:
    from sentence_transformers import SentenceTransformer, SentenceTransformerTrainer, SentenceTransformerTrainingArguments
    from sentence_transformers.losses import MultipleNegativesRankingLoss
    from datasets import Dataset as HFDataset
except ImportError:
    SentenceTransformer = None

def train_retriever_mnrl(
    model_name: str = "CODE4LIFEOFFICIAL/huydang-dek21-embedding-v2",
    citations_path: str = "artifacts/task2/data/qa_citations.parquet",
    chunks_path: str = "artifacts/task2/data/legal_chunks.parquet",
    output_dir: str = "artifacts/task2/checkpoints/retriever",
    epochs: int = 2,
    batch_size: int = 32,
    lr: float = 2e-5
):
    print(f"=== Starting MNRL Fine-Tuning for Retriever: {model_name} ===")
    if SentenceTransformer is None:
        print("sentence-transformers not installed. Skipping training execution.")
        return

    print(f"Loading citations from {citations_path} and chunks from {chunks_path}...")
    df_cit = pd.read_parquet(citations_path)
    df_chunks = pd.read_parquet(chunks_path)

    # Build chunk map
    chunk_map = {}
    for _, r in df_chunks.iterrows():
        key = (str(r.get("doc_name", "")), str(r.get("article_number", "")))
        if key not in chunk_map:
            chunk_map[key] = r.get("text_raw", "")

    anchors, positives, negatives = [], [], []
    for _, row in tqdm(df_cit.iterrows(), total=min(3000, len(df_cit)), desc="Assembling Triplets"):
        q = row.get("question", "")
        doc_num = row.get("doc_number", "")
        art = row.get("article", "")

        pos_text = chunk_map.get((doc_num, art)) or chunk_map.get(("", art))
        if q and pos_text:
            anchors.append(tokenize_vietnamese(q))
            positives.append(tokenize_vietnamese(pos_text[:1000]))

    if not anchors:
        print("No valid positive triplets found. Check citation mapping.")
        return

    triplets = HFDataset.from_dict({"anchor": anchors, "positive": positives})
    print(f"Built {len(triplets)} training pairs.")

    os.makedirs(output_dir, exist_ok=True)
    model = SentenceTransformer(model_name)
    loss = MultipleNegativesRankingLoss(model)

    training_args = SentenceTransformerTrainingArguments(
        output_dir=os.path.join(output_dir, "runs"),
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        learning_rate=lr,
        warmup_ratio=0.1,
        save_strategy="no",
        report_to="none"
    )

    trainer = SentenceTransformerTrainer(
        model=model,
        args=training_args,
        train_dataset=triplets,
        loss=loss
    )

    print("Training started...")
    trainer.train()
    model.save_pretrained(output_dir)
    print(f"Retriever fine-tuning complete. Checkpoint saved to {output_dir}")

def main():
    parser = argparse.ArgumentParser(description="Fine-tune Dense Retriever with MNRL")
    parser.add_argument("--model", default="CODE4LIFEOFFICIAL/huydang-dek21-embedding-v2")
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=2e-5)
    args = parser.parse_args()

    train_retriever_mnrl(
        model_name=args.model,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr
    )

if __name__ == "__main__":
    main()
