import argparse
import os
import sys
import json
import pandas as pd
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.common.bm25 import BM25Retriever
from src.task2.article_stitcher import ArticleStitcher
from src.task2.source_snap import clean_statutory_text

def prepare_mlx_data(qa_path: str, index_dir: str, out_dir: str):
    os.makedirs(out_dir, exist_ok=True)
    df = pd.read_parquet(qa_path)

    print(f"Loading BM25 index from {index_dir} for RAG context generation...")
    bm25 = BM25Retriever.load(index_dir)
    stitcher = ArticleStitcher(bm25.corpus)

    train_lines = []
    val_lines = []

    print(f"Formatting {len(df)} QA pairs with retrieved legal evidence...")
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Formatting ChatML"):
        q = row["question_raw"]
        a = row["answer_raw"]

        seeds = bm25.search(q, top_k=5)
        stitched = stitcher.stitch(seeds, max_chars=2500)
        evidence = clean_statutory_text(stitched.get("stitched_text") or (seeds[0]["text_raw"] if seeds else ""))

        text = (
            f"<|im_start|>system\n"
            f"Bạn là trợ lý pháp luật chuyên nghiệp. Hãy trả lời câu hỏi dựa trên căn cứ pháp lý được cung cấp. "
            f"Giữ nguyên các số hiệu văn bản, điều, khoản, số tiền phạt, ngày tháng và thuật ngữ pháp lý.<|im_end|>\n"
            f"<|im_start|>user\n"
            f"[CĂN CỨ PHÁP LÝ]\n{evidence}\n\n"
            f"[CÂU HỎI]\n{q}<|im_end|>\n"
            f"<|im_start|>assistant\n{a}<|im_end|>"
        )
        entry = json.dumps({"text": text}, ensure_ascii=False)
        if idx % 10 == 0:
            val_lines.append(entry)
        else:
            train_lines.append(entry)

    with open(os.path.join(out_dir, "train.jsonl"), "w", encoding="utf-8") as f:
        f.write("\n".join(train_lines) + "\n")
    with open(os.path.join(out_dir, "valid.jsonl"), "w", encoding="utf-8") as f:
        f.write("\n".join(val_lines) + "\n")

    print(f"RAG MLX dataset saved to {out_dir}: {len(train_lines)} train, {len(val_lines)} valid samples.")

def run_mlx_training(
    model_name: str = "Qwen/Qwen2.5-3B-Instruct",
    data_dir: str = "artifacts/task2/data/mlx_data",
    adapter_dir: str = "artifacts/task2/checkpoints/generator/mlx_adapter",
    batch_size: int = 2,
    num_layers: int = 16,
    iters: int = 600,
    lr: float = 1e-4
):
    print("=== Starting Local Apple Silicon MLX Fine-Tuning ===")
    print(f"Base Model: {model_name}")
    print(f"Hardware: Apple Silicon M3 Pro with Unified Memory")

    cmd = (
        f"python -m mlx_lm.lora "
        f"--model {model_name} "
        f"--train "
        f"--data {data_dir} "
        f"--batch-size {batch_size} "
        f"--lora-layers {num_layers} "
        f"--iters {iters} "
        f"--learning-rate {lr} "
        f"--adapter-path {adapter_dir}"
    )
    print(f"\nExecute MLX training command:\n{cmd}\n")

def main():
    parser = argparse.ArgumentParser(description="MLX LoRA Fine-Tuning for Apple Silicon")
    parser.add_argument("--qa_path", default="artifacts/task2/data/qa_unique.parquet")
    parser.add_argument("--index_dir", default="artifacts/task2/indexes/bm25")
    parser.add_argument("--data_dir", default="artifacts/task2/data/mlx_data")
    parser.add_argument("--model", default="Qwen/Qwen2.5-3B-Instruct")
    parser.add_argument("--iters", type=int, default=600)
    args = parser.parse_args()

    if os.path.exists(args.qa_path) and os.path.exists(args.index_dir):
        prepare_mlx_data(args.qa_path, args.index_dir, args.data_dir)
        run_mlx_training(model_name=args.model, data_dir=args.data_dir, iters=args.iters)
    else:
        print(f"Data file {args.qa_path} or index {args.index_dir} not found.")

if __name__ == "__main__":
    main()
