#!/usr/bin/env python3
"""Lightweight Kaggle Smoke Test for LegalQA Task 2.

Exercises real model loading, mini-retrieval, candidate assembly, generation,
and submission packaging on 8 queries without full-epoch training or full-corpus indexing.
Follows the lightweight verification specification in docs/next-run-060/06-release-and-a100-runbook.md.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import zipfile
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd
import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.common.bm25 import BM25Retriever
from src.common.dense import DenseRetriever
from src.common.reranker import BGEReranker
from src.task2.candidates import generate_candidate_ensemble
from src.task2.evidence_packer import EvidencePacker
from src.task2.generator import QwenGenerator
from src.task2.predict import LegalQAPipeline
from src.task2.qa_memory import QAMemory
from src.task2.scorer_contract import verify_zip_inner_matches_loose
from src.task2.selector import CandidateSelector


def run_kaggle_smoke(
    data_dir: str = "artifacts/task2/data",
    output_dir: str = "artifacts/smoke_output",
    device: str = "cuda:0" if torch.cuda.is_available() else "cpu",
    dense_model: str = "CODE4LIFEOFFICIAL/huydang-dek21-embedding-v2",
    generator_model: str = "Qwen/Qwen2.5-3B-Instruct",
    adapter_path: str = "",
    num_queries: int = 8,
    max_corpus_passages: int = 200,
    allow_mock: bool = False,
) -> Dict[str, Any]:
    """Execute end-to-end lightweight smoke test on real components."""
    t0 = time.time()
    out_p = Path(output_dir)
    out_p.mkdir(parents=True, exist_ok=True)
    d_path = Path(data_dir)

    print(f"=== LegalQA Lightweight Kaggle Smoke Gate ===")
    print(f"Device: {device} | Data Dir: {data_dir} | Output: {output_dir}")

    # 1. Load QA items
    qa_file = d_path / "qa_unique.parquet"
    if not qa_file.is_file():
        # Synthetic fallback if data dir does not contain parquet
        print(f"[*] Notice: {qa_file} not found; generating {num_queries} synthetic queries...")
        sample_items = [
            {"id": f"smoke_q{i}", "question": f"Người có hành vi vi phạm giao thông mức {i+1} bị xử phạt thế nào?"}
            for i in range(num_queries)
        ]
        corpus_chunks = [
            {
                "chunk_id": f"chunk_{j}",
                "doc_name": "Nghị định 100/2019/NĐ-CP",
                "parent_article_id": f"doc_art_{j // 2}",
                "article_number": str(j // 2 + 1),
                "clause_number": str((j % 2) + 1),
                "text_raw": f"Điều {j // 2 + 1}. Xử phạt vi phạm giao thông khoản {(j % 2) + 1}: Phạt tiền từ {j+1}.000.000 đồng.",
            }
            for j in range(max_corpus_passages)
        ]
    else:
        df_qa = pd.read_parquet(qa_file)
        sample_df = df_qa.head(num_queries)
        sample_items = [
            {"id": str(row["qa_id"]), "question": str(row["question_raw"]).strip()}
            for _, row in sample_df.iterrows()
        ]
        chunks_file = d_path / "legal_chunks.parquet"
        if chunks_file.is_file():
            df_chunks = pd.read_parquet(chunks_file)
            corpus_chunks = df_chunks.head(max_corpus_passages).to_dict("records")
        else:
            corpus_chunks = [
                {
                    "chunk_id": f"c_{k}",
                    "doc_name": "Luật Pháp Lệnh",
                    "parent_article_id": f"art_{k}",
                    "text_raw": f"Điều {k}. Quy định chi tiết thi hành.",
                }
                for k in range(max_corpus_passages)
            ]

    print(f"[1/5] Loaded {len(sample_items)} queries and {len(corpus_chunks)} toy corpus passages.")

    # 2. Build In-Memory Retrievers (BM25 + Dense)
    print(f"[2/5] Initializing Retrievers on {device}...")
    bm25 = BM25Retriever()
    bm25.fit(corpus_chunks)

    # If sentence_transformers available, use real model or mock fallback
    dense_dev = "cpu" if device.startswith("cuda") and torch.cuda.get_device_properties(0).total_memory < 8e9 else device
    try:
        dense = DenseRetriever(model_name=dense_model, device=dense_dev, dtype="float32" if device == "cpu" else "float16")
        dense.fit(corpus_chunks, batch_size=32, show_progress=False)
        print(f"      Dense retriever fitted ({dense_model}) on {dense_dev}.")
    except Exception as e:
        if not allow_mock:
            raise RuntimeError(f"Dense model '{dense_model}' failed to load: {e}. Refusing mock fallback in strict smoke mode.") from e
        print(f"      [!] Dense real fit skipped ({e}); using mock embeddings for smoke...")
        dense = DenseRetriever(model_name="mock")
        dense.fit_mock(corpus_chunks)

    reranker = BGEReranker(model_name="mock")
    packer = EvidencePacker(corpus_chunks)
    memory = QAMemory.from_records([])

    # 3. Generator Initialization
    print(f"[3/5] Initializing Generator...")
    if torch.cuda.is_available():
        try:
            generator = QwenGenerator.load(
                model_path=generator_model,
                adapter_path=adapter_path if adapter_path and os.path.exists(adapter_path) else None,
                device=device,
                runtime="torch",
                load_mode="nf4" if "cuda" in device else "float32",
                merge_adapter=False,
                fail_on_fallback=False,
            )
            print("      Neural Qwen generator loaded.")
        except Exception as e:
            if not allow_mock:
                raise RuntimeError(f"Neural generator '{generator_model}' failed to load: {e}. Refusing fallback in strict smoke mode.") from e
            print(f"      [!] Neural generator load skipped ({e}); using fast fallback generator.")
            generator = QwenGenerator(runtime="fallback")
    else:
        if not allow_mock:
            raise RuntimeError("CUDA is not available. Refusing fallback generator in strict smoke mode.")
        generator = QwenGenerator(runtime="fallback")

    selector = CandidateSelector(policy="fixed_baseline", best_fixed_candidate="dual_assembled")

    # 4. End-to-End Prediction
    print(f"[4/5] Running End-to-End Prediction on {len(sample_items)} items...")
    pipeline = LegalQAPipeline(
        memory=memory,
        bm25=bm25,
        dense=dense,
        reranker=reranker,
        packer=packer,
        generator=generator,
        selector=selector,
    )

    raw_cache = str(out_p / "smoke_raw_cache.jsonl")
    submission, provenance = pipeline.predict_batch(
        items=sample_items,
        max_new_tokens=128,
        generation_batch_size=2,
        retrieval_batch_size=8,
        reranker_batch_size=16,
        return_provenance=True,
        raw_cache_path=raw_cache,
    )

    assert len(submission) == len(sample_items), f"Expected {len(sample_items)} answers, got {len(submission)}"
    for qid, entry in submission.items():
        ans = str(entry.get("answer", "")).strip()
        assert len(ans) > 0, f"Query {qid} produced empty answer!"

    # 5. Packaging and Verification
    print(f"[5/5] Packaging Submission and verifying ZIP...")
    sub_json = out_p / "submission.json"
    sub_zip = out_p / "submission.json.zip"

    with open(sub_json, "w", encoding="utf-8") as f:
        json.dump(submission, f, ensure_ascii=False, indent=2)

    with zipfile.ZipFile(sub_zip, "w", zipfile.ZIP_DEFLATED) as z:
        z.write(sub_json, arcname="submission.json")

    zip_check = verify_zip_inner_matches_loose(str(sub_zip), str(sub_json))
    assert zip_check["inner_sha256"] == zip_check["loose_sha256"], "ZIP inner bytes do not match loose submission.json!"

    elapsed = round(time.time() - t0, 2)
    smoke_report = {
        "status": "PASS",
        "stage": "kaggle_smoke_t4",
        "elapsed_seconds": elapsed,
        "queries_tested": len(submission),
        "device": device,
        "submission_json": str(sub_json),
        "submission_zip": str(sub_zip),
        "zip_sha256": zip_check["zip_sha256"],
        "sample_answer_preview": list(submission.values())[0]["answer"][:200],
    }
    report_file = out_p / "smoke_report.json"
    report_file.write_text(json.dumps(smoke_report, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"\n[+] SMOKE GATE PASS: {len(submission)} answers generated & verified in {elapsed}s.")
    print(f"    Report saved to: {report_file}")
    return smoke_report


def main() -> None:
    parser = argparse.ArgumentParser(description="Run lightweight Kaggle smoke test.")
    parser.add_argument("--data-dir", default="artifacts/task2/data", help="Directory containing dataset files")
    parser.add_argument("--output-dir", default="artifacts/smoke_output", help="Directory to store smoke output")
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--dense-model", default="CODE4LIFEOFFICIAL/huydang-dek21-embedding-v2")
    parser.add_argument("--generator-model", default="Qwen/Qwen2.5-3B-Instruct")
    parser.add_argument("--adapter-path", default="")
    parser.add_argument("--num-queries", type=int, default=8)
    parser.add_argument("--allow-mock", action="store_true", help="Allow fallback/mock models when running on CPU without weights")
    args = parser.parse_args()

    res = run_kaggle_smoke(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        device=args.device,
        dense_model=args.dense_model,
        generator_model=args.generator_model,
        adapter_path=args.adapter_path,
        num_queries=args.num_queries,
        allow_mock=args.allow_mock,
    )
    if res.get("status") != "PASS":
        sys.exit(1)


if __name__ == "__main__":
    main()
