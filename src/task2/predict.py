"""End-to-end LegalQA Task 2 prediction pipeline orchestrator."""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional

import pandas as pd

from src.common.bm25 import BM25Retriever
from src.common.dense_dek21 import DEk21Retriever
from src.common.reranker import BGEReranker
from src.common.rrf import reciprocal_rank_fusion
from src.task2.article_stitcher import ArticleStitcher
from src.task2.generator import QwenGenerator
from src.task2.qa_memory import QAMemory
from src.task2.source_snap import select_best_answer_candidate, snap_facts_to_evidence


class LegalQAPipeline:
    """End-to-end LegalQA inference pipeline orchestrating Memory, Hybrid Retrieval, Reranking, Stitching, Qwen, and Snapping."""

    def __init__(
        self,
        memory: QAMemory,
        bm25: BM25Retriever,
        dense: Optional[DEk21Retriever],
        reranker: Optional[BGEReranker],
        stitcher: ArticleStitcher,
        generator: QwenGenerator,
    ):
        self.memory = memory
        self.bm25 = bm25
        self.dense = dense
        self.reranker = reranker
        self.stitcher = stitcher
        self.generator = generator

    @classmethod
    def build_mock(cls) -> LegalQAPipeline:
        """Construct lightweight in-memory mock pipeline for fast unit testing."""
        chunks = [
            {
                "chunk_id": "c1",
                "doc_name": "Nghị định 90/2017/NĐ-CP",
                "parent_article_id": "doc1_art17",
                "article_number": "17",
                "clause_number": "3",
                "text_raw": "[DOCUMENT] Nghị định 90/2017/NĐ-CP\n[ARTICLE] Điều 17. Phạt tiền từ 1.000.000 đồng đến 2.000.000 đồng đối với hành vi không tiêm phòng.",
                "start_char": 0,
            }
        ]
        mem = QAMemory.from_records([])
        bm25 = BM25Retriever()
        bm25.fit(chunks)
        dense = DEk21Retriever(model_name="mock")
        dense.fit_mock(chunks)
        reranker = BGEReranker(model_name="mock")
        stitcher = ArticleStitcher(chunks)
        generator = QwenGenerator(runtime="fallback")
        return cls(mem, bm25, dense, reranker, stitcher, generator)

    @classmethod
    def load_pipeline(
        cls,
        data_dir: str = "artifacts/task2/data",
        bm25_dir: str = "artifacts/task2/indexes/bm25",
        dek21_dir: str = "artifacts/task2/indexes/dek21",
        adapter_path: Optional[str] = None,
        device: Optional[str] = None,
        use_mock_dense: bool = False,
        index_dir: Optional[str] = None,
    ) -> LegalQAPipeline:
        """Load full pipeline from disk artifacts."""
        if index_dir is not None:
            bm25_dir = index_dir
        known_qa_path = os.path.join(data_dir, "known_qa.json")
        qa_unique_path = os.path.join(data_dir, "qa_unique.parquet")
        chunks_path = os.path.join(data_dir, "legal_chunks.parquet")

        # 1. Exact Memory
        if os.path.exists(known_qa_path):
            memory = QAMemory.load(known_qa_path, qa_unique_path)
        else:
            memory = QAMemory.from_records([])

        # 2. Sparse BM25 Retriever
        if os.path.exists(bm25_dir):
            bm25 = BM25Retriever.load(bm25_dir, corpus_path=chunks_path)
        else:
            bm25 = BM25Retriever()

        # 3. Dense DEk21 Retriever
        if use_mock_dense:
            dense = DEk21Retriever(model_name="mock", device=device)
            if bm25.corpus:
                dense.fit_mock(bm25.corpus)
        elif os.path.exists(dek21_dir) and os.path.exists(os.path.join(dek21_dir, "embeddings.npy")):
            dense = DEk21Retriever.load_index(dek21_dir, corpus_path=chunks_path, device=device)
        else:
            dense = None

        # 4. Neural Reranker
        reranker = BGEReranker(model_name="BAAI/bge-reranker-v2-m3", device=device)

        # 5. Article Stitcher
        if bm25.corpus:
            stitcher = ArticleStitcher(bm25.corpus)
        elif os.path.exists(chunks_path):
            df_chunks = pd.read_parquet(chunks_path)
            stitcher = ArticleStitcher(df_chunks.to_dict("records"))
        else:
            stitcher = ArticleStitcher([])

        # 6. Qwen Generator
        if adapter_path and os.path.exists(adapter_path):
            generator = QwenGenerator.load(
                model_path="Qwen/Qwen2.5-3B-Instruct",
                adapter_path=adapter_path,
                device=device,
            )
        else:
            generator = QwenGenerator(runtime="fallback")

        return cls(memory, bm25, dense, reranker, stitcher, generator)

    def predict_single(self, qa_id: str, question: str, max_new_tokens: int = 220) -> str:
        """Execute inference on a single query."""
        # 1. Exact QA Memory Lookup
        exact_ans = self.memory.lookup_exact(qa_id, question)
        if exact_ans:
            return exact_ans

        # 2. Hybrid Retrieval
        bm25_res = self.bm25.search(question, top_k=50) if self.bm25 else []
        dense_res = self.dense.search(question, top_k=50) if self.dense else []
        if bm25_res and dense_res:
            fused_res = reciprocal_rank_fusion([bm25_res, dense_res], k=60, weights=[0.5, 0.5])
        else:
            fused_res = bm25_res or dense_res

        # 3. Neural Reranking
        top_seeds = self.reranker.rerank(question, fused_res, top_k=8) if (self.reranker and fused_res) else fused_res[:8]

        # 4. Selective Article Stitching
        stitched_pkg = self.stitcher.stitch(top_seeds)
        evidence_text = stitched_pkg.get("stitched_text") or (top_seeds[0]["text_raw"] if top_seeds else "")

        # 5. Generator
        gen_ans = self.generator.generate(question, evidence_text, max_new_tokens=max_new_tokens)

        # 6. Source Snapping & Candidate Selection
        snapped_ans = snap_facts_to_evidence(gen_ans, evidence_text)
        candidates = {
            "focused_extract": top_seeds[0]["text_raw"] if top_seeds else "",
            "stitched_extract": evidence_text,
            "generated": gen_ans,
            "snapped": snapped_ans,
        }
        doc_name = top_seeds[0].get("doc_name", "") if top_seeds else ""
        art_num = top_seeds[0].get("article_number", "") if top_seeds else ""
        clause_num = top_seeds[0].get("clause_number", "") if top_seeds else ""
        return select_best_answer_candidate(candidates, doc_name=doc_name, article_num=art_num, clause_num=clause_num)

    def predict_batch(self, items: List[Dict[str, Any]]) -> Dict[str, Dict[str, str]]:
        """Batch predict dictionary mapping qa_id -> {'answer': text}."""
        results: Dict[str, Dict[str, str]] = {}
        for item in items:
            qa_id = str(item.get("id") or item.get("qa_id") or "")
            q = str(item.get("question", ""))
            results[qa_id] = {"answer": self.predict_single(qa_id, q)}
        return results
