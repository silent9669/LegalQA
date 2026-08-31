"""End-to-end LegalQA Task 2 prediction pipeline orchestrator with explicit Dual-T4 device placement."""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

from src.common.bm25 import BM25Retriever
from src.common.dense import DenseRetriever
from src.common.reranker import BGEReranker
from src.common.rrf import reciprocal_rank_fusion
from src.task2.candidates import generate_candidate_ensemble
from src.task2.evidence_packer import EvidencePacker
from src.task2.generator import QwenGenerator
from src.task2.production_config import policy_requires_generator
from src.task2.qa_memory import QAMemory
from src.task2.selector import CandidateSelector


class LegalQAPipeline:
    """End-to-end LegalQA inference pipeline orchestrating Memory, Hybrid Retrieval, Reranking, Evidence Packing, Qwen, and Selection."""

    def __init__(
        self,
        memory: QAMemory,
        bm25: BM25Retriever,
        dense: Optional[DenseRetriever],
        reranker: Optional[BGEReranker],
        packer: EvidencePacker,
        generator: Optional[QwenGenerator] = None,
        selector: Optional[CandidateSelector] = None,
    ):
        self.memory = memory
        self.bm25 = bm25
        self.dense = dense
        self.reranker = reranker
        self.packer = packer
        self.stitcher = packer  # Alias for backward compatibility
        self.generator = generator
        self.selector = selector or CandidateSelector(policy="fixed_baseline", best_fixed_candidate="stitched_extract")

    @property
    def policy_needs_generator(self) -> bool:
        """Check if current selector configuration requires neural generation."""
        if self.selector is None:
            return False
        policy = getattr(self.selector, "policy", "fixed_baseline")
        best_fixed = getattr(self.selector, "best_fixed_candidate", "stitched_extract")
        return policy_requires_generator(policy, best_fixed)

    @classmethod
    def build_mock(cls) -> LegalQAPipeline:
        """Construct lightweight in-memory mock pipeline for fast unit testing."""
        chunks = [
            {
                "chunk_id": "c1",
                "doc_name": "Nghị định 90/2017/NĐ-CP",
                "parent_article_id": "doc1_art17",
                "article_id": "doc1_art17",
                "article_number": "17",
                "clause_number": "3",
                "text_raw": "[DOCUMENT] Nghị định 90/2017/NĐ-CP\n[ARTICLE] Điều 17. Phạt tiền từ 1.000.000 đồng đến 2.000.000 đồng đối với hành vi không tiêm phòng.",
                "start_char": 0,
            }
        ]
        mem = QAMemory.from_records([])
        bm25 = BM25Retriever()
        bm25.fit(chunks)
        dense = DenseRetriever(model_name="mock")
        dense.fit_mock(chunks)
        reranker = BGEReranker(model_name="mock")
        packer = EvidencePacker(chunks)
        generator = QwenGenerator(runtime="fallback")
        selector = CandidateSelector(policy="fixed_baseline", best_fixed_candidate="stitched_extract")
        return cls(mem, bm25, dense, reranker, packer, generator, selector)

    @classmethod
    def load_pipeline(
        cls,
        data_dir: str = "artifacts/task2/data",
        bm25_dir: str = "artifacts/task2/indexes/bm25",
        dense_dir: str = "artifacts/task2/indexes/dek21",
        model_path: Optional[str] = "Qwen/Qwen2.5-3B-Instruct",
        adapter_path: Optional[str] = None,
        generator_runtime: str = "auto",
        device: Optional[str] = None,
        gen_device: Optional[str] = None,
        retrieval_device: Optional[str] = None,
        use_mock_dense: bool = False,
        index_dir: Optional[str] = None,
        selector_model_path: Optional[str] = None,
        fail_on_missing_index: bool = False,
        fail_on_model_fallback: bool = False,
        require_adapter: bool = False,
        load_generator: bool = True,
    ) -> LegalQAPipeline:
        """Load full pipeline from disk artifacts with explicit Dual-T4 GPU placement."""
        if index_dir is not None:
            bm25_dir = index_dir
        known_qa_path = os.path.join(data_dir, "known_qa.json")
        qa_unique_path = os.path.join(data_dir, "qa_unique.parquet")
        chunks_path = os.path.join(data_dir, "legal_chunks.parquet")

        # 1. Exact & Similar Memory
        if os.path.exists(known_qa_path):
            memory = QAMemory.load(known_qa_path, qa_unique_path)
        else:
            memory = QAMemory.from_records([])

        # 2. Sparse BM25 Retriever on CPU
        if os.path.exists(bm25_dir):
            bm25 = BM25Retriever.load(bm25_dir, corpus_path=chunks_path, fail_on_missing_index=fail_on_missing_index)
        else:
            if fail_on_missing_index:
                raise FileNotFoundError(f"FINAL_PIPELINE_ERROR: BM25 index missing at {bm25_dir}")
            bm25 = BM25Retriever()

        # 3. Dense Retriever on retrieval_device (e.g. cuda:1)
        r_dev = retrieval_device or device or "cuda:1"
        if use_mock_dense:
            dense = DenseRetriever(model_name="mock", device=r_dev)
            if bm25.corpus:
                dense.fit_mock(bm25.corpus)
        elif os.path.exists(dense_dir) and os.path.exists(os.path.join(dense_dir, "embeddings.npy")):
            dense = DenseRetriever.load_index(
                dense_dir,
                corpus_path=chunks_path,
                device=r_dev,
                expected_dtype="float16",
                final_mode=fail_on_model_fallback,
            )
        else:
            if fail_on_missing_index:
                raise FileNotFoundError(f"FINAL_PIPELINE_ERROR: Dense corpus index missing at {dense_dir}")
            dense = None

        # 4. Neural Cross-Encoder Reranker on retrieval_device (e.g. cuda:1)
        reranker = BGEReranker(model_name="BAAI/bge-reranker-v2-m3", device=r_dev)

        # 5. Structured Evidence Packer
        if bm25.corpus:
            packer = EvidencePacker(bm25.corpus)
        elif os.path.exists(chunks_path):
            df_chunks = pd.read_parquet(chunks_path)
            packer = EvidencePacker(df_chunks.to_dict("records"))
        else:
            packer = EvidencePacker([])

        # 6. Candidate Selector
        if selector_model_path and os.path.exists(selector_model_path):
            selector = CandidateSelector.load(selector_model_path)
        else:
            selector = CandidateSelector(policy="fixed_baseline", best_fixed_candidate="stitched_extract")

        # 7. Qwen Generator on gen_device (e.g. cuda:0) - Optional if extractive policy
        generator: Optional[QwenGenerator] = None
        if load_generator and model_path:
            g_dev = gen_device or device or "cuda:0"
            generator = QwenGenerator.load(
                model_path=model_path,
                adapter_path=adapter_path,
                device=g_dev,
                runtime=generator_runtime,
                fail_on_fallback=fail_on_model_fallback,
                final_mode=fail_on_model_fallback,
                require_adapter=require_adapter,
            )

        return cls(memory, bm25, dense, reranker, packer, generator, selector)

    def retrieve_and_rerank(self, question: str, top_k_rerank: int = 8) -> Dict[str, Any]:
        """Perform hybrid retrieval (BM25 + GPU Dense) and Cross-Encoder reranking returning full trace (P0-7)."""
        bm25_res = self.bm25.search(question, top_k=50) if self.bm25 else []
        dense_res = self.dense.search(question, top_k=50) if self.dense else []
        if bm25_res and dense_res:
            fused_res = reciprocal_rank_fusion([bm25_res, dense_res], k=60, weights=[0.5, 0.5])
        else:
            fused_res = bm25_res or dense_res

        top_seeds = self.reranker.rerank(question, fused_res, top_k=top_k_rerank) if (self.reranker and fused_res) else fused_res[:top_k_rerank]

        pack_multi = self.packer.pack_evidence(top_seeds, pack_type="multi_seed_2500_chars", max_chars=3500)
        primary_evidence = pack_multi.get("text") or (top_seeds[0]["text_raw"] if top_seeds else "")

        r1 = float(top_seeds[0].get("rerank_score", top_seeds[0].get("score", 0.0))) if top_seeds else 0.0
        r2 = float(top_seeds[1].get("rerank_score", top_seeds[1].get("score", 0.0))) if len(top_seeds) > 1 else r1

        retrieval_meta = {
            "rerank_top1": r1,
            "rerank_margin": r1 - r2,
            "bm25_top1": float(bm25_res[0].get("score", 0.0)) if bm25_res else 0.0,
            "dense_top1": float(dense_res[0].get("score", 0.0)) if dense_res else 0.0,
        }

        return {
            "bm25_results": bm25_res,
            "dense_results": dense_res,
            "fused_results": fused_res,
            "reranked_results": top_seeds,
            "primary_evidence": primary_evidence,
            "retrieval_meta": retrieval_meta,
            "pack_multi": pack_multi,
        }

    def predict_single(
        self,
        qa_id: str,
        question: str,
        max_new_tokens: int = 384,
        return_candidates: bool = False,
        return_trace: bool = False,
    ) -> Any:
        """Execute inference on a single query with optional candidates and retrieval trace (P0-7)."""
        # 1. Exact QA Memory Lookup
        exact_ans = self.memory.lookup_exact(qa_id, question)
        if exact_ans:
            if return_trace:
                empty_trace = {
                    "bm25_results": [],
                    "dense_results": [],
                    "fused_results": [],
                    "reranked_results": [],
                    "primary_evidence": exact_ans,
                    "retrieval_meta": {"is_exact_memory": True},
                }
                return exact_ans, {"exact_memory": exact_ans}, empty_trace
            if return_candidates:
                return exact_ans, {"exact_memory": exact_ans}, exact_ans
            return exact_ans

        # 2. Similar QA Memory Lookup
        fuzzy_hit = self.memory.lookup_fuzzy(question, threshold=0.90)
        fuzzy_ans = fuzzy_hit["answer"] if fuzzy_hit else ""

        # 3. Hybrid Retrieval & Reranking Trace (P0-7)
        trace = self.retrieve_and_rerank(question, top_k_rerank=8)
        top_seeds = trace["reranked_results"]
        primary_evidence = trace["primary_evidence"]
        retrieval_meta = trace["retrieval_meta"]
        retrieval_meta["fuzzy_sim"] = float(fuzzy_hit["similarity"]) if fuzzy_hit else 0.0
        pack_multi = trace.get("pack_multi", {})

        # 4. Multi-Granularity Evidence Packing
        pack_focused = self.packer.pack_evidence(top_seeds, pack_type="focused_clause")
        pack_full_art = self.packer.pack_evidence(top_seeds, pack_type="primary_full_article")
        pack_top2_rel = self.packer.pack_evidence(top_seeds, pack_type="relevance_selected_top2_articles", max_chars=2500)

        # 5. Generator Candidate (Optional if generator loaded & needed)
        gen_ans = ""
        if self.generator is not None and (self.policy_needs_generator or return_candidates):
            gen_ans = self.generator.generate(question, primary_evidence, max_new_tokens=max_new_tokens)

        # 6. Candidate Ensemble
        top_doc = pack_multi.get("top_doc_name", "")
        top_art = pack_multi.get("top_article_num", "")
        top_clause = pack_multi.get("top_clause_num", "")

        evidence_packs = {
            "focused": pack_focused.get("text", ""),
            "full_article": pack_full_art.get("text", ""),
            "top2_relevance": pack_top2_rel.get("text", ""),
        }

        candidates = generate_candidate_ensemble(
            gen_ans=gen_ans,
            evidence=primary_evidence,
            exact_ans="",
            fuzzy_ans=fuzzy_ans,
            doc_name=top_doc,
            art_num=top_art,
            clause_num=top_clause,
            evidence_packs=evidence_packs,
        )

        selected = self.selector.select(
            candidates=candidates,
            question=question,
            evidence=primary_evidence,
            retrieval_meta=retrieval_meta,
            features=fuzzy_hit,
        )

        if return_trace:
            return selected, candidates, trace

        if return_candidates:
            return selected, candidates, primary_evidence

        return selected

    def predict_batch(
        self,
        items: List[Dict[str, Any]],
        max_new_tokens: int = 384,
        retrieval_batch_size: int = 32,
        reranker_batch_size: int = 32,
        generation_batch_size: int = 4,
    ) -> Dict[str, Dict[str, str]]:
        """High-throughput batch prediction orchestrating BM25, batched dense GPU search, batched reranking, and batched generation."""
        results: Dict[str, Dict[str, str]] = {}
        unseen_items: List[Tuple[str, str]] = []

        # 1. Exact Memory Pre-pass
        for item in items:
            qa_id = str(item.get("id") or item.get("qa_id") or "").strip()
            q = str(item.get("question", "")).strip()
            exact_ans = self.memory.lookup_exact(qa_id, q)
            if exact_ans:
                results[qa_id] = {"answer": exact_ans}
            else:
                unseen_items.append((qa_id, q))

        if not unseen_items:
            return results

        unseen_queries = [q for _, q in unseen_items]

        # 2. Batched Dense Retrieval
        dense_results: List[List[Dict[str, Any]]] = []
        if self.dense:
            dense_results = self.dense.search_batch(unseen_queries, top_k=50, batch_size=retrieval_batch_size)
        else:
            dense_results = [[] for _ in unseen_queries]

        # 3. Collect BM25 and fused candidates for all queries
        fused_candidate_lists: List[List[Dict[str, Any]]] = []
        bm25_scores: List[float] = []
        dense_scores: List[float] = []
        fuzzy_hits: List[Optional[Dict[str, Any]]] = []

        for idx, (qa_id, question) in enumerate(unseen_items):
            f_hit = self.memory.lookup_fuzzy(question, threshold=0.90)
            fuzzy_hits.append(f_hit)

            bm25_res = self.bm25.search(question, top_k=50) if self.bm25 else []
            dense_res = dense_results[idx] if idx < len(dense_results) else []

            bm25_scores.append(float(bm25_res[0].get("score", 0.0)) if bm25_res else 0.0)
            dense_scores.append(float(dense_res[0].get("score", 0.0)) if dense_res else 0.0)

            if bm25_res and dense_res:
                fused = reciprocal_rank_fusion([bm25_res, dense_res], k=60, weights=[0.5, 0.5])
            else:
                fused = bm25_res or dense_res
            fused_candidate_lists.append(fused)

        # 4. Batch Rerank across all queries
        if self.reranker and hasattr(self.reranker, "rerank_batch"):
            all_top_seeds = self.reranker.rerank_batch(
                unseen_queries,
                fused_candidate_lists,
                top_k=8,
                batch_size=reranker_batch_size,
            )
        elif self.reranker:
            all_top_seeds = [
                self.reranker.rerank(q, cands, top_k=8)
                for q, cands in zip(unseen_queries, fused_candidate_lists)
            ]
        else:
            all_top_seeds = [cands[:8] for cands in fused_candidate_lists]

        # 5. Build Evidence Packs
        evidence_records: List[Dict[str, Any]] = []
        for idx, (qa_id, question) in enumerate(unseen_items):
            top_seeds = all_top_seeds[idx]
            fuzzy_hit = fuzzy_hits[idx]
            fuzzy_ans = fuzzy_hit["answer"] if fuzzy_hit else ""

            pack_multi = self.packer.pack_evidence(top_seeds, pack_type="multi_seed_2500_chars", max_chars=3500)
            pack_focused = self.packer.pack_evidence(top_seeds, pack_type="focused_clause")
            pack_full_art = self.packer.pack_evidence(top_seeds, pack_type="primary_full_article")
            pack_top2_rel = self.packer.pack_evidence(top_seeds, pack_type="relevance_selected_top2_articles", max_chars=2500)

            primary_evidence = pack_multi.get("text") or (top_seeds[0]["text_raw"] if top_seeds else "")

            r1 = float(top_seeds[0].get("rerank_score", top_seeds[0].get("score", 0.0))) if top_seeds else 0.0
            r2 = float(top_seeds[1].get("rerank_score", top_seeds[1].get("score", 0.0))) if len(top_seeds) > 1 else r1

            retrieval_meta = {
                "rerank_top1": r1,
                "rerank_margin": r1 - r2,
                "bm25_top1": bm25_scores[idx],
                "dense_top1": dense_scores[idx],
                "fuzzy_sim": float(fuzzy_hit["similarity"]) if fuzzy_hit else 0.0,
            }

            evidence_records.append({
                "qa_id": qa_id,
                "question": question,
                "primary_evidence": primary_evidence,
                "fuzzy_ans": fuzzy_ans,
                "fuzzy_hit": fuzzy_hit,
                "top_doc": pack_multi.get("top_doc_name", ""),
                "top_art": pack_multi.get("top_article_num", ""),
                "top_clause": pack_multi.get("top_clause_num", ""),
                "evidence_packs": {
                    "focused": pack_focused.get("text", ""),
                    "full_article": pack_full_art.get("text", ""),
                    "top2_relevance": pack_top2_rel.get("text", ""),
                },
                "retrieval_meta": retrieval_meta,
            })

        # 6. Batched Qwen Generation (Only if generator loaded and needed!)
        if self.generator is not None and self.policy_needs_generator:
            pairs = [(rec["question"], rec["primary_evidence"]) for rec in evidence_records]
            gen_answers = self.generator.generate_batch(pairs, max_new_tokens=max_new_tokens, batch_size=generation_batch_size)
        else:
            gen_answers = ["" for _ in evidence_records]

        # 7. Candidate Ensembles & Selection
        for rec, gen_ans in zip(evidence_records, gen_answers):
            candidates = generate_candidate_ensemble(
                gen_ans=gen_ans,
                evidence=rec["primary_evidence"],
                exact_ans="",
                fuzzy_ans=rec["fuzzy_ans"],
                doc_name=rec["top_doc"],
                art_num=rec["top_art"],
                clause_num=rec["top_clause"],
                evidence_packs=rec["evidence_packs"],
            )

            selected = self.selector.select(
                candidates=candidates,
                question=rec["question"],
                evidence=rec["primary_evidence"],
                retrieval_meta=rec["retrieval_meta"],
                features=rec["fuzzy_hit"],
            )
            results[rec["qa_id"]] = {"answer": selected}

        return results
