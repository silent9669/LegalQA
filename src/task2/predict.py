"""End-to-end LegalQA Task 2 prediction pipeline orchestrator with explicit Dual-T4 device placement."""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

from src.common.bm25 import BM25Retriever
from src.common.dense import DenseRetriever
from src.common.legal_reference import search_legal_references
from src.common.query_rewrite import has_legal_reference, rewrite_query_for_retrieval
from src.common.reranker import BGEReranker
from src.common.rrf import reciprocal_rank_fusion
from src.task2.candidates import generate_candidate_ensemble
from src.task2.evidence_packer import EvidencePacker
from src.task2.generator import QwenGenerator
from src.task2.production_config import policy_requires_generator
from src.task2.qa_memory import QAMemory
from src.task2.selector import CandidateSelector


def retrieval_options_from_config(cfg: Any) -> Tuple[Dict[str, Any], Dict[str, float]]:
    """Split a resolved candidate's retrieval recipe into options + weights."""
    retrieval = cfg.algorithm.retrieval
    options = {
        "use_legal_reference": bool(retrieval.use_legal_reference),
        "use_acronyms": bool(retrieval.use_acronyms),
        "use_weighted_rrf": bool(retrieval.use_weighted_rrf),
        "diversify_context": bool(retrieval.diversify_context),
        "lost_in_middle": bool(retrieval.lost_in_middle),
        "max_parts_per_article": int(retrieval.max_parts_per_article),
        "rrf_k": int(retrieval.rrf_k),
        "candidate_pool": int(retrieval.candidate_pool),
    }
    weights = {
        "w_bm25_plain": float(retrieval.w_bm25_plain),
        "w_dense_plain": float(retrieval.w_dense_plain),
        "w_bm25_lex": float(retrieval.w_bm25_lex),
        "w_dense_lex": float(retrieval.w_dense_lex),
        "w_lexref_lex": float(retrieval.w_lexref_lex),
    }
    return options, weights


class LegalQAPipeline:
    """End-to-end LegalQA inference pipeline orchestrating Memory, Hybrid Retrieval, Reranking, Evidence Packing, Qwen, and Selection."""

    #: Default retrieval options. Every experiment flag defaults off, so the
    #: production recipe (BM25 + dense RRF, uniform weights) is unchanged
    #: unless a candidate explicitly opts in.
    DEFAULT_RETRIEVAL_OPTIONS: Dict[str, Any] = {
        "use_legal_reference": False,
        "use_acronyms": False,
        "use_weighted_rrf": False,
        "diversify_context": False,
        "lost_in_middle": False,
        "max_parts_per_article": 2,
        "rrf_k": 60,
        "candidate_pool": 50,
    }

    #: Default RRF weights reproduce the legacy uniform recipe. Score-affecting
    #: overrides arrive via retrieval_weights (hashed in the candidate).
    DEFAULT_RETRIEVAL_WEIGHTS: Dict[str, float] = {
        "w_bm25_plain": 0.5,
        "w_dense_plain": 0.5,
        "w_bm25_lex": 1.0 / 3.0,
        "w_dense_lex": 1.0 / 3.0,
        "w_lexref_lex": 1.0 / 3.0,
    }

    def __init__(
        self,
        memory: QAMemory,
        bm25: BM25Retriever,
        dense: Optional[DenseRetriever],
        reranker: Optional[BGEReranker],
        packer: EvidencePacker,
        generator: Optional[QwenGenerator] = None,
        selector: Optional[CandidateSelector] = None,
        legal_index: Optional[Dict[str, Any]] = None,
        legal_rows: Optional[List[Dict[str, Any]]] = None,
        retrieval_options: Optional[Dict[str, Any]] = None,
        retrieval_weights: Optional[Dict[str, float]] = None,
    ):
        self.memory = memory
        self.bm25 = bm25
        self.dense = dense
        self.reranker = reranker
        self.packer = packer
        self.stitcher = packer  # Alias for backward compatibility
        self.generator = generator
        self.selector = selector or CandidateSelector(policy="fixed_baseline", best_fixed_candidate="stitched_extract")
        self.legal_index = legal_index
        self.legal_rows = legal_rows
        self.retrieval_options = dict(self.DEFAULT_RETRIEVAL_OPTIONS)
        if retrieval_options:
            unknown = set(retrieval_options) - set(self.DEFAULT_RETRIEVAL_OPTIONS)
            if unknown:
                raise ValueError(f"unknown retrieval options: {sorted(unknown)}")
            self.retrieval_options.update(retrieval_options)
        self.retrieval_weights = dict(self.DEFAULT_RETRIEVAL_WEIGHTS)
        if retrieval_weights:
            unknown_w = set(retrieval_weights) - set(self.DEFAULT_RETRIEVAL_WEIGHTS)
            if unknown_w:
                raise ValueError(f"unknown retrieval weights: {sorted(unknown_w)}")
            self.retrieval_weights.update({k: float(v) for k, v in retrieval_weights.items()})
        self._validate_retrieval_weights()

    def _validate_retrieval_weights(self) -> None:
        w = self.retrieval_weights
        if any(v < 0 for v in w.values()):
            raise ValueError("retrieval RRF weights must be non-negative")
        if abs(w["w_bm25_plain"] + w["w_dense_plain"] - 1.0) > 1e-6:
            raise ValueError("retrieval plain weights must sum to 1.0")
        if abs(w["w_bm25_lex"] + w["w_dense_lex"] + w["w_lexref_lex"] - 1.0) > 1e-6:
            raise ValueError("retrieval lex weights must sum to 1.0")

    def rrf_arm_weights(self, present_arms: List[str], lex_query: bool) -> List[float]:
        """RRF weights for the present arms, renormalized to sum 1.

        Weighted policy (use_weighted_rrf + reference-bearing query) uses the
        lex row, else the plain row; the legacy path (flag off) stays uniform.
        """
        w = self.retrieval_weights
        if self.retrieval_options["use_weighted_rrf"] and lex_query:
            table = {"bm25": w["w_bm25_lex"], "dense": w["w_dense_lex"], "lexref": w["w_lexref_lex"]}
        elif self.retrieval_options["use_weighted_rrf"]:
            table = {"bm25": w["w_bm25_plain"], "dense": w["w_dense_plain"], "lexref": w["w_lexref_lex"]}
        else:
            return [1.0 / len(present_arms)] * len(present_arms) if present_arms else []
        picked = [table[name] for name in present_arms]
        total = sum(picked)
        if total <= 0:
            raise ValueError("retrieval RRF weights sum to zero for present arms")
        return [v / total for v in picked]

    @property
    def policy_needs_generator(self) -> bool:
        """Check if current selector configuration requires neural generation."""
        if self.selector is None:
            return False
        policy = getattr(self.selector, "policy", "fixed_baseline")
        best_fixed = getattr(self.selector, "best_fixed_candidate", "stitched_extract")
        return policy_requires_generator(policy, best_fixed)

    @classmethod
    def build_mock(
        cls,
        retrieval_options: Optional[Dict[str, Any]] = None,
        retrieval_weights: Optional[Dict[str, float]] = None,
    ) -> LegalQAPipeline:
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
        return cls(mem, bm25, dense, reranker, packer, generator, selector,
                   retrieval_options=retrieval_options, retrieval_weights=retrieval_weights)

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
        use_legal_reference: bool = False,
        resolved_config: Optional[Any] = None,
    ) -> LegalQAPipeline:
        """Load full pipeline from disk artifacts with explicit Dual-T4 GPU placement.

        With resolved_config, the candidate's retrieval recipe (flags, pool,
        RRF weights) and legal-reference arm bind to this pipeline; explicit
        arguments win over the resolved recipe.
        """
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

        # 8. Legal-reference arm (opt-in experiment): positions index over the
        # same ordered corpus rows the BM25/dense arms use.
        legal_index: Optional[Dict[str, Any]] = None
        legal_rows: Optional[List[Dict[str, Any]]] = None
        retrieval_options: Optional[Dict[str, Any]] = None
        retrieval_weights: Optional[Dict[str, float]] = None
        if resolved_config is not None:
            retrieval_options, retrieval_weights = retrieval_options_from_config(resolved_config)
            use_legal_reference = use_legal_reference or bool(retrieval_options["use_legal_reference"])
        if use_legal_reference:
            from src.common.legal_reference import build_legal_reference_index

            legal_rows = list(bm25.corpus) if bm25.corpus else []
            legal_index, lex_report = build_legal_reference_index(legal_rows)
            if lex_report["is_empty"]:
                raise ValueError("legal-reference arm enabled but the index is empty")

        return cls(
            memory,
            bm25,
            dense,
            reranker,
            packer,
            generator,
            selector,
            legal_index=legal_index,
            legal_rows=legal_rows,
            retrieval_options=retrieval_options or ({"use_legal_reference": True} if use_legal_reference else None),
            retrieval_weights=retrieval_weights,
        )

    def retrieve_and_rerank(
        self,
        question: str,
        top_k_rerank: int = 8,
        **option_overrides: Any,
    ) -> Dict[str, Any]:
        """Hybrid retrieval (BM25 + Dense [+ legal-reference]) and reranking.

        Retrieval experiments (legal-reference arm, acronym expansion,
        weighted RRF, diversification, lost-in-middle) are opt-in via
        constructor retrieval_options or per-call overrides; defaults
        reproduce the production recipe exactly.
        """
        options = dict(self.retrieval_options)
        unknown = set(option_overrides) - set(self.DEFAULT_RETRIEVAL_OPTIONS)
        if unknown:
            raise ValueError(f"unknown retrieval options: {sorted(unknown)}")
        options.update(option_overrides)

        pool = int(options["candidate_pool"])
        rrf_k = int(options["rrf_k"])
        retrieval_query = rewrite_query_for_retrieval(question, use_acronyms=options["use_acronyms"])
        lex_query = has_legal_reference(question)

        bm25_res = self.bm25.search(retrieval_query, top_k=pool) if self.bm25 else []
        dense_res = self.dense.search(retrieval_query, top_k=pool) if self.dense else []
        lex_res: List[Dict[str, Any]] = []
        if options["use_legal_reference"] and self.legal_index is not None and self.legal_rows is not None:
            lex_res = search_legal_references([retrieval_query], self.legal_index, self.legal_rows, k=pool)[0]

        arms: List[List[Dict[str, Any]]] = []
        arm_names: List[str] = []
        if bm25_res:
            arms.append(bm25_res)
            arm_names.append("bm25")
        if dense_res:
            arms.append(dense_res)
            arm_names.append("dense")
        if lex_res:
            arms.append(lex_res)
            arm_names.append("lexref")
        # Scores live in the candidate (algorithm config); an empty lexref
        # arm is a no-op and fusion falls back to the present arms.
        arm_weights = self.rrf_arm_weights(arm_names, lex_query)
        if len(arms) >= 2:
            fused_res = reciprocal_rank_fusion(arms, k=rrf_k, weights=arm_weights)
        else:
            fused_res = arms[0] if arms else []

        top_seeds = self.reranker.rerank(question, fused_res, top_k=top_k_rerank) if (self.reranker and fused_res) else fused_res[:top_k_rerank]
        top_seeds = self.packer.prepare_seeds(
            top_seeds,
            diversify=options["diversify_context"],
            lost_in_middle=options["lost_in_middle"],
            max_parts_per_article=int(options["max_parts_per_article"]),
        )

        pack_multi = self.packer.pack_evidence(top_seeds, pack_type="multi_seed_2500_chars", max_chars=3500)
        primary_evidence = pack_multi.get("text") or (top_seeds[0]["text_raw"] if top_seeds else "")

        r1 = float(top_seeds[0].get("rerank_score", top_seeds[0].get("score", 0.0))) if top_seeds else 0.0
        r2 = float(top_seeds[1].get("rerank_score", top_seeds[1].get("score", 0.0))) if len(top_seeds) > 1 else r1

        retrieval_meta = {
            "rerank_top1": r1,
            "rerank_margin": r1 - r2,
            "bm25_top1": float(bm25_res[0].get("score", 0.0)) if bm25_res else 0.0,
            "dense_top1": float(dense_res[0].get("score", 0.0)) if dense_res else 0.0,
            "lexref_fired": bool(lex_res),
            "lexref_hits": len(lex_res),
            "rrf_weights": list(arm_weights),
            "query_rewritten": retrieval_query != question,
        }

        return {
            "bm25_results": bm25_res,
            "dense_results": dense_res,
            "lexref_results": lex_res,
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
        return_provenance: bool = False,
    ) -> Any:
        """High-throughput batch prediction orchestrating BM25, batched dense GPU search, batched reranking, and batched generation.

        With return_provenance=True, returns (results, provenance) where
        provenance records the winning source per query (exact / fuzzy /
        generated / extractive) plus counts. Default returns results only.
        """
        results: Dict[str, Dict[str, str]] = {}
        sources: Dict[str, str] = {}
        unseen_items: List[Tuple[str, str]] = []

        # 1. Exact Memory Pre-pass
        for item in items:
            qa_id = str(item.get("id") or item.get("qa_id") or "").strip()
            q = str(item.get("question", "")).strip()
            exact_ans = self.memory.lookup_exact(qa_id, q)
            if exact_ans:
                results[qa_id] = {"answer": exact_ans}
                sources[qa_id] = "exact_memory"
            else:
                unseen_items.append((qa_id, q))

        if not unseen_items:
            return (results, self._provenance_report(sources)) if return_provenance else results

        options = dict(self.retrieval_options)
        pool = int(options.get("candidate_pool", 50))
        rrf_k = int(options.get("rrf_k", 60))

        unseen_queries = [q for _, q in unseen_items]
        dense_queries = [
            rewrite_query_for_retrieval(q, use_acronyms=options.get("use_acronyms", False))
            for q in unseen_queries
        ]

        # 2. Batched Dense Retrieval
        dense_results: List[List[Dict[str, Any]]] = []
        if self.dense:
            dense_results = self.dense.search_batch(dense_queries, top_k=pool, batch_size=retrieval_batch_size)
        else:
            dense_results = [[] for _ in unseen_queries]

        # 3. Collect BM25, Lexref, and Fused Candidates for all queries
        fused_candidate_lists: List[List[Dict[str, Any]]] = []
        bm25_scores: List[float] = []
        dense_scores: List[float] = []
        fuzzy_hits: List[Optional[Dict[str, Any]]] = []

        for idx, (qa_id, question) in enumerate(unseen_items):
            f_hit = self.memory.lookup_fuzzy(question, threshold=0.90)
            fuzzy_hits.append(f_hit)

            ret_q = dense_queries[idx]
            bm25_res = self.bm25.search(ret_q, top_k=pool) if self.bm25 else []
            dense_res = dense_results[idx] if idx < len(dense_results) else []

            lex_res: List[Dict[str, Any]] = []
            if options.get("use_legal_reference") and self.legal_index is not None and self.legal_rows is not None:
                lex_res = search_legal_references([ret_q], self.legal_index, self.legal_rows, k=pool)[0]

            bm25_scores.append(float(bm25_res[0].get("score", 0.0)) if bm25_res else 0.0)
            dense_scores.append(float(dense_res[0].get("score", 0.0)) if dense_res else 0.0)

            arms: List[List[Dict[str, Any]]] = []
            arm_names: List[str] = []
            if bm25_res:
                arms.append(bm25_res)
                arm_names.append("bm25")
            if dense_res:
                arms.append(dense_res)
                arm_names.append("dense")
            if lex_res:
                arms.append(lex_res)
                arm_names.append("lexref")

            lex_query = has_legal_reference(question)
            arm_weights = self.rrf_arm_weights(arm_names, lex_query)
            if len(arms) >= 2:
                fused = reciprocal_rank_fusion(arms, k=rrf_k, weights=arm_weights)
            else:
                fused = arms[0] if arms else []
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

        # 5. Build Evidence Packs with Context Preparation
        evidence_records: List[Dict[str, Any]] = []
        for idx, (qa_id, question) in enumerate(unseen_items):
            raw_top_seeds = all_top_seeds[idx]
            top_seeds = self.packer.prepare_seeds(
                raw_top_seeds,
                diversify=options.get("diversify_context", False),
                lost_in_middle=options.get("lost_in_middle", False),
                max_parts_per_article=int(options.get("max_parts_per_article", 2)),
            )
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

            selected, source = self.selector.select_with_source(
                candidates=candidates,
                question=rec["question"],
                evidence=rec["primary_evidence"],
                retrieval_meta=rec["retrieval_meta"],
                features=rec["fuzzy_hit"],
            )
            results[rec["qa_id"]] = {"answer": selected}
            sources[rec["qa_id"]] = source

        if return_provenance:
            return results, self._provenance_report(sources)
        return results

    @staticmethod
    def _provenance_report(sources: Dict[str, str]) -> Dict[str, Any]:
        """Bucket winning selector keys into exact/fuzzy/generated/extractive."""
        buckets: Dict[str, str] = {}
        for qa_id, source in sources.items():
            if source == "exact_memory":
                buckets[qa_id] = "exact"
            elif source == "fuzzy_memory":
                buckets[qa_id] = "fuzzy"
            elif source in ("generated", "snapped") or source.startswith("strategy_f_"):
                buckets[qa_id] = "generated"
            else:
                buckets[qa_id] = "extractive"
        counts = {"exact": 0, "fuzzy": 0, "generated": 0, "extractive": 0}
        for bucket in buckets.values():
            counts[bucket] += 1
        return {"sources": buckets, "counts": counts, "num_predictions": len(buckets)}
