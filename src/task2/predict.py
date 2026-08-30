import os
import json
from src.common.bm25 import BM25Retriever
from src.common.dense_dek21 import DEk21Retriever
from src.common.rrf import reciprocal_rank_fusion
from src.common.reranker import BGEReranker
from src.task2.qa_memory import QAMemory
from src.task2.article_stitcher import ArticleStitcher
from src.task2.generator import QwenGenerator
from src.task2.source_snap import snap_facts_to_evidence, select_best_answer_candidate

class LegalQAPipeline:
    def __init__(self, memory: QAMemory, bm25: BM25Retriever, dense: DEk21Retriever, reranker: BGEReranker, stitcher: ArticleStitcher, generator: QwenGenerator):
        self.memory = memory
        self.bm25 = bm25
        self.dense = dense
        self.reranker = reranker
        self.stitcher = stitcher
        self.generator = generator

    @classmethod
    def build_mock(cls):
        chunks = [
            {"chunk_id": "c1", "doc_name": "Nghị định 90/2017/NĐ-CP", "parent_article_id": "art17", "text_raw": "[DOCUMENT] Nghị định 90/2017/NĐ-CP\n[ARTICLE] Điều 17. Phạt tiền từ 1.000.000 đồng đến 2.000.000 đồng đối với hành vi không tiêm phòng.", "start_char": 0}
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
    def load_pipeline(cls, data_dir: str = "artifacts/task2/data", index_dir: str = "artifacts/task2/indexes/bm25"):
        known_qa_path = os.path.join(data_dir, "known_qa.json")
        qa_unique_path = os.path.join(data_dir, "qa_unique.parquet")
        chunks_path = os.path.join(data_dir, "legal_chunks.parquet")

        if os.path.exists(known_qa_path):
            memory = QAMemory.load(known_qa_path, qa_unique_path)
        else:
            memory = QAMemory.from_records([])

        if os.path.exists(index_dir):
            bm25 = BM25Retriever.load(index_dir)
        else:
            bm25 = BM25Retriever()

        dense = DEk21Retriever(model_name="mock")
        reranker = BGEReranker(model_name="BAAI/bge-reranker-v2-m3")

        if bm25.corpus:
            stitcher = ArticleStitcher(bm25.corpus)
        elif os.path.exists(chunks_path):
            import pandas as pd
            df_chunks = pd.read_parquet(chunks_path)
            stitcher = ArticleStitcher(df_chunks.to_dict("records"))
        else:
            stitcher = ArticleStitcher([])

        mlx_adapter = "artifacts/task2/checkpoints/generator/mlx_adapter"
        if os.path.exists(mlx_adapter):
            generator = QwenGenerator.load(
                model_path="Qwen/Qwen2.5-3B-Instruct",
                adapter_path=mlx_adapter,
                runtime="auto"
            )
        else:
            generator = QwenGenerator(runtime="fallback")

        return cls(memory, bm25, dense, reranker, stitcher, generator)

    def predict_single(self, qa_id: str, question: str, max_new_tokens: int = 220) -> str:
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

        # 3. Neural Reranker
        top_seeds = self.reranker.rerank(question, fused_res, top_k=8) if (self.reranker and fused_res) else fused_res[:8]

        # 4. Article Stitcher
        stitched_pkg = self.stitcher.stitch(top_seeds)
        evidence_text = stitched_pkg.get("stitched_text") or (top_seeds[0]["text_raw"] if top_seeds else "")

        # 5. Generator
        gen_ans = self.generator.generate(question, evidence_text, max_new_tokens=max_new_tokens)

        # 6. Source Snap & Candidate Selection
        snapped_ans = snap_facts_to_evidence(gen_ans, evidence_text)
        candidates = {
            "focused_extract": top_seeds[0]["text_raw"] if top_seeds else "",
            "stitched_extract": evidence_text,
            "generated": gen_ans,
            "snapped": snapped_ans
        }
        doc_name = top_seeds[0].get("doc_name", "") if top_seeds else ""
        art_num = top_seeds[0].get("article_number", "") if top_seeds else ""
        clause_num = top_seeds[0].get("clause_number", "") if top_seeds else ""
        return select_best_answer_candidate(candidates, doc_name=doc_name, article_num=art_num, clause_num=clause_num)

    def predict_batch(self, items: list[dict]) -> dict:
        results = {}
        for item in items:
            qa_id = str(item.get("id") or item.get("qa_id") or "")
            q = str(item.get("question", ""))
            results[qa_id] = {"answer": self.predict_single(qa_id, q)}
        return results
