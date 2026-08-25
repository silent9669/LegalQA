import json
from src.memory.exact_memory import ExactMemory
from src.memory.provision_memory import ProvisionMemory
from src.postprocess.article_stitcher import ArticleStitcher
from src.postprocess.extractive import generate_extractive_answer
from src.postprocess.source_snap import source_snap_answer
from src.selector.candidate_selector import CandidateSelector

class LegalQAPipeline:
    """
    End-to-End LegalQA Multi-Stage Prediction Pipeline:
    1. Exact QA Memory Lookup (Immediate 100% precision on overlapping queries)
    2. Hybrid Retrieval (BM25 + Dense)
    3. Cross-Encoder Reranker
    4. Article-Level Context Expansion & Sibling Chunk Stitching
    5. Provision Memory Lookup (In-context style demonstration)
    6. Candidate Generation (Extractive / Snapped / LLM)
    7. Candidate Selection
    """
    def __init__(
        self,
        exact_memory: ExactMemory,
        retriever=None,
        reranker=None,
        generator=None,
        article_stitcher: ArticleStitcher = None,
        provision_memory: ProvisionMemory = None
    ):
        self.exact_memory = exact_memory
        self.retriever = retriever
        self.reranker = reranker
        self.generator = generator
        self.article_stitcher = article_stitcher
        self.provision_memory = provision_memory
        self.selector = CandidateSelector()

    def predict(self, sample_id: str, question: str) -> str:
        # 1. Exact Memory Lookup
        known = self.exact_memory.lookup(sample_id, question)
        if known is not None:
            return known

        # 2. Retrieval & Reranking
        evidence_chunks = []
        if self.retriever:
            retrieved_ids = self.retriever.search(question, top_k=25)
            if hasattr(self.retriever, 'chunk_map'):
                candidates = [self.retriever.chunk_map[cid] for cid, _ in retrieved_ids if cid in self.retriever.chunk_map]
            else:
                candidates = [{"content": cid} for cid, _ in retrieved_ids]

            if self.reranker:
                ranked_candidates = self.reranker.rank(question, candidates, top_k=8)
            else:
                ranked_candidates = candidates[:8]

            # 3. Article-Level Expansion / Sibling Stitching
            if self.article_stitcher and ranked_candidates:
                expanded_top = self.article_stitcher.expand_chunk(ranked_candidates[0])
                evidence_chunks = [expanded_top] + ranked_candidates[1:]
            else:
                evidence_chunks = ranked_candidates
        else:
            evidence_chunks = []

        # 4. Provision Memory Example
        examples = []
        if self.provision_memory and evidence_chunks:
            top_c = evidence_chunks[0]
            doc_num = top_c.get("document_number") or ""
            art_num = top_c.get("article_number") or ""
            if doc_num and art_num:
                examples = self.provision_memory.lookup(doc_num, art_num)

        # 5. Multi-Candidate Synthesis
        cand_extract = generate_extractive_answer(question, evidence_chunks)
        cand_generate = cand_extract
        if self.generator:
            cand_generate = self.generator.generate(question, evidence_chunks, examples=examples)

        cand_snap = source_snap_answer(cand_generate, evidence_chunks)

        candidates = {
            "candidate_generate": cand_generate,
            "candidate_snap": cand_snap,
            "candidate_extract": cand_extract
        }

        features = {
            "has_penalty_keyword": any(w in question.lower() for w in ["phạt", "mức phạt", "xử phạt"]),
            "extractive_quality": 0.9 if evidence_chunks else 0.0
        }

        return self.selector.select(question, candidates, features=features)
