# LegalQA Kaggle Max-Score V2 — Architectural Design Specification

- **Date:** 2026-08-31
- **Target Platform:** Kaggle Dual NVIDIA T4 GPUs (16 GB + 16 GB VRAM)
- **Primary Metric:** Official Task 2 Whitespace-Tokenized METEOR
- **Parameter Limit:** Strictly `< 4,000,000,000` total learned parameters loaded at inference

---

## 1. Overview & Core Philosophy

The LegalQA system is an **extractive-first, retrieval-dominant RAG pipeline** where statutory precision is paramount. Because Vietnamese legal QA evaluation heavily weights exact statutory phrasing, the architecture treats LLM generation (Qwen2.5) as **one candidate producer** within a rich candidate ensemble, rather than unconditionally outputting generated text.

The system enforces:
1. **Zero-Leakage Grouped Folds**: Near-duplicate and identical legal questions are blocked across cross-validation folds. Sampled OOF evaluation isolates all fold records from memory prior to sampling.
2. **True BM25S Lexical Retrieval**: Memory-mapped BM25S index without redundant Python postings reconstruction.
3. **Exact GPU Dense Search**: L2-normalized FP16 corpus embeddings on `cuda:1` using `torch.matmul` and `torch.topk`.
4. **Evidence Candidate Packing**: Multi-granularity statutory packing (clause-focused, primary article siblings, full article, relevance-selected top-2 articles, and multi-seed budgets).
5. **Chat Parity & Completion SFT**: Tokenizer-native chat template for training/inference, completion-only loss masking, and answer-preserving truncation.
6. **Cross-Fitted Meta-Selector with Fixed Guardrail**: Meta-OOF cross-fitted selector gated against the strongest fixed candidate baseline (e.g. relevance/stitched extract).
7. **Canonical Lifecycle Notebook**: Self-contained Kaggle notebook with preflight, optional reranker/QLoRA training, reload smoke verification, dev evaluation, dual-GPU inference, and strict 1000-ID submission validation.

---

## 2. Model Stack Architectures

### Stack A (Generator-Capacity Oriented)
- **Sparse**: BM25S (0 parameters)
- **Dense**: DEk21 v2 (`CODE4LIFEOFFICIAL/huydang-dek21-embedding-v2`, ~100M parameters, 768-dim)
- **Reranker**: BGE Reranker v2 M3 (`BAAI/bge-reranker-v2-m3`, ~568M parameters)
- **Generator**: Qwen2.5-3B-Instruct (`Qwen/Qwen2.5-3B-Instruct`, ~3.09B parameters)
- **Total Parameters**: ~3.758B (compliant with `< 4.0B`)

### Stack B (Retrieval-Capacity Oriented)
- **Sparse**: BM25S (0 parameters)
- **Dense**: BGE-M3 (`BAAI/bge-m3`, ~568M parameters, 1024-dim)
- **Reranker**: BGE Reranker v2 M3 (`BAAI/bge-reranker-v2-m3`, ~568M parameters)
- **Generator**: Qwen2.5-1.5B-Instruct (`Qwen/Qwen2.5-1.5B-Instruct`, ~1.54B parameters)
- **Total Parameters**: ~2.676B (compliant with `< 4.0B`)

### Promotion Rule
Promotion is empirical: the stack achieving higher 5-fold OOF METEOR under leakage-safe cross-validation is deployed. If generator candidates do not consistently outperform extractive candidates, the extractive baseline is promoted.

---

## 3. Component Details & Invariants

### 3.1. Data & Fold Grouping (`src/common/normalize.py`, `scripts/prepare_data.py`)
- Near-duplicate clustering using char TF-IDF (3-5 ngrams) + cosine similarity threshold $\ge 0.92$ + matching doc/article signals.
- Graph connected components assign all related queries to the same `fold_id` (0..4).
- Invariant: $\text{Train\_IDs} \cap \text{Val\_IDs} = \emptyset$ and $\text{Train\_Question\_Norms} \cap \text{Val\_Question\_Norms} = \emptyset$.

### 3.2. QA Memory (`src/task2/qa_memory.py`)
- `lookup_exact`: Matches normalized question with unique answer; ID match verified against question consistency.
- `lookup_fuzzy`: Returns similar QA record with similarity score and legal entity consistency flags.
- `filter_fold`: Strictly excludes all records for the entire held-out fold.

### 3.3. Sparse Retrieval (`src/common/bm25.py`)
- BM25S load uses `mmap=True` without Python dictionary rebuilding.
- Computes `bm25_raw_score` and separate `legal_boost` without mutating raw scores.

### 3.4. Dense Retrieval (`src/common/dense.py`)
- Abstract `DenseRetriever` interface with `fit`, `load_index`, `save_index`, `search`, and `search_batch`.
- FP16 corpus tensor (~1.15 GB) on `cuda:1` searched via `torch.topk`.
- Fallback to chunked GPU matrix multiplication if memory constrained.
- Strict manifest hash and row count check on load.

### 3.5. Evidence Candidate Packing (`src/task2/evidence_packer.py`)
- Packs statutory units preserving hierarchical structure (`[DOCUMENT]`, `[ARTICLE]`, `[CLAUSE]`).
- Candidate packs: `focused_clause`, `primary_article_relevant_siblings`, `primary_full_article`, `relevance_selected_top2_articles`, `multi_seed_2500_chars`, `multi_seed_4000_chars`.

### 3.6. Generator & SFT Training (`src/task2/generator.py`, `src/task2/training/train_generator.py`)
- Native chat formatting: `tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)`.
- Loss mask on assistant response tokens only.
- Answer-preserving truncation: Truncates evidence before gold answer.
- Final mode fails loudly on model load failure or missing weights.

### 3.7. Candidate Ensemble & Selection (`src/task2/candidates.py`, `src/task2/selector.py`)
- Candidate family: `exact_memory`, `fuzzy_memory`, `focused_clause_extract`, `relevance_extract`, `primary_article_extract`, `multi_seed_extract`, `generated_base`, `generated_qlora`, `source_safe_generated`, `strategy_f_300`, `strategy_f_600`, `strategy_f_1000`, `strategy_f_1500`.
- Meta-OOF cross-fitted `HistGradientBoostingRegressor` selector.
- Guardrail: Fallback to best fixed candidate baseline if meta-OOF METEOR of learned selector is lower.

### 3.8. Canonical Kaggle Notebook (`kaggle_kernel/legalqa_gpu_pipeline.ipynb`)
- End-to-end lifecycle:
  1. Config flags (`RUN_RERANKER_TRAINING`, `RUN_GENERATOR_TRAINING`, `RUN_DEV_EVALUATION`, `RUN_PUBLIC_INFERENCE`, `REUSE_EXISTING_CHECKPOINTS`, `FINAL_STACK`).
  2. Environment detection & safe secret retrieval (`HF_TOKEN`).
  3. Packaged code/data resolution (`code/LegalQA`).
  4. Preflight validation (`scripts/preflight_kaggle.py`).
  5. Reranker training (optional).
  6. Generator QLoRA training & reload smoke test (optional).
  7. Dev evaluation & regression check.
  8. Dual-T4 inference: GPU 0 (Qwen), GPU 1 (Dense + Reranker), CPU (BM25 + Memory).
  9. Strict 1000-ID submission validation (`submission.json` & `submission.json.zip`).

---

## 4. Verification Plan

1. **Unit Tests**: Full `pytest` suite passing across all modules.
2. **Leakage Tests**: Verify fold isolation on exact and sampled OOF runs.
3. **Retrieval Benchmark**: Compare BM25, DEk21, BGE-M3, and reranker on Recall@K and MRR.
4. **Candidate & Selector Ablation**: Verify candidate family scores and confirm selector guardrail.
5. **Parameter Audit**: Verify total learned parameters strictly `< 4.0B`.
6. **Kaggle Packaging & Preflight**: Ensure dataset packager produces complete staged runtime and preflight passes.
