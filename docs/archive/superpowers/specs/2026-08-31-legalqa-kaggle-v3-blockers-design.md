# LegalQA Kaggle V3 — Final Blocker Fix & Production Stack A Design

- **Date:** 2026-08-31
- **Focus:** Production Execution Readiness on Kaggle Dual-T4
- **Primary Stack:** **Stack A** (BM25S + DEk21 v2 + BGE-reranker-v2-m3 + Qwen2.5-3B-Instruct QLoRA)
- **Parameter Target:** Strictly `< 4,000,000,000` total learned parameters loaded at inference

---

## 1. Core Architecture Principles

1. **Production Stack A Focus**: Strip fake runtime switching for Stack B from the production notebook. Keep Stack A as the single tested, reproducible production pipeline.
2. **Self-Contained Runtime Packaging**: Package `src/`, `scripts/`, `configs/`, `requirements-kaggle.txt`, and `reranker_training_pairs.parquet` with complete SHA256 code and dataset manifests.
3. **No Silent Training Skips**: In `train_and_submit` profile, if reranker or generator training fails or is skipped, the notebook fails immediately with a fatal error.
4. **Token-Aware Truncation & Chat Parity**: Tokenize answer + assistant framing first; reserve answer token budget; pack evidence units into remaining token budget; render prompts via native `tokenizer.apply_chat_template`.
5. **Reranker Training with Validation & Best Checkpoint**: Evaluate on held-out fold pairs per epoch, track validation loss and pairwise accuracy, and save the best checkpoint.
6. **Dense Index Integrity & FP16 Preservation**: Memory-map FP16 embeddings without expanding to FP32 in RAM; verify chunk-ID hashes and row count; fail loudly on GPU allocation/search error in final mode.
7. **True Batched Dual-T4 Inference**: Batch BM25 candidate retrieval, `DenseRetriever.search_batch()`, batched reranking, and `QwenGenerator.generate_batch()` across queries.
8. **Dynamic Parameter Audit**: Count base model parameters and actual trained PEFT adapter parameters dynamically from `training_manifest.json`.

---

## 2. Component Specifications

### 2.1 Packaging (`scripts/package_kaggle_dataset.py`)
- Stages:
  * `code/LegalQA/src/**`
  * `code/LegalQA/scripts/**`
  * `code/LegalQA/configs/**`
  * `code/LegalQA/requirements-kaggle.txt` (including `pyvi>=0.1.1`)
  * `code_manifest.json` (hashes of all python/yaml files in `src/`, `scripts/`, `configs/`)
  * `data/legal_chunks.parquet`
  * `data/qa_unique.parquet`
  * `data/known_qa.json`
  * `data/qa_citations.parquet`
  * `data/retrieval_labels.parquet`
  * `data/fold_assignments.parquet`
  * `data/reranker_training_pairs.parquet`
  * `indexes/bm25/**`
  * `indexes/dek21/**`

### 2.2 BM25 & Dense Search (`src/common/bm25.py`, `src/common/dense.py`)
- `bm25.py`: Ensure `import sys` is present; support `fail_on_missing_index` flag.
- `dense.py`: Load FP16 using `np.load(emb_path, mmap_mode="r")`; check chunk-ID hash and row count; move to `cuda:1` FP16 tensor; fail loudly in final mode.

### 2.3 Reranker Training (`src/task2/training/train_reranker.py`)
- Training & validation DataLoader on `PairDataset`.
- Computes validation loss and pairwise ranking accuracy (`score(pos) > score(neg)`).
- Saves best validation checkpoint to `output_dir`.
- Fails loudly on data error or GPU failure.

### 2.4 Generator SFT (`src/task2/generator.py`, `src/task2/training/train_generator.py`)
- Prompts rendered using `tokenizer.apply_chat_template`.
- `build_sft_example`: Token-aware answer preservation.
- Strict reload smoke test: if generated text is empty or raises exception, raise `RuntimeError`.
- Persists `adapter_trainable_params` in `training_manifest.json`.

### 2.5 Inference Orchestrator (`src/task2/predict.py`)
- `predict_batch`: True multi-query batching across BM25, GPU dense search, reranker scoring, and generator batching.
- Feature extraction uses `rerank_score` instead of fused `score`.

### 2.6 Canonical Kaggle Notebook (`kaggle_kernel/legalqa_gpu_pipeline.ipynb`)
- `EXECUTION_PROFILE = "train_and_submit"` default.
- Installs missing dependencies from `requirements-kaggle.txt`.
- Preflight with resolved paths.
- Reranker training $\to$ QLoRA training $\to$ Smoke verification $\to$ Checkpoint validation $\to$ Batched inference $\to$ Strict 1000-ID submission.
