# LegalQA Kaggle V3 Final Blockers Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Remediate all 15 audit findings in `LEGALQA_KAGGLE_FINAL_BLOCKERS_FIX_V3.md` and deliver a 100% production-ready, reproducible Kaggle Dual-T4 notebook and codebase for Stack A.

**Architecture:** Stack A production RAG system (BM25S + DEk21 v2 + BGE-reranker-v2-m3 + Qwen2.5-3B QLoRA) with self-contained runtime packaging, true batched inference, token-aware SFT, validated reranker training, and strict failure gates.

**Tech Stack:** Python, PyTorch, Transformers, PEFT/QLoRA, TRL, SentenceTransformers, BM25S, scikit-learn, Pandas, Parquet, NLTK METEOR, Kaggle dual NVIDIA T4.

**Spec:** `docs/superpowers/specs/2026-08-31-legalqa-kaggle-v3-blockers-design.md` and `LEGALQA_KAGGLE_FINAL_BLOCKERS_FIX_V3.md`.

## Global Constraints
- Maximum total learned parameters: `< 4,000,000,000` (strict exclusive limit, including adapter).
- Optimization target: Official Task 2 whitespace-tokenized METEOR.
- Kaggle target environment: Dual NVIDIA T4 (16 GB + 16 GB VRAM).
- Fail loudly: No silent fallback or silent skips in `train_and_submit` profile.

---

### Task 1: Runtime Packaging & Requirements

**Files:**
- Modify: `requirements-kaggle.txt`
- Modify: `scripts/package_kaggle_dataset.py`
- Modify: `tests/test_kaggle_packaging.py`

- [ ] **Step 1: Update `requirements-kaggle.txt` to include `pyvi>=0.1.1`**
- [ ] **Step 2: Update `scripts/package_kaggle_dataset.py` to package `scripts/`, `src/`, `configs/`, `reranker_training_pairs.parquet`, and hash both `src/` and `scripts/`**
- [ ] **Step 3: Update `tests/test_kaggle_packaging.py` to verify staged runtime imports scripts and src**
- [ ] **Step 4: Run `pytest tests/test_kaggle_packaging.py -v` and verify PASS**

---

### Task 2: BM25 & Dense Index Integrity and FP16 mmap

**Files:**
- Modify: `src/common/bm25.py`
- Modify: `src/common/dense.py`
- Modify: `tests/test_common_bm25.py`
- Modify: `tests/test_common_dense_and_rrf.py`

- [ ] **Step 1: Ensure `import sys` in `src/common/bm25.py` and add `fail_on_missing_index` support**
- [ ] **Step 2: Update `src/common/dense.py` to preserve FP16 on mmap load and verify chunk-ID hash in final mode**
- [ ] **Step 3: Run `pytest tests/test_common_bm25.py tests/test_common_dense_and_rrf.py -v` and verify PASS**

---

### Task 3: Reranker Trainer with Validation & Best Checkpoint Saving

**Files:**
- Modify: `src/task2/training/train_reranker.py`
- Modify: `tests/test_reranker_training_data.py`

- [ ] **Step 1: Implement validation loop with loss and pairwise accuracy evaluation in `train_bge_reranker`**
- [ ] **Step 2: Save best validation checkpoint and fail loudly on missing pairs or runtime failure**
- [ ] **Step 3: Run `pytest tests/test_reranker_training_data.py -v` and verify PASS**

---

### Task 4: QLoRA Prompt Parity, Token-Aware Truncation & Smoke Enforcement

**Files:**
- Modify: `src/task2/generator.py`
- Modify: `src/task2/training/train_generator.py`
- Modify: `tests/test_generator_training_data.py`
- Modify: `tests/test_task2_generator.py`

- [ ] **Step 1: Update `QwenGenerator` and `train_generator.py` to use native tokenizer chat template**
- [ ] **Step 2: Implement token-aware `build_sft_example` reserving gold answer tokens first**
- [ ] **Step 3: Enforce strict `RuntimeError` on failed reload smoke test and record `adapter_trainable_params`**
- [ ] **Step 4: Run `pytest tests/test_generator_training_data.py tests/test_task2_generator.py -v` and verify PASS**

---

### Task 5: Selector Reranker Metadata & Feature Distribution Parity

**Files:**
- Modify: `src/task2/predict.py`
- Modify: `src/task2/selector.py`
- Modify: `tests/test_task2_selector.py`

- [ ] **Step 1: Update `predict.py` to extract `rerank_score` for `rerank_top1` and `rerank_margin`**
- [ ] **Step 2: Ensure deterministic meta-fold grouping by `qa_id` and consistent feature distribution in `selector.py`**
- [ ] **Step 3: Run `pytest tests/test_task2_selector.py -v` and verify PASS**

---

### Task 6: True Batched Dual-T4 Inference

**Files:**
- Modify: `src/task2/predict.py`
- Modify: `tests/test_task2_end_to_end.py`

- [ ] **Step 1: Implement true batched retrieval and generation in `LegalQAPipeline.predict_batch`**
- [ ] **Step 2: Verify `predict_batch` parity and correctness in `tests/test_task2_end_to_end.py`**
- [ ] **Step 3: Run `pytest tests/test_task2_end_to_end.py -v` and verify PASS**

---

### Task 7: Preflight & Dynamic Adapter Parameter Audit

**Files:**
- Modify: `scripts/preflight_kaggle.py`
- Modify: `scripts/audit_parameters.py`
- Modify: `tests/test_config_and_preflight.py`

- [ ] **Step 1: Update `preflight_kaggle.py` to validate resolved Kaggle paths and reranker pairs file**
- [ ] **Step 2: Update `audit_parameters.py` to support dynamic adapter parameter reading from manifest**
- [ ] **Step 3: Run `pytest tests/test_config_and_preflight.py -v` and verify PASS**

---

### Task 8: Canonical Notebook & Full Test Suite Pass

**Files:**
- Modify: `kaggle_kernel/legalqa_gpu_pipeline.ipynb`
- Modify: `README.md`
- Run: full pytest suite

- [ ] **Step 1: Rebuild `kaggle_kernel/legalqa_gpu_pipeline.ipynb` with `EXECUTION_PROFILE = "train_and_submit"`, dependency installation, strict training failure gates, and true batched inference**
- [ ] **Step 2: Update `README.md` with accurate Stack A production documentation**
- [ ] **Step 3: Run full pytest suite across `tests/` and verify 100% PASS**
