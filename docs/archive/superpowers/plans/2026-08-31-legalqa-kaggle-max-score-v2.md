# LegalQA Kaggle Max-Score V2 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Turn `silent9669/LegalQA` into a clean, reproducible, correctly trained, Kaggle T4×2 LegalQA system whose architecture is selected by leakage-safe official-METEOR experiments within a strict `< 4.0B` parameter budget.

**Architecture:** Extractive-first, retrieval-dominant LegalQA with exact/similar QA memory, hybrid BM25S + GPU exact dense retrieval, task-tuned BGE reranker, structured evidence candidate packing, optional Qwen2.5 QLoRA generation, rich candidate ensemble, and cross-fitted candidate selector with fixed baseline guardrail.

**Tech Stack:** Python, PyTorch, Transformers, PEFT/QLoRA, TRL, SentenceTransformers, BM25S, scikit-learn, Pandas, Parquet, NLTK METEOR, Kaggle dual NVIDIA T4.

**Spec:** `docs/superpowers/specs/2026-08-31-legalqa-kaggle-max-score-v2-design.md` and `LEGALQA_KAGGLE_MAX_SCORE_FIX_V2.md`.

## Global Constraints
- Maximum total learned parameters loaded at inference: `< 4,000,000,000` (strict exclusive limit).
- Optimization target: Official Task 2 whitespace-tokenized METEOR.
- Kaggle target environment: Dual NVIDIA T4 (16 GB + 16 GB VRAM).
- Zero data leakage: Near-duplicate and identical queries blocked across folds; full held-out fold excluded from memory on sampled validation.
- Fail loudly: No silent fallback to mock models or BM25-only in final competition mode.

---

### Task 1: Establish Canonical Configuration and Preflight

**Files:**
- Modify: `configs/pipeline.yaml`
- Modify: `configs/models.yaml`
- Create: `configs/experiments.yaml`
- Create: `scripts/preflight_kaggle.py`
- Create: `tests/test_config_and_preflight.py`

- [ ] **Step 1: Write `tests/test_config_and_preflight.py`**
- [ ] **Step 2: Update `configs/pipeline.yaml` with canonical paths and fields**
- [ ] **Step 3: Update `configs/models.yaml` and create `configs/experiments.yaml`**
- [ ] **Step 4: Implement `scripts/preflight_kaggle.py`**
- [ ] **Step 5: Run tests and verify PASS**

---

### Task 2: Fix Fold Construction and Leakage-Safe Evaluation

**Files:**
- Modify: `scripts/prepare_data.py`
- Modify: `scripts/run_oof_validation.py`
- Create: `tests/test_oof_isolation.py`

- [ ] **Step 1: Write `tests/test_oof_isolation.py` testing near-duplicate grouping and sampled fold isolation**
- [ ] **Step 2: Implement near-duplicate grouping in `scripts/prepare_data.py`**
- [ ] **Step 3: Update `scripts/run_oof_validation.py` to isolate the entire validation fold before sampling**
- [ ] **Step 4: Run tests and verify PASS**

---

### Task 3: Safe Exact QA Memory and Useful Similar-QA Lookup

**Files:**
- Modify: `src/task2/qa_memory.py`
- Modify: `tests/test_task2_qa_memory.py`

- [ ] **Step 1: Write tests for ID-question consistency check and strict fold filtering**
- [ ] **Step 2: Update `src/task2/qa_memory.py` with verified exact lookup and feature-rich fuzzy search**
- [ ] **Step 3: Run tests and verify PASS**

---

### Task 4: Real, Memory-Efficient BM25S Lexical Engine

**Files:**
- Modify: `src/common/bm25.py`
- Modify: `tests/test_common_bm25.py`

- [ ] **Step 1: Write tests ensuring BM25S mmap load does not rebuild Python postings in memory**
- [ ] **Step 2: Refactor `src/common/bm25.py` to use pure BM25S mmap and separate legal boosts**
- [ ] **Step 3: Run tests and verify PASS**

---

### Task 5: Generic Dense Retriever with GPU Exact Search

**Files:**
- Create: `src/common/dense.py`
- Modify: `src/common/dense_dek21.py` (backward-compatibility wrapper)
- Modify: `tests/test_common_dense_and_rrf.py`

- [ ] **Step 1: Write tests for `DenseRetriever` row verification, hash validation, and FP16 topk search**
- [ ] **Step 2: Implement `src/common/dense.py` with multi-model support (DEk21, BGE-M3) and PyTorch topk**
- [ ] **Step 3: Run tests and verify PASS**

---

### Task 6: Retrieval Supervision and Hard Negative Mining

**Files:**
- Modify: `src/common/evidence.py`
- Create: `scripts/mine_retrieval_negatives.py`
- Modify: `tests/test_common_evidence.py`

- [ ] **Step 1: Write tests for hard negative miner excluding all resolved positives**
- [ ] **Step 2: Update `src/common/evidence.py` and `scripts/mine_retrieval_negatives.py`**
- [ ] **Step 3: Run tests and verify PASS**

---

### Task 7: Reranker Fine-Tuning Module

**Files:**
- Create: `src/task2/training/__init__.py`
- Create: `src/task2/training/train_reranker.py`
- Create: `scripts/train_reranker.py`
- Create: `tests/test_reranker_training_data.py`

- [ ] **Step 1: Write tests for reranker training data preparation and batch formatting**
- [ ] **Step 2: Implement `train_reranker.py` module and thin CLI**
- [ ] **Step 3: Run tests and verify PASS**

---

### Task 8: Structured Evidence Candidate Packer

**Files:**
- Create: `src/task2/evidence_packer.py`
- Modify: `src/task2/article_stitcher.py` (backward-compatibility alias)
- Create: `tests/test_task2_evidence_packer.py`

- [ ] **Step 1: Write tests for multi-granularity evidence packing without clause duplication**
- [ ] **Step 2: Implement `src/task2/evidence_packer.py`**
- [ ] **Step 3: Run tests and verify PASS**

---

### Task 9: Native Chat Parity and Answer-Preserving Generator SFT

**Files:**
- Modify: `src/task2/generator.py`
- Create: `src/task2/training/train_generator.py`
- Modify: `scripts/train_generator_qlora.py`
- Modify: `tests/test_task2_generator.py`
- Create: `tests/test_generator_training_data.py`

- [ ] **Step 1: Write tests for chat-template parity, assistant completion loss masking, and answer-preserving truncation**
- [ ] **Step 2: Refactor `src/task2/generator.py` and implement `train_generator.py`**
- [ ] **Step 3: Update `scripts/train_generator_qlora.py`**
- [ ] **Step 4: Run tests and verify PASS**

---

### Task 10: Candidate Ensemble Engine

**Files:**
- Create: `src/task2/candidates.py`
- Modify: `src/task2/source_snap.py`
- Create: `tests/test_task2_candidates.py`

- [ ] **Step 1: Write tests for candidate family generation and fact snapping**
- [ ] **Step 2: Implement `src/task2/candidates.py`**
- [ ] **Step 3: Run tests and verify PASS**

---

### Task 11: Cross-Fitted Candidate Selector with Baseline Guardrail

**Files:**
- Create: `src/task2/selector.py`
- Create: `tests/test_task2_selector.py`

- [ ] **Step 1: Write tests for selector feature extraction, meta-OOF fitting, and guardrail fallback**
- [ ] **Step 2: Implement `src/task2/selector.py`**
- [ ] **Step 3: Run tests and verify PASS**

---

### Task 12: End-to-End Pipeline Orchestration & Dual-GPU Placement

**Files:**
- Modify: `src/task2/predict.py`
- Modify: `tests/test_task2_end_to_end.py`

- [ ] **Step 1: Write tests for `LegalQAPipeline` with candidates, selector, and dual-device routing**
- [ ] **Step 2: Update `src/task2/predict.py` to integrate all components cleanly**
- [ ] **Step 3: Run tests and verify PASS**

---

### Task 13: Exact Parameter & Manifest Audit Tooling

**Files:**
- Modify: `scripts/audit_parameters.py`
- Modify: `tests/test_validation_and_audit.py`

- [ ] **Step 1: Write tests for exact parameter counts, adapter accounting, and budget verification**
- [ ] **Step 2: Refactor `scripts/audit_parameters.py`**
- [ ] **Step 3: Run tests and verify PASS**

---

### Task 14: Deterministic Kaggle Dataset Packager

**Files:**
- Modify: `scripts/package_kaggle_dataset.py`
- Create: `requirements-kaggle.txt`
- Modify: `kaggle_dataset/dataset-metadata.json`
- Create: `tests/test_kaggle_packaging.py`

- [ ] **Step 1: Write tests verifying dataset packaging stages code, configs, dependencies, and manifests**
- [ ] **Step 2: Implement full runtime staging in `scripts/package_kaggle_dataset.py`**
- [ ] **Step 3: Run tests and verify PASS**

---

### Task 15: Canonical Train -> Validate -> Infer Kaggle Notebook

**Files:**
- Modify: `kaggle_kernel/legalqa_gpu_pipeline.ipynb`
- Modify: `kaggle_kernel/kernel-metadata.json`

- [ ] **Step 1: Rebuild `kaggle_kernel/legalqa_gpu_pipeline.ipynb` with all 14 required cells**
- [ ] **Step 2: Verify notebook syntax and imports**

---

### Task 16: Complete Test Suite & Validation Regression Pass

**Files:**
- Run full pytest suite with `.venv-ml`
- Verify parameter audit and preflight checks

- [ ] **Step 1: Run pytest across the entire `tests/` directory**
- [ ] **Step 2: Run parameter audit on both Stack A and Stack B**
- [ ] **Step 3: Clean up temporary files and update repository documentation**
