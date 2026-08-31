# LegalQA Task 2 Optimization Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Turn `silent9669/LegalQA` into the strongest possible DSC 2026 Task 2 LegalQA system maximizing official whitespace METEOR while strictly adhering to the <4.0B parameter budget and dual-T4 Kaggle GPU constraints.

**Architecture:** A hybrid retrieval and grounded generation pipeline: Exact & Similar-QA Memory -> Sparse BM25 + Dense DEk21 v2 -> RRF Fusion -> BGE-Reranker-v2-m3 -> Structured Evidence Packing -> Qwen2.5-3B-Instruct (GPU 0) / DEk21+BGE (GPU 1) -> Multi-Candidate Snapping & Strategy F Generation -> OOF-Tuned Candidate Selector -> Verified Submission.

**Tech Stack:** Python 3.12, PyTorch, Transformers, Sentence-Transformers, bm25s, PEFT, TRL, Pandas, NumPy, NLTK (official METEOR), ROUGE-Score.

**Spec:** `task.md`

## Global Constraints

- Primary Metric: Official whitespace-tokenized METEOR: `meteor_score([reference.split()], prediction.split())`.
- Parameter Budget: Strictly < 4.0B learned parameters (DEk21 100M + BGE Reranker 568M + Qwen2.5-3B 3.09B = 3.758B, leaving ~242M margin).
- Hardware Target: Kaggle Dual NVIDIA T4 GPUs (16GB VRAM each). Gen on GPU 0 (`cuda:0`), Retrieval/Reranking on GPU 1 (`cuda:1`).
- Secret: `HF_TOKEN` from Kaggle secrets/environment. Never log, commit, or print tokens.
- Submission Format: `submission.json` and `submission.json.zip` containing all 1,000 public test IDs mapping to `{"answer": "..."}`.
- Zero Data Leakage: Cross-validation and similar-QA memory must strictly isolate held-out folds.

---

### Task 1: Reconcile Configs, Dependencies, and Parameter Audit

**Files:**
- Modify: `configs/pipeline.yaml`
- Modify: `configs/models.yaml`
- Modify: `scripts/audit_parameters.py`
- Modify: `requirements.txt`
- Test: `tests/test_validation_and_audit.py`

**Interfaces:**
- `audit_parameter_budget(config_path: str) -> dict`: Returns `{"total_learned_parameters": int, "limit": int, "is_compliant": bool, "breakdown": dict}`.

- [ ] **Step 1: Write failing test for config consistency and parameter audit with adapter**
Ensure `test_validation_and_audit.py` tests that `configs/pipeline.yaml` matches `configs/models.yaml` and parameter audit includes adapter overhead.

- [ ] **Step 2: Run test to verify failure**
Run: `.venv-ml/bin/pytest tests/test_validation_and_audit.py -v`

- [ ] **Step 3: Update `configs/pipeline.yaml`, `configs/models.yaml`, `scripts/audit_parameters.py`**
Align dense model to `CODE4LIFEOFFICIAL/huydang-dek21-embedding-v2`, reranker to `BAAI/bge-reranker-v2-m3`, generator to `Qwen/Qwen2.5-3B-Instruct`.

- [ ] **Step 4: Run test to verify pass**
Run: `.venv-ml/bin/pytest tests/test_validation_and_audit.py -v`

- [ ] **Step 5: Commit**
Run: `git commit -m "fix(config): reconcile model configs and parameter audit under 4B budget"`

---

### Task 2: Robust BM25, Dense DEk21, and RRF Fusion

**Files:**
- Modify: `src/common/bm25.py`
- Modify: `src/common/dense_dek21.py`
- Modify: `src/common/rrf.py`
- Test: `tests/test_common_bm25.py`
- Test: `tests/test_common_dense_and_rrf.py`

**Interfaces:**
- `BM25Retriever.search(query: str, top_k: int = 60) -> List[Dict[str, Any]]`: Full corpus BM25 without truncation, with legal entity boosts.
- `DEk21Retriever.search(query: str, top_k: int = 60) -> List[Dict[str, Any]]`: L2-normalized dense cosine retrieval with FP16 support.
- `reciprocal_rank_fusion(run_list: List[List[Dict[str, Any]]], k: int = 60, weights: List[float] = None) -> List[Dict[str, Any]]`: Deterministic RRF.

- [ ] **Step 1: Write test for BM25 hand-computed scoring, no posting cap, and dense row alignment**
- [ ] **Step 2: Run test to verify behavior**
- [ ] **Step 3: Update `src/common/bm25.py`, `src/common/dense_dek21.py`, `src/common/rrf.py`**
- [ ] **Step 4: Run test to verify pass**
- [ ] **Step 5: Commit**

---

### Task 3: Similar-QA Memory Branch with Strict Fold Isolation

**Files:**
- Modify: `src/task2/qa_memory.py`
- Test: `tests/test_task2_qa_memory.py`

**Interfaces:**
- `QAMemory.lookup_fuzzy(question: str, threshold: float = 0.90) -> Optional[Dict[str, Any]]`: Finds high-confidence nearest training QA using normalized n-gram similarity + entity consistency.
- `QAMemory.filter_fold(val_qa_ids: Set[str], val_questions: Optional[Set[str]] = None) -> QAMemory`: Zero-leakage fold isolation.

- [ ] **Step 1: Write tests for fuzzy QA memory and fold isolation**
- [ ] **Step 2: Run test to verify failure**
- [ ] **Step 3: Implement fuzzy matching in `src/task2/qa_memory.py`**
- [ ] **Step 4: Run test to verify pass**
- [ ] **Step 5: Commit**

---

### Task 4: Structured Article Stitching and Evidence Packing

**Files:**
- Modify: `src/task2/article_stitcher.py`
- Modify: `src/common/evidence.py`
- Test: `tests/test_task2_article_stitcher.py`
- Test: `tests/test_common_evidence.py`

**Interfaces:**
- `ArticleStitcher.pack_evidence(seed_chunks: List[Dict[str, Any]], max_chars: int = 3500) -> Dict[str, Any]`: Structured legal evidence packing preserving document title, article, clause, sibling clauses, and metadata.

- [ ] **Step 1: Write tests for multi-chunk structured evidence packing**
- [ ] **Step 2: Run test to verify failure**
- [ ] **Step 3: Implement structured packing in `src/task2/article_stitcher.py`**
- [ ] **Step 4: Run test to verify pass**
- [ ] **Step 5: Commit**

---

### Task 5: Generator Parity, Fallback Fix, and Dynamic Output Length

**Files:**
- Modify: `src/task2/generator.py`
- Modify: `src/task2/predict.py`
- Test: `tests/test_task2_generator.py`
- Test: `tests/test_task2_end_to_end.py`

**Interfaces:**
- `QwenGenerator.load(model_path: str, adapter_path: Optional[str] = None, device: Optional[str] = None, runtime: str = "auto") -> QwenGenerator`: Loads base model when adapter_path is None.
- `QwenGenerator.format_prompt(question: str, evidence: str) -> str`: Shared chat template.
- `LegalQAPipeline.load_pipeline(...) -> LegalQAPipeline`: Initializes full pipeline with proper model and device.

- [ ] **Step 1: Write test for generator base loading when adapter_path is None**
- [ ] **Step 2: Run test to verify failure**
- [ ] **Step 3: Fix `src/task2/generator.py` and `src/task2/predict.py`**
- [ ] **Step 4: Run test to verify pass**
- [ ] **Step 5: Commit**

---

### Task 6: Source Snapping Fixes, Candidate Generation, and Selector

**Files:**
- Modify: `src/task2/source_snap.py`
- Test: `tests/test_task2_source_snap.py`

**Interfaces:**
- `snap_facts_to_evidence(generated_text: str, evidence_text: str) -> str`: Multi-date, money, entity snapping.
- `generate_candidate_ensemble(gen_ans: str, evidence: str, exact_ans: str = "", fuzzy_ans: str = "", doc_name: str = "", art_num: str = "", clause_num: str = "") -> Dict[str, str]`: Generates all candidate variations (extracts, generated, snapped, Strategy F 300/600/1000/1500).
- `select_best_answer_candidate(candidates: Dict[str, str], features: Optional[Dict[str, Any]] = None, ...) -> str`: OOF-tuned candidate selector using test-time features.

- [ ] **Step 1: Write test for candidate generation, Strategy F variations, and selector rules**
- [ ] **Step 2: Run test to verify failure**
- [ ] **Step 3: Implement candidate ensemble and calibrated selector in `src/task2/source_snap.py`**
- [ ] **Step 4: Run test to verify pass**
- [ ] **Step 5: Commit**

---

### Task 7: Fix Dataset Packaging Script for Precomputed Indexes

**Files:**
- Modify: `scripts/package_kaggle_dataset.py`
- Test: `tests/test_data_pipeline_integration.py`

**Interfaces:**
- `package_kaggle_dataset(...)`: Staged copy of all required parquet files and optional BM25/DEk21 index folders.

- [ ] **Step 1: Write test for index packaging**
- [ ] **Step 2: Update `scripts/package_kaggle_dataset.py`**
- [ ] **Step 3: Run test to verify pass**
- [ ] **Step 4: Commit**

---

### Task 8: Comprehensive OOF Validation with Fast & Full Modes and Candidate Logging

**Files:**
- Modify: `scripts/run_oof_validation.py`
- Test: `tests/test_evaluation.py`
- Test: `tests/test_validation_and_audit.py`

**Interfaces:**
- `run_oof_validation(mode: str = "fast"|"full", ...) -> Dict[str, Any]`: Computes METEOR, ROUGE-L, candidate scores, and oracle upper bounds across 5 folds.

- [ ] **Step 1: Write test for official METEOR scorer parity and fast/full OOF modes**
- [ ] **Step 2: Update `scripts/run_oof_validation.py`**
- [ ] **Step 3: Run test to verify pass**
- [ ] **Step 4: Commit**

---

### Task 9: Rebuild Kaggle GPU Pipeline Notebook

**Files:**
- Modify: `kaggle_kernel/legalqa_gpu_pipeline.ipynb`
- Modify: `kaggle_kernel/kernel-metadata.json`
- Test: `tests/test_task2_end_to_end.py`

**Interfaces:**
- Clean, thin notebook using `src/` modules, dual-T4 placement, batched inference, deterministic model discovery, and submission zipping.

- [ ] **Step 1: Rebuild notebook with structured cells importing `src/`**
- [ ] **Step 2: Test end-to-end flow with mock and sample data**
- [ ] **Step 3: Verify submission schema check and zero-leakage constraints**
- [ ] **Step 4: Commit**

---

### Task 10: Full 15-Item Test Suite and Verification

**Files:**
- Update all test files in `tests/` to guarantee coverage of all 15 required verification items from `task.md`.

- [ ] **Step 1: Run complete test suite `.venv-ml/bin/pytest tests/ -v`**
- [ ] **Step 2: Ensure all tests pass with zero warnings/regressions**
- [ ] **Step 3: Commit**

---

### Task 11: Documentation and Final Deliverables

**Files:**
- Modify: `README.md`
- Documentation deliverables in final summary.

- [ ] **Step 1: Update README.md with dual-T4 architecture, Kaggle execution steps, and parameter budget**
- [ ] **Step 2: Commit**
