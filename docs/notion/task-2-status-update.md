# DSC 2026 Task 2 — LegalQA Pipeline Status Update (2026-08-26)

**Purpose:** Comprehensive, evidence-backed status report and paste-ready update for the Task 2 Notion wiki (`https://dangphuc.notion.site/task-2` / `https://app.notion.com/p/dangphuc/DSC-2026-Task-2-LegalQA-Maximum-Score-Pipeline-3c4519f69b98815d8569c8dba22fde81`).

---

## 1. Executive Summary & Component Status

Every component in the repository is categorized using the strict 5-tier status taxonomy:

| Component | Status | Verified Evidence & Details |
|---|---|---|
| **Exact QA Memory** | **Implemented** | `ExactMemory` in `src/memory/exact_memory.py`. Exact ID + normalized question lookup across 7,113 canonical QA pairs ($100\%$ precision). |
| **BM25 Lexical Retriever** | **Implemented** | `SimpleBM25` in `src/retrieval/bm25_retriever.py` indexing all 365,046 canonical chunks (425.5 MB, 18 columns). |
| **Lexical Overlap Reranker** | **Implemented** | `SimpleLexicalReranker` in `src/reranking/cross_encoder.py` using token overlap + Jaccard similarity. |
| **Article Stitcher** | **Implemented** | `ArticleStitcher` in `src/postprocess/article_stitcher.py` assembling multi-part sibling clauses into complete legal articles. |
| **Extractive Answer Generator** | **Implemented** | `generate_extractive_answer` in `src/postprocess/extractive.py` extracting grounded text from top provisions. |
| **Source Snapper** | **Implemented** | `source_snap_answer` in `src/postprocess/source_snap.py` aligning numbers, penalties, dates, and citations to source text. |
| **Rule-Based Selector** | **Implemented** | `CandidateSelector` in `src/selector/candidate_selector.py` picking candidates via penalty keywords and confidence. |
| **CodaBench Evaluation Scorer** | **Implemented** | `evaluate_predictions` in `src/evaluation/codabench_eval.py` mirroring exact whitespace-tokenized NLTK METEOR and unstemmed ROUGE-L. |
| **Artifact Manifests & Audit** | **Implemented** | `scripts/audit_artifacts.py` auditing 8,544 canonical artifacts; SHA-256 integrity in `artifacts/manifests/artifacts.json`. |
| **Model Manifest & Downloader** | **Implemented** | `load_model_manifest` in `src/utils/manifest.py` (< 4B check) and atomic downloader `scripts/verify_download.py`. |
| **Submission v1 (Baseline)** | **Measured** | Official CodaBench score: `METEOR = 0.2952`, `ROUGE-L = 0.3957` (stored in `logs/submission_v1_bm25/`). |
| **Submission v2 (Article Stitcher)** | **Measured** | Official CodaBench score: `METEOR = 0.3781`, `ROUGE-L = 0.3718` (stored in `logs/submission_v2_stitcher/`, **Top 26 Leaderboard**). |
| **Sampled OOF Validation (Legacy)** | **Measured (Legacy Signal)** | `0.3542 ± 0.0223` on 100-sample random split; legacy diagnostic signal only, not a promotion baseline. |
| **Hybrid RRF Fusion** | **Configured but unused** | `src/retrieval/hybrid_fusion.py` implemented; inactive in current default prediction pipeline. |
| **Neural Configurations** | **Configured but unused** | YAML configs in `configs/` (`models.yaml`, `retrieval.yaml`, `reranking.yaml`, `generation.yaml`). |
| **Dense Retriever (`Qwen3-Embedding-0.6B`)** | **Planned** | 595.8M parameters; planned for post-validation neural bake-off. |
| **Neural Cross-Encoder (`Qwen3-Reranker-0.6B`)** | **Planned** | 595.8M parameters; planned for post-validation neural bake-off. |
| **Grounded Generator (`Qwen3-1.7B`)** | **Planned** | 2,031.7M parameters; planned for post-validation neural bake-off. |
| **Task 2-Only LoRA Fine-Tuning** | **Planned** | Fold-isolated PEFT/LoRA using only Task 2 training provisions. |
| **Provision Demonstration Memory** | **Planned** | In-context demonstration lookup from grounded training provisions (`src/memory/provision_memory.py`). |
| **Learned Candidate Selector** | **Planned** | ML-based candidate selection replacing heuristic rules. |
| **Benchmark-Equivalent Full OOF Validation** | **Planned** | Full 7,113-row question-blocked and document-held-out 5-fold cross-validation suite. |
| **Pretrained Checkpoint Rule Legality** | **Unverified** | Requires official confirmation from organizers regarding external pretrained weights. |

---

## 2. Active Production Pipeline

The active offline prediction pipeline is 100% deterministic and runs without neural dependencies:

$$\text{Test Query} \rightarrow \text{Exact Memory} \rightarrow \text{BM25 (Top 25)} \rightarrow \text{Lexical Rerank (Top 8)} \rightarrow \text{Article Stitcher} \rightarrow \text{Extractive + Snap} \rightarrow \text{Rule Selector} \rightarrow \text{Answer}$$

- **Execution time:** $\sim 0.5$ seconds per query on CPU.
- **Reproducibility:** 100 unit/integration tests passing (`pytest tests/ -v`).

---

## 3. Artifact Governance & Cleanup Summary

1. **Authoritative Hierarchy Established**:
   - `artifacts/` is the sole source of truth.
   - Raw data: 7,000 train, 500 warmup, 1,000 public test questions, and complete `selected-contexts/`.
   - Canonical chunks: `artifacts/chunks/legal_chunks.parquet` (365,046 rows, 18 columns, 425.5 MB).
   - Canonical QA: `artifacts/data/qa_unique.parquet` (7,113 rows).
   - Manifests: `artifacts/manifests/artifacts.json` (8,544 audited files).

2. **Legacy Cleanup & Quarantine Completed**:
   - Incompatible legacy parquet (344,301 rows, 14 columns) quarantined with provenance to `artifacts/archive/trung-legacy/`.
   - Differing legacy memory file quarantined to `artifacts/archive/trung-legacy/`.
   - Byte-identical duplicates (`qa_unique.parquet`, `retrieval_labels.parquet`) verified by SHA-256 and deleted.
   - Directory `trung_artifacts/` removed cleanly.
   - `artifacts/archive/` added to `.gitignore` and excluded from canonical manifests.

3. **Fallback Retirement**:
   - All silent fallbacks to `data/raw/*`, `data/intermediate/*`, or root files removed from `validation.py`, `scripts/predict.py`, `scripts/prepare_artifacts.py`, `scripts/run_oof_validation.py`, and `scripts/test_codabench_submission.py`.
   - Missing paths raise clear, actionable `FileNotFoundError`.

4. **Model Budget & Download Security**:
   - Shipped candidate stack: `Qwen/Qwen3-Embedding-0.6B` ($595.8\text{M}$) + `Qwen/Qwen3-Reranker-0.6B` ($595.8\text{M}$) + `Qwen/Qwen3-1.7B` ($2{,}031.7\text{M}$) = $3{,}223{,}292{,}928$ total ($< 4\text{B}$, fully compliant).
   - Checksum-verified atomic downloader (`scripts/verify_download.py`) with redirect-chain validation, temporary directory safety, and credentials filtering.

---

## 4. Next Phase: Benchmark-Equivalent Validation & Neural Bake-Off

Before training or submitting neural models, the following execution order is established:

1. **Phase 1: Full Leakage-Resistant Validation Suite**:
   - Implement exact key equality check in evaluation.
   - Build source-aware QA audit and multiple citation resolver.
   - Construct 5-fold question-blocked and document-held-out splits over all 7,113 canonical QA rows.
   - Add structured prediction traces and retrieval/fidelity metrics.
   - Measure true full-data baseline performance.

2. **Phase 2: Neural Retriever & Reranker Bake-Off**:
   - Build offline dense index using `Qwen/Qwen3-Embedding-0.6B` and controls (`BAAI/bge-m3`).
   - Compare BM25, Dense, and Hybrid RRF on identical candidate pools.
   - Benchmark neural cross-encoder (`Qwen/Qwen3-Reranker-0.6B`) against lexical reranker.

3. **Phase 3: Generator & Provision Memory**:
   - Benchmark grounded generation with `Qwen/Qwen3-1.7B`.
   - Fine-tune Task 2-only LoRA adapters with fold isolation.
   - Enable provision demonstration memory and learned candidate selection.
   - Promote neural submissions only if passing the $+0.005$ METEOR promotion gate with 95% bootstrap confidence.
