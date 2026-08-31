# DSC 2026 Task 2 — LegalQA System & Kaggle Training Pipeline

This repository implements the end-to-end Legal Question Answering (LegalQA) system for DSC 2026 Task 2, designed to maximize the official **whitespace-tokenized METEOR** score under strict competition constraints:
- **Parameter budget**: Total learned parameters strictly $< 4.0\text{B}$ (Audited: **3.758B**).
- **Data constraint**: Task 2/BTC legal contexts and QA records only (zero external data or external APIs).
- **Target runtime**: Kaggle dual NVIDIA T4 GPU environment (`GPU T4 x 2`).

---

## 1. Quickstart — Kaggle Execution

The single authoritative file for Task 2 on Kaggle is:

👉 **`kaggle_kernel/legalqa_gpu_pipeline.ipynb`**  
*(Available on Kaggle at [**`kaggle.com/code/phucdangg/legalqa-training`**](https://www.kaggle.com/code/phucdangg/legalqa-training))*

### Kaggle Environment Configuration
1. **Accelerator**: Select **`GPU T4 x 2`** in the right-hand sidebar.
2. **Internet**: Toggle **`Internet on`** in the session options.
3. **Mounted Inputs**:
   - **Dataset**: `LegalQA` (`phucdangg/legalqa-task2-clean-data`)
   - **Model**: `Qwen2.5 · qwen2.5 · V1` (or dynamically loads `Qwen/Qwen2.5-3B-Instruct`)
4. **Secrets Setup** (Optional for public models, recommended for private access):
   - Go to **Add-ons → Secrets**.
   - Add secret with label `HF_TOKEN` and your Hugging Face access token.

---

## 2. How to Download the Submission Artifacts

When the Kaggle notebook finishes running, it produces:
- `/kaggle/working/submission.json` (1,000 predicted QA answers)
- `/kaggle/working/submission.json.zip` (CodaBench-ready submission archive)

### Method A: Download via Kaggle Web UI
1. In the Kaggle notebook editor, expand the **Output** section in the right sidebar (`/kaggle/working`).
2. Click the refresh icon on `/kaggle/working`.
3. Click the three dots `⋮` next to `submission.json.zip` (or `submission.json`) and select **Download**.
4. Upload `submission.json.zip` directly to the **CodaBench competition portal**.

### Method B: Download and Validate via CLI (Automated)
Run the automated downloader from your local terminal:
```bash
.venv-ml/bin/python scripts/fetch_kaggle_submission.py --slug phucdangg/legalqa-training
```
This tool:
1. Checks that the remote kernel status is `COMPLETE`.
2. Downloads the output to a safe temporary directory.
3. Strictly validates all 1,000 query IDs and ensures zero empty answers.
4. Atomically places the verified files into `artifacts/task2/submissions/submission.json.zip`.

---

## 3. End-to-End Architecture

```text
Official Task 2 Data
  ├── Raw Contexts (8,532 JSONs) -> Hierarchical Parser -> legal_chunks.parquet (801,863 chunks)
  └── QA Pairs (7,500 records)   -> Citation Resolution  -> retrieval_labels.parquet (6,399 labels)
                                                        -> fold_assignments.parquet (5-fold zero-leakage)

Inference Flow:
  Query
   ├── 1. Exact QA Memory Lookup (Conflict-Safe, 41 Public Test Hits)
   └── 2. Hybrid Retrieval:
            ├── BM25 Sparse Search (with statutory legal entity boosts)
            └── DEk21 v2 Dense Search (huydang-dek21-embedding-v2)
                 └── RRF Fusion (k=60)
                      └── BGE-Reranker v2 M3 Cross-Encoder (Top-8 Evidence)
                           └── Article Stitcher (Context Expansion)
                                └── Qwen2.5-3B-Instruct (FP16 Batched Generation)
                                     └── Source Snapping (Date & Statutory Entity Preservation)
                                          └── submission.json.zip (1,000 items)
```

---

## 4. Parameter Budget Compliance

Checked via `scripts/audit_parameters.py` against `configs/models.yaml`:

| Component | Model Identifier | Parameter Count |
| :--- | :--- | :--- |
| **Dense Retriever** | `CODE4LIFEOFFICIAL/huydang-dek21-embedding-v2` | `100,000,000` (100M) |
| **Reranker** | `BAAI/bge-reranker-v2-m3` | `568,000,000` (568M) |
| **Generator** | `Qwen/Qwen2.5-3B-Instruct` | `3,090,000,000` (3.09B) |
| **Total System Parameters** | | **`3,758,000,000`** (~3.76B) |
| **Competition Parameter Limit** | | **`4,000,000,000`** (4.0B) |
| **Compliance Status** | | **COMPLIANT** |

---

## 5. Local Scripts & CLI Reference

All Python scripts are located in `scripts/`:

```bash
# 1. Prepare canonical chunks, citations, retrieval labels, and 5-fold splits
.venv-ml/bin/python scripts/prepare_data.py

# 2. Build BM25 index and DEk21 dense embeddings
.venv-ml/bin/python scripts/build_indexes.py

# 3. Stage clean dataset artifacts for Kaggle upload
.venv-ml/bin/python scripts/package_kaggle_dataset.py

# 4. Run 5-fold OOF cross-validation with exact official METEOR score
.venv-ml/bin/python scripts/run_oof_validation.py --samples 100 --folds 5

# 5. Fine-tune Qwen2.5-3B with QLoRA on dual T4 GPUs
.venv-ml/bin/python scripts/train_generator_qlora.py --epochs 1 --batch_size 1 --grad_accum 8

# 6. Verify parameter budget compliance (<4.0B)
.venv-ml/bin/python scripts/audit_parameters.py

# 7. Monitor remote Kaggle kernel execution
.venv-ml/bin/python scripts/monitor_kaggle.py --slug phucdangg/legalqa-training

# 8. Run unit & integration test suite (30 tests)
.venv-ml/bin/pytest -v
```

---

## 6. Repository Layout

```text
LegalQA/
├── kaggle_kernel/
│   ├── legalqa_gpu_pipeline.ipynb   # Canonical Kaggle training & inference notebook
│   └── kernel-metadata.json         # Kaggle kernel configuration
├── kaggle_dataset/
│   └── dataset-metadata.json        # Kaggle dataset metadata (title: "LegalQA")
├── configs/
│   ├── models.yaml                  # Model parameter registry & compliance manifest
│   └── task2.yaml                   # Task 2 hyperparameters and path configurations
├── src/
│   ├── common/                      # Reusable core RAG modules
│   │   ├── normalize.py             # Legal text cleaner & canonical identifier parser
│   │   ├── legal_parser.py          # Hierarchy parser with exact offset spans
│   │   ├── bm25.py                  # BM25 retriever with statutory boosts
│   │   ├── dense_dek21.py           # Persistent DEk21 dense retriever
│   │   ├── rrf.py                   # Reciprocal Rank Fusion
│   │   ├── reranker.py              # BGE v2 M3 cross-encoder wrapper
│   │   ├── evidence.py              # Citation resolution & negative label miner
│   │   └── security.py              # Secret scanner & preflight validator
│   └── task2/                       # Task 2 specialized components
│       ├── qa_memory.py             # Conflict-safe exact QA memory
│       ├── article_stitcher.py      # Sibling clause stitcher
│       ├── generator.py             # Qwen2.5-3B-Instruct generator wrapper
│       ├── source_snap.py           # Date, fine, and statutory entity snapper
│       └── predict.py               # End-to-end inference orchestrator
├── scripts/                         # Operational CLI scripts
└── tests/                           # Unit and integration test suite
```
