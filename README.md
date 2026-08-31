# DSC 2026 Task 2 — LegalQA System & Kaggle Pipeline

This repository implements the end-to-end Legal Question Answering (LegalQA) system for DSC 2026 Task 2, engineered to maximize the official **whitespace-tokenized METEOR** score under strict competition constraints:
- **Parameter Budget**: Total learned parameters strictly $< 4.0\text{B}$ (Audited: **3.758B** base, leaving 242M margin for LoRA/adapters).
- **Data Constraint**: Task 2 organizer-provided legal contexts and QA records only (zero external legal corpora or external answer APIs).
- **Target Runtime**: Kaggle dual NVIDIA T4 GPU environment (`GPU T4 x 2`, 16GB VRAM each).

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
   - **Model**: `qwen-lm/qwen2.5/transformers/3b-instruct/1` (or loads `Qwen/Qwen2.5-3B-Instruct` via HF)
4. **Secrets Setup** (Optional for public models, recommended for private access):
   - Go to **Add-ons → Secrets**.
   - Add secret with label `HF_TOKEN` and your Hugging Face access token.

---

## 2. End-to-End Architecture

```text
Official Task 2 Data
  ├── Raw Contexts (8,532 JSONs) -> Hierarchical Parser -> legal_chunks.parquet (801,863 chunks)
  └── QA Pairs (7,500 records)   -> Citation Resolution  -> retrieval_labels.parquet (6,399 labels)
                                                        -> fold_assignments.parquet (5-fold zero-leakage)

Inference Flow:
  Query
   ├── 1. Exact QA Memory Lookup (41 Public Test Hits)
   ├── 2. Similar QA Memory Lookup (Fuzzy near-duplicate retrieval with legal entity matching)
   └── 3. Hybrid Retrieval:
            ├── BM25 Sparse Search (Term saturation, length normalization, legal signal booster)
            └── DEk21 v2 Dense Search (huydang-dek21-embedding-v2, 768-dim normalized, FP16 on GPU 1)
                 └── RRF Fusion (k=60, equal weights)
                      └── BGE-Reranker-v2-m3 Cross-Encoder (Top-8 seeds on GPU 1)
                           └── Structured Article Stitcher (Sibling clause packing, 3500 char budget)
                                └── Qwen2.5-3B-Instruct (FP16 Batched Generation on GPU 0, max_new_tokens=384)
                                     └── Multi-Candidate Snapping & Strategy F Selection
                                          └── submission.json.zip (1,000 items)
```

---

## 3. Parameter Budget Compliance

Audited via `scripts/audit_parameters.py` against `configs/models.yaml` and `configs/pipeline.yaml`:

| Component | Model Identifier | Parameters | Device | Role |
| :--- | :--- | :---: | :---: | :--- |
| **Dense Retriever** | `CODE4LIFEOFFICIAL/huydang-dek21-embedding-v2` | `100,000,000` (100M) | `cuda:1` | Dense query/chunk embeddings |
| **Reranker** | `BAAI/bge-reranker-v2-m3` | `568,000,000` (568M) | `cuda:1` | Cross-encoder pair scoring |
| **Generator** | `Qwen/Qwen2.5-3B-Instruct` | `3,090,000,000` (3.09B) | `cuda:0` | Conditioned statutory generation |
| **Total Learned Parameters** | | **`3,758,000,000`** (~3.76B) | | |
| **Budget Limit** | | **`< 4,000,000,000`** (4.0B) | | |
| **Safe Margin** | | **`+242,000,000`** (242M) | | Leaves margin for LoRA/adapter |
| **Compliance Status** | | **COMPLIANT** | | |

---

## 4. 5-Fold OOF Validation & Candidate Ablation Results

Evaluated on 5-Fold Cross-Validation using exact official whitespace METEOR:

| Candidate Family | Mean METEOR | Mean ROUGE-L | Strategy / Description |
| :--- | :---: | :---: | :--- |
| **generated** | 0.0749 | 0.1820 | Vanilla short generation |
| **snapped** | 0.0749 | 0.1820 | Fact-snapped generation |
| **strategy_f_300** | 0.1172 | 0.2291 | Generated + 300 chars evidence |
| **strategy_f_600** | 0.1606 | 0.2774 | Generated + 600 chars evidence |
| **strategy_f_1000** | 0.2009 | 0.3216 | Generated + 1000 chars evidence |
| **strategy_f_1500** | 0.2339 | 0.3582 | Generated + 1500 chars evidence |
| **focused_extract** | 0.2026 | 0.3150 | Top-1 chunk statutory extract |
| **stitched_extract** | 0.3051 | 0.4012 | Multi-clause stitched statutory extract |
| **Selected System** | **0.2009** | **0.3216** | Calibrated OOF selector |
| **Oracle Best Upper Bound** | **0.3112** | **0.4105** | Theoretical best per-query candidate |

---

## 5. Local Scripts & CLI Reference

All Python scripts are located in `scripts/`:

```bash
# 1. Prepare canonical chunks, citations, retrieval labels, and 5-fold splits
.venv-ml/bin/python scripts/prepare_data.py

# 2. Build BM25 index and DEk21 dense embeddings
.venv-ml/bin/python scripts/build_indexes.py

# 3. Stage clean dataset artifacts (data + indexes) for Kaggle upload
.venv-ml/bin/python scripts/package_kaggle_dataset.py

# 4. Run 5-fold OOF cross-validation with exact official METEOR score
.venv-ml/bin/python scripts/run_oof_validation.py --mode fast --samples 100 --folds 5

# 5. Full OOF validation on GPU with Qwen2.5-3B
.venv-ml/bin/python scripts/run_oof_validation.py --mode full --samples 100 --device cuda

# 6. Fine-tune Qwen2.5-3B with QLoRA on dual T4 GPUs
.venv-ml/bin/python scripts/train_generator_qlora.py --epochs 1 --batch_size 1 --grad_accum 8

# 7. Verify parameter budget compliance (<4.0B)
.venv-ml/bin/python scripts/audit_parameters.py

# 8. Run unit & integration test suite (48 tests)
.venv-ml/bin/pytest tests/ -v
```

---

## 6. Repository Layout

```text
LegalQA/
├── kaggle_kernel/
│   ├── legalqa_gpu_pipeline.ipynb   # Canonical Kaggle dual-T4 pipeline notebook
│   └── kernel-metadata.json         # Kaggle kernel configuration
├── kaggle_dataset/
│   ├── dataset-metadata.json        # Kaggle dataset metadata (title: "LegalQA")
│   └── staged/                      # Staged parquet files and precomputed indexes
├── configs/
│   ├── models.yaml                  # Model parameter registry & compliance manifest
│   └── pipeline.yaml                # Unified pipeline configurations
├── src/
│   ├── common/                      # Reusable core RAG modules
│   │   ├── normalize.py             # Legal text cleaner & canonical identifier parser
│   │   ├── legal_parser.py          # Hierarchy parser with exact offset spans
│   │   ├── bm25.py                  # Full-corpus BM25 retriever with statutory boosts
│   │   ├── dense_dek21.py           # Persistent DEk21 v2 dense retriever (FP16/batched)
│   │   ├── rrf.py                   # Reciprocal Rank Fusion
│   │   ├── reranker.py              # BGE-reranker-v2-m3 cross-encoder wrapper
│   │   ├── evidence.py              # Citation resolution & negative label miner
│   │   └── security.py              # Secret scanner & preflight validator
│   └── task2/                       # Task 2 specialized components
│       ├── qa_memory.py             # Exact and Similar-QA Memory with fold isolation
│       ├── article_stitcher.py      # Structured statutory article & clause stitcher
│       ├── generator.py             # Qwen2.5-3B-Instruct generator wrapper
│       ├── source_snap.py           # Entity snapping, Strategy F, candidate ensemble
│       └── predict.py               # End-to-end pipeline orchestrator
├── scripts/                         # Command-line tools and pipelines
└── tests/                           # 48 comprehensive unit and integration tests
```
