# DSC 2026 Task 2 — LegalQA System & Kaggle Pipeline (V3 Production)

This repository implements the end-to-end Legal Question Answering (LegalQA) system for DSC 2026 Task 2, engineered to maximize the official **whitespace-tokenized METEOR** score under strict competition constraints:
- **Parameter Budget**: Total learned parameters strictly $< 4.0\text{B}$ (Audited: **3.758B** base, leaving 242M safe margin for adapters).
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
   - **Dataset**: `LegalQA` (`phucdangg/legalqa-task2-clean-data`) packaged via `scripts/package_kaggle_dataset.py`
   - **Model**: `qwen-lm/qwen2.5/transformers/3b-instruct/1` (or loads `Qwen/Qwen2.5-3B-Instruct` via HF)
4. **Secrets Setup**:
   - Go to **Add-ons → Secrets**.
   - Add secret with label `HF_TOKEN` and your Hugging Face access token.

### Execution Profiles (Cell 1)
- `EXECUTION_PROFILE = "train_and_submit"` *(Default)*: Preflight $\to$ Task-Tuned Reranker Fine-Tuning $\to$ Qwen2.5-3B QLoRA SFT $\to$ Reload Smoke Verification $\to$ Held-Out Sanity Evaluation $\to$ Dual-T4 Batched Inference $\to$ Strict Submission Validation.
- `EXECUTION_PROFILE = "reuse_checkpoints_and_submit"`: Reuses previously trained checkpoints and runs full Dual-T4 inference.
- `EXECUTION_PROFILE = "smoke_only"`: Quick hardware & checkpoint training smoke verification.

---

## 2. End-to-End Architecture (Stack A)

```text
Official Task 2 Data
  ├── Raw Contexts (8,532 JSONs) -> Hierarchical Parser -> legal_chunks.parquet (801,863 chunks)
  └── QA Pairs (7,500 records)   -> Citation Resolution  -> retrieval_labels.parquet (6,399 labels)
                                                        -> reranker_training_pairs.parquet
                                                        -> fold_assignments.parquet (5-fold near-duplicate grouping)

Inference Flow:
  Query
   ├── 1. Exact QA Memory Lookup (Verified Question & ID Consistency)
   ├── 2. Similar QA Memory Lookup (Near-duplicate retrieval with legal entity matching)
   └── 3. Hybrid Retrieval:
            ├── BM25S Sparse Search (mmap, zero truncation, legal signal booster on CPU)
            └── DEk21 v2 Dense Search (huydang-dek21-embedding-v2, 768-dim FP16 exact top-K on GPU 1)
                 └── RRF Fusion (k=60, equal weights)
                      └── Task-Tuned BGE-Reranker-v2-m3 Cross-Encoder (Top-8 seeds on GPU 1)
                           └── Structured Evidence Packer (Multi-granularity statutory packing)
                                └── Qwen2.5-3B-Instruct (4-bit NF4 QLoRA Batched Generation on GPU 0)
                                     └── Candidate Ensemble & Selector Guardrail
                                          └── submission.json.zip (1,000 public IDs)
```

---

## 3. Parameter Budget Compliance

Audited via `scripts/audit_parameters.py` against `configs/models.yaml`:

| Component | Model Identifier | Parameters | Device | Role |
| :--- | :--- | :---: | :---: | :--- |
| **Dense Retriever** | `CODE4LIFEOFFICIAL/huydang-dek21-embedding-v2` | `100,000,000` (100M) | `cuda:1` | Dense query/chunk embeddings |
| **Cross-Encoder Reranker** | `BAAI/bge-reranker-v2-m3` | `568,000,000` (568M) | `cuda:1` | Fine-tuned pair reranker |
| **Base Generator** | `Qwen/Qwen2.5-3B-Instruct` | `3,090,000,000` (3.09B) | `cuda:0` | 4-bit quantized base model |
| **QLoRA Adapter** | Task-2 LoRA rank 16 adapter | `~20,000,000` (20M) | `cuda:0` | Trainable LoRA adapter |
| **Total Learned Parameters** | — | **~3,778,000,000** | — | **COMPLIANT ($< 4.0\text{B}$)** |
| **Remaining Safe Margin** | — | **~222,000,000** | — | Parameters below 4.0B cap |

---

## 4. Local Testing & Verification

Run the comprehensive test suite across all subsystems:

```bash
# Run full pytest suite
.venv-ml/bin/pytest tests/ -v

# Run preflight diagnostics
.venv-ml/bin/python scripts/preflight_kaggle.py --pipeline_config configs/pipeline.yaml --models_config configs/models.yaml

# Run parameter audit
.venv-ml/bin/python scripts/audit_parameters.py
```
