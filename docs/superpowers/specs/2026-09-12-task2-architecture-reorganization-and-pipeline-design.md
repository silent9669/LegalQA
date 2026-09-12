# Task 2 Architecture Reorganization, Clean Pipeline & Dual-Target Execution Design

## 1. Executive Summary & Authoritative Context

This document establishes the authoritative system architecture for **LegalQA (Task 2)** in compliance with the **DSC 2026 Reproducible Training Workflow** specification (`https://dangphuc.notion.site/dscc`).

### Core Problem & Motivation
The previous repository iteration mixed code and data by bundling the Python codebase directly inside the Kaggle dataset package (`code/LegalQA`), tangled 15 versions of ad-hoc scratchpad docs and 40+ flat test files, and executed both initial screening and heavy final training on Kaggle GPUs. 

The new workflow separates roles and environments:
1. **Data Owner Release**: Releases canonical clean dataset artifacts directly to Kaggle (`phucdangg/legalqa-task2-clean-data`) with complete schema validation, referential integrity, and cryptographic manifest verification. **No application code is bundled inside the dataset.**
2. **Training Owner (Kaggle Dual-T4 Smoke Gate)**: Tests the exact Git commit using Kaggle Dual NVIDIA T4 GPUs for rapid, cheap validation of real CUDA execution, VRAM safety, worst-case sequence lengths, endurance stability, and mini-evaluation. Exports `kaggle_smoke_report.json`.
3. **Training Owner (Google Colab A100 Production Training)**: Once the `(dataset_version, git_sha, config, base_model, smoke_report)` tuple is frozen and approved, full expensive training runs on Google Colab with NVIDIA A100 GPU (leveraging BF16, larger batch size, gradient accumulation, and full epochs). Exports the complete Run Bundle to Hugging Face.

---

## 2. System Architecture & Lifecycle

```
+---------------------------+
| Official BTC Task 2 Data  |
+---------------------------+
              |
              v
+---------------------------+
| Data Owner: Clean/Build   |
+---------------------------+
              |
              v
+---------------------------+     FAIL
| Dataset Validation Script | ------------+
+---------------------------+             |
              | PASS                      |
              v                           v
+---------------------------+     +---------------+
| Publish Kaggle Dataset vN |     | Fix Data Bug  |
| (Pure Data + Manifest)    |     +---------------+
+---------------------------+
              |
              v
+---------------------------+
| GitHub: Notebooks + Code  |
+---------------------------+
              |
              v
+---------------------------+     FAIL
| Pre-Push Gate / Local CI  | ------------+
+---------------------------+             |
              | PASS                      |
              v                           v
+---------------------------+     +---------------+
| Kaggle Dual-T4 CUDA Smoke |     | Fix Code /    |
| (Worst-case + Endurance)  |     | Memory Bug    |
+---------------------------+     +---------------+
              | PASS
              v
+---------------------------+
| Freeze Tuple & Checksums  |
+---------------------------+
              |
              v
+---------------------------+
| Google Colab A100 Train   |
| (BF16, Full Training)     |
+---------------------------+
              |
              v
+---------------------------+
| Metric Validation         |
| (Official METEOR / ROUGE) |
+---------------------------+
              |
              v
+---------------------------+
| Hugging Face Model + Logs |
+---------------------------+
```

---

## 3. Radical Workspace Cleanup & Reorganization

All historical scratchpads, legacy versioned tests (`test_v6` to `test_v15`), obsolete configs, and dataset code-packaging bloat are deleted to produce a compact, production-grade workspace:

### 3.1 Files & Directories to Delete
1. **Historical Documentation**:
   - `docs/archive/` (all 25 legacy markdown files).
   - `docs/active/v16/` (all 21 old notes).
   - `docs/LEGALQA_V16_ARCHITECTURE_REVIEW_AND_FIX_INSTRUCTIONS.md`.
   - `docs/LEGALQA_V16_FINAL_PREFLIGHT_HARDENING.md`.
   - `docs/OPERATING_RULES.md`.
2. **Redundant Configurations**:
   - `configs/experiments.yaml`
   - `configs/models.yaml`
   - `configs/pipeline.yaml`
   - `configs/production_selection.yaml`
   - `configs/runtime_api.yaml`
   - `configs/task2.yaml`
3. **Dataset Code Bundles**:
   - `kaggle_dataset/staged/code/`
   - `kaggle_dataset/staged/code_manifest.json`
4. **Obsolete Scripts**:
   - `scripts/run_colab_smoke.py`
   - `scripts/fetch_kaggle_submission.py`
   - `scripts/monitor_kaggle.py`
   - `scripts/verify_inherited_dataset.py`
   - `scripts/train_retriever_mnrl.py`
   - `scripts/preflight_kaggle.py`
   - `scripts/build_indexes.py`
   - `scripts/mine_retrieval_negatives.py`
   - `scripts/evaluate_checkpoint.py`
   - `scripts/promote_production_selection.py`
   - `scripts/run_oof_validation.py`
5. **Flat Test Clutter** (Delete all 40+ flat root tests, keeping only new structured test modules):
   - `tests/test_v*`
   - `tests/test_common_*`
   - `tests/test_task2_*`
   - `tests/test_colab_smoke_contract.py`
   - `tests/test_data_pipeline_integration.py`
   - `tests/test_evaluation.py`
   - `tests/test_generator_training_data.py`
   - `tests/test_kaggle_packaging.py`
   - `tests/test_oof_isolation.py`
   - `tests/test_path_resolver_kaggle_nested.py`
   - `tests/test_production_config_and_manifest.py`
   - `tests/test_real_kaggle_screen_gate.py`
   - `tests/test_reranker_training_data.py`
   - `tests/test_security_preflight.py`
   - `tests/test_validation_and_audit.py`

### 3.2 Canonical Target Structure
```text
LegalQA/
├── configs/
│   ├── kaggle_smoke_t4.yaml         # Dual-T4 smoke test profile
│   ├── colab_train_a100.yaml        # A100 production training profile
│   └── dataset_schema.yaml          # Canonical dataset schema invariants
│
├── src/
│   └── task2/
│       ├── common/                  # BM25S, normalization, legal parsing, RRF
│       ├── dataset/                 # Dataset loader, validation, manifest generator
│       ├── retrieval/               # Hybrid BM25S + DEk21 dense retrieval
│       ├── reranker/                # BGE-reranker-v2-m3 model & inference
│       ├── generation/              # Qwen2.5-3B-Instruct, QLoRA, Liger-kernel backend
│       ├── pipeline/                # Unified runner & profile executor
│       └── provenance/              # Freeze-tuple hashing, run manifest, integrity
│
├── notebooks/
│   ├── kaggle_smoke_test.ipynb      # Kaggle 2xT4 smoke launcher (synced with GitHub)
│   └── colab_train_a100.ipynb       # Colab A100 full training launcher
│
├── kaggle_dataset/
│   ├── dataset-metadata.json        # Slug: phucdangg/legalqa-task2-clean-data
│   └── staged/                      # Pure data artifacts (Parquet, JSON, indexes, manifest)
│
├── tests/
│   ├── dataset/
│   │   └── test_dataset_integrity.py
│   ├── notebooks/
│   │   └── test_notebook_contracts.py
│   ├── unit/
│   │   ├── test_liger_backend.py
│   │   ├── test_evidence_packer.py
│   │   ├── test_retrieval_and_bm25.py
│   │   └── test_checkpoint_resolver.py
│   ├── contracts/
│   │   ├── test_runtime_contracts.py
│   │   └── test_profile_resolution.py
│   └── gpu/
│       └── test_liger_qlora_gpu.py
│
├── scripts/
│   ├── pre_push_check.py            # Local developer pre-push gate
│   ├── validate_dataset.py          # Standalone dataset validation tool
│   ├── package_kaggle_dataset.py    # Clean dataset packaging & Kaggle metadata builder
│   └── run_pipeline.py              # CLI entry point for local & Colab execution
│
├── docs/
│   ├── ARCHITECTURE.md              # System architecture & team roles
│   ├── KAGGLE_SMOKE_GATE.md         # Smoke test instructions & criteria
│   ├── COLAB_A100_TRAINING.md       # Production Colab A100 training guide
│   └── DATASET_SPEC.md              # Dataset schema & manifest contract
│
├── .github/workflows/
│   └── tests.yml                    # Clean CI running tests & pre-push checks
│
├── requirements.txt                 # Unified project dependencies
├── requirements-kaggle.txt          # Minimal Kaggle environment overrides
├── requirements-colab.txt           # Minimal Colab A100 environment overrides
├── pytest.ini                       # Test runner configuration
└── README.md                        # High-level repository orientation
```

---

## 4. Configuration Contracts

### 4.1 `configs/kaggle_smoke_t4.yaml` (Dual-T4 CUDA Smoke Gate)
```yaml
profile_name: kaggle_smoke_t4
target_hardware: "NVIDIA T4 x2"
seed: 42

devices:
  generator: "cuda:0"
  retrieval: "cuda:1"

data:
  dataset_slug: "phucdangg/legalqa-task2-clean-data"
  runtime_root: "/kaggle/input/legalqa-task2-clean-data"
  max_seq_len: 2048

generator:
  model_id: "Qwen/Qwen2.5-3B-Instruct"
  load_in_4bit: true
  use_liger_kernel: true
  lora_r: 16
  lora_alpha: 32
  lora_dropout: 0.05
  per_device_train_batch_size: 1
  gradient_accumulation_steps: 4

probes:
  run_worstcase_probe: true
  worstcase_steps: 3
  run_endurance_probe: true
  endurance_steps: 30
  run_mini_eval: true
  mini_eval_samples: 10

outputs:
  report_path: "/kaggle/working/kaggle_smoke_report.json"
  telemetry_path: "/kaggle/working/gpu_telemetry.json"
```

### 4.2 `configs/colab_train_a100.yaml` (Google Colab A100 Production Training)
```yaml
profile_name: colab_train_a100
target_hardware: "NVIDIA A100"
seed: 42

devices:
  generator: "cuda:0"
  retrieval: "cuda:0"

data:
  dataset_slug: "phucdangg/legalqa-task2-clean-data"
  runtime_root: "/content/data/legalqa-task2-clean-data"
  max_seq_len: 2048

generator:
  model_id: "Qwen/Qwen2.5-3B-Instruct"
  load_in_4bit: false                # Full precision / BF16 QLoRA or native adapter
  torch_dtype: "bfloat16"
  use_liger_kernel: true
  lora_r: 32
  lora_alpha: 64
  lora_dropout: 0.05
  learning_rate: 2.0e-4
  num_train_epochs: 3
  per_device_train_batch_size: 4     # Leverages A100 40GB/80GB VRAM
  gradient_accumulation_steps: 4
  warmup_ratio: 0.05
  logging_steps: 10
  save_strategy: "epoch"
  save_total_limit: 2

huggingface:
  repo_id: "silent9669/legalqa-qwen2.5-3b-adapter"
  private: true
  upload_run_bundle: true
```

### 4.3 `configs/dataset_schema.yaml`
Specifies the canonical tables, column names, datatypes, and integrity constraints for `legal_chunks.parquet`, `qa_unique.parquet`, `qa_citations.parquet`, `fold_assignments.parquet`, `retrieval_labels.parquet`, and `known_qa.json`.

---

## 5. Clean Dataset Release for Task 2

The dataset package `phucdangg/legalqa-task2-clean-data` contains only pure data artifacts and metadata:
- `legal_chunks.parquet`: Deduplicated legal law/article chunks with canonical `chunk_id`.
- `qa_unique.parquet`: Deduplicated questions and ground truth answers.
- `known_qa.json`: Exact-match memory cache.
- `qa_citations.parquet`: Ground-truth citation links (`qa_id` -> `article_id`, `chunk_id`).
- `retrieval_labels.parquet`: Binary/graded relevance labels for retrieval training.
- `fold_assignments.parquet`: Deterministic 5-fold cross-validation assignments.
- `reranker_training_pairs.parquet`: Hard-negative mining pairs for BGE reranker.
- `public-official.json`: Public evaluation queries.
- `indexes/`: Precomputed BM25S and DEk21 indices aligned with `legal_chunks`.
- `dataset_manifest.json`: Manifest recording artifact sizes, record counts, and SHA256 hashes.
- `dataset-metadata.json`: Kaggle metadata for dataset upload.

**Zero code files (`.py`, `.sh`, `.git`) are included in the Kaggle dataset release.**

---

## 6. Notebook Contracts

### 6.1 `notebooks/kaggle_smoke_test.ipynb`
1. **Cell 1**: Environment check, disables async weight materialization (`HF_DEACTIVATE_ASYNC_LOAD=1`), validates Dual-T4 GPUs.
2. **Cell 2**: Code resolution — uses local workspace directory (synced from GitHub) and verifies `src/task2` and `configs/`.
3. **Cell 3**: Dataset mount resolution — locates `/kaggle/input/legalqa-task2-clean-data` and verifies `dataset_manifest.json` hashes.
4. **Cell 4**: Dependency bootstrap — checks Liger-kernel, transformers, peft, and bitsandbytes.
5. **Cell 5**: Smoke Probe Execution — executes worst-case token length probe (3 steps), short endurance probe (30 steps), and mini-evaluation.
6. **Cell 6**: Evidence Export — writes `kaggle_smoke_report.json`, `environment.txt`, `nvidia-smi.txt`, and logs to `/kaggle/working/`.

### 6.2 `notebooks/colab_train_a100.ipynb`
1. **Cell 1**: GPU verification — asserts `nvidia-smi` reports NVIDIA A100.
2. **Cell 2**: Repo synchronization — checks out frozen Git commit SHA.
3. **Cell 3**: Dataset verification — downloads/mounts `legalqa-task2-clean-data` and verifies SHA256 manifest match.
4. **Cell 4**: Freeze Tuple check — verifies `kaggle_smoke_report.json` indicates PASS for the same commit and dataset.
5. **Cell 5**: Production Training — executes full QLoRA generator training with BF16 and high throughput.
6. **Cell 6**: Full Evaluation — computes official whitespace-tokenized METEOR score and ROUGE-L on validation folds.
7. **Cell 7**: Run Bundle packaging and Hugging Face upload.

---

## 7. Pre-Push Verification & Testing Suite

### 7.1 Pre-Push Verification Gate (`scripts/pre_push_check.py`)
Run before any `git push` to guarantee repository integrity:
1. Validates all JSON/YAML files for syntax and schema compliance.
2. Runs dataset contract tests against `kaggle_dataset/staged/`.
3. Runs notebook contract tests to ensure no broken cells or unresolved paths.
4. Runs unit and contract tests via `pytest`.
5. Fails immediately if any contract is broken.

### 7.2 Structured Test Suite (`tests/`)
- `tests/dataset/test_dataset_integrity.py`: Verifies parquet schemas, non-empty text, foreign-key citation resolution to chunks, fold coverage, and absence of target leakage.
- `tests/notebooks/test_notebook_contracts.py`: Verifies valid notebook JSON format, required cells, profile parameter injection, and Kaggle/Colab path resolution.
- `tests/unit/test_liger_backend.py`: Verifies Liger fused-linear cross-entropy import and configuration.
- `tests/unit/test_evidence_packer.py`: Verifies evidence formatting and prompt construction.
- `tests/contracts/test_runtime_contracts.py`: Verifies configuration schemas and freeze-tuple SHA generation.
- `tests/gpu/test_liger_qlora_gpu.py`: Gated CUDA test for forward/backward gradients.

---

## 8. Definition of Done & Success Criteria

1. **Workspace Cleanliness**: Zero legacy scratchpad docs, zero obsolete configs, zero dead versioned tests, zero code packaged inside dataset staging.
2. **Dataset Release Ready**: `kaggle_dataset/staged/` contains only valid data artifacts with verified `dataset_manifest.json` hashes.
3. **Kaggle Smoke Compatibility**: `notebooks/kaggle_smoke_test.ipynb` executes cleanly on Kaggle Dual-T4 with PASS status.
4. **Colab A100 Compatibility**: `notebooks/colab_train_a100.ipynb` runs with BF16 and high batch throughput on A100.
5. **Green Pre-Push Check**: `python scripts/pre_push_check.py` passes 100% locally.
6. **Green CI/CD**: GitHub Actions `.github/workflows/tests.yml` passes 100%.
7. **Memory & Docs Updated**: `MEMORY.md` and `docs/` reflect the new architecture.
