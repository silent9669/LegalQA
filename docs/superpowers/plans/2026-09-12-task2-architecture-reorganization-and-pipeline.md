# Task 2 Architecture Reorganization, Clean Pipeline & Dual-Target Execution Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Reorganize the LegalQA Task 2 repository into a compact, production-grade workspace, establish a dedicated Kaggle Dual-T4 CUDA smoke test gate and Google Colab A100 production training pipeline, clean out all legacy artifacts, and deliver a modular validation suite with a pre-push verification gate.

**Architecture:** The system decouples dataset from code: clean canonical dataset artifacts are published directly to Kaggle (`phucdangg/legalqa-task2-clean-data`) without bundled application code. The GitHub repository serves as the single source of truth for code and notebooks. Kaggle Dual-T4 GPUs execute a thin smoke test launcher verifying real CUDA/QLoRA/Liger execution on worst-case and short endurance probes. Following smoke PASS, the frozen tuple is promoted to Google Colab A100 for high-throughput production training, packaging, and Hugging Face release.

**Tech Stack:** Python 3.10+, PyTorch 2.4+/CUDA, HuggingFace Transformers 5.0+, PEFT, bitsandbytes, Liger-kernel 0.8.2, BM25S, PyArrow/Parquet, Pytest, GitHub Actions.

**Spec:** `docs/superpowers/specs/2026-09-12-task2-architecture-reorganization-and-pipeline-design.md`

## Global Constraints

- **Zero Code Bundled in Dataset**: `kaggle_dataset/staged/` contains only pure Parquet/JSON/index data artifacts; no `.py`, `.sh`, or code directories.
- **Model Parameter Budget**: Must strictly respect competition rules: open-source model with parameters strictly below 4.0B (Qwen2.5-3B-Instruct).
- **Metric Fidelity**: Validation must use official whitespace-tokenized METEOR and secondary ROUGE-L.
- **Freeze Tuple Provenance**: Colab A100 production run requires an exact frozen tuple `(dataset_version, dataset_manifest_sha256, git_commit_sha, config_fingerprint, base_model_id_revision, kaggle_smoke_report_pass)`.
- **Pre-Push Guarantee**: `scripts/pre_push_check.py` must run and pass before any push to `origin main`.

---

### Task 1: Radical Workspace Cleanup & Staging Pruning

**Files:**
- Delete:
  - `docs/archive/`
  - `docs/active/`
  - `docs/LEGALQA_V16_ARCHITECTURE_REVIEW_AND_FIX_INSTRUCTIONS.md`
  - `docs/LEGALQA_V16_FINAL_PREFLIGHT_HARDENING.md`
  - `docs/OPERATING_RULES.md`
  - `configs/experiments.yaml`
  - `configs/models.yaml`
  - `configs/pipeline.yaml`
  - `configs/production_selection.yaml`
  - `configs/runtime_api.yaml`
  - `configs/task2.yaml`
  - `kaggle_dataset/staged/code/`
  - `kaggle_dataset/staged/code_manifest.json`
  - `notebooks/colab_t4_smoke.ipynb`
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
  - All flat `tests/test_v*.py` and `tests/test_common_*.py`
- Modify: `.gitignore`

**Interfaces:**
- Consumes: Existing legacy workspace files.
- Produces: Clean, pruned file tree with zero dead documentation, zero obsolete configs, and clean dataset staging without code.

- [ ] **Step 1: Remove legacy docs, obsolete configs, and dead scripts**

Run:
```bash
rm -rf docs/archive docs/active
rm -f docs/LEGALQA_V16_ARCHITECTURE_REVIEW_AND_FIX_INSTRUCTIONS.md docs/LEGALQA_V16_FINAL_PREFLIGHT_HARDENING.md docs/OPERATING_RULES.md
rm -f configs/experiments.yaml configs/models.yaml configs/pipeline.yaml configs/production_selection.yaml configs/runtime_api.yaml configs/task2.yaml
rm -rf kaggle_dataset/staged/code kaggle_dataset/staged/code_manifest.json
rm -f notebooks/colab_t4_smoke.ipynb
rm -f scripts/run_colab_smoke.py scripts/fetch_kaggle_submission.py scripts/monitor_kaggle.py scripts/verify_inherited_dataset.py scripts/train_retriever_mnrl.py scripts/preflight_kaggle.py scripts/build_indexes.py scripts/mine_retrieval_negatives.py scripts/evaluate_checkpoint.py scripts/promote_production_selection.py scripts/run_oof_validation.py
```

- [ ] **Step 2: Remove legacy flat tests**

Run:
```bash
rm -f tests/test_v*.py tests/test_common_*.py tests/test_task2_*.py tests/test_colab_smoke_contract.py tests/test_data_pipeline_integration.py tests/test_evaluation.py tests/test_generator_training_data.py tests/test_kaggle_packaging.py tests/test_oof_isolation.py tests/test_path_resolver_kaggle_nested.py tests/test_production_config_and_manifest.py tests/test_real_kaggle_screen_gate.py tests/test_reranker_training_data.py tests/test_security_preflight.py tests/test_validation_and_audit.py
```

- [ ] **Step 3: Update `.gitignore` to prevent tracking temp artifacts and `.DS_Store`**

Ensure `.gitignore` contains entries for `.playwright-mcp`, `.remember`, `artifacts/raw`, and `.DS_Store`.

- [ ] **Step 4: Verify working tree status**

Run: `git status`
Expected: Staged/untracked cleanup visible; no accidental deletions of `src/task2/` core logic or `kaggle_dataset/staged/*.parquet`.

- [ ] **Step 5: Commit workspace pruning**

```bash
git add -u
git commit -m "refactor(cleanup): prune legacy docs, obsolete configs, and dead scripts

Co-Authored-By: Claude Code <noreply@anthropic.com>"
```

---

### Task 2: Canonical Configuration Suite

**Files:**
- Create: `configs/kaggle_smoke_t4.yaml`
- Create: `configs/colab_train_a100.yaml`
- Create: `configs/dataset_schema.yaml`
- Test: `tests/contracts/test_configs.py`

**Interfaces:**
- Consumes: Design spec section 4 definitions.
- Produces: YAML config files providing typed runtime parameters for Kaggle smoke testing, Colab A100 training, and dataset schema validation.

- [ ] **Step 1: Write failing test for configuration loading and validation**

Create `tests/contracts/test_configs.py`:
```python
import os
import yaml
import pytest

def test_kaggle_smoke_config_structure():
    path = "configs/kaggle_smoke_t4.yaml"
    assert os.path.exists(path), f"Missing {path}"
    with open(path, "r") as f:
        cfg = yaml.safe_load(f)
    assert cfg["profile_name"] == "kaggle_smoke_t4"
    assert cfg["devices"]["generator"] == "cuda:0"
    assert cfg["devices"]["retrieval"] == "cuda:1"
    assert cfg["generator"]["load_in_4bit"] is True
    assert cfg["probes"]["run_worstcase_probe"] is True
    assert cfg["probes"]["worstcase_steps"] == 3
    assert cfg["probes"]["endurance_steps"] == 30

def test_colab_train_config_structure():
    path = "configs/colab_train_a100.yaml"
    assert os.path.exists(path), f"Missing {path}"
    with open(path, "r") as f:
        cfg = yaml.safe_load(f)
    assert cfg["profile_name"] == "colab_train_a100"
    assert cfg["target_hardware"] == "NVIDIA A100"
    assert cfg["generator"]["torch_dtype"] == "bfloat16"
    assert cfg["generator"]["num_train_epochs"] >= 1
    assert "huggingface" in cfg

def test_dataset_schema_config_structure():
    path = "configs/dataset_schema.yaml"
    assert os.path.exists(path), f"Missing {path}"
    with open(path, "r") as f:
        cfg = yaml.safe_load(f)
    assert "tables" in cfg
    assert "legal_chunks" in cfg["tables"]
    assert "qa_unique" in cfg["tables"]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/contracts/test_configs.py -v`
Expected: FAIL (files missing).

- [ ] **Step 3: Create `configs/kaggle_smoke_t4.yaml`, `configs/colab_train_a100.yaml`, and `configs/dataset_schema.yaml`**

Create `configs/kaggle_smoke_t4.yaml`:
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

Create `configs/colab_train_a100.yaml`:
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
  load_in_4bit: false
  torch_dtype: "bfloat16"
  use_liger_kernel: true
  lora_r: 32
  lora_alpha: 64
  lora_dropout: 0.05
  learning_rate: 2.0e-4
  num_train_epochs: 3
  per_device_train_batch_size: 4
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

Create `configs/dataset_schema.yaml`:
```yaml
schema_version: "2.0.0"
tables:
  legal_chunks:
    required_file: "legal_chunks.parquet"
    primary_key: "chunk_id"
    required_columns:
      chunk_id: "string"
      article_id: "string"
      text: "string"
  qa_unique:
    required_file: "qa_unique.parquet"
    primary_key: "qa_id"
    required_columns:
      qa_id: "string"
      question: "string"
      answer: "string"
  qa_citations:
    required_file: "qa_citations.parquet"
    required_columns:
      qa_id: "string"
      article_id: "string"
      chunk_id: "string"
  fold_assignments:
    required_file: "fold_assignments.parquet"
    required_columns:
      qa_id: "string"
      fold: "int64"
  retrieval_labels:
    required_file: "retrieval_labels.parquet"
    required_columns:
      qa_id: "string"
      chunk_id: "string"
      relevance: "int64"
  known_qa:
    required_file: "known_qa.json"
    type: "json"
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/contracts/test_configs.py -v`
Expected: PASS.

- [ ] **Step 5: Commit configuration suite**

```bash
git add configs/ tests/contracts/test_configs.py
git commit -m "feat(config): add canonical configs for kaggle smoke, colab train, and dataset schema

Co-Authored-By: Claude Code <noreply@anthropic.com>"
```

---

### Task 3: Dataset Validation & Clean Packaging System

**Files:**
- Create: `src/task2/dataset/validator.py`
- Create: `scripts/validate_dataset.py`
- Create: `scripts/package_kaggle_dataset.py`
- Modify: `kaggle_dataset/dataset-metadata.json`
- Test: `tests/dataset/test_dataset_integrity.py`

**Interfaces:**
- Consumes: `configs/dataset_schema.yaml` and `kaggle_dataset/staged/` artifacts.
- Produces: `validate_dataset(data_dir, schema_path) -> dict` returning PASS/FAIL status and report.
- Produces: `scripts/package_kaggle_dataset.py` to regenerate `dataset_manifest.json` with accurate SHA256 sums of pure data.

- [ ] **Step 1: Write failing test for dataset validation**

Create `tests/dataset/test_dataset_integrity.py`:
```python
import os
import pytest
from src.task2.dataset.validator import validate_dataset

def test_dataset_integrity_on_staged():
    staged_dir = "kaggle_dataset/staged"
    if not os.path.exists(os.path.join(staged_dir, "qa_unique.parquet")):
        pytest.skip("Staged dataset artifacts not found locally.")
    
    report = validate_dataset(data_dir=staged_dir, schema_path="configs/dataset_schema.yaml")
    assert report["status"] == "PASS", f"Dataset validation failed: {report.get('errors')}"
    assert report["manifest_verified"] is True
    assert "code" not in report.get("detected_directories", [])
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/dataset/test_dataset_integrity.py -v`
Expected: FAIL (validator module missing).

- [ ] **Step 3: Implement `src/task2/dataset/validator.py`**

Implement schema checking, parquet column/null auditing, referential foreign-key check (`qa_citations.chunk_id` in `legal_chunks.chunk_id`), manifest SHA256 verification, and asserting zero code files in `data_dir`.

- [ ] **Step 4: Implement `scripts/validate_dataset.py` and `scripts/package_kaggle_dataset.py`**

- `scripts/validate_dataset.py`: CLI taking `--data-dir` and `--schema-path`, printing structured summary and writing `validation_report.json`.
- `scripts/package_kaggle_dataset.py`: Computes SHA256 hashes for all data files, builds clean `dataset_manifest.json` without any `code/` references, and formats `dataset-metadata.json`.

- [ ] **Step 5: Run dataset packaging and validation**

Run:
```bash
python scripts/package_kaggle_dataset.py --source-dir kaggle_dataset/staged
python scripts/validate_dataset.py --data-dir kaggle_dataset/staged
pytest tests/dataset/test_dataset_integrity.py -v
```
Expected: PASS with 100% valid schema and manifest check.

- [ ] **Step 6: Commit dataset tools and updated manifest**

```bash
git add src/task2/dataset/ scripts/validate_dataset.py scripts/package_kaggle_dataset.py kaggle_dataset/ tests/dataset/
git commit -m "feat(dataset): add clean dataset packaging and integrity validator

Co-Authored-By: Claude Code <noreply@anthropic.com>"
```

---

### Task 4: Kaggle Dual-T4 Smoke Test Notebook & Profile

**Files:**
- Create: `notebooks/kaggle_smoke_test.ipynb`
- Modify: `kaggle_kernel/kernel-metadata.json`
- Modify: `kaggle_kernel/legalqa_gpu_pipeline.ipynb` (synced copy for Kaggle kernel push)
- Modify: `src/task2/pipeline/profiles.py` & `src/task2/pipeline/runner.py`
- Test: `tests/notebooks/test_notebook_contracts.py`

**Interfaces:**
- Consumes: `configs/kaggle_smoke_t4.yaml` and GitHub repo code.
- Produces: `kaggle_smoke_report.json` with execution metrics for worstcase probe, endurance probe, and mini-eval.

- [ ] **Step 1: Write failing test for notebook contract**

Create `tests/notebooks/test_notebook_contracts.py`:
```python
import json
import os
import pytest

def test_kaggle_smoke_notebook_contract():
    nb_path = "notebooks/kaggle_smoke_test.ipynb"
    assert os.path.exists(nb_path), f"Missing {nb_path}"
    with open(nb_path, "r") as f:
        nb = json.load(f)
    cells = nb.get("cells", [])
    assert len(cells) >= 5
    source_all = "\n".join("".join(c.get("source", [])) for c in cells)
    assert "kaggle_smoke_t4.yaml" in source_all
    assert "kaggle_smoke_report.json" in source_all
    assert "HF_DEACTIVATE_ASYNC_LOAD" in source_all
    assert "/kaggle/input/**/code/LegalQA" not in source_all, "Notebook must not expect code inside dataset!"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/notebooks/test_notebook_contracts.py -v`
Expected: FAIL (notebook missing).

- [ ] **Step 3: Implement `notebooks/kaggle_smoke_test.ipynb` and sync to `kaggle_kernel/legalqa_gpu_pipeline.ipynb`**

Build the thin launcher notebook:
- Cell 1: Environment & Async Load override (`HF_DEACTIVATE_ASYNC_LOAD=1`), Dual-T4 verification.
- Cell 2: Code detection (uses current workspace clone or repo root).
- Cell 3: Mount resolution for `/kaggle/input/legalqa-task2-clean-data` and manifest validation.
- Cell 4: Dependency check (Liger, Transformers 5, PEFT, bitsandbytes).
- Cell 5: Execution of `kaggle_smoke_t4` profile via `run_pipeline()`.
- Cell 6: Export summary and write `kaggle_smoke_report.json`.

Sync to `kaggle_kernel/legalqa_gpu_pipeline.ipynb` and update `kaggle_kernel/kernel-metadata.json`.

- [ ] **Step 4: Update `src/task2/pipeline/profiles.py` and `src/task2/pipeline/runner.py`**

Ensure `resolve_execution_profile` supports `kaggle_smoke_t4` profile directly from YAML config and executes the worst-case, endurance, and mini-eval sequence.

- [ ] **Step 5: Run notebook contract test**

Run: `pytest tests/notebooks/test_notebook_contracts.py -v`
Expected: PASS.

- [ ] **Step 6: Commit Kaggle smoke notebook and runner updates**

```bash
git add notebooks/ kaggle_kernel/ src/task2/pipeline/ tests/notebooks/
git commit -m "feat(smoke): add canonical kaggle dual-t4 smoke notebook and profile

Co-Authored-By: Claude Code <noreply@anthropic.com>"
```

---

### Task 5: Google Colab A100 Production Training Pipeline & Notebook

**Files:**
- Create: `notebooks/colab_train_a100.ipynb`
- Create: `scripts/run_pipeline.py`
- Modify: `src/task2/provenance/freeze_tuple.py` (or `runtime_integrity.py`)
- Test: `tests/contracts/test_profile_resolution.py`

**Interfaces:**
- Consumes: `configs/colab_train_a100.yaml`, frozen tuple SHA, verified smoke report.
- Produces: `scripts/run_pipeline.py --config configs/colab_train_a100.yaml` for CLI/Colab execution.
- Produces: Full Run Bundle generator & Hugging Face upload integration.

- [ ] **Step 1: Write failing test for profile resolution and freeze tuple verification**

Create `tests/contracts/test_profile_resolution.py`:
```python
import pytest
from src.task2.pipeline.profiles import load_profile_from_yaml
from src.task2.provenance.freeze_tuple import compute_freeze_tuple_hash

def test_resolve_colab_a100_profile():
    profile = load_profile_from_yaml("configs/colab_train_a100.yaml")
    assert profile.name == "colab_train_a100"
    assert profile.torch_dtype == "bfloat16"
    assert profile.batch_size == 4

def test_freeze_tuple_computation():
    freeze_hash = compute_freeze_tuple_hash(
        dataset_manifest_sha="abc",
        git_commit_sha="def",
        config_hash="ghi",
        base_model_revision="jkl",
    )
    assert isinstance(freeze_hash, str) and len(freeze_hash) == 64
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/contracts/test_profile_resolution.py -v`
Expected: FAIL (methods missing).

- [ ] **Step 3: Implement `src/task2/provenance/freeze_tuple.py` and update `src/task2/pipeline/profiles.py`**

- Add `compute_freeze_tuple_hash` to generate deterministic SHA256 from freeze tuple elements.
- Add `load_profile_from_yaml` to parse `kaggle_smoke_t4.yaml` and `colab_train_a100.yaml` into typed `ExecutionProfile` dataclasses.

- [ ] **Step 4: Implement `scripts/run_pipeline.py` and `notebooks/colab_train_a100.ipynb`**

- `scripts/run_pipeline.py`: Unified CLI entry point accepting `--config`, `--run-mode` (smoke vs production), `--output-dir`, and `--check-smoke-report`.
- `notebooks/colab_train_a100.ipynb`: Interactive launcher designed for Colab with A100 GPU, executing `scripts/run_pipeline.py` or calling modules directly with BF16 throughput and HF model hub export.

- [ ] **Step 5: Run profile tests**

Run: `pytest tests/contracts/test_profile_resolution.py -v`
Expected: PASS.

- [ ] **Step 6: Commit Colab A100 pipeline components**

```bash
git add notebooks/colab_train_a100.ipynb scripts/run_pipeline.py src/task2/provenance/ src/task2/pipeline/ tests/contracts/test_profile_resolution.py
git commit -m "feat(colab): add colab a100 training pipeline, cli runner, and freeze tuple resolver

Co-Authored-By: Claude Code <noreply@anthropic.com>"
```

---

### Task 6: Modularized Test Suite Refactoring

**Files:**
- Create: `tests/unit/test_liger_backend.py`
- Create: `tests/unit/test_evidence_packer.py`
- Create: `tests/unit/test_checkpoint_resolver.py`
- Create: `tests/unit/test_retrieval_and_bm25.py`
- Create: `tests/contracts/test_runtime_contracts.py`
- Create: `tests/gpu/test_liger_qlora_gpu.py`
- Test: All tests in `tests/`

**Interfaces:**
- Consumes: Cleaned `src/task2` modules.
- Produces: Fast, modular, deterministic test suite covering units, integration, dataset, contracts, and GPU.

- [ ] **Step 1: Write unit tests for Liger backend and evidence packer**

Consolidate essential component checks into `tests/unit/test_liger_backend.py` (version 0.8.2 check, kernel application) and `tests/unit/test_evidence_packer.py` (token budget, structure formatting).

- [ ] **Step 2: Write unit tests for retrieval and checkpoint resolver**

Consolidate into `tests/unit/test_retrieval_and_bm25.py` and `tests/unit/test_checkpoint_resolver.py`.

- [ ] **Step 3: Write runtime contract tests**

In `tests/contracts/test_runtime_contracts.py`: test parameter budget check (<4B parameters), dependency lock check, and API version 16 binding.

- [ ] **Step 4: Keep CUDA-gated GPU test in `tests/gpu/test_liger_qlora_gpu.py`**

Ensure `tests/gpu/test_liger_qlora_gpu.py` tests forward/backward loss parity with cosine similarity >= 0.99 and is cleanly skipped when CUDA is not available.

- [ ] **Step 5: Run entire unit & contract test suite**

Run: `pytest tests/unit tests/contracts tests/dataset tests/notebooks -v`
Expected: 100% PASS.

- [ ] **Step 6: Commit modularized test suite**

```bash
git add tests/
git commit -m "test: reorganize and modularize test suite into unit, dataset, contracts, and notebooks

Co-Authored-By: Claude Code <noreply@anthropic.com>"
```

---

### Task 7: Pre-Push Verification Gate & CI/CD Pipeline

**Files:**
- Create: `scripts/pre_push_check.py`
- Modify: `.github/workflows/tests.yml`
- Test: `scripts/pre_push_check.py`

**Interfaces:**
- Consumes: Full repository state (configs, code, staged dataset, notebooks).
- Produces: Exit code 0 if ready for `git push`, non-zero otherwise.

- [ ] **Step 1: Implement `scripts/pre_push_check.py`**

Write a pre-push script that performs:
1. Syntax & lint check: validates all `.yaml` and `.json` files.
2. Dataset contract check: validates `kaggle_dataset/staged/` (or checks schema if large files absent).
3. Notebook contract check: verifies notebook JSON integrity and required profile paths.
4. Pytest suite: executes `pytest tests/unit tests/contracts tests/dataset tests/notebooks -v`.
5. Prints clear summary banner: `=== ALL PRE-PUSH CHECKS PASSED ===`.

- [ ] **Step 2: Update `.github/workflows/tests.yml`**

Streamline GitHub Actions workflow to:
- Run preflight & lint checks (`python scripts/pre_push_check.py`).
- Run unit and contract tests on Python 3.10 and 3.11.
- Test exact Kaggle HF dependency stack (`transformers==5.0.0`, `peft==0.19.1`, `liger-kernel==0.8.2`).

- [ ] **Step 3: Run `scripts/pre_push_check.py` locally**

Run: `python scripts/pre_push_check.py`
Expected: PASS with exit code 0.

- [ ] **Step 4: Commit pre-push gate and CI/CD workflow**

```bash
git add scripts/pre_push_check.py .github/workflows/tests.yml
git commit -m "ci: add local pre-push check gate and clean up github actions workflow

Co-Authored-By: Claude Code <noreply@anthropic.com>"
```

---

### Task 8: Authoritative Documentation & Memory Synchronization

**Files:**
- Create: `docs/ARCHITECTURE.md`
- Create: `docs/KAGGLE_SMOKE_GATE.md`
- Create: `docs/COLAB_A100_TRAINING.md`
- Create: `docs/DATASET_SPEC.md`
- Modify: `README.md`
- Modify: `/Users/phucdang/.claude/projects/-Users-phucdang-Documents-LegalQA---Public-Test/memory/kaggle-and-colab-workflow.md`
- Modify: `/Users/phucdang/.claude/projects/-Users-phucdang-Documents-LegalQA---Public-Test/memory/MEMORY.md`

**Interfaces:**
- Consumes: Notion spec and verified codebase implementation.
- Produces: Authoritative, compact documentation and updated memory files.

- [ ] **Step 1: Author `docs/ARCHITECTURE.md`**

Document the DSC 2026 system architecture, roles (Data Owner, Training Owner, Lead), artifact storage, and authoritative handoff flow.

- [ ] **Step 2: Author `docs/KAGGLE_SMOKE_GATE.md`**

Document the Kaggle Dual-T4 smoke test procedure: environment preflight, worst-case token length probe, 30-step endurance probe, mini-eval, and `kaggle_smoke_report.json` generation.

- [ ] **Step 3: Author `docs/COLAB_A100_TRAINING.md`**

Document the Colab A100 production training procedure: freeze tuple authorization, BF16 throughput settings, checkpoint cadence, METEOR evaluation, and Hugging Face artifact release.

- [ ] **Step 4: Author `docs/DATASET_SPEC.md`**

Document the `phucdangg/legalqa-task2-clean-data` schema, Parquet tables, integrity checks, and rule against bundling code.

- [ ] **Step 5: Refresh `README.md`**

Update `README.md` to cleanly present the new architecture, how to run pre-push checks, how to smoke test on Kaggle, and how to train on Colab A100.

- [ ] **Step 6: Update persistent memory files**

Update `/Users/phucdang/.claude/projects/-Users-phucdang-Documents-LegalQA---Public-Test/memory/kaggle-and-colab-workflow.md` and `MEMORY.md` with the new Kaggle Smoke vs Colab A100 training lifecycle and pre-push requirements.

- [ ] **Step 7: Run final verification**

Run: `python scripts/pre_push_check.py`
Expected: 100% PASS.

- [ ] **Step 8: Commit documentation and memory**

```bash
git add docs/ README.md
git commit -m "docs: add authoritative architecture, kaggle smoke, colab training, and dataset specs

Co-Authored-By: Claude Code <noreply@anthropic.com>"
```
