# LegalQA Task 2 — Unified Modal A100 Training & Release Workflow

Authoritative production workflow for **LegalQA Task 2 (DSC 2026)** utilizing Modal A100 for scalable, reproducible GPU training and inference.

---

## 1. High-Level Architecture Overview

```text
[Clean Dataset] ──> phucdangg/legalqa-task2-clean-data (Kaggle Dataset v1)
       │
       ▼
[Local Verification] ──> python scripts/pre_push_check.py --mode full (261/261 PASS)
       │
       ▼
[Candidate Freeze] ────> artifacts/candidates/<candidate_id>/candidate_manifest.json
       │
       ▼
[Kaggle Dual-T4 Gate] ─> artifacts/gates/<candidate_id>/kaggle_t4x2_report.json (PASS, 8/8)
       │
       ▼
[Modal A100 Execution] (via ./scripts/run_modal.sh)
       ├─> Ingress & Volume Caching (legalqa-data-vol)
       ├─> BM25 & Dense Index Verification / Auto-Rebuild
       ├─> Dual-T4 Gate Verification (kaggle_t4x2_report.json)
       ├─> Modal A100 Micro-Probe (a100_micro_probe_report.json)
       ├─> Full SFT 2-Epoch Training (A100-40GB, bfloat16, Liger fused CE)
       ├─> Private Set Inference (1,918 queries, batch 16/128, 1536 token ceiling)
       ├─> Automated Packaging of Audited Run Bundle
       ├─> Automated Hugging Face Release (runs/<run_id>/)
       └─> Automated Download of submission.json.zip to project root
```

---

## 2. Infrastructure & Workspace Setup

### 2.1. Modal Configuration
The project connects to Modal serverless infrastructure using persistent volumes and workspace secrets:
- **Profile**: Configured via `modal setup`.
- **Persistent Volumes**:
  - `legalqa-data-vol` (mounted at `/data`): caches `phucdangg/legalqa-task2-clean-data` and prebuilt indexes (`indexes/bm25`, `indexes/dek21_rebuilt`).
  - `legalqa-runs-vol` (mounted at `/runs`): retains trained checkpoints, execution logs, and run bundles.
- **Workspace Secrets**:
  - `kaggle-secret`: contains `KAGGLE_USERNAME` and `KAGGLE_KEY`.
  - `huggingface-secret`: contains `HF_TOKEN`.

*Note: `./scripts/run_modal.sh` automatically checks and provisions any missing secrets or volumes directly from `.env` upon execution.*

---

## 3. Production Execution Protocol

### Step 1: Pre-Flight Verification (Local)
Run the full local gate to verify Python syntax, YAML configs, secrets scan, notebook ASTs, parameter budget (<4B), and test suite:
```bash
./.venv311/bin/python scripts/pre_push_check.py --mode full
```

### Step 2: Modal A100 Micro-Probe (~2 minutes)
Validates A100 allocation, VRAM limits, and establishes parent gate reports:
```bash
./scripts/run_modal.sh micro_probe
```
- Automatically executes the Dual-T4 root gate on Modal if `kaggle_t4x2_report.json` is missing.
- Runs the 2-step A100 micro-probe.
- Automatically saves `a100_micro_probe_report.json` to `artifacts/gates/<candidate_id>/`.

### Step 3: Modal A100 Full Training & Inference (~2.2 hours)
Executes the production training run and generates the private competition submission:
```bash
./scripts/run_modal.sh full
```
- Trains Qwen2.5-3B-Instruct for 2 epochs on deduplicated, evidence-grounded dataset using 4-bit NF4 QLoRA and selective Liger fused linear cross-entropy with `train_sampling_strategy='group_by_length'`.
- Runs high-throughput batched retrieval and generation for all **1,918 private queries** (`private-official.json`) in native `bfloat16` with merged adapter weights and length-sorted batching.
- Automatically bundles all artifacts, manifests, and logs, then publishes them to Hugging Face (`dangphuc2109/legalqa-qwen2.5-3b-adapter`).
- Automatically downloads `./submission.json.zip` directly to the project root, ready for upload to Codabench.

---

## 4. Key Performance & Quality Optimizations

1. **1-Epoch SFT Alignment**:
   - Training for 1 epoch (889 optimizer steps) on 1,325 canonical QA rows preserves general legal reasoning while teaching exact statutory citation format, preventing the catastrophic memorization that occurred with 4 epochs.
2. **CUDA Allocation Efficiency**:
   - Eliminating per-step `torch.cuda.empty_cache()` removes synchronization bubbles, allowing A100 Tensor Cores to run at continuous hardware bus throughput.
3. **Batched Inference Acceleration**:
   - Dense retrieval batch size 32, reranker batch size 128, and generator batch size 16 reduce 1,918-query inference from >2.5 hours to ~60–80 minutes with recursive order-preserving OOM halving.
4. **Retrieval Precision Recipe**:
   - Candidate pool of 50 chunks.
   - Legal reference arm extracting explicit articles and decrees.
   - Weighted RRF prioritizing lexical and reference matches for anchored queries.
   - Context diversification capping citations at 2 parts per article.

## 5. Dataset Contract (Essentials)

- **Slug**: `phucdangg/legalqa-task2-clean-data` — pure data only, zero code (`*.py`, `*.sh`, `.git` rejected by `validate_dataset` and `test_zero_code_in_dataset`).
- **Key artifacts**: `legal_chunks.parquet` (801,863 rows, `chunk_id`), `qa_unique.parquet` (`qa_id`, `fold_id`), `qa_citations.parquet`, `fold_assignments.parquet`, `retrieval_labels.parquet` (multi-positive, 4,759 labeled QA IDs), `reranker_training_pairs.parquet`, `known_qa.json`, `public-official.json` (1,000), `private-official.json` (1,918).
- **Indexes**: `indexes/bm25/` (verified order-bound bm25s) ships in the dataset; `indexes/dek21/` is excluded while quarantined and cold-rebuilt on the A100 with a pinned encoder revision + self-consistency gate.
- **Manifest**: `dataset_manifest.json` carries SHA-256 per file; any content change requires a new candidate freeze + re-gate.

## 6. Kaggle Dual-T4 Gate Essentials

- **Hardware**: 2× Tesla T4 (generator `cuda:0`, retrieval `cuda:1`), FP16.
- **Probes**: worst-case 2048-token probe (3 optimizer steps, finite loss, weight update) + 30-step endurance (VRAM stability) + adapter save/reload + 10-query mini-eval.
- **Report**: `kaggle_t4x2_report.json` with 8/8 checks, bound to the exact candidate SHA; verified locally with `verify_gate_report` before any promotion.
- A PASS covers only what it measures (CUDA training mechanics) — never retrieval quality or scores.
