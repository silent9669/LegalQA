# Kaggle Dual-T4 CUDA Gate Guide

Authoritative specification for **Gate 2: Kaggle Dual-T4 CUDA Gate** (`B2.1 Task 2 — Kaggle 2×T4 Smoke Gate`).

---

## 1. Purpose

The Kaggle Dual-T4 gate proves that the exact LegalQA candidate executes the real CUDA / QLoRA / Liger / retrieval path on dual NVIDIA T4 GPUs before consuming Google Colab GPU credits.

The gate must **fail fast** on memory regressions, dependency mismatches, and control-flow issues.

---

## 2. Hardware & Environment Setup

- **Platform**: Kaggle Notebook (Kernel)
- **Accelerators**: GPU T4 x2 (Dual NVIDIA Tesla T4 16GB)
- **Device Layout**:
  - `cuda:0`: Qwen2.5-3B-Instruct Generator (4-bit QLoRA with Liger fused-linear CE)
  - `cuda:1`: Dense Retrieval (DEk21) + BGE Reranker v2 m3
  - `CPU`: BM25S, Evidence Packer, Memory
- **Mandatory Environment Flag**:
  ```python
  import os
  os.environ["HF_DEACTIVATE_ASYNC_LOAD"] = "1"
  os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True,max_split_size_mb:128"
  ```
  *Why*: Transformers 5.0 async weight materialization transiently spikes VRAM faster than 4-bit quantization consumes weights, causing OOMs on 16GB GPUs during model initialization.

---

## 3. Required Staged Probes

The standalone gate runner (`scripts/run_gpu_gate.py --stage kaggle_t4x2`) or notebook (`notebooks/kaggle_smoke.ipynb`) executes four deterministic verification probes:

### Probe 1: Worst-Case Token Length Probe (3 Steps)
- Filters training examples for the longest representative input prompts and completions (close to 2048 tokens).
- Executes 3 real optimizer steps with backpropagation and weight updates.
- **Pass condition**: No OOM, finite loss, trainable weights demonstrably change.

### Probe 2: Endurance Probe (30 Steps)
- Runs 30 consecutive optimizer steps on diverse data samples.
- Monitored by the GPU telemetry tracker.
- **Pass condition**: Stable step time, zero monotonic VRAM creep or uncollected tensor fragmentation.

### Probe 3: Adapter Save & Reload
- Saves adapter weights to disk, cleans up GPU allocations, reloads the adapter, and verifies inference generation.

### Probe 4: End-to-End Mini Evaluation (10 Queries)
- Executes the full inference pipeline: BM25S + DEk21 dense retrieval -> RRF fusion -> BGE reranking -> Structured evidence packing -> Qwen 3B generation.
- Computes official whitespace METEOR and secondary ROUGE-L on 10 deterministic test questions.

---

## 4. Required Output: `kaggle_t4x2_report.json`

The gate generates `/kaggle/working/kaggle_t4x2_report.json`:

```json
{
  "schema_version": 1,
  "stage": "kaggle_t4x2",
  "status": "PASS",
  "candidate_id": "<16-hex>",
  "identity": {
    "git_commit_sha": "<40-hex>",
    "dataset_slug": "phucdangg/legalqa-task2-clean-data",
    "dataset_version": 1,
    "dataset_manifest_sha256": "<64-hex>",
    "algorithm_sha256": "<64-hex>",
    "runtime_profile_sha256": "<64-hex>",
    "dependency_lock_sha256": "<64-hex>"
  },
  "checks": {
    "dataset_verified": true,
    "config_verified": true,
    "model_revisions_verified": true,
    "finite_loss": true,
    "trainable_weight_changed": true,
    "checkpoint_saved": true,
    "checkpoint_reloaded": true,
    "mini_eval_completed": true
  }
}
```

---

## 5. Invalidation & Gate Rules

A smoke `PASS` applies strictly to the frozen candidate.
Any modification to model architecture, loss function, sequence lengths, tokenization, or dataset creates a new candidate manifest and requires re-running the Kaggle gate.
