# Kaggle Dual-T4 CUDA Smoke Gate Guide

Authoritative specification for the **Kaggle 2×T4 Smoke Gate** (`B2.1 Task 2 — Kaggle 2×T4 Smoke Gate`).

---

## 1. Purpose

The Kaggle Smoke Gate proves that the exact LegalQA release executes the real CUDA / QLoRA / Liger / retrieval path on dual NVIDIA T4 GPUs before consuming expensive Google Colab A100 credits.

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
  ```
  *Why*: Transformers 5.0 async weight materialization transiently spikes VRAM faster than 4-bit quantization consumes weights, causing OOMs on 16GB GPUs during model initialization.

---

## 3. Staged Smoke Probes

The smoke notebook (`notebooks/kaggle_smoke_test.ipynb`) executes three deterministic verification probes:

### Probe 1: Worst-Case Token Length Probe (3 Steps)
- Filters training examples for the longest representative input prompts and completions (close to 2048 tokens).
- Executes 3 real optimizer steps with backpropagation and weight updates.
- **Pass condition**: No OOM, finite loss, trainable weights demonstrably change.

### Probe 2: Short Endurance Probe (30 Steps)
- Runs 30 consecutive optimizer steps on diverse data samples.
- Monitored by the GPU telemetry tracker.
- **Pass condition**: Stable step time, zero VRAM creep or uncollected tensor fragmentation.

### Probe 3: End-to-End Mini Evaluation (10 Queries)
- Executes the full inference pipeline: BM25S + DEk21 dense retrieval -> RRF fusion -> BGE reranking -> Structured evidence packing -> Qwen 3B generation.
- Computes official whitespace METEOR and secondary ROUGE-L on 10 deterministic test questions.
- **Pass condition**: Pipeline runs end-to-end and outputs formatted answer strings without exceptions.

---

## 4. Required Output: `kaggle_smoke_report.json`

At the end of Cell 6, the notebook generates `/kaggle/working/kaggle_smoke_report.json`:

```json
{
  "status": "PASS",
  "profile": "kaggle_smoke_t4",
  "hardware": {
    "gpu_count": 2,
    "gen_device": "cuda:0",
    "retrieval_device": "cuda:1"
  },
  "dataset_verified": true,
  "stages_executed": [
    "environment_preflight",
    "dataset_provenance",
    "worstcase_probe",
    "endurance_probe",
    "mini_evaluation"
  ]
}
```

---

## 5. Invalidation & Gate Rules

A smoke `PASS` applies strictly to the frozen tuple:
`(Git commit SHA, Kaggle dataset version/hash, config fingerprint, base model revision)`.

Any modification to model architecture, loss function, sequence lengths, tokenization, or dataset requires re-running the smoke gate.
