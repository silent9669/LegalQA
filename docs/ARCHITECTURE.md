# LegalQA Task 2 — Reproducible Architecture Specification

Authoritative architecture specification for **LegalQA (Task 2)** in compliance with the **DSC 2026 Reproducible Training Workflow** (`https://dangphuc.notion.site/dscc`).

---

## 1. System Architecture & Lifecycle

```
[Official BTC Data]
        │
        ▼
[Data Owner: Parse/Clean/Normalize]
        │
        ▼
[Dataset Schema & Integrity Validation] ──(FAIL)──> [Fix Data Bug]
        │ (PASS)
        ▼
[Publish Kaggle Dataset vN] (Pure Data Artifacts + Manifest)
        │
        ▼
[GitHub: Notebooks + Code + Configs]
        │
        ▼
[Pre-Push Check Gate / Local CI] ────────(FAIL)──> [Fix Code / Contracts]
        │ (PASS)
        ▼
[Kaggle GPU Dual-T4 CUDA Smoke Gate] ────(FAIL)──> [Fix CUDA/VRAM Bug]
        │ (PASS)
        ▼
[Freeze Tuple: Dataset Hash + Git SHA + Config Fingerprint + Model Rev]
        │
        ▼
[Lead Approval / Run Authorization]
        │
        ▼
[Google Colab NVIDIA A100 Production Training] (BF16, Full Training)
        │
        ▼
[Official Metric Evaluation: Whitespace METEOR + ROUGE-L]
        │
        ▼
[Hugging Face Release: Model Adapter + Logs + Run Bundle]
```

---

## 2. Team Boundaries & Responsibilities

| Role | Scope | Canonical Output |
| --- | --- | --- |
| **Data Owner** | Prepares, cleans, validates, and releases the canonical Kaggle dataset. Schema, manifest, foreign-key integrity, and provenance. | Kaggle Dataset `phucdangg/legalqa-task2-clean-data` vN (pure data, zero code bundled). |
| **Training Owner (Kaggle)** | Maintains GitHub repo and runs the thin smoke launcher on Kaggle Dual-T4 GPUs to prove real CUDA/VRAM/Liger safety. | Git commit SHA + `kaggle_smoke_report.json` = PASS. |
| **Training Owner (Colab A100)** | Consumes frozen tuple, executes full production training on Google Colab A100, exports Run Bundle. | Trained QLoRA adapter + metrics + TensorBoard logs uploaded to Hugging Face. |
| **Lead / Integration** | Reviews handoffs, verifies freeze tuples, authorizes A100 consumption, and manages competition submission mapping. | Approved Run ID + audit traceability. |

---

## 3. Storage Map & Artifact Invariants

| Artifact | Canonical Storage | Invariant |
| --- | --- | --- |
| **Canonical Dataset** | Kaggle Dataset | Direct upload to Kaggle. Must contain `dataset_manifest.json` with SHA-256 hashes. Never bundle application code inside dataset. |
| **Source Code & Notebooks** | GitHub (`silent9669/LegalQA`) | All logic lives in Git. Kaggle pulls notebook from linked GitHub repo. |
| **CUDA Smoke Gate** | Kaggle Dual-T4 | Executes worst-case sequence length probe, 30-step endurance probe, and mini-eval. Produces `kaggle_smoke_report.json`. |
| **Production Training** | Google Colab A100 | Executes full training only after smoke PASS. Leverages native BF16 throughput and larger batch sizes. |
| **Release Artifacts** | Hugging Face Hub | Model adapter checkpoints, `run_manifest.json`, training logs, and metrics. |

---

## 4. Freeze Tuple Contract

An A100 training run is authorized only when the following tuple is frozen:
```text
RUN_TUPLE:
├── task: "LegalQA"
├── kaggle_dataset_slug: "phucdangg/legalqa-task2-clean-data"
├── kaggle_dataset_version: "vN"
├── dataset_manifest_sha256: "<64-hex>"
├── git_repository: "silent9669/LegalQA"
├── git_commit_sha: "<40-hex>"
├── config_fingerprint: "<64-hex>"
├── base_model_id: "Qwen/Qwen2.5-3B-Instruct"
├── base_model_revision: "main"
└── kaggle_smoke_report_status: "PASS"
```

---

## 5. Minimum Run Bundle

Every production run produces a standardized audit bundle:
```text
RUN_ID/
├── run_manifest.json               # Provenance map linking model -> run -> code -> dataset
├── config.yaml                     # Frozen configuration used for training
├── dataset_manifest.json           # Cryptographic manifest of consumed data
├── validation_report.json          # Dataset validation pass evidence
├── kaggle_smoke_report.json        # Upstream Kaggle smoke pass report
├── environment.txt                 # Pinned pip environment dump
├── nvidia-smi.txt                  # Captured A100 GPU hardware telemetry
├── train.log                       # Training console output
├── metrics.json                    # Final evaluation scores (METEOR, ROUGE-L)
├── trainer_state.json              # Checkpoint progression and loss history
├── tensorboard/                    # Training curves
└── final_adapter/                  # SafeTensors trained LoRA weights & adapter_config.json
```
