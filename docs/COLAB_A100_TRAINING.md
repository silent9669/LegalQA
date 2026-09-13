# Google Colab A100 Production Training Guide

Authoritative specification for **Gate 4: Google Colab A100 Production Training** (`B2.2 Task 2 — Colab A100 Real Run & Evidence`).

---

## 1. Purpose

Google Colab with NVIDIA A100 GPU (40GB or 80GB VRAM) serves as the **production training environment**. A100 compute is expensive and strictly reserved for fully verified, frozen candidate versions that have achieved `PASS` status on both:
1. **Gate 2: Kaggle Dual-T4 CUDA Gate** (`kaggle_t4x2_report.json`)
2. **Gate 3: Google Colab Single-T4 Gate** (`colab_t4_report.json`)

---

## 2. Prerequisites & Authorization Gate

Before executing the Colab A100 launcher:
1. **GitHub Commit Frozen**: Exact Git commit SHA verified.
2. **GitHub CI PASS**: `scripts/verify_ci_status.py --sha <commit_sha>` reports `PASS`.
3. **Candidate Manifest Frozen**: `artifacts/candidates/<candidate_id>/candidate_manifest.json` present.
4. **Prior Gate Reports Attached**:
   - `artifacts/gates/<candidate_id>/kaggle_t4x2_report.json` = `PASS`
   - `artifacts/gates/<candidate_id>/colab_t4_report.json` = `PASS` (cryptographically chained to Kaggle report)

---

## 3. Production Configuration (`configs/task2/runtime/colab_a100.yaml`)

- **Hardware**: NVIDIA A100 (Single-GPU)
- **Precision**: Native `bfloat16` (`compute_dtype: bfloat16`)
- **Kernel**: Liger Fused Linear Cross-Entropy loss enabled (`loss_type: nll`)
- **LoRA Parameters**: Rank $r=16$, Alpha $\alpha=32$, Dropout $0.0$ (all 7 projection modules)
- **Batch Size**: `per_device_train_batch_size: 4`, `gradient_accumulation_steps: 2` (effective batch size = 8)
- **Training Scope**: All allowed training data (`training_scope: all_allowed_train`, `val_fold: null`)
- **Epochs**: 3 full epochs

---

## 4. Execution Workflow

### Automated CLI Execution (Recommended)
Run the stage-aware Colab session orchestrator from your local terminal:
```bash
python scripts/launch_colab_training.py \
  --stage a100 \
  --candidate artifacts/candidates/<candidate_id>/candidate_manifest.json \
  --kaggle-report artifacts/gates/<candidate_id>/kaggle_t4x2_report.json \
  --colab-t4-report artifacts/gates/<candidate_id>/colab_t4_report.json
```

What the launcher executes:
1. **Preflight**: Validates `.env` credentials, candidate manifest, and prior gate report hashes.
2. **Provisioning**: Runs `colab new -s <session> --gpu A100`.
3. **Upload**: Uploads `run_request.json`, `candidate_manifest.json`, and parent reports to `/content/legalqa_bootstrap/`.
4. **Remote Entry**: Executes `colab exec -s <session> -f scripts/colab_remote_entry.py` (no `--timeout` passed to CLI).
5. **Micro-Probe**: Verifies A100 hardware and executes 2 real optimizer steps -> `a100_micro_probe_report.json`.
6. **Full Training**: Trains Qwen2.5-3B QLoRA on all allowed data (`val_fold=None`).
7. **Packaging & Release**: Builds audited Run Bundle, computes `checksums.sha256`, and uploads to Hugging Face Hub under `runs/<run_id>/`.
8. **Download & Verification**: Downloads control artifacts to local `artifacts/gates/<candidate_id>/` and verifies them.
9. **Teardown**: Executes `colab stop -s <session>` in a `finally` block to prevent credit leakage.

---

## 5. Standardized Production Run Bundle

The output bundle is uploaded under `runs/<run_id>/` in Hugging Face repository `dangphuc2109/legalqa-qwen2.5-3b-adapter`:

```text
runs/<run_id>/
├── candidate_manifest.json
├── production_run_manifest.json
├── algorithm.resolved.json
├── runtime.resolved.json
├── dataset_manifest.json
├── dataset_validation_report.json
├── gate_reports/
│   ├── kaggle_t4x2_report.json
│   ├── colab_t4_report.json
│   └── a100_micro_probe_report.json
├── environment/
│   ├── python.txt
│   ├── pip-freeze.txt
│   ├── nvidia-smi.txt
│   └── runtime.json
├── logs/
│   └── train.log
├── metrics.json
├── trainer_state.json
├── telemetry.json
├── final_adapter/
│   ├── adapter_config.json
│   ├── adapter_model.safetensors
│   └── tokenizer files
├── model_card.md
└── checksums.sha256
```

---

## 6. Provenance Store (gitignored artifacts)

`artifacts/candidates/`, `artifacts/gates/`, `*.parquet`, and `*_output.ipynb` are gitignored by design (heavy binaries must never enter git). The promotable evidence therefore lives in two places:

1. **Hugging Face** `dangphuc2109/legalqa-qwen2.5-3b-adapter` under `runs/<run_id>/` — immutable release containing `candidate_manifest.json`, all three gate reports, resolved configs, metrics, and checksums (Lead approval source of truth).
2. **Local run host** `artifacts/gates/<candidate_id>/` — staging copy downloaded by `launch_colab_training.py` and verified with `verify_gate_report` before promotion.

The root `kaggle_smoke_report.json` is a legacy stub kept for backward compatibility only and is NOT promotable evidence. Never approve an A100 run from it; require the `artifacts/gates/<candidate_id>/kaggle_t4x2_report.json` + `colab_t4_report.json` chain for the exact frozen candidate.

## 7. Notebook Roles

- `notebooks/kaggle_smoke.ipynb` — Kaggle Dual-T4 CUDA gate launcher (Gate 2).
- `notebooks/colab_a100_train.ipynb` — canonical A100 production training notebook (Gate 4). Uses authoritative `configs/task2/algorithm.yaml` + `configs/task2/runtime/colab_a100.yaml` and the single HF target `src/task2/hf_uploader.py::DEFAULT_HF_REPO`. Legacy flat configs (`configs/kaggle_smoke_t4.yaml`, `configs/colab_train_a100.yaml`) are frozen for compatibility only.
- `Colab_A100_Master_Pipeline.ipynb` — INFERENCE-ONLY Drive pipeline (retrieval + merged-bf16 generation). Not a training path; do not spend training credit from it.
