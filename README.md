# LegalQA Task 2 — Reproducible Pipeline (DSC 2026)

End-to-end Legal Question Answering (LegalQA) system for DSC 2026 Task 2, engineered for reproducible training, GPU credit efficiency, and verifiable provenance in compliance with the **DSC 2026 Reproducible Training Workflow** (`https://dangphuc.notion.site/dscc`).

- **Parameter Budget**: Total learned parameters strictly $< 4.0\text{B}$ (`Qwen/Qwen2.5-3B-Instruct` base).
- **Data Constraint**: Task 2 official organizer data only (zero external legal corpora or external answer APIs).
- **Dual-Target Execution**:
  - **Kaggle GPU Dual-T4**: CUDA smoke test gate (worst-case sequence probe, short endurance stability, mini-eval).
  - **Google Colab NVIDIA A100**: Production training environment (native BF16, high-throughput batching, full epochs, Hugging Face release).
- **Loss Backend**: Liger fused-linear cross-entropy (`liger-kernel==0.8.2`, `loss_type=nll`).

---

## 1. System Architecture & Dual-Target Workflow

```text
[Official BTC Data]
        │
        ▼
[Data Owner] ───> [Publish Kaggle Dataset vN] (phucdangg/legalqa-task2-clean-data)
                          │ (Pure data, zero code bundled)
                          ▼
[Training Owner] ─> [GitHub Repo: silent9669/LegalQA]
                          │
                          ▼
             [Pre-Push Check Gate: python scripts/pre_push_check.py]
                          │ (PASS)
                          ▼
             [Kaggle GPU Dual-T4 Smoke Gate] ───> kaggle_smoke_report.json = PASS
                          │
                          ▼
             [Freeze Tuple: Dataset Hash + Git SHA + Config + Model Rev]
                          │
                          ▼
             [Google Colab A100 Production Train] (BF16, Full Training)
                          │
                          ▼
             [Hugging Face Release: Model Adapter + Run Bundle]
```

---

## 2. Quickstart Guides

### A. Pre-Push Verification (Local Development)
Before pushing any commit to GitHub, run the canonical verification gate:
```bash
python scripts/pre_push_check.py
```
This automatically validates all configuration YAMLs, notebooks, dataset staging invariants, and runs the unit/contract test suite.

### B. Kaggle Dual-T4 CUDA Smoke Gate
The smoke test notebook runs on Kaggle with Dual NVIDIA T4 GPUs:
- **Notebook**: `notebooks/kaggle_smoke_test.ipynb` (mirrored to `kaggle_kernel/legalqa_gpu_pipeline.ipynb`)
- **Kaggle URL**: [kaggle.com/code/phucdangg/legalqa-training](https://www.kaggle.com/code/phucdangg/legalqa-training)
- **Config**: `configs/kaggle_smoke_t4.yaml`
- **Output**: Generates `/kaggle/working/kaggle_smoke_report.json` with PASS status for the worst-case and endurance probes.

### C. Google Colab A100 Production Training
The production training notebook executes full training on NVIDIA A100:
- **Notebook**: `notebooks/colab_train_a100.ipynb`
- **Config**: `configs/colab_train_a100.yaml`
- **CLI Runner**:
  ```bash
  python scripts/run_pipeline.py \
    --config configs/colab_train_a100.yaml \
    --data-dir /content/data/legalqa-task2-clean-data \
    --output-dir /content/runs/current \
    --require-smoke-pass kaggle_smoke_report.json \
    --allow-single-gpu
  ```
- **Output**: Packages the full Run Bundle and uploads trained QLoRA adapters to Hugging Face (`silent9669/legalqa-qwen2.5-3b-adapter`).

### D. Dataset Packaging & Direct Kaggle Release
To package a new dataset version:
```bash
# 1. Package and hash data files (ensures zero code is bundled)
python scripts/package_kaggle_dataset.py --source-dir kaggle_dataset/staged

# 2. Validate schema and referential integrity
python scripts/validate_dataset.py --data-dir kaggle_dataset/staged

# 3. Direct upload via Kaggle CLI
cd kaggle_dataset/staged && kaggle datasets version -m "Release clean Task 2 data" -p .
```

---

## 3. Directory Layout

```text
LegalQA/
├── configs/
│   ├── kaggle_smoke_t4.yaml         # Dual-T4 smoke test profile
│   ├── colab_train_a100.yaml        # Colab A100 production training profile
│   └── dataset_schema.yaml          # Canonical dataset schema invariants
├── src/task2/                       # Modular Python domain packages
├── notebooks/                       # Kaggle smoke & Colab A100 production notebooks
├── kaggle_dataset/staged/           # Pure data artifacts for direct Kaggle upload
├── tests/                           # Modular test suite (contracts, dataset, unit, gpu)
├── scripts/                         # Pre-push gate, validator, packager, runner
└── docs/                            # Authoritative architecture and operating runbooks
```

---

## 4. Documentation References

- [Architecture & Governance](docs/ARCHITECTURE.md) — System architecture, roles, and handoffs.
- [Kaggle Smoke Gate Runbook](docs/KAGGLE_SMOKE_GATE.md) — Dual-T4 probe execution and pass criteria.
- [Colab A100 Production Guide](docs/COLAB_A100_TRAINING.md) — A100 training, BF16 throughput, and HF release.
- [Dataset Specification](docs/DATASET_SPEC.md) — Data Owner contracts, parquet schemas, and release rules.
