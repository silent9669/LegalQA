# LegalQA Colab-First GPU Smoke Workflow

This document outlines the standard operational procedure for verifying GPU-dependent training and quantization on a single Tesla T4 GPU in Google Colab before manual execution on Kaggle Dual T4 GPUs.

## Overview

Kaggle Dual-T4 execution is reserved exclusively for manual execution by the user. To prevent runtime failures and regressions, all GPU-dependent model changes undergo single-T4 smoke testing on Google Colab after passing GitHub Actions CI.

```text
CODE CHANGE
   ↓
LOCAL TESTS
   ↓
PUSH GITHUB
   ↓
GITHUB ACTIONS GREEN
   ↓
COLAB T4 QUICK COMPONENT SMOKE (3 steps)
   ↓
COLAB T4 FULL COMPONENT SMOKE (30 steps)
   ↓
STATIC RELEASE/DATA AUDIT
   ↓
USER-ONLY MANUAL KAGGLE T4x2 INTEGRATION SMOKE
   ↓
USER-ONLY FULL KAGGLE TRAINING
```

---

## Google Colab Smoke Procedure

1. **Push Source to GitHub**: Commit all code, tests, and configuration changes to `origin main`.
2. **Verify Green CI**: Confirm all GitHub Actions matrix jobs pass.
3. **Open Google Colab**: Navigate to Google Colab using the configured browser MCP or interactive browser session.
4. **Select Accelerator**: Select **Runtime → Change runtime type → T4 GPU**.
5. **Clone Repository**:
   ```bash
   !git clone https://github.com/silent9669/LegalQA.git
   %cd LegalQA
   !git checkout <FINAL_HEAD>
   ```
6. **Install Runtime Dependencies**:
   Install exact Kaggle user-space ML packages without modifying Colab's baseline PyTorch and CUDA runtime:
   ```bash
   !pip install --upgrade-strategy only-if-needed -r requirements-colab-smoke.txt
   ```
7. **Mount or Copy Required Data Files**:
   Ensure `/content/legalqa-data` contains:
   - `qa_unique.parquet`
   - `retrieval_labels.parquet`
   - `legal_chunks.parquet`
   - *(Optional for reranker smoke)*: `reranker_training_pairs.parquet`
8. **Run Generator Quick Smoke (3 Steps)**:
   ```bash
   !python scripts/run_colab_smoke.py \
       --data-root /content/legalqa-data \
       --component generator \
       --mode quick
   ```
9. **Run Generator Full Smoke (30 Steps)**:
   If quick smoke passes, run:
   ```bash
   !python scripts/run_colab_smoke.py \
       --data-root /content/legalqa-data \
       --component generator \
       --mode full
   ```
10. **Run Reranker Smoke (When Applicable)**:
    If code modifications touch `train_reranker.py` or reranker configs, run:
    ```bash
    !python scripts/run_colab_smoke.py \
        --data-root /content/legalqa-data \
        --component reranker \
        --mode full
    ```
11. **Capture Report**:
    Verify that `colab_smoke_report.json` records `status: "PASS"`, `peak_vram_mb`, and `adapter_reload: "pass"`.
12. **Report Status**: Report `COLAB GPU SMOKE PASS` or `BLOCKED`.
13. **Strict No-Kaggle Isolation**: Do **NOT** interact with Kaggle kernels or push notebooks.
14. **Manual Kaggle Execution**: The user manually triggers the Kaggle Dual-T4 notebook.
