# LegalQA Task 2 — Five-Gate Reproducible Production Pipeline (DSC 2026)

End-to-end Legal Question Answering (LegalQA) system for DSC 2026 Task 2, engineered for fail-closed reproducibility, GPU credit efficiency, and verifiable cryptographic provenance across a five-gate promotion ladder.

- **Competition Parameter Budget**: Total learned parameters strictly $< 4.0\text{B}$ (`Qwen/Qwen2.5-3B-Instruct` base).
- **Data Constraint**: Task 2 official organizer data only (`phucdangg/legalqa-task2-clean-data`, pure data, zero code bundled).
- **Authoritative Configuration**: Two-layer config architecture separating score-affecting `algorithm.yaml` from hardware runtime profiles (`runtime/*.yaml`).
- **Five-Gate Promotion Ladder**:
  1. **Gate 0 (Local)**: CPU pre-push verification (`scripts/pre_push_check.py --mode fast|full`).
  2. **Gate 1 (GitHub CI)**: Exact-commit SHA verification across Python 3.10/3.12, exact GPU user-space stack, and security checks.
  3. **Gate 2 (Kaggle Dual-T4)**: Real CUDA 2048-token worst-case sequence probe, 30-step endurance probe, and mini-eval -> `kaggle_t4x2_report.json`.
  4. **Gate 3 (Google Colab Single-T4)**: Colab CLI/session/upload/download/exact checkout/versioned data lifecycle -> `colab_t4_report.json`.
  5. **Gate 4 (Google Colab A100)**: 2-step in-session micro-probe -> full all-data production train (`val_fold=None`) -> audited immutable Run Bundle release to Hugging Face Hub under `runs/<run_id>/`.
- **Loss Backend**: Liger fused-linear cross-entropy (`liger-kernel==0.8.2`, `loss_type=nll`).

---

## 1. Five-Gate Promotion Ladder

```text
[Official BTC Data]
        │
        ▼
[Publish Kaggle Dataset vN] (phucdangg/legalqa-task2-clean-data, pure data)
        │
        ▼
[Gate 0: Local Verification] ────(FAIL)──> [Fix Code / Config]
  python scripts/pre_push_check.py --mode full
        │ (PASS)
        ▼
[Gate 1: GitHub CI on Exact SHA] ─(FAIL)──> [Fix CI Matrix]
  python scripts/verify_ci_status.py --sha <sha>
        │ (PASS)
        ▼
[Candidate Freeze] ───────────────────────> artifacts/candidates/<candidate_id>/candidate_manifest.json
  python scripts/freeze_candidate.py
        │
        ▼
[Gate 2: Kaggle Dual-T4 CUDA Gate] ─(FAIL)─> [Fix CUDA/VRAM Bug]
  scripts/run_gpu_gate.py --stage kaggle_t4x2
        │ (PASS) ──> kaggle_t4x2_report.json
        ▼
[Gate 3: T4 Gate on Modal Tesla T4] ───(FAIL)─> [Fix T4 Bootstrap]
  ./scripts/run_modal.sh colab_t4 (stage name kept for chain continuity)
        │ (PASS) ──> colab_t4_report.json (parent: kaggle_t4x2)
        ▼
[Gate 4: Modal A100 Production] ──(FAIL)─> [Fix A100 Micro-Probe]
  ./scripts/run_modal.sh micro_probe, then full ...
        │ (PASS)
        ├─> A100 Micro-Probe (2 steps) -> a100_micro_probe_report.json
        ├─> Full Production Training (All allowed data, val_fold=None)
        ├─> Audited Run Bundle & Checksums Generation
        └─> Hugging Face Hub Immutable Release: runs/<run_id>/
```

---

## 2. Quickstart Guides

### A. Gate 0: Local Pre-Push Verification
Before pushing any commit to GitHub, run the canonical pre-push verification script:
```bash
# Fast mode (CPU checks, syntax, configs, secret scan, git hygiene, notebooks)
python scripts/pre_push_check.py --mode fast

# Full mode (Complete contract, provenance, unit, and integration test suite)
python scripts/pre_push_check.py --mode full
```

### B. Gate 1: GitHub CI Verification
Verify that the exact Git commit SHA passed all 5 required CI jobs:
```bash
python scripts/verify_ci_status.py --sha <commit_sha>
```

### C. Freeze Candidate Manifest
Once code is committed and pushed, freeze an immutable candidate manifest:
```bash
python scripts/freeze_candidate.py
```
This cryptographically binds the Git commit SHA, Kaggle dataset version/manifest, Hugging Face model commit SHAs, algorithm hash, runtime profile hashes, and `constraints-gpu.txt` dependency lock.

### D. Gate 2: Kaggle Dual-T4 CUDA Gate
Run the worst-case 2048-token probe and 30-step endurance test on Kaggle Dual NVIDIA T4 GPUs:
- **Notebook**: `notebooks/kaggle_smoke.ipynb`
- **Config**: `configs/task2/runtime/kaggle_t4x2.yaml`
- **Output**: Generates `/kaggle/working/kaggle_t4x2_report.json` with cryptographic telemetry.

### E. Gate 3: T4 Gate on Modal Tesla T4
Runs the `colab_t4` stage (1×T4 smoke config; stage names are hardware
profiles, not vendors) on Modal, chained from the Kaggle report:
```bash
./scripts/run_modal.sh colab_t4
```
Produces `artifacts/gates/<candidate_id>/colab_t4_report.json` chained from the Kaggle report.

### F. Gate 4: Modal A100 Production Training
Executes 2-step micro-probe, then full all-data production training (`val_fold=None`), packages the audited run bundle, and publishes to Hugging Face Hub:
```bash
./scripts/run_modal.sh micro_probe
./scripts/run_modal.sh full --test-path private-official.json
```

---

## 3. Directory Layout

```text
LegalQA/
├── configs/
│   ├── dataset_schema.yaml              # Canonical dataset schema invariants
│   └── task2/
│       ├── algorithm.yaml               # Authoritative score-affecting hyperparameters
│       └── runtime/
│           ├── kaggle_t4x2.yaml         # Dual-T4 hardware profile
│           ├── colab_t4.yaml            # Single-T4 gate profile (executes on Modal T4)
│           └── modal_a100.yaml          # Modal A100 production profile
│
├── src/task2/
│   ├── config/                          # Typed config schemas and canonical loader
│   ├── provenance/                      # Candidate manifests, gate reports, checksums, run bundles
│   ├── pipeline/                        # Profile resolution and pipeline execution
│   └── generation/                      # QLoRA trainer, SFT datasets, memory hygiene, Liger backend
│
├── scripts/
│   ├── pre_push_check.py                # Local Gate 0 (--mode fast|full)
│   ├── freeze_candidate.py              # Candidate manifest freeze utility
│   ├── verify_ci_status.py              # GitHub Actions CI check run validator
│   ├── run_gpu_gate.py                  # Standalone GPU gate runner
│   ├── modal_app.py                     # Modal A100/T4 remote runner
│   ├── run_modal.sh                     # One-click Modal dispatcher
│   ├── audit_parameters.py              # Competition parameter budget auditor (< 4.0B)
│   └── run_pipeline.py                  # Standalone pipeline CLI runner
│
├── notebooks/
│   └── kaggle_smoke.ipynb               # Kaggle Dual-T4 thin launcher
│
├── constraints-gpu.txt                  # Exact GPU user-space dependency lock
├── requirements.txt                     # Core dependencies
└── tests/                               # Comprehensive test suite (contracts, provenance, unit, integration)
```
