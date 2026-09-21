# LegalQA Task 2 — Reproducible Production Pipeline (DSC 2026)

End-to-end Legal Question Answering (LegalQA) system for DSC 2026 Task 2, engineered for fail-closed reproducibility, GPU credit efficiency, and verifiable cryptographic provenance across a rigorous gate promotion ladder.

- **Competition Parameter Budget**: Total learned parameters strictly $< 4.0\text{B}$ (`Qwen/Qwen2.5-3B-Instruct` base). Compliant: 3,788,891,136 / 4,000,000,000 (211M safe margin).
- **Data Constraint**: Task 2 official organizer data only (`phucdangg/legalqa-task2-clean-data`, pure data, zero code bundled).
- **Authoritative Configuration**: Two-layer config architecture separating score-affecting `algorithm.yaml` from hardware runtime profiles (`runtime/*.yaml`).
- **Autonomous Gate Promotion Ladder**:
  1. **Gate 0 (Local)**: CPU pre-push verification (`./test.sh` / `scripts/pre_push_check.py --mode full`, 261/261 tests passing).
  2. **Gate 1 (GitHub CI)**: Exact-commit SHA verification across Python 3.10/3.12, exact GPU user-space stack, and security checks.
  3. **Gate 2 (Dual-T4 CUDA Gate)**: Real CUDA 2048-token worst-case sequence probe, 30-step endurance probe, and mini-eval -> `kaggle_t4x2_report.json` (executes on Kaggle Dual-T4 or Modal Dual-T4 via `--stage kaggle_t4x2`).
  4. **Gate 3 (Modal A100 Production)**: 2-step in-session micro-probe -> full 2-epoch production train (`val_fold=None`) -> private set inference (1,918 queries, native bfloat16, 1536 token ceiling, prompt-keyed persistent caching) -> audited immutable Run Bundle release to Hugging Face Hub under `runs/<run_id>/`.
- **Loss Backend**: Liger fused-linear cross-entropy (`liger-kernel==0.8.2`, `loss_type=nll`).
- **Active Production Candidate**: `0589ce35af4300d4` (bound to commit `ba1b370`).

---

## 1. Autonomous Gate Promotion Ladder

```text
[Official BTC Data]
        │
        ▼
[Publish Kaggle Dataset vN] (phucdangg/legalqa-task2-clean-data, pure data)
        │
        ▼
[Gate 0: Local Verification] ────(FAIL)──> [Fix Code / Config]
  ./test.sh (scripts/pre_push_check.py --mode full)
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
[Gate 2: Dual-T4 CUDA Gate] ──────(FAIL)─> [Fix CUDA/VRAM Bug]
  scripts/modal_app.py --stage kaggle_t4x2 (or notebooks/kaggle_smoke.ipynb)
        │ (PASS) ──> kaggle_t4x2_report.json
        ▼
[Gate 3: Modal A100 Production] ──(FAIL)─> [Fix A100 Micro-Probe]
  ./scripts/run_modal.sh full --test-path private-official.json
        │ (PASS)
        ├─> A100 Micro-Probe (2 steps) -> a100_micro_probe_report.json
        ├─> Full Production Training (2 Epochs, val_fold=None, group_by_length=True)
        ├─> Private Set Inference (1,918 queries, native bfloat16, 1536 token ceiling)
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

# Full mode (Complete contract, provenance, unit, and integration test suite — 261 tests)
./test.sh
```

### B. Gate 1: GitHub CI Verification
Verify that the exact Git commit SHA passed all required CI jobs:
```bash
python scripts/verify_ci_status.py --sha <commit_sha>
```

### C. Freeze Candidate Manifest
Once code is committed and pushed, freeze an immutable candidate manifest:
```bash
python scripts/freeze_candidate.py
```
This cryptographically binds the Git commit SHA, Kaggle dataset version/manifest, Hugging Face model commit SHAs, algorithm hash, runtime profile hashes, and `constraints-gpu.txt` dependency lock.

### D. Gate 2: Dual-T4 CUDA Gate
Run the worst-case 2048-token probe and 30-step endurance test on Dual NVIDIA T4 GPUs:
- **On Modal (One-Click)**:
  ```bash
  modal run scripts/modal_app.py --stage kaggle_t4x2
  ```
- **On Kaggle**: `notebooks/kaggle_smoke.ipynb` (Dual-T4 accelerator).
- **Output**: Generates `artifacts/gates/<candidate_id>/kaggle_t4x2_report.json` with cryptographic telemetry.

### E. Gate 3: Modal A100 Production Training & Submission
Executes 2-step micro-probe, then full all-data production training (`val_fold=None`), packages the audited run bundle, publishes to Hugging Face Hub, and auto-downloads `submission.json.zip`:
```bash
# Autonomous one-click dispatcher (auto-resolves missing parent gate reports)
./scripts/run_modal.sh full --test-path private-official.json
```

---

## 3. Key Performance & Architecture Highlights

1. **Dual-Part Statutory Answer Assembly**:
   - Pairs clean reasoning prose with the primary statutory citation block (`Trích dẫn quy định:\n`).
   - SFT target trains the model end-to-end on full 3-part statutory answers (Căn cứ pháp lý -> Trích dẫn điều luật -> Kết luận).
2. **Native Bfloat16 Inference & 1536-Token Ceiling**:
   - Eliminates 4-bit NF4 dequantization overhead on A100; merges LoRA adapter weights for 3.5x faster decode throughput.
   - Expanded token ceiling covers 99.5% of gold legal answers, pushing metric recall ceiling to 0.9979.
3. **Training Efficiency via Length Grouping**:
   - `train_sampling_strategy='group_by_length'` groups sequences by length, cutting padding waste by ~50%.
   - Filtered out 2,354 evidence-free examples to prevent hallucination on ungrounded questions.
4. **Resilient Pipeline & Prompt-Keyed Caching**:
   - `predict_batch` streams generated prose into `gen_raw_cache.jsonl` keyed by exact prompt SHA-256; interrupted runs resume without wasting GPU quota.
   - Degenerate answer guards prevent empty or bare header-only answers from contaminating submissions.

---

## 4. Directory Layout

```text
LegalQA/
├── configs/
│   ├── dataset_schema.yaml              # Canonical dataset schema invariants
│   └── task2/
│       ├── algorithm.yaml               # Authoritative score-affecting hyperparameters (2 epochs)
│       └── runtime/
│           ├── kaggle_t4x2.yaml         # Dual-T4 hardware profile
│           └── modal_a100.yaml          # Modal A100 production profile (bfloat16, 1536 tokens)
│
├── src/task2/
│   ├── config/                          # Typed config schemas and canonical loader
│   ├── provenance/                      # Candidate manifests, gate reports, checksums, run bundles
│   ├── pipeline/                        # Profile resolution and pipeline execution
│   ├── generation/                      # QLoRA trainer, SFT datasets, memory hygiene, Liger backend
│   ├── candidates.py                    # Candidate generation and dual-part statutory assembly
│   ├── predict.py                       # Batched inference with prompt-keyed persistent caching
│   └── selector.py                      # Candidate selector and fallback hierarchy
│
├── scripts/
│   ├── pre_push_check.py                # Local Gate 0 (--mode fast|full)
│   ├── freeze_candidate.py              # Candidate manifest freeze utility
│   ├── verify_ci_status.py              # GitHub Actions CI check run validator
│   ├── run_gpu_gate.py                  # Standalone GPU gate runner
│   ├── modal_app.py                     # Modal A100 / Dual-T4 remote runner
│   ├── run_modal.sh                     # Autonomous one-click Modal dispatcher
│   ├── sweep_assembly.py                # Offline assembly strategy sweep (official NLTK METEOR)
│   └── audit_parameters.py              # Competition parameter budget auditor (< 4.0B)
│
├── notebooks/
│   ├── DSC2026_LegalQA_Pipeline_v13.ipynb # Reference v13.2 inference & lab pipeline
│   └── kaggle_smoke.ipynb               # Kaggle Dual-T4 thin launcher
│
├── constraints-gpu.txt                  # Exact GPU user-space dependency lock
├── requirements.txt                     # Core dependencies
└── tests/                               # 261 automated tests (contracts, provenance, unit, integration)
```
