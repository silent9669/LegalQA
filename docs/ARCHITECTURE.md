# LegalQA Task 2 — Four-Gate Reproducible Architecture Specification

Authoritative architecture specification for **LegalQA (Task 2)** in compliance with the **DSC 2026 Reproducible Training Workflow** (`https://dangphuc.notion.site/dscc`).

---

## 1. System Architecture & Four-Gate Promotion Ladder

```text
[Official BTC Data]
        │
        ▼
[Publish Kaggle Dataset vN] (phucdangg/legalqa-task2-clean-data, pure data)
        │
        ▼
[Gate 0: Local Verification] ────(FAIL)──> [Fix Code / Config]
  ./test.sh (scripts/pre_push_check.py --mode full, 261/261 PASS)
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

Colab tooling was completely retired (see HISTORY.md and docs/a100-readiness/); all GPU execution runs on Modal or Kaggle.
```

---

## 2. Configuration Authority Architecture

Configuration is strictly divided into two orthogonal layers:

1. **Algorithm Contract (`configs/task2/algorithm.yaml`)**:
   - Contains all score-affecting model and optimization parameters.
   - Model IDs and revision policies (`generator`, `reranker`, `dense`).
   - QLoRA rank ($r=16$), alpha ($\alpha=32$), dropout ($0.0$), and target projection modules.
   - Max sequence length ($2048$ tokens), quantization (`4bit_nf4`), double quantization.
   - Effective batch size ($8$), learning rate ($1.0\times 10^{-4}$), scheduler (`cosine`), warmup ($0.05$).
   - Training scope (`all_allowed_train`, `val_fold: null`).
   - Produces a single cryptographic digest: `algorithm_sha256`.

2. **Runtime Hardware Profiles (`configs/task2/runtime/*.yaml`)**:
   - `kaggle_t4x2.yaml`: 2x T4 GPUs, generator on `cuda:0`, retrieval on `cuda:1`, FP16, per-device batch 1, grad accum 8.
   - `colab_t4.yaml`: 1x T4 GPU, single-GPU placement on `cuda:0`, FP16, per-device batch 1, grad accum 8.
   - `modal_a100.yaml`: 1x A100 GPU, single-GPU placement on `cuda:0`, BF16, per-device batch 4, grad accum 2, gen batch 16 / rerank batch 128.
   - `colab_a100.yaml`: retired Colab profile, kept only for historical candidate hashes.
   - Runtime profiles are forbidden from modifying protected algorithm fields.
   - Effective batch size invariant ($batch \times accum = 8$) is strictly enforced.

---

## 3. Team Boundaries & Responsibilities

| Role | Scope | Canonical Output |
| --- | --- | --- |
| **Data Owner** | Prepares, cleans, validates, and releases the canonical Kaggle dataset. Schema, manifest, foreign-key integrity, and provenance. | Kaggle Dataset `phucdangg/legalqa-task2-clean-data` vN (pure data, zero code bundled). |
| **Release Lead** | Manages local pre-push checks, GitHub CI validation, candidate freezing, and gate report verification. | `candidate_manifest.json` and promotion authorizations. |
| **Training Owner (Kaggle)** | Executes Kaggle Dual-T4 CUDA gate (worst-case sequence probe, 30-step endurance probe, mini-eval). | `kaggle_t4x2_report.json` with status PASS. |
| **Training Owner (Dual-T4)** | Validates Dual-T4 CUDA execution (Kaggle or Modal Dual-T4), 2048-tok probe & endurance. | `kaggle_t4x2_report.json`. |
| **Training Owner (Colab A100)** | Runs in-session micro-probe, full all-data production training (`val_fold=None`), and exports audited run bundle. | Trained QLoRA adapter + run bundle uploaded to Hugging Face Hub under `runs/<run_id>/`. |

---

## 4. Immutable Candidate Manifest

Frozen by `scripts/freeze_candidate.py`:
```text
CandidateManifest:
├── schema_version: 1
├── candidate_id: <16-hex>
├── task: "task2"
├── git_repository: "https://github.com/silent9669/LegalQA.git"
├── git_commit_sha: "<40-hex>"
├── dataset:
│   ├── slug: "phucdangg/legalqa-task2-clean-data"
│   ├── version: 1
│   └── manifest_sha256: "<64-hex>"
├── algorithm_sha256: "<64-hex>"
├── runtime_profile_sha256:
│   ├── kaggle_t4x2: "<64-hex>"
│   ├── colab_t4: "<64-hex>"
│   └── colab_a100: "<64-hex>"
├── config_bundle_sha256: "<64-hex>"
├── models:
│   ├── generator: {id: "Qwen/Qwen2.5-3B-Instruct", revision: "<sha>"}
│   ├── reranker: {id: "BAAI/bge-reranker-v2-m3", revision: "<sha>"}
│   └── dense: {id: "CODE4LIFEOFFICIAL/huydang-dek21-embedding-v2", revision: "<sha>"}
├── dependency_lock_sha256: "<64-hex>"
├── seed: 42
└── created_at_utc: "<ISO-8601>"
```

---

## 5. Standardized Production Run Bundle

Every successful A100 training run produces an audited bundle:
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
