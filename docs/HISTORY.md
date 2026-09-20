# LegalQA Task 2 — Architecture & Evolution History

A chronological history of architecture decisions, empirical findings, and lessons learned across LegalQA Task 2 iterations.

---

## 1. Timeline of Major Iterations

### v6 Baseline (Historical Reported Score: ~0.5486)
- **Architecture**: Flat JSON configs (`run_config_v6.json`), unbatched sequential prediction, standard BM25 + DEk21 dense search.
- **Issues Identified**:
  - Lack of exact cryptographic hash linkage between configs, data, and models.
  - Silent answer truncation when sequence lengths exceeded standard defaults.
  - Per-step GPU cache clearing causing substantial execution slowdown.
  - Vulnerability to dataset permutation and misalignment.

### Re-Architecture (Tasks 1–11 / Notion DSC 2026 Specification)
- **Separation of Concerns**:
  - `algorithm.yaml`: Authoritative, score-affecting hyperparameters (Hashed to `algorithm_sha256`).
  - `runtime/*.yaml`: Hardware-specific device mappings and batch configurations.
- **Strict Cryptographic Gates**:
  - `CandidateManifest`: Canonical SHA-256 over algorithms, runtime profiles, clean dataset manifests, and 40-hex model commits.
  - Chained gate verification: `kaggle_t4x2` -> `colab_t4` -> `a100_micro_probe` -> `full`.
  - Offline scorer separation: local evaluations keep official public scores `None` until genuine receipt verification.

### Transition to Modal A100 & Production Scaling
- **Harnessing Modal Serverless GPU**:
  - Google Colab was deprecated for production training due to interactive OAuth challenges, session timeout constraints, and lack of reproducible volume persistence.
  - Modal A100-40GB selected as the unified training and inference environment.
  - Added persistent volume caching (`legalqa-data-vol`, `legalqa-runs-vol`) to eliminate repeated dataset and index downloads.
- **Training Duration & Overfitting Optimization**:
  - Training scaled from 4 epochs down to 1 epoch.
  - Avoids memorization on small legal QA datasets while cutting training wall-clock time by ~75%.
  - Memory callback configured to avoid continuous CUDA synchronization stalls during full training.
- **Private Round Submission Alignment**:
  - Teammate submission failure (`submission-3.json`) diagnosed: submission contained 1,000 public queries instead of the 1,918 private queries expected by the portal, causing `eval_qa` count mismatch exception.
  - Default inference target bound strictly to `private-official.json` with 1,918-query parity contracts and automated binary ZIP validation.
- **Automated Hub Release**:
  - Automatic packaging and direct datacenter upload to Hugging Face Hub (`dangphuc2109/legalqa-qwen2.5-3b-adapter`) under immutable paths (`runs/<run_id>/`).
  - Strict authorship proof generated from measured trainer outputs and verified gate hashes.

### Modal-Only Migration (2026-09-20)
- **Colab retired**: `colab_remote_entry.py`, `launch_colab_training.py`, and `colab_a100_train.ipynb` removed; all GPU execution runs on Modal (gates) or Kaggle (smoke).
- **Stage names are hardware profiles, not vendors**: the `colab_t4` gate stage now executes on Modal Tesla T4 with the identical runtime profile and DAG position (`kaggle_t4x2` → `colab_t4` → `a100_micro_probe` → full). No DAG bypass was granted.
- **Dense index quarantined**: checked-in `embeddings.npy` failed identical-pair self-consistency (cosine ~0.0012); excluded from the dataset release, cold-rebuilt on-GPU with pinned revision + gate.
- **BM25 rebuilt**: fresh order-bound bm25s index (k1=1.5/b=0.75 continuity, 5/5 top-5 agreement with predecessor) published in the dataset; loader enforces doc_ids order.
- **Model pins corrected**: previous pins 404'd upstream; verified live SHAs now used.

### DAG Migration: Direct Kaggle-to-A100 Chain (2026-09-20)
- **Decision**: the A100 microprobe accepts a `kaggle_t4x2` parent directly; the `colab_t4` T4 rehearsal stays available as an optional stage (`modal run --stage colab_t4`) but no longer blocks promotion.
- **Rationale**: Modal-only team; single-GPU placement is probed by the microprobe itself (cuda:0); a cheap T4 rehearsal remains one command away.
- **Single source of truth**: `GATE_PARENTS` in `scripts/run_gpu_gate.py`; `validate_parent_gate`, the gate runner, and `modal_app` all derive from it (covered by `test_parent_rules_have_single_source_of_truth`).
