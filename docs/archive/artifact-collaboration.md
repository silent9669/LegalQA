# LegalQA Artifact Governance and Collaboration Workflow

This document defines the storage boundaries, team collaboration procedures, and bootstrap protocols for the DSC 2026 Task 2 Vietnamese LegalQA pipeline.

---

## 1. Storage Boundaries

To maintain reproducibility, prevent repository bloat, and comply with competition rules, repository assets are strictly separated across three tiers:

| Tier | Host | Contents | Versioning & Integrity |
|---|---|---|---|
| **Code & Manifests** | **GitHub** | Source code (`src/`), CLI scripts (`scripts/`), configurations (`configs/`), test suite (`tests/`), deterministic JSON manifests (`artifacts/manifests/`), checksums, and documentation. | Git commits, PR reviews, tags. |
| **Model Checkpoints & Adapters** | **Private Hugging Face Hub** | Fine-tuned LoRA adapters, model checkpoints, tokenizer configurations. | Pinned 40-character revision SHA, Model Manifest (`models.json`). |
| **Data Bundles & Heavy Artifacts** | **Cloudflare R2 / S3-compatible Object Storage** | Raw context archives (`selected-contexts.zip`), canonical chunk parquet (`legal_chunks.parquet`), dense vector indexes, full OOF prediction traces, submission packages. | Streaming SHA-256 digests, Artifact Manifest (`artifacts.json`). |
| **Git LFS** | *Optional Fallback* | Small binary fixtures if needed. Not preferred for multi-GB frequently updated indexes or weights. | Git LFS pointers. |

> **Strict Rule:** Never commit multi-GB model weights, vector indexes, or raw corpora to Git history. Never commit credentials, private tokens, or pre-signed URLs.

---

## 2. Authoritative Artifact Hierarchy

All local runtime and evaluation components read exclusively from the canonical `artifacts/` tree:

```text
artifacts/
├── raw/                         # Official Task 2 organizer inputs
│   ├── train.json               # 7,000 QA training pairs
│   ├── warmup.json              # 500 QA warmup pairs
│   ├── public-official.json     # 1,000 public test questions
│   ├── selected-contexts.zip    # Official archive of legal context documents
│   └── selected-contexts/       # Extracted context hierarchy
├── chunks/
│   ├── legal_chunks.parquet     # Canonical runtime corpus (365,046 rows, 18 columns, 425.5 MB)
│   └── chunks_output.jsonl      # Reconstruction intermediate (retained for lineage)
├── data/
│   ├── qa_unique.parquet        # 7,113 canonical QA pairs with normalized questions
│   └── known_qa.json            # Exact QA memory dictionary (by_id & by_question)
├── labels/
│   └── retrieval_labels.parquet # 7,113 retrieval supervision records
├── manifests/
│   ├── artifacts.json           # SHA-256 manifest of all 8,544 canonical files
│   └── models.json              # Model parameter manifest and budget audit
├── submissions/
│   ├── submission.json          # Generated public predictions
│   └── submission.json.zip      # CodaBench submission package
└── archive/                     # Quarantined legacy artifacts (ignored by runtime & git)
    └── trung-legacy/
        └── provenance.json      # Provenance metadata for quarantined legacy files
```

---

## 3. Team Member Bootstrap Flow

When setting up a fresh development workspace or syncing with teammates:

### Step 1: Environment Setup
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Step 2: Download or Restore Artifact Bundles
If downloading remote bundles from team R2/S3 storage, use the atomic checksum-verifying downloader:
```bash
# Verify and install an individual bundle atomically
python scripts/verify_download.py \
  --source "https://<storage-domain>/bundles/legal_chunks.parquet" \
  --destination artifacts/chunks/legal_chunks.parquet \
  --sha256 9b5ac8d12a0b43bcd09a0722ed2b3d4c58fac70f7c0d4a695c445838dd746876 \
  --size-bytes 425466880

# Or verify from a dedicated remote bundle manifest JSON
python scripts/verify_download.py --manifest path/to/remote_bundle_manifest.json
```

### Step 3: Audit Canonical Artifact Integrity
Run the structural audit to verify that all canonical files, schemas, and counts match the exact specification:
```bash
python scripts/audit_artifacts.py --output artifacts/manifests/artifacts.json
```

### Step 4: Run Automated Test Suite
```bash
pytest tests/ -v
```

---

## 4. Model Parameter Budget Accounting

Competition rules enforce a strict upper bound:
$$\sum \text{Learned Parameters Loaded at Inference} < 4{,}000{,}000{,}000$$

### Primary Stack Accounting:
- **Dense Retriever (`Qwen/Qwen3-Embedding-0.6B`)**: `595,776,512` parameters
- **Neural Reranker (`Qwen/Qwen3-Reranker-0.6B`)**: `595,776,512` parameters
- **Generator (`Qwen/Qwen3-1.7B`)**: `2,031,739,904` parameters (conservative upper bound)
- **Total Loaded Learned Parameters**: `3,223,292,928` ($< 4\text{B}$, exactly compliant)

### Accounting Rules:
1. Every independently loaded model checkpoint, adapter, and task-specific head is counted additively.
2. Architecture sharing is not deduplicated unless models share the exact same in-memory weights.
3. Unloaded controls (e.g. baseline BGE-M3 or Qwen2.5) are tracked with `loaded_at_inference: false` and do not count toward inference totals.
4. Model manifests are validated automatically by `load_model_manifest()` in `src/utils/manifest.py`.
