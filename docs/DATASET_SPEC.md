# LegalQA Task 2 — Canonical Dataset Specification

Authoritative specification for the canonical Task 2 dataset release (`A2. Task 2 — LegalQA Data Owner`).

---

## 1. Governance & Data Ownership

- **Dataset Slug**: `phucdangg/legalqa-task2-clean-data`
- **Data Owner**: Controls parsing, cleaning, normalization, citation links, fold assignments, and manifest generation.
- **Training Owner**: Consumes the immutable Kaggle dataset release without modifying parquet or json files in the notebook.

### Invariant: Pure Data Release (Zero Code)
The Kaggle dataset package contains **only** clean data files and cryptographic metadata. **No application code (`src/`, `scripts/`, `.py`, `.sh`) may ever be packaged inside the Kaggle dataset release.**

---

## 2. Canonical Data Artifacts

| File Name | Format | Primary Key / Index | Description |
| --- | --- | --- | --- |
| `legal_chunks.parquet` | Parquet | `chunk_id` | Normalized legal law/article chunks. |
| `qa_unique.parquet` | Parquet | `qa_id` | Deduplicated QA pairs with raw & normalized text, answer, and `fold_id`. |
| `qa_citations.parquet` | Parquet | `qa_id` | Ground-truth document, article, and clause references. |
| `fold_assignments.parquet` | Parquet | `qa_id` | Deterministic 5-fold cross validation assignments. |
| `retrieval_labels.parquet` | Parquet | `qa_id` | Positive chunk targets and mined hard negatives. |
| `reranker_training_pairs.parquet` | Parquet | `qa_id` | Triplet pairs `(question, positive_chunk, negative_chunk)` for BGE training. |
| `known_qa.json` | JSON | `question` | Exact/near-match memory cache. |
| `public-official.json` | JSON | `id` | Organizer public test set queries. |
| `indexes/bm25/` | Directory | N/A | Prebuilt BM25S index files aligned with `legal_chunks`. |
| `indexes/dek21/` | Directory | N/A | Prebuilt DEk21 dense vector embeddings (`embeddings.npy`). |
| `dataset_manifest.json` | JSON | N/A | Cryptographic manifest with SHA-256 hashes and sizes for every artifact. |
| `dataset-metadata.json` | JSON | N/A | Kaggle dataset metadata for CLI upload. |

---

## 3. Schema Invariants & Referential Integrity

1. **Foreign Key Integrity**:
   - Every `qa_id` in `qa_citations`, `retrieval_labels`, and `fold_assignments` must exist in `qa_unique`.
   - Every `positive_chunk_id` in `retrieval_labels` and `reranker_training_pairs` must exist in `legal_chunks`.
2. **Deterministic Folds**:
   - `fold_id` spans integers `0..4`. Each `qa_id` is assigned to exactly one fold.
3. **No External Data**:
   - All text must derive strictly from official organizer data. Manual labeling or external augmentation is prohibited.

---

## 4. Packaging & Release Procedure

To prepare a new dataset version for direct upload to Kaggle:

```bash
# 1. Package clean dataset and compute SHA-256 manifest
python scripts/package_kaggle_dataset.py --source-dir kaggle_dataset/staged

# 2. Validate dataset against schema contract
python scripts/validate_dataset.py --data-dir kaggle_dataset/staged

# 3. Upload directly to Kaggle using Kaggle CLI
cd kaggle_dataset/staged
kaggle datasets version -m "Release Task 2 clean canonical data vN" -p .
```
