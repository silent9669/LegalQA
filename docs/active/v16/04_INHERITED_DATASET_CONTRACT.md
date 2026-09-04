# Inherited Dataset Contract

## Purpose

V16 must inherit the verified V15 data and indexes byte-for-byte. Runtime refactoring must not silently regenerate or mutate competition data.

## Authoritative inherited dataset

```text
slug: legalqa-task2-clean-data
owner: phucdangg
profile: final_training
source runtime API: 15
source Git SHA: 151313fc3126615ec11c08ca68f154d5b0c5406f
```

## Core file hashes

| File | SHA256 | Manifest size |
|---|---|---:|
| `legal_chunks.parquet` | `6b34c7f338871f3786b94f947c89f6eff000f5bf21785ebee59dc267a00c164e` | 327.24 MB |
| `qa_unique.parquet` | `5f0055113346cba7faa1ee92f105b29d0570473d3033c214a1f52e8b77176e7a` | 6.17 MB |
| `known_qa.json` | `1b453f36f5db10d913f6e1b04b48e081170ff9b4f88a3cb008abb376c971d00c` | 29.57 MB |
| `qa_citations.parquet` | `0772c6b3e0d4c1b72711331ed924c0a753c94a49e2f0f26aa5d7891f20b22e19` | 0.51 MB |
| `retrieval_labels.parquet` | `f11d0e46185e06d2aa8b997817877694938f51034b678fa071665891cb73004c` | 0.71 MB |
| `fold_assignments.parquet` | `1422fd1587fce02941c1e31846d6e2f4dcb9b88b6b65890686137aad85a9f0c3` | 0.45 MB |
| `reranker_training_pairs.parquet` | `d67274b1b5cb2fd4892b55a327ed9b30de26140692e103c78eec38cd3ba4a126` | 13.33 MB |
| `public-official.json` | `5f68ca901cb20798559538bef60fa7c32bd7d0df59f5bf31a37eb220c9e00df5` | 0.18 MB |

The downloaded `qa_unique.parquet` and `retrieval_labels.parquet` copies were independently SHA256-checked against these manifest values during this refresh audit.

## BM25 inherited index

Must contain exactly the manifest-declared 7 files:

```text
indexes/bm25/
├── bm25_manifest.json
├── corpus_meta.parquet
└── bm25s_index/
    ├── params.index.json
    ├── vocab.index.json
    ├── indices.csc.index.npy
    ├── data.csc.index.npy
    └── indptr.csc.index.npy
```

## DEk21 inherited index

Must contain:

```text
indexes/dek21/
├── dek21_manifest.json
├── dense_manifest.json
└── embeddings.npy
```

Known verified matrix shape from runtime preflight:

```text
(801863, 768)
dtype=float16
```

## Data invariants

The V16 verifier must fail if any of the following change unexpectedly:

```text
legal chunk count != 801,863
public query count != 1,000
BM25 document count != 801,863
DEk21 row count != 801,863
DEk21 dimension != 768
core file hash mismatch
missing fold_id in QA data
missing positive labels required for SFT/reranker training
```

## V16 packaging behavior

API16 packaging should copy/reuse these data bytes and build **new manifests** containing:

```text
runtime_api_version = 16
git_sha = final V16 HEAD
```

The data hashes themselves remain the V15 hashes above unless the user explicitly approves a dataset regeneration project.

## Forbidden actions during refresh

Do not:

- regenerate legal chunks;
- rebuild BM25;
- recompute DEk21 embeddings;
- rewrite QA fold assignment;
- remine reranker pairs;
- normalize/retokenize gold answers in stored data;
- overwrite the inherited dataset in place.

## Provenance layout

Keep a small immutable provenance copy in the repository:

```text
artifacts/inherited/dataset_manifest_v15.json
```

The large binary data stays outside Git and is attached through Kaggle/Drive.

## Verification command contract

Add:

```bash
python scripts/verify_inherited_dataset.py   --root <dataset-root>   --manifest artifacts/inherited/dataset_manifest_v15.json
```

Exit `0` only when all hashes and structural checks pass.
