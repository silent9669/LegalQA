# Release and Packaging — Runtime API 16

## Why API16 is required

V16 changes packaged runtime behavior:

- new generator module;
- Liger dependency;
- new loss backend;
- new execution profiles;
- new runtime contracts.

Therefore the notebook must reject API15 packages.

## Required bindings

Set:

```text
configs/runtime_api.yaml
  runtime_api_version: 16

src/task2/runtime_integrity.py
  EXPECTED_RUNTIME_API_VERSION = 16

kaggle notebook
  REQUIRED_RUNTIME_API_VERSION = 16
```

## Manifest parity

The final clean V16 HEAD must appear in:

```text
kaggle_dataset/staged/dataset_manifest.json
kaggle_dataset/staged/code_manifest.json
kaggle_dataset/staged/code/LegalQA/code_manifest.json
```

All three:

```text
runtime_api_version = 16
git_sha = exact 40-character final HEAD
```

## Inherited artifact check

Before packaging:

```bash
python scripts/verify_inherited_dataset.py --root <source-root>   --manifest artifacts/inherited/dataset_manifest_v15.json
```

Do not package if hashes differ.

## Package contents

Must include:

```text
legal_chunks.parquet
qa_unique.parquet
known_qa.json
qa_citations.parquet
retrieval_labels.parquet
fold_assignments.parquet
reranker_training_pairs.parquet
public-official.json
indexes/bm25/**
indexes/dek21/**
code/LegalQA/**
dataset_manifest.json
code_manifest.json
```

## Packaged code requirements

The package must include:

```text
src/task2/generation/*
src/task2/pipeline/*
src/task2/runtime/*
exact Kaggle requirements with liger-kernel==0.8.2
runtime API16 config
```

## Secret hygiene

Fail packaging when any staged file contains:

```text
HF_TOKEN=
KAGGLE_KEY=
API key literal
private credential
.env
```

Do not package logs containing secret values.

## Remote deployment audit

After uploading a new Kaggle dataset version:

1. inspect remote dataset version metadata;
2. inspect remote `dataset_manifest.json`;
3. inspect remote root code manifest;
4. inspect remote nested code manifest;
5. verify API16/final SHA parity;
6. verify `requirements-kaggle.txt`/canonical requirements contains Liger pin;
7. verify generator source hash matches manifest;
8. verify all BM25/DEk21 files exist.

Only then:

```text
READY FOR USER MANUAL KAGGLE GENERATOR PROBE
```

## Notebook publishing rule

Agent may prepare notebook source in GitHub.

The user controls Kaggle notebook execution/publishing. Do not trigger Save & Run All through automation.
