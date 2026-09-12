import os
import glob
import json
import pytest
import pandas as pd
from src.task2.dataset.validator import validate_dataset, compute_sha256

DATASET_DIR = "kaggle_dataset"
SCHEMA_PATH = "configs/dataset_schema.yaml"

@pytest.fixture
def dataset_data():
    if not os.path.exists(os.path.join(DATASET_DIR, "qa_unique.parquet")):
        pytest.skip("Dataset artifacts not present locally.")
    return DATASET_DIR

def test_dataset_integrity_validator(dataset_data):
    report = validate_dataset(data_dir=dataset_data, schema_path=SCHEMA_PATH)
    assert report["status"] == "PASS", f"Dataset validation failed: {report.get('errors')}"
    assert report["manifest_verified"] is True
    assert "code" not in report.get("detected_directories", [])

def test_zero_code_in_dataset(dataset_data):
    # Strictly zero code files or code directories in dataset package
    subdirs = [d for d in os.listdir(dataset_data) if os.path.isdir(os.path.join(dataset_data, d))]
    assert "code" not in subdirs, "Found code/ directory inside dataset staging"
    assert "src" not in subdirs, "Found src/ directory inside dataset staging"

    py_files = glob.glob(os.path.join(dataset_data, "**/*.py"), recursive=True)
    sh_files = glob.glob(os.path.join(dataset_data, "**/*.sh"), recursive=True)
    assert len(py_files) == 0, f"Found .py files in dataset: {py_files}"
    assert len(sh_files) == 0, f"Found .sh files in dataset: {sh_files}"

def test_parquet_referential_integrity(dataset_data):
    qa_df = pd.read_parquet(os.path.join(dataset_data, "qa_unique.parquet"))
    legal_df = pd.read_parquet(os.path.join(dataset_data, "legal_chunks.parquet"))
    citations_df = pd.read_parquet(os.path.join(dataset_data, "qa_citations.parquet"))
    labels_df = pd.read_parquet(os.path.join(dataset_data, "retrieval_labels.parquet"))
    folds_df = pd.read_parquet(os.path.join(dataset_data, "fold_assignments.parquet"))

    valid_qa_ids = set(qa_df["qa_id"].dropna().unique())
    valid_chunk_ids = set(legal_df["chunk_id"].dropna().unique())

    # Every citation qa_id must exist in qa_unique
    citation_qa_ids = set(citations_df["qa_id"].dropna().unique())
    assert citation_qa_ids.issubset(valid_qa_ids), "Citations refer to nonexistent qa_ids"

    # Every retrieval label qa_id must exist in qa_unique
    label_qa_ids = set(labels_df["qa_id"].dropna().unique())
    assert label_qa_ids.issubset(valid_qa_ids), "Retrieval labels refer to nonexistent qa_ids"

    # Every positive chunk must exist in legal_chunks
    pos_chunks = set(labels_df["positive_chunk_id"].dropna().unique())
    assert pos_chunks.issubset(valid_chunk_ids), "Retrieval labels refer to nonexistent chunk_ids"

    # Folds must cover qa_unique exactly
    fold_qa_ids = set(folds_df["qa_id"].dropna().unique())
    assert fold_qa_ids == valid_qa_ids, "Fold assignments do not match qa_unique exactly"
    assert set(folds_df["fold_id"].unique()) == {0, 1, 2, 3, 4}, "Folds must span 0..4"

def test_manifest_checksums_match_actual_bytes(dataset_data):
    manifest_path = os.path.join(dataset_data, "dataset_manifest.json")
    assert os.path.exists(manifest_path)
    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    assert "code" not in manifest, "dataset_manifest.json must not reference code"

    for fname, meta in manifest.get("files", {}).items():
        fpath = os.path.join(dataset_data, fname)
        assert os.path.exists(fpath), f"File {fname} declared in manifest missing on disk"
        expected_sha = meta["sha256"]
        actual_sha = compute_sha256(fpath)
        assert actual_sha == expected_sha, f"SHA mismatch for {fname}"
