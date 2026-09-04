import json
import hashlib
from pathlib import Path
import pytest

from src.task2.data_contract import (
    INHERITED_CORE_HASHES,
    INHERITED_BM25_FILES,
    INHERITED_DEK21_FILES,
    INHERITED_DATA_INVARIANTS,
    compute_file_sha256,
    verify_inherited_dataset,
    DatasetVerificationResult,
)


def test_manifest_v15_exists_and_matches_contract():
    manifest_path = Path("artifacts/inherited/dataset_manifest_v15.json")
    assert manifest_path.exists(), "dataset_manifest_v15.json must exist in artifacts/inherited/"

    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    assert manifest["runtime_api_version"] == 15
    assert manifest["git_sha"] == "151313fc3126615ec11c08ca68f154d5b0c5406f"
    assert manifest["slug"] == "legalqa-task2-clean-data"

    # Check all 8 core files are present with exact expected hashes
    files = manifest.get("files", {})
    for filename, expected_sha in INHERITED_CORE_HASHES.items():
        assert filename in files, f"Missing {filename} in manifest files"
        assert files[filename]["sha256"] == expected_sha, (
            f"Hash mismatch for {filename}: {files[filename]['sha256']} != {expected_sha}"
        )

    # Check BM25 index entry
    indexes = manifest.get("indexes", {})
    assert "indexes/bm25" in indexes
    bm25_entry = indexes["indexes/bm25"]
    assert bm25_entry["files_count"] == 7
    assert set(bm25_entry["files"]) == set(INHERITED_BM25_FILES)

    # Check DEk21 index entry
    assert "indexes/dek21" in indexes
    dek21_entry = indexes["indexes/dek21"]
    assert dek21_entry["files_count"] == 3
    assert set(dek21_entry["files"]) == set(INHERITED_DEK21_FILES)


def test_streaming_sha256_computation(tmp_path):
    test_file = tmp_path / "sample.bin"
    content = b"LegalQA V16 Data Contract Test Stream"
    test_file.write_bytes(content)

    expected = hashlib.sha256(content).hexdigest()
    actual = compute_file_sha256(test_file)
    assert actual == expected


def test_verify_inherited_dataset_detects_missing_files(tmp_path):
    # Empty directory should fail verification with missing files
    result = verify_inherited_dataset(root=tmp_path, manifest_path="artifacts/inherited/dataset_manifest_v15.json")
    assert not result.is_valid
    assert len(result.missing_files) > 0


def test_verify_inherited_dataset_detects_corrupted_hash(tmp_path):
    # Set up directory with corrupted file
    data_dir = tmp_path / "data"
    data_dir.mkdir(parents=True)
    bad_file = data_dir / "legal_chunks.parquet"
    bad_file.write_bytes(b"corrupted content")

    # Also create other files as empty
    for fname in INHERITED_CORE_HASHES:
        if fname != "legal_chunks.parquet":
            if fname == "public-official.json":
                p = tmp_path / "artifacts" / "raw"
                p.mkdir(parents=True, exist_ok=True)
                (p / fname).write_bytes(b"{}")
            else:
                (data_dir / fname).write_bytes(b"dummy")

    # Create index files
    bm25_dir = tmp_path / "indexes" / "bm25" / "bm25s_index"
    bm25_dir.mkdir(parents=True, exist_ok=True)
    (tmp_path / "indexes" / "bm25" / "bm25_manifest.json").write_text("{}")
    (tmp_path / "indexes" / "bm25" / "corpus_meta.parquet").write_bytes(b"")
    for f in ["params.index.json", "vocab.index.json", "indices.csc.index.npy", "data.csc.index.npy", "indptr.csc.index.npy"]:
        (bm25_dir / f).write_text("{}")

    dek21_dir = tmp_path / "indexes" / "dek21"
    dek21_dir.mkdir(parents=True, exist_ok=True)
    for f in INHERITED_DEK21_FILES:
        (dek21_dir / f).write_text("{}")

    result = verify_inherited_dataset(root=tmp_path, manifest_path="artifacts/inherited/dataset_manifest_v15.json")
    assert not result.is_valid
    assert any("legal_chunks.parquet" in m for m in result.hash_mismatches)


def test_data_invariants_constants():
    assert INHERITED_DATA_INVARIANTS["legal_chunk_count"] == 801863
    assert INHERITED_DATA_INVARIANTS["public_query_count"] == 1000
    assert INHERITED_DATA_INVARIANTS["bm25_doc_count"] == 801863
    assert INHERITED_DATA_INVARIANTS["dek21_row_count"] == 801863
    assert INHERITED_DATA_INVARIANTS["dek21_dim"] == 768
