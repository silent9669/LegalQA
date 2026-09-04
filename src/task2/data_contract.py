"""Data contract and verification for inherited competition dataset in LegalQA Task 2.

V16 must inherit the verified V15 dataset and indexes byte-for-byte without mutation.
"""

from dataclasses import dataclass, field
import hashlib
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Set

logger = logging.getLogger(__name__)

# Verified authoritative file hashes from V15 manifest (legalqa-task2-clean-data)
INHERITED_CORE_HASHES: Dict[str, str] = {
    "legal_chunks.parquet": "6b34c7f338871f3786b94f947c89f6eff000f5bf21785ebee59dc267a00c164e",
    "qa_unique.parquet": "5f0055113346cba7faa1ee92f105b29d0570473d3033c214a1f52e8b77176e7a",
    "known_qa.json": "1b453f36f5db10d913f6e1b04b48e081170ff9b4f88a3cb008abb376c971d00c",
    "qa_citations.parquet": "0772c6b3e0d4c1b72711331ed924c0a753c94a49e2f0f26aa5d7891f20b22e19",
    "retrieval_labels.parquet": "f11d0e46185e06d2aa8b997817877694938f51034b678fa071665891cb73004c",
    "fold_assignments.parquet": "1422fd1587fce02941c1e31846d6e2f4dcb9b88b6b65890686137aad85a9f0c3",
    "reranker_training_pairs.parquet": "d67274b1b5cb2fd4892b55a327ed9b30de26140692e103c78eec38cd3ba4a126",
    "public-official.json": "5f68ca901cb20798559538bef60fa7c32bd7d0df59f5bf31a37eb220c9e00df5",
}

# The 7 files comprising the BM25 index layout
INHERITED_BM25_FILES: List[str] = [
    "bm25_manifest.json",
    "corpus_meta.parquet",
    "bm25s_index/params.index.json",
    "bm25s_index/vocab.index.json",
    "bm25s_index/indices.csc.index.npy",
    "bm25s_index/data.csc.index.npy",
    "bm25s_index/indptr.csc.index.npy",
]

# The 3 files comprising the DEk21 dense index layout
INHERITED_DEK21_FILES: List[str] = [
    "dek21_manifest.json",
    "dense_manifest.json",
    "embeddings.npy",
]

# Invariants specified in 04_INHERITED_DATASET_CONTRACT.md
INHERITED_DATA_INVARIANTS: Dict[str, int] = {
    "legal_chunk_count": 801863,
    "public_query_count": 1000,
    "bm25_doc_count": 801863,
    "dek21_row_count": 801863,
    "dek21_dim": 768,
}


@dataclass
class DatasetVerificationResult:
    is_valid: bool
    missing_files: List[str] = field(default_factory=list)
    hash_mismatches: List[str] = field(default_factory=list)
    structural_mismatches: List[str] = field(default_factory=list)
    verified_files: List[str] = field(default_factory=list)
    summary: str = ""


def compute_file_sha256(file_path: Path | str, chunk_size: int = 1048576) -> str:
    """Stream-compute the SHA-256 hex digest of a file to prevent OOM on large files."""
    path = Path(file_path)
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(chunk_size):
            h.update(chunk)
    return h.hexdigest()


def find_file_in_root(root: Path, relative_name: str, manifest_source: Optional[str] = None) -> Optional[Path]:
    """Resolve file location in dataset root supporting both flat and structured layouts."""
    candidates = []
    if manifest_source:
        candidates.append(root / manifest_source)
    candidates.append(root / relative_name)
    candidates.append(root / "data" / relative_name)
    candidates.append(root / "artifacts" / "raw" / relative_name)

    for c in candidates:
        if c.exists() and c.is_file():
            return c
    return None


def verify_inherited_dataset(
    root: str | Path,
    manifest_path: str | Path,
    check_hashes: bool = True,
    check_structural: bool = False,
) -> DatasetVerificationResult:
    """Verify that the dataset at root matches the verified inherited dataset contract."""
    root_path = Path(root)
    manifest_file = Path(manifest_path)

    if not manifest_file.exists():
        return DatasetVerificationResult(
            is_valid=False,
            summary=f"Dataset manifest not found at: {manifest_file}",
        )

    with open(manifest_file, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    missing_files: List[str] = []
    hash_mismatches: List[str] = []
    structural_mismatches: List[str] = []
    verified_files: List[str] = []

    files_section = manifest.get("files", {})

    # 1. Verify core files
    for filename, expected_sha in INHERITED_CORE_HASHES.items():
        file_meta = files_section.get(filename, {})
        manifest_source = file_meta.get("source")
        target_path = find_file_in_root(root_path, filename, manifest_source)

        if target_path is None:
            missing_files.append(filename)
            continue

        if check_hashes:
            actual_sha = compute_file_sha256(target_path)
            if actual_sha != expected_sha:
                hash_mismatches.append(
                    f"{filename}: expected {expected_sha}, got {actual_sha}"
                )
            else:
                verified_files.append(filename)
        else:
            verified_files.append(filename)

    # 2. Verify BM25 index files layout
    bm25_root = root_path / "indexes" / "bm25"
    for bm25_file in INHERITED_BM25_FILES:
        target = bm25_root / bm25_file
        if not target.exists():
            missing_files.append(f"indexes/bm25/{bm25_file}")
        else:
            verified_files.append(f"indexes/bm25/{bm25_file}")

    # 3. Verify DEk21 index files layout
    dek21_root = root_path / "indexes" / "dek21"
    for dek21_file in INHERITED_DEK21_FILES:
        target = dek21_root / dek21_file
        if not target.exists():
            missing_files.append(f"indexes/dek21/{dek21_file}")
        else:
            verified_files.append(f"indexes/dek21/{dek21_file}")

    is_valid = len(missing_files) == 0 and len(hash_mismatches) == 0 and len(structural_mismatches) == 0
    summary = (
        f"Verified {len(verified_files)} files. "
        f"Missing: {len(missing_files)}, Hash mismatches: {len(hash_mismatches)}, "
        f"Structural errors: {len(structural_mismatches)}"
    )

    return DatasetVerificationResult(
        is_valid=is_valid,
        missing_files=missing_files,
        hash_mismatches=hash_mismatches,
        structural_mismatches=structural_mismatches,
        verified_files=verified_files,
        summary=summary,
    )
