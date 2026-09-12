"""SHA-256 checksum generation and verification for run bundles and artifacts."""

from __future__ import annotations

import hashlib
import os
from pathlib import Path
from typing import Dict, List, Sequence, Union


def compute_file_sha256(file_path: Union[Path, str], chunk_size: int = 65536) -> str:
    """Compute SHA-256 hash of a single file in chunks."""
    p = Path(file_path)
    if not p.is_file():
        raise FileNotFoundError(f"File not found for checksum calculation: {p}")

    hasher = hashlib.sha256()
    with open(p, "rb") as f:
        while chunk := f.read(chunk_size):
            hasher.update(chunk)
    return hasher.hexdigest()


def compute_directory_checksums(
    dir_path: Union[Path, str],
    exclude_names: Sequence[str] = ("checksums.sha256", ".DS_Store"),
) -> Dict[str, str]:
    """Recursively compute SHA-256 for all files in a directory, sorted deterministically."""
    root = Path(dir_path)
    if not root.is_dir():
        raise NotADirectoryError(f"Directory not found: {root}")

    exclude_set = set(exclude_names)
    results: Dict[str, str] = {}

    for dirpath, _, filenames in os.walk(root):
        for fname in sorted(filenames):
            if fname in exclude_set:
                continue
            full_path = Path(dirpath) / fname
            rel_path = str(full_path.relative_to(root)).replace("\\", "/")
            results[rel_path] = compute_file_sha256(full_path)

    # Return sorted by relative path
    return dict(sorted(results.items()))


def write_checksums_file(
    dir_path: Union[Path, str],
    output_path: Union[Path, str],
    exclude_names: Sequence[str] = ("checksums.sha256", ".DS_Store"),
) -> Dict[str, str]:
    """Compute directory checksums and write in standard 'hash  filename' format."""
    checksums = compute_directory_checksums(dir_path, exclude_names=exclude_names)
    out_p = Path(output_path)
    out_p.parent.mkdir(parents=True, exist_ok=True)

    lines = [f"{digest}  {rel_path}\n" for rel_path, digest in checksums.items()]
    out_p.write_text("".join(lines), encoding="utf-8")
    return checksums


def verify_checksums_file(
    dir_path: Union[Path, str],
    checksums_path: Union[Path, str],
) -> bool:
    """Verify all files recorded in checksums.sha256 against actual files in directory."""
    root = Path(dir_path)
    cs_p = Path(checksums_path)
    if not cs_p.is_file():
        raise FileNotFoundError(f"Checksums file not found: {cs_p}")

    lines = cs_p.read_text(encoding="utf-8").splitlines()
    for line in lines:
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split(None, 1)
        if len(parts) != 2:
            raise ValueError(f"Malformed checksum line: {line}")
        expected_hash, rel_path = parts[0].strip(), parts[1].strip()
        target_file = root / rel_path
        if not target_file.is_file():
            raise FileNotFoundError(f"Missing file recorded in checksums: {rel_path}")
        actual_hash = compute_file_sha256(target_file)
        if actual_hash != expected_hash:
            raise ValueError(
                f"Checksum mismatch for {rel_path}: expected {expected_hash}, got {actual_hash}"
            )
    return True
