"""Chunked streaming hashing utilities for large files and datasets."""

from __future__ import annotations

import hashlib
import os
from pathlib import Path
from typing import Union


def sha256_file(path: Union[str, Path], chunk_size: int = 8 * 1024 * 1024) -> str:
    """Compute SHA256 checksum of a file using chunked streaming (default 8MB chunks).

    Prevents loading multi-gigabyte files (e.g. dense embeddings, weights) into RAM.
    """
    path_str = str(path)
    if not os.path.exists(path_str):
        raise FileNotFoundError(f"File not found for hashing: {path_str}")

    h = hashlib.sha256()
    with open(path_str, "rb") as f:
        while chunk := f.read(chunk_size):
            h.update(chunk)
    return h.hexdigest()
