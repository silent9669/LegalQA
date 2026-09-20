"""Kaggle input-mount resolution: flat and nested layouts with sync wait.

Newer kernels nest datasets at /kaggle/input/datasets/<owner>/<slug> and
may sync file contents asynchronously after the container starts. Fixed
single-path lookups fail there; recursive search with a bounded wait does
not. Pure stdlib, CPU-testable.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import List, Optional


def find_mounted_file(
    filename: str,
    roots: Optional[List[str]] = None,
    timeout_seconds: float = 180.0,
    poll_interval: float = 5.0,
) -> Path:
    """Locate filename under Kaggle input mounts, waiting for sync.

    Searches each root recursively; polls until timeout for lazy mounts.
    Raises FileNotFoundError listing what WAS visible (diagnosable).
    """
    search_roots = [Path(r) for r in (roots or ["/kaggle/input"])]
    deadline = time.monotonic() + max(0.0, float(timeout_seconds))
    visible: List[str] = []
    while True:
        for root in search_roots:
            if not root.is_dir():
                continue
            try:
                found = sorted(root.rglob(filename))
            except OSError:
                continue
            hits = [p for p in found if p.is_file()]
            if hits:
                return hits[0]
            try:
                visible = sorted(str(p) for p in root.rglob("*") if p.is_dir())[:25]
            except OSError:
                visible = []
        if time.monotonic() >= deadline:
            raise FileNotFoundError(
                f"{filename} not mounted under {[str(r) for r in search_roots]} "
                f"after {timeout_seconds}s; visible dirs: {visible}"
            )
        time.sleep(min(poll_interval, max(0.0, deadline - time.monotonic())))


def find_candidate_manifest(timeout_seconds: float = 180.0) -> Path:
    """Locate the candidate manifest on Kaggle input mounts."""
    return find_mounted_file("candidate_manifest.json", timeout_seconds=timeout_seconds)
