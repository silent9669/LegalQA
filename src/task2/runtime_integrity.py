"""Strict packaged runtime identity and manifest provenance verification for Kaggle (V11)."""

from __future__ import annotations

import glob
import json
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

EXPECTED_RUNTIME_API_VERSION: int = 11
GIT_SHA_RE = re.compile(r"^[0-9a-f]{40}$")


def _require_git_sha(value: object, field: str) -> str:
    """Validate that a field contains a real 40-character lowercase hexadecimal Git commit SHA."""
    text = str(value or "").strip().lower()
    if not GIT_SHA_RE.fullmatch(text):
        raise RuntimeError(
            f"{field} must contain a real 40-character lowercase Git commit SHA; found {value!r}."
        )
    return text


def find_packaged_code_roots(base_input_dir: str = "/kaggle/input") -> List[str]:
    """Find all valid packaged LegalQA code directories containing src/ and scripts/."""
    if not os.path.exists(base_input_dir):
        return []

    code_candidates = glob.glob(f"{base_input_dir}/**/code/LegalQA", recursive=True)
    valid_roots = []
    for cand in sorted(list(set(code_candidates))):
        if os.path.isdir(os.path.join(cand, "src")) and os.path.isdir(os.path.join(cand, "scripts")):
            valid_roots.append(os.path.abspath(cand))

    return valid_roots


def resolve_packaged_code_root(base_input_dir: str = "/kaggle/input", *, strict: bool = True) -> str:
    """Deterministically resolve the single packaged LegalQA code root.

    When strict=True, fails loud on 0 or >1 packaged code roots, refusing fallback to local development paths.
    """
    roots = find_packaged_code_roots(base_input_dir)

    if len(roots) == 1:
        return roots[0]
    elif len(roots) > 1:
        raise RuntimeError(
            f"Ambiguous packaged LegalQA code roots found under '{base_input_dir}': {roots}. "
            f"Expected exactly 1 packaged code root."
        )
    else:
        if strict:
            raise RuntimeError(
                f"No packaged LegalQA code root found under '{base_input_dir}' in strict Kaggle mode. "
                f"Ensure the Kaggle dataset is mounted with code/LegalQA containing src/ and scripts/."
            )
        # Development fallback only
        for fallback in [".", "/kaggle/working", "/kaggle/working/LegalQA"]:
            if os.path.isdir(os.path.join(fallback, "src")) and os.path.isdir(os.path.join(fallback, "scripts")):
                return os.path.abspath(fallback)
        return os.path.abspath(".")


def validate_runtime_manifests(
    runtime_root: str,
    code_root: str,
    *,
    expected_api_version: int = EXPECTED_RUNTIME_API_VERSION,
    expected_git_sha: Optional[str] = None,
) -> Dict[str, Any]:
    """Strictly validate dataset_manifest.json and code_manifest.json provenance and API versions.

    Rules:
    - Missing dataset_manifest.json or code_manifest.json is fatal.
    - runtime_api_version must match expected_api_version in both manifests.
    - git_sha must be a valid 40-character SHA and match between dataset_manifest and code_manifest.
    - expected_git_sha must match if provided.
    """
    # 1. Dataset Manifest
    ds_man_path = os.path.join(runtime_root, "dataset_manifest.json")
    if not os.path.exists(ds_man_path):
        # Also check parent directory if runtime_root is task2/
        alt_ds_man = os.path.join(os.path.dirname(runtime_root), "dataset_manifest.json")
        if os.path.exists(alt_ds_man):
            ds_man_path = alt_ds_man
        else:
            raise RuntimeError(f"Missing dataset_manifest.json under runtime root '{runtime_root}'.")

    with open(ds_man_path, "r", encoding="utf-8") as f:
        ds_man = json.load(f)

    # 2. Code Manifest
    code_man_path = os.path.join(code_root, "code_manifest.json")
    if not os.path.exists(code_man_path):
        alt_code_man = os.path.join(os.path.dirname(code_root), "code_manifest.json")
        if os.path.exists(alt_code_man):
            code_man_path = alt_code_man
        else:
            alt_root_code = os.path.join(runtime_root, "code_manifest.json")
            if os.path.exists(alt_root_code):
                code_man_path = alt_root_code
            else:
                raise RuntimeError(f"Missing code_manifest.json under code root '{code_root}'.")

    with open(code_man_path, "r", encoding="utf-8") as f:
        code_man = json.load(f)

    # 3. Validate runtime API versions
    ds_api = ds_man.get("runtime_api_version")
    if ds_api != expected_api_version:
        raise RuntimeError(
            f"dataset_manifest.json runtime_api_version mismatch: found {ds_api}, expected {expected_api_version}."
        )

    code_api = code_man.get("runtime_api_version")
    if code_api != expected_api_version:
        raise RuntimeError(
            f"code_manifest.json runtime_api_version mismatch: found {code_api}, expected {expected_api_version}."
        )

    # 4. Validate real 40-character Git SHAs (Task 2)
    ds_sha = _require_git_sha(ds_man.get("git_sha"), "dataset_manifest.json 'git_sha'")
    code_sha = _require_git_sha(code_man.get("git_sha"), "code_manifest.json 'git_sha'")

    if ds_sha != code_sha:
        raise RuntimeError(
            f"Git SHA divergence between dataset ({ds_sha}) and code ({code_sha}) manifests! "
            f"Dataset and code must be staged together in a single package."
        )

    # 5. Validate expected git sha if specified
    if expected_git_sha:
        expected_norm = _require_git_sha(expected_git_sha, "expected_git_sha")
        if code_sha != expected_norm:
            raise RuntimeError(
                f"Code git_sha mismatch: expected '{expected_norm}', found '{code_sha}' in code_manifest.json."
            )

    print(
        f"Verified Runtime Integrity: API v{expected_api_version} | "
        f"Git SHA: {code_sha[:10]}"
    )

    return {
        "dataset_manifest": ds_man,
        "code_manifest": code_man,
        "runtime_api_version": expected_api_version,
        "git_sha": code_sha,
    }
