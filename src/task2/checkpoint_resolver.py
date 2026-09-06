"""Deterministic and provenance-bound checkpoint resolver for LegalQA V16."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import List, Optional, Sequence


def _validate_candidate(
    candidate_path: Path,
    component: str,
    expected_base_model: str,
    expected_runtime_api: int = 16,
) -> None:
    if not candidate_path.is_dir():
        raise RuntimeError(f"Candidate {candidate_path} is not a directory.")

    manifest_name = f"{component}_manifest.json"
    manifest_file = candidate_path / manifest_name
    if not manifest_file.exists():
        raise RuntimeError(f"Missing required component manifest '{manifest_name}' in {candidate_path}")

    try:
        with open(manifest_file, "r", encoding="utf-8") as f:
            manifest_data = json.load(f)
    except Exception as exc:
        raise RuntimeError(f"Failed to parse manifest {manifest_file}: {exc}") from exc

    api_v = manifest_data.get("runtime_api_version")
    if api_v is not None and int(api_v) != expected_runtime_api:
        raise RuntimeError(
            f"Checkpoint at {candidate_path} has runtime_api_version={api_v} != expected {expected_runtime_api}"
        )

    base_model = manifest_data.get("base_model") or manifest_data.get("model")
    if base_model and base_model != expected_base_model:
        raise RuntimeError(
            f"Checkpoint at {candidate_path} expected base model '{expected_base_model}', got '{base_model}'"
        )

    if component == "generator":
        weights_exist = (
            (candidate_path / "adapter_model.safetensors").exists()
            or (candidate_path / "adapter_model.bin").exists()
        )
        if not weights_exist:
            raise RuntimeError(f"Generator adapter weights missing in {candidate_path}")
        if not (candidate_path / "adapter_config.json").exists():
            raise RuntimeError(f"adapter_config.json missing in {candidate_path}")
    elif component == "reranker":
        weights_exist = (
            (candidate_path / "model.safetensors").exists()
            or (candidate_path / "pytorch_model.bin").exists()
            or any(candidate_path.glob("model-*.safetensors"))
        )
        if not weights_exist:
            raise RuntimeError(f"Reranker model weights missing in {candidate_path}")


def resolve_component_checkpoint(
    *,
    component: str,
    expected_base_model: str,
    preferred_path: Optional[str] = None,
    search_roots: Optional[Sequence[str | Path]] = None,
    expected_runtime_api: int = 16,
) -> str:
    """Resolve an unambiguous, validated checkpoint path.

    Never uses arbitrary first-match globbing. If preferred_path is valid, returns it.
    If multiple valid candidates exist with no preferred match, raises RuntimeError for ambiguity.
    If no valid candidate exists, raises FileNotFoundError.
    """
    if preferred_path:
        p = Path(preferred_path)
        if p.exists() and p.is_dir():
            _validate_candidate(p, component, expected_base_model, expected_runtime_api)
            return str(p.resolve())

    # Build search roots
    roots = [Path(r) for r in (search_roots or ["/kaggle/input", "checkpoints", "artifacts/task2/checkpoints"])]
    target_rel = "checkpoints/generator/hf_adapter" if component == "generator" else "checkpoints/reranker/best"

    discovered: List[Path] = []
    for root in roots:
        if not root.exists():
            continue
        # Direct relative path under root
        direct = root / target_rel
        if direct.exists() and direct.is_dir():
            discovered.append(direct)
        # Search mounted subtrees up to depth 4
        pattern = f"**/{target_rel}"
        for sub in root.glob(pattern):
            if sub.is_dir() and sub.resolve() not in [d.resolve() for d in discovered]:
                discovered.append(sub)

    valid_candidates: List[Path] = []
    validation_errors = []
    for cand in discovered:
        try:
            _validate_candidate(cand, component, expected_base_model, expected_runtime_api)
            valid_candidates.append(cand)
        except Exception as err:
            validation_errors.append(f"{cand}: {err}")

    if not valid_candidates:
        err_msg = f"No valid {component} checkpoint found."
        if validation_errors:
            err_msg += " Rejected candidates:\n" + "\n".join(validation_errors)
        raise FileNotFoundError(err_msg)

    if len(valid_candidates) == 1:
        return str(valid_candidates[0].resolve())

    # If preferred_path matches one of them exactly:
    if preferred_path:
        resolved_pref = str(Path(preferred_path).resolve())
        for vc in valid_candidates:
            if str(vc.resolve()) == resolved_pref:
                return str(vc.resolve())

    raise RuntimeError(
        f"Ambiguous {component} checkpoint candidates found: {[str(c) for c in valid_candidates]}. "
        f"Specify an exact preferred path in production selection provenance."
    )
