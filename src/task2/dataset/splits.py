"""Group-isolated split assignment with existing-fold retention.

Groups (qa_group_id) are the exclusion boundary: no group may appear in more
than one split. When every row already carries a fold_id and no group spans
multiple folds, the stored folds are retained and mapped to roles
(folds 2-4 -> train, fold 1 -> dev, fold 0 -> lockbox). Otherwise groups are
assigned deterministically from sha256(seed, qa_group_id) into 60/20/20
train/dev/lockbox buckets unless explicit ratios are given.
"""

from __future__ import annotations

import hashlib
from collections import defaultdict
from typing import Any, Dict, List, Tuple

from src.task2.config.loader import canonical_sha256

_FOLD_TO_SPLIT = {2: "train", 3: "train", 4: "train", 1: "dev", 0: "lockbox"}


def _bucket_for_group(qa_group_id: str, seed: int) -> float:
    digest = hashlib.sha256(f"{seed}:{qa_group_id}".encode("utf-8")).hexdigest()
    return int(digest[:16], 16) / float(0xFFFFFFFFFFFFFFFF)


def assign_group_splits(
    rows: List[Dict[str, Any]],
    seed: int = 42,
    ratios: Dict[str, float] | None = None,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Assign a group-isolated ``split`` to each row.

    Returns (rows_with_split, report). The report records whether existing
    folds were retained, per-split group/example counts, and any group that
    would cross splits (which raises ValueError instead of being split).
    """
    if ratios is None:
        ratios = {"train": 0.6, "dev": 0.2, "lockbox": 0.2}
    total_ratio = sum(ratios.values())
    if abs(total_ratio - 1.0) > 1e-9:
        raise ValueError(f"split ratios must sum to 1.0, got {ratios}")
    for required in ("train", "dev", "lockbox"):
        if required not in ratios:
            raise ValueError(f"split ratios missing '{required}': {ratios}")

    for row in rows:
        if not row.get("qa_group_id"):
            raise ValueError("assign_group_splits requires canonical rows with qa_group_id")

    # Audit existing fold assignments when present.
    group_folds: Dict[str, set] = defaultdict(set)
    has_any_fold = False
    for row in rows:
        fold = row.get("fold_id", row.get("fold"))
        if fold is not None:
            has_any_fold = True
            group_folds[row["qa_group_id"]].add(int(fold))
    cross_fold_groups = sorted(g for g, folds in group_folds.items() if len(folds) > 1)

    out: List[Dict[str, Any]] = []
    if has_any_fold and not cross_fold_groups:
        # Retain stored folds: every group maps to exactly one fold.
        for row in rows:
            fold = row.get("fold_id", row.get("fold"))
            if fold is None:
                raise ValueError("mixed rows with and without fold_id cannot retain existing folds")
            split = _FOLD_TO_SPLIT.get(int(fold))
            if split is None:
                raise ValueError(f"unknown fold_id retained from stored folds: {fold}")
            item = dict(row)
            item["split"] = split
            out.append(item)
        report = _report(out, seed, ratios, retained_existing_folds=True, cross_fold_groups=[])
        return out, report
    if cross_fold_groups:
        raise ValueError(f"cross-fold group leakage blocks split retention: {cross_fold_groups[:5]}")

    train_cut = ratios["train"]
    dev_cut = ratios["train"] + ratios["dev"]
    group_ids = sorted({row["qa_group_id"] for row in rows})
    group_split = {}
    for gid in group_ids:
        bucket = _bucket_for_group(gid, seed)
        if bucket < train_cut:
            group_split[gid] = "train"
        elif bucket < dev_cut:
            group_split[gid] = "dev"
        else:
            group_split[gid] = "lockbox"
    for row in rows:
        item = dict(row)
        item["split"] = group_split[row["qa_group_id"]]
        out.append(item)
    report = _report(out, seed, ratios, retained_existing_folds=False, cross_fold_groups=[])
    return out, report


def _report(
    rows: List[Dict[str, Any]],
    seed: int,
    ratios: Dict[str, float],
    retained_existing_folds: bool,
    cross_fold_groups: List[str],
) -> Dict[str, Any]:
    by_split: Dict[str, set] = defaultdict(set)
    examples_by_split: Dict[str, int] = defaultdict(int)
    for row in rows:
        by_split[row["split"]].add(row["qa_group_id"])
        examples_by_split[row["split"]] += 1
    # Group isolation invariant: pairwise intersections must be empty.
    splits = list(by_split)
    for i in range(len(splits)):
        for j in range(i + 1, len(splits)):
            overlap = by_split[splits[i]] & by_split[splits[j]]
            if overlap:
                raise ValueError(f"group leakage across splits {splits[i]}/{splits[j]}: {sorted(overlap)[:5]}")
    return {
        "seed": seed,
        "ratios": dict(ratios),
        "retained_existing_folds": retained_existing_folds,
        "cross_fold_groups": list(cross_fold_groups),
        "num_groups": sum(len(v) for v in by_split.values()),
        "groups_by_split": {k: len(v) for k, v in sorted(by_split.items())},
        "examples_by_split": {k: int(examples_by_split[k]) for k in sorted(examples_by_split)},
        "split_fingerprint": split_fingerprint(rows),
    }


def split_fingerprint(rows: List[Dict[str, Any]]) -> str:
    """Deterministic fingerprint over (qa_group_id, qa_id/qa_row_id, split)."""
    keys = []
    for row in rows:
        keys.append(
            {
                "qa_group_id": row.get("qa_group_id", ""),
                "qa_id": row.get("qa_id", row.get("qa_row_id", "")),
                "split": row.get("split", ""),
                "fold_id": row.get("fold_id", row.get("fold", "")),
            }
        )
    keys.sort(key=lambda item: (str(item["qa_group_id"]), str(item["qa_id"])))
    return canonical_sha256(keys)
