#!/usr/bin/env python3
"""Score labelled predictions with the strict offline scorer contract.

Official scoring remains labelled-only: this entry point refuses to score a
public submission without references. Use --offline with explicit reference
and prediction JSON files for development/lockbox reporting.

Usage:
  python scripts/score_official.py --pred runs/<run>/submission.json --refs <labelled-reference> --offline
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.task2.scorer_contract import (  # noqa: E402
    build_offline_metric_report,
    load_predictions_json,
    score_labelled_rows,
)


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description="Strict offline official-scorer entry point.")
    parser.add_argument("--pred", required=True, help="Predictions JSON ({id: {answer: str}})")
    parser.add_argument("--refs", required=True, help="Labelled reference JSON ({id: str})")
    parser.add_argument("--offline", action="store_true", help="Required flag: labelled offline scoring only")
    args = parser.parse_args()

    if not args.offline:
        raise SystemExit("refusing to score without --offline: public submissions have no local references")

    scorer_path = REPO_ROOT / "Scoring-Program-Task-LegalQA" / "scoring.py"
    scorer_sha = _sha256_file(scorer_path)

    predictions = load_predictions_json(args.pred)
    references_raw = json.loads(Path(args.refs).read_text(encoding="utf-8"))
    if not isinstance(references_raw, dict):
        raise SystemExit("reference file must be a JSON object keyed by ID")

    pred_ids = sorted(predictions)
    ref_ids = sorted(str(k) for k in references_raw)
    if pred_ids != ref_ids:
        raise SystemExit(f"reference/prediction ID mismatch: {len(pred_ids)} preds vs {len(ref_ids)} refs")

    refs = [str(references_raw[qid]) for qid in pred_ids]
    preds = [str(predictions[qid]["answer"]) for qid in pred_ids]
    for qid in pred_ids:
        value = predictions[qid]
        if not isinstance(value, dict) or not str(value.get("answer", "")).strip():
            raise SystemExit(f"empty or invalid answer for ID: {qid}")

    scores = score_labelled_rows(refs, preds)
    report = build_offline_metric_report(
        {"meteor": scores["meteor"], "rouge": scores["rouge"], "num_rows": scores["num_rows"]},
        scorer_sha,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
