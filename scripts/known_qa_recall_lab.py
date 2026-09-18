#!/usr/bin/env python3
"""Known-QA recall lab: group-isolated leave-one-out exact/fuzzy economics.

Inherited from the v10 recall lab. For each sampled dev question, the query's
OWN canonical group is excluded from memory (leave-one-group-out), then exact
and threshold-free fuzzy lookup are measured. A threshold sweep prices each
operating point as v10 does: Δ = fires/N × (E(t) − baseline).

The baseline default (0.549) is the v10 generated-family reference, recorded
as such — it is not our measured deployable score. A recommendation is made
only for max Δ > 0.002, else exact-only.

Usage:
  python scripts/known_qa_recall_lab.py --data-dir kaggle_dataset --sample 200 --out artifacts/labs/recall.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import pandas as pd

from src.common.normalize import normalize_question
from src.task2.dataset.identity import canonicalize_qa_rows
from src.task2.qa_memory import QAMemory
from src.task2.scorer_contract import score_labelled_rows

THRESHOLDS = (0.95, 0.90, 0.85, 0.80, 0.75, 0.70, 0.60)
REFERENCE_BASELINE = 0.549  # v10 generated-family reference; not our measured score.


def main() -> None:
    parser = argparse.ArgumentParser(description="Group-isolated known-QA recall economics.")
    parser.add_argument("--data-dir", default="kaggle_dataset")
    parser.add_argument("--sample", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--baseline", type=float, default=REFERENCE_BASELINE)
    parser.add_argument("--out", default="artifacts/labs/recall.json")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    df = pd.read_parquet(data_dir / "qa_unique.parquet")
    rows = [
        {"qa_id": str(r[0]), "question_raw": str(r[1]), "answer_raw": str(r[2])}
        for r in df[["qa_id", "question_raw", "answer_raw"]].itertuples(index=False)
    ]
    canon, canon_report = canonicalize_qa_rows(rows, conflict_policy="quarantine")
    fold_of = {str(r[0]): int(r[1]) for r in df[["qa_id", "fold_id"]].itertuples(index=False)}
    dev = [r for r in canon if fold_of.get(r["qa_id"]) == 1 or fold_of.get(r["qa_id"]) in (None,)]
    # Strict dev: only fold 1 rows whose qa_id is known.
    dev = [r for r in canon if fold_of.get(r["qa_id"]) == 1]
    rng_sample = sorted(dev, key=lambda r: r["qa_row_id"])
    step = max(1, len(rng_sample) // args.sample)
    sampled = rng_sample[::step][: args.sample]

    memory = QAMemory.from_records(
        [
            {"qa_id": r["qa_id"], "question_raw": r["question_raw"],
             "answer_raw": r["answer_raw"], "qa_group_id": r["qa_group_id"]}
            for r in canon
        ]
    )
    # Exact-match coverage analog: dev questions whose normalized form exists
    # in ANOTHER canonical group.
    norm_groups: dict = {}
    for r in canon:
        norm_groups.setdefault(r["question_norm"], set()).add(r["qa_group_id"])
    exact_coverable = sum(1 for r in sampled if len(norm_groups.get(r["question_norm"], set())) > 1)

    per_query = []
    for record in sampled:
        loo = memory.filter_groups({record["qa_group_id"]})
        exact = loo.lookup_exact(None, record["question_raw"])
        fuzzy = loo.lookup_fuzzy(record["question_raw"], threshold=0.50, require_entity_match=True)
        per_query.append(
            {
                "qa_id": record["qa_id"],
                "exact_hit": exact is not None,
                "exact_answer": exact,
                "fuzzy_sim": float(fuzzy["similarity"]) if fuzzy else 0.0,
                "fuzzy_answer": fuzzy["answer"] if fuzzy else "",
                "own_answer": record["answer_raw"],
            }
        )

    # METEOR of fuzzy answers vs own references (labelled, offline).
    sims = [q["fuzzy_sim"] for q in per_query]
    sweep = []
    for threshold in THRESHOLDS:
        fired = [q for q in per_query if q["fuzzy_sim"] >= threshold]
        if not fired:
            sweep.append({"threshold": threshold, "fires": 0, "mean_meteor": 0.0, "delta": 0.0})
            continue
        refs = [q["own_answer"] for q in fired]
        preds = [q["fuzzy_answer"] for q in fired]
        try:
            scores = score_labelled_rows(refs, preds)
            mean_meteor = scores["meteor"]
        except Exception:
            mean_meteor = 0.0
        delta = len(fired) / len(per_query) * (mean_meteor - args.baseline)
        sweep.append(
            {"threshold": threshold, "fires": len(fired), "mean_meteor": round(mean_meteor, 4),
             "delta": round(delta, 5)}
        )
    best = max(sweep, key=lambda s: s["delta"])
    recommendation = (
        {"mode": "fuzzy", "threshold": best["threshold"], "expected_delta": best["delta"]}
        if best["delta"] > 0.002
        else {"mode": "exact_only", "threshold": None, "expected_delta": 0.0}
    )
    report = {
        "num_canonical": len(canon),
        "num_dev_sampled": len(sampled),
        "quarantined_rows": canon_report["quarantined_row_count"],
        "exact_coverable": exact_coverable,
        "exact_coverable_rate": round(exact_coverable / max(1, len(sampled)), 4),
        "baseline_reference": args.baseline,
        "baseline_note": "v10 generated-family reference, not a measured deployable score",
        "threshold_sweep": sweep,
        "recommendation": recommendation,
    }
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"dev sampled: {len(sampled)} | exact-coverable: {exact_coverable} ({report['exact_coverable_rate']:.1%})")
    print(f"{'thr':>6} {'fires':>6} {'meteor':>8} {'delta':>9}")
    for s in sweep:
        print(f"{s['threshold']:>6} {s['fires']:>6} {s['mean_meteor']:>8.4f} {s['delta']:>+9.5f}")
    print(f"recommendation: {recommendation}")


if __name__ == "__main__":
    main()
