"""Offline assembly strategy sweep against the official LegalQA scorer.

Measures competing answer assembly and citation stitching policies over cached
raw prose on CPU, using exact official NLTK METEOR (alpha=0.9) and ROUGE-L.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any, Dict, List, Optional

import nltk
import numpy as np
from nltk.translate.meteor_score import meteor_score
from rouge_score import rouge_scorer

from src.task2.candidates import (
    build_citation_header,
    clean_model_prose,
    clean_statutory_text,
    snap_facts_to_evidence,
    trim_at_complete_sentence,
    apply_strategy_f,
)


def evaluate_assembly_strategies(
    items: List[Dict[str, Any]],
    citation_budgets: Optional[List[int]] = None,
) -> Dict[str, Dict[str, float]]:
    """Score candidate assembly strategies on paired (reference, prose, evidence) items."""
    try:
        nltk.data.find("corpora/wordnet.zip")
    except Exception:
        nltk.download("wordnet", quiet=True)
        nltk.download("omw-1.4", quiet=True)

    if citation_budgets is None:
        citation_budgets = [2000, 3000, 4000, 6000]

    rs = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=False)

    # Strategies to evaluate
    strat_names = ["generated", "snapped", "focused_complete_clause", "strategy_f_1000"]
    for bud in citation_budgets:
        strat_names.append(f"cite_only_{bud}")
        strat_names.append(f"dual_assembled_{bud}")

    strat_answers: Dict[str, List[str]] = {s: [] for s in strat_names}
    references: List[str] = []

    for item in items:
        ref = str(item.get("reference", "")).strip()
        raw_prose = str(item.get("prose", "")).strip()
        raw_ev = str(item.get("evidence", "")).strip()
        doc_name = str(item.get("doc_name", "")).strip()
        art_num = str(item.get("art_num", "")).strip()
        clause_num = str(item.get("clause_num", "")).strip()

        references.append(ref)

        clean_ev = clean_statutory_text(raw_ev)
        header = build_citation_header(doc_name, art_num, clause_num)
        gen_prose = clean_model_prose(raw_prose)
        snapped = snap_facts_to_evidence(gen_prose, clean_ev) if gen_prose else ""
        prose_active = snapped or gen_prose

        strat_answers["generated"].append(gen_prose)
        strat_answers["snapped"].append(snapped)
        strat_answers["focused_complete_clause"].append(
            f"{header}\n{trim_at_complete_sentence(clean_ev, max_chars=800)}".strip() if clean_ev else header
        )
        strat_answers["strategy_f_1000"].append(
            apply_strategy_f(prose_active, clean_ev, max_chars=1000)
        )

        for bud in citation_budgets:
            cit_block = f"{header}\n{clean_ev[:bud]}".strip() if clean_ev else ""
            strat_answers[f"cite_only_{bud}"].append(cit_block)
            dual = (
                (prose_active + "\n\nTrích dẫn quy định:\n" + cit_block).strip()
                if (prose_active and cit_block)
                else (prose_active or cit_block)
            )
            strat_answers[f"dual_assembled_{bud}"].append(dual)

    results: Dict[str, Dict[str, float]] = {}
    for s_name in strat_names:
        hyps = strat_answers[s_name]
        m_scores = [
            meteor_score([r.split()], h.split()) if h.strip() else 0.0
            for r, h in zip(references, hyps)
        ]
        r_scores = [
            rs.score(r, h)["rougeL"].fmeasure if h.strip() else 0.0
            for r, h in zip(references, hyps)
        ]
        word_counts = [len(h.split()) for h in hyps]

        results[s_name] = {
            "meteor": float(np.mean(m_scores)) if m_scores else 0.0,
            "rouge_l": float(np.mean(r_scores)) if r_scores else 0.0,
            "mean_words": float(np.mean(word_counts)) if word_counts else 0.0,
            "p90_words": float(np.percentile(word_counts, 90)) if word_counts else 0.0,
        }

    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Sweep assembly strategies over cached prose or evaluation trace.")
    parser.add_argument("--cache-jsonl", required=True, help="Path to gen_raw_cache.jsonl or eval_trace.jsonl")
    parser.add_argument("--qa-path", default=None, help="Optional path to qa_unique.parquet to backfill references")
    parser.add_argument("--output-json", default="artifacts/labs/assembly_sweep.json", help="Path to save sweep report")
    args = parser.parse_args()

    if not os.path.exists(args.cache_jsonl):
        print(f"Error: cache file not found at {args.cache_jsonl}", file=sys.stderr)
        sys.exit(1)

    items = []
    with open(args.cache_jsonl, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                items.append(json.loads(line))

    # If raw cache format (has 'raw' and 'qa_id' but lacks 'reference'), backfill from qa-path
    if items and not any(it.get("reference") for it in items):
        qa_p = args.qa_path or "artifacts/task2/data/qa_unique.parquet"
        if os.path.exists(qa_p):
            import pandas as pd
            df_qa = pd.read_parquet(qa_p)
            id_to_ref = dict(zip(df_qa["qa_id"].astype(str), df_qa["answer_raw"].astype(str)))
            for it in items:
                qid = str(it.get("qa_id", ""))
                if qid in id_to_ref:
                    it["reference"] = id_to_ref[qid]
                if "prose" not in it and "raw" in it:
                    it["prose"] = it["raw"]

    if not items or not any(it.get("reference") for it in items):
        print("Error: Input items lack 'reference' answers for evaluation. Pass an evaluation trace or provide a valid --qa-path.", file=sys.stderr)
        sys.exit(1)

    sweep_res = evaluate_assembly_strategies(items)
    os.makedirs(os.path.dirname(os.path.abspath(args.output_json)), exist_ok=True)
    with open(args.output_json, "w", encoding="utf-8") as f:
        json.dump(sweep_res, f, indent=2, ensure_ascii=False)
    print(f"Saved assembly sweep results to {args.output_json}")


if __name__ == "__main__":
    main()
