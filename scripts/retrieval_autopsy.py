#!/usr/bin/env python3
"""Retrieval autopsy: label resolution, recall vs ordering, per-arm, pool depth.

Inherited from the v10 autopsy cell. Uses the production components
(BM25 mmap index, legal-reference arm, optional dense) on a labelled dev
slice. Reports, in order:
  1. label resolution (unresolved golds read exactly like retrieval failure),
  2. fused-pool hit@K curve (the reranker ceiling),
  3. per-arm hit@pool + fire rate (a 0%-fire arm is mis-keyed, not unhelpful),
  4. pool-depth curve (what a wider candidate pool would buy).

Dense CPU search is ~12 s/query, so dense runs on a smaller deterministic
subset (default 20, batched). Article-level hits use parent_article_id.

Usage:
  python scripts/retrieval_autopsy.py --data-dir kaggle_dataset --sample 100 --pool 100 --out artifacts/labs/autopsy.json
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import pandas as pd

from src.common.bm25 import BM25Retriever
from src.common.legal_reference import build_legal_reference_index, search_legal_references
from src.common.rrf import reciprocal_rank_fusion


def _resolve_positions(labels: pd.DataFrame, chunk_id_to_positions: dict) -> pd.DataFrame:
    labels = labels.copy()
    labels["gold_positions"] = labels["positive_chunk_id"].astype(str).map(
        lambda cid: list(chunk_id_to_positions.get(cid, []))
    )
    return labels


def main() -> None:
    parser = argparse.ArgumentParser(description="CPU retrieval autopsy on a labelled dev slice.")
    parser.add_argument("--data-dir", default="kaggle_dataset")
    parser.add_argument("--sample", type=int, default=100)
    parser.add_argument("--pool", type=int, default=100)
    parser.add_argument("--dense-subset", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", default="artifacts/labs/autopsy.json")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    t0 = time.time()
    chunks = pd.read_parquet(
        data_dir / "legal_chunks.parquet",
        columns=["chunk_id", "doc_name", "parent_article_id", "article_number", "text_raw"],
    )
    chunk_rows = chunks.to_dict("records")
    chunk_id_to_positions: dict = {}
    for pos, cid in enumerate(chunks["chunk_id"].astype(str)):
        chunk_id_to_positions.setdefault(cid, []).append(pos)

    labels = pd.read_parquet(data_dir / "retrieval_labels.parquet")
    dev_labels = labels[labels["fold_id"] == 1].reset_index(drop=True)
    dev_labels = _resolve_positions(dev_labels, chunk_id_to_positions)

    npos = dev_labels["gold_positions"].map(len).to_numpy()
    print(f"dev labelled rows: {len(dev_labels)}")
    print(f"1. LABEL RESOLUTION: mean {npos.mean():.1f} positions/row, "
          f"zero for {int((npos == 0).sum())}/{len(dev_labels)}")
    unresolved_rate = round(float((npos == 0).mean()), 4)

    # 0. INDEX SELF-CONSISTENCY (no encoder needed): identical-text duplicate
    # pairs must score cosine >= 0.95. A failed matrix poisons every section.
    from src.common.dense import duplicate_text_pairs, embedding_self_consistency

    consistency_report: dict = {"status": "skipped", "reason": "embeddings unavailable"}
    try:
        import numpy as _np

        _emb = _np.load(str(data_dir / "indexes" / "dek21" / "embeddings.npy"), mmap_mode="r")
        _pairs = duplicate_text_pairs([{"chunk_id": c, "text_raw": t} for c, t in
                                       zip(chunks["chunk_id"].astype(str), chunks["text_raw"].astype(str))])
        consistency_report = embedding_self_consistency(_emb, _pairs)
        consistency_report["status"] = "measured"
        print(f"0. SELF-CONSISTENCY: {consistency_report}")
    except Exception as exc:
        consistency_report = {"status": "failed", "reason": str(exc)[:200]}
        print(f"0. SELF-CONSISTENCY check failed to run: {consistency_report['reason']}")

    sample = dev_labels.sample(n=min(args.sample, len(dev_labels)), random_state=args.seed).reset_index(drop=True)
    questions = sample["question"].astype(str).tolist()
    gold_art_lists = sample["positive_article_id"].astype(str).map(
        lambda v: set(str(v).split("|")) if v and v != "nan" else set()
    ).tolist()

    bm25 = BM25Retriever.load(str(data_dir / "indexes" / "bm25"), corpus_path=str(data_dir / "legal_chunks.parquet"))
    print(f"   bm25 index: k1={bm25.k1} b={bm25.b} (from stored manifest)")
    lex_index, lex_report = build_legal_reference_index(chunk_rows)
    print(f"   lexref index: {lex_report['num_doc_keys']} doc keys, empty={lex_report['is_empty']}")

    def _row_hit(pool_ids: list, gold: list) -> bool:
        gold_set = set(gold)
        return any(p in gold_set for p in pool_ids)

    def _art_hit(pool_ids: list, gold_arts: set) -> bool:
        if not gold_arts:
            return False
        arts = {chunk_rows[p].get("parent_article_id", "") for p in pool_ids if 0 <= p < len(chunk_rows)}
        return bool(arts & gold_arts)

    bm25_pools = []
    for q in questions:
        b = bm25.search(q, top_k=args.pool)
        b_positions = []
        for hit in b:
            b_positions.extend(chunk_id_to_positions.get(str(hit.get("chunk_id", "")), []))
        bm25_pools.append(b_positions[: args.pool])
    # lexref search returns chunk dicts; map back to ordered positions.
    lex_pools = []
    for q in questions:
        hits = search_legal_references([q], lex_index, chunk_rows, k=args.pool)[0]
        positions = []
        for hit in hits:
            positions.extend(chunk_id_to_positions.get(str(hit.get("chunk_id", "")), []))
        # preserve lexref order, dedup
        seen, ordered = set(), []
        for p in positions:
            if p not in seen:
                seen.add(p)
                ordered.append(p)
        lex_pools.append(ordered[: args.pool])

    fused_pools = []
    for b_pool, l_pool in zip(bm25_pools, lex_pools):
        b_dicts = [{"chunk_id": f"pos:{p}"} for p in b_pool]
        l_dicts = [{"chunk_id": f"pos:{p}"} for p in l_pool]
        if b_dicts and l_dicts:
            fused = reciprocal_rank_fusion([b_dicts, l_dicts], k=60, weights=[0.5, 0.5])
            fused_pools.append([int(item["chunk_id"].split(":")[1]) for item in fused[: args.pool]])
        else:
            fused_pools.append((b_pool + l_pool)[: args.pool])

    gold_lists = sample["gold_positions"].tolist()

    def _curve(pools: list) -> dict:
        curve = {}
        for k in (1, 3, 8, 20, 50, 100):
            if k > args.pool:
                continue
            curve[f"hit@{k}"] = round(float(np.mean([_row_hit(p[:k], g) for p, g in zip(pools, gold_lists)])), 4)
        return curve

    def _art_curve(pools: list, k: int = 8) -> float:
        return round(float(np.mean([_art_hit(p[:k], g) for p, g in zip(pools, gold_art_lists)])), 4)

    bm25_curve, lex_curve, fused_curve = _curve(bm25_pools), _curve(lex_pools), _curve(fused_pools)
    ceiling = fused_curve.get(f"hit@{min(100, args.pool)}", 0.0)
    print("2. FUSED POOL (bm25+lexref, pre-rerank):", fused_curve, f"article_hit@8={_art_curve(fused_pools)}")
    print("   BM25 alone:", bm25_curve, f"article_hit@8={_art_curve(bm25_pools)}")
    print("   lexref alone:", lex_curve)

    def _arm_stats(pools: list) -> dict:
        return {
            "fire_rate": round(float(np.mean([len(p) > 0 for p in pools])), 4),
            f"hit@{min(100, args.pool)}": round(
                float(np.mean([_row_hit(p, g) if p else False for p, g in zip(pools, gold_lists)])), 4),
        }

    per_arm = {"bm25": _arm_stats(bm25_pools), "lexref": _arm_stats(lex_pools)}

    # Dense on a deterministic subset (batched).
    dense_stats: dict = {"status": "skipped", "reason": "dense_subset=0"}
    if args.dense_subset > 0:
        try:
            from src.common.dense import DenseRetriever

            dense = DenseRetriever.load_index(
                str(data_dir / "indexes" / "dek21"),
                corpus_path=str(data_dir / "legal_chunks.parquet"),
                device="cpu",
            )
            sub = sample.head(args.dense_subset)
            sub_q = sub["question"].astype(str).tolist()
            sub_gold = sub["gold_positions"].tolist()
            t1 = time.time()
            d_lists = dense.search_batch(sub_q, top_k=args.pool, batch_size=8)
            d_positions = []
            for d in d_lists:
                positions = []
                for hit in d:
                    positions.extend(chunk_id_to_positions.get(str(hit.get("chunk_id", "")), []))
                d_positions.append(positions[: args.pool])
            hits = [any(p in set(g) for p in pool) if pool else False for pool, g in zip(d_positions, sub_gold)]
            dense_stats = {
                "status": "measured",
                "n": len(sub_q),
                "fire_rate": round(float(np.mean([len(p) > 0 for p in d_positions])), 4),
                f"hit@{min(100, args.pool)}": round(float(np.mean(hits)), 4),
                "seconds": round(time.time() - t1, 1),
            }
            print("   dense subset:", dense_stats)
        except Exception as exc:
            dense_stats = {"status": "failed", "reason": str(exc)[:200]}
            print("   dense subset failed:", dense_stats["reason"])

    # Pool depth on the fused pool (positions already capped at pool).
    verdict = (
        "RECALL problem (ceiling < 0.35): the right chunk is usually absent; "
        "the reranker cannot fix it."
        if ceiling < 0.35
        else "ORDERING headroom: the chunk is often in the pool; reranking/packing is where headroom sits."
    )
    print(f"3. VERDICT: {verdict}")
    report = {
        "dev_labelled_rows": len(dev_labels),
        "sampled": len(sample),
        "pool": args.pool,
        "self_consistency": consistency_report,
        "unresolved_gold_rate": unresolved_rate,
        "bm25_params": {"k1": bm25.k1, "b": bm25.b},
        "lexref_report": lex_report,
        "bm25_curve": bm25_curve,
        "lexref_curve": lex_curve,
        "fused_curve": fused_curve,
        "fused_article_hit@8": _art_curve(fused_pools),
        "per_arm": per_arm,
        "dense": dense_stats,
        "verdict": verdict,
        "elapsed_seconds": round(time.time() - t0, 1),
    }
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"wrote {out_path} in {report['elapsed_seconds']}s")


if __name__ == "__main__":
    main()
