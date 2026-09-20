#!/usr/bin/env python3
"""Cold rebuild of the BM25S index with order-hash manifest + load gate.

Mirrors scripts/rebuild_dense_index.py: encode-free lexical build in corpus
order (k1=1.5, b=0.75 production continuity), atomic publish, strict reload
(doc_ids order + params) before acceptance.

Usage:
  python scripts/rebuild_bm25_index.py --corpus <legal_chunks.parquet> --out <index-dir> [--force]
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))


def build_verified_bm25(corpus_path: str, out_dir: str, k1: float = 1.5, b: float = 0.75) -> dict:
    """Fit BM25 on the corpus in order, save atomically, reload strictly."""
    import shutil
    import tempfile

    import pandas as pd

    from src.common.bm25 import BM25Retriever

    t0 = time.time()
    df = pd.read_parquet(corpus_path, columns=["chunk_id", "text_raw", "text_norm"])
    corpus = df.to_dict("records")
    print(f"corpus rows: {len(corpus)} (order preserved, no dedup)")

    retriever = BM25Retriever(k1=k1, b=b)
    retriever.fit(corpus)
    if retriever.bm25s_index is None:
        raise RuntimeError("bm25s build failed: no searchable index produced")

    destination = Path(out_dir)
    tmp_parent = destination.parent if destination.parent.exists() else Path(".")
    tmp_dir = Path(tempfile.mkdtemp(prefix="bm25-rebuild-", dir=str(tmp_parent)))
    try:
        retriever.save(str(tmp_dir))
        check = BM25Retriever.load(str(tmp_dir), corpus_path=str(corpus_path), fail_on_missing_index=True)
        probe = check.search("Theo Nghị định 100/2019 Điều 17 phạt bao nhiêu?", top_k=3)
        if not probe:
            raise RuntimeError("rebuilt BM25 index returned zero results on probe query")
        manifest = {
            "k1": k1,
            "b": b,
            "corpus": str(corpus_path),
            "corpus_rows": len(corpus),
            "probe_top1": {k: v for k, v in probe[0].items() if k in ("chunk_id", "bm25_score")},
            "seconds": round(time.time() - t0, 1),
        }
        (tmp_dir / "rebuild_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        if destination.exists():
            shutil.rmtree(destination)
        destination.parent.mkdir(parents=True, exist_ok=True)
        tmp_dir.rename(destination)
    except BaseException:
        shutil.rmtree(str(tmp_dir), ignore_errors=True)
        raise
    manifest["index_dir"] = str(destination)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Cold rebuild a verified BM25S index.")
    parser.add_argument("--corpus", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--k1", type=float, default=1.5)
    parser.add_argument("--b", type=float, default=0.75)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    if Path(args.out).exists() and any(Path(args.out).iterdir()) and not args.force:
        raise SystemExit(f"refusing to overwrite existing index dir without --force: {args.out}")
    manifest = build_verified_bm25(args.corpus, args.out, args.k1, args.b)
    print(f"PASS: verified BM25 index at {manifest['index_dir']} in {manifest['seconds']}s")


if __name__ == "__main__":
    main()
