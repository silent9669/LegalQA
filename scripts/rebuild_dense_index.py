#!/usr/bin/env python3
"""Cold rebuild of the DEK21 dense index with order-hash manifest + gate.

The checked-in dense matrix failed the identical-pair self-consistency test
(mean cosine ~0.0 on 400 identical-text pairs), so it must never be used for
retrieval. This script re-encodes the corpus IN ORDER with the pinned encoder
revision, writes the order-hash manifest, and reloads with the consistency
gate enabled (final_mode). Refuses floating model revisions.

Usage (A100 host):
  python scripts/rebuild_dense_index.py --corpus <legal_chunks.parquet> \\
      --out <index-dir> --revision <40-hex-commit> [--force]
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import sys
import tempfile
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

_COMMIT_RE = re.compile(r"^[0-9a-f]{40}$")


def require_pinned_revision(revision: str) -> str:
    """Fail closed on floating encoder revisions."""
    normalized = str(revision or "").strip().lower()
    if not _COMMIT_RE.match(normalized):
        raise ValueError(f"refusing floating encoder revision: {revision}")
    return normalized


def check_dense_alignment(index_dir: str, corpus_path: str, max_pairs: int = 400) -> dict:
    """Query-free alignment check of a staged index (mmap, no encoder needed)."""
    import numpy as np
    import pandas as pd

    from src.common.dense import duplicate_text_pairs, embedding_self_consistency

    embeddings_path = Path(index_dir) / "embeddings.npy"
    if not embeddings_path.is_file():
        return {"status": "missing", "aligned": False, "reason": "embeddings.npy absent"}
    df = pd.read_parquet(corpus_path, columns=["chunk_id", "text_raw"])
    embeddings = np.load(str(embeddings_path), mmap_mode="r")
    if embeddings.shape[0] != len(df):
        return {"status": "shape_mismatch", "aligned": False,
                "reason": f"rows {embeddings.shape} vs corpus {len(df)}"}
    pairs = duplicate_text_pairs(
        [{"chunk_id": c, "text_raw": t} for c, t in
         zip(df["chunk_id"].astype(str), df["text_raw"].astype(str))],
        max_pairs=max_pairs,
    )
    report = embedding_self_consistency(embeddings, pairs)
    report["status"] = "measured"
    return report


def build_verified_index(
    corpus_path: str,
    out_dir: str,
    model_id: str,
    revision: str,
    batch_size: int = 256,
    device: str = "cuda:0",
    dtype: str = "float16",
) -> dict:
    """Encode the corpus in order, write the order-hash manifest, gate reload."""
    import pandas as pd

    from src.common.dense import DenseRetriever

    revision = require_pinned_revision(revision)
    t0 = time.time()
    df = pd.read_parquet(corpus_path)
    if "chunk_id" not in df.columns or "text_raw" not in df.columns:
        raise ValueError("corpus must contain chunk_id and text_raw columns")
    corpus = df.to_dict("records")
    print(f"corpus rows: {len(corpus)} (order preserved, no dedup)")

    retriever = DenseRetriever(model_name=model_id, revision=revision, device=device, dtype=dtype)
    retriever.fit(corpus, batch_size=batch_size, show_progress=True)

    destination = Path(out_dir)
    destination.parent.mkdir(parents=True, exist_ok=True)
    tmp_dir = Path(tempfile.mkdtemp(prefix="dense-rebuild-", dir=str(destination.parent)))
    try:
        retriever.save_index(str(tmp_dir), dtype=dtype)
        check = DenseRetriever.load_index(
            str(tmp_dir),
            corpus_path=str(corpus_path),
            model_name=model_id,
            device="cpu",
            dtype=dtype,
            final_mode=True,
            expected_model_name=model_id,
            verify_self_consistency=True,
        )
        report = getattr(check, "self_consistency_report", {})
        print(f"self-consistency gate: {report}")
        build_manifest = {
            "model_id": model_id,
            "revision": revision,
            "corpus": str(corpus_path),
            "corpus_rows": len(corpus),
            "dtype": dtype,
            "seconds": round(time.time() - t0, 1),
            "self_consistency": report,
        }
        (tmp_dir / "rebuild_manifest.json").write_text(json.dumps(build_manifest, indent=2), encoding="utf-8")
        if destination.exists():
            shutil.rmtree(destination)
        try:
            tmp_dir.rename(destination)
        except OSError:
            shutil.move(str(tmp_dir), str(destination))
    except BaseException:
        shutil.rmtree(str(tmp_dir), ignore_errors=True)
        raise
    build_manifest["index_dir"] = str(destination)
    return build_manifest


def select_verified_dense_index(
    candidate_dirs: list,
    corpus_path: str,
    expected: dict,
    max_pairs: int = 400,
) -> dict:
    """Pick the first staged index that strictly binds the expected encoder.

    ``expected`` carries model_id, revision, encoder_weights_sha256 and
    preprocessing (see ``verify_dense_manifest_identity``). For each
    candidate dir: manifest identity must verify, ``embeddings.npy`` SHA
    must match the manifest, corpus order/content hash must match the live
    corpus, and the self-consistency diagnostic must pass. Anything else
    (legacy manifest without identity, tampered bytes/order, stale
    same-family weights) returns ``{"action": "rebuild_needed", ...}``
    instead of reusing — never overwrites the old production index.
    """
    import numpy as np
    import pandas as pd

    from src.common.dense import (
        build_embedding_row_keys,
        compute_embedding_order_hash,
        duplicate_text_pairs,
        embedding_self_consistency,
        verify_dense_manifest_identity,
    )

    failures = []
    for candidate in candidate_dirs:
        index_dir = Path(candidate)
        manifest_path = index_dir / "dense_manifest.json"
        if not manifest_path.is_file():
            manifest_path = index_dir / "dek21_manifest.json"
        if not manifest_path.is_file():
            failures.append(f"{index_dir.name}: no dense manifest")
            continue
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            verify_dense_manifest_identity(manifest, expected)
            emb_path = index_dir / "embeddings.npy"
            if not emb_path.is_file():
                raise ValueError("embeddings.npy absent")
            import hashlib as _hashlib

            with open(emb_path, "rb") as f:
                current_emb_sha = _hashlib.sha256(f.read()).hexdigest()
            if manifest.get("embeddings_sha256") and current_emb_sha != manifest["embeddings_sha256"]:
                raise ValueError("embeddings bytes differ from manifest digest")
            df = pd.read_parquet(str(corpus_path), columns=["chunk_id", "text_raw"])
            embeddings = np.load(str(emb_path), mmap_mode="r")
            if embeddings.shape[0] != len(df):
                raise ValueError(f"rows {embeddings.shape} vs corpus {len(df)}")
            corpus = [{"chunk_id": c, "text_raw": t} for c, t in
                      zip(df["chunk_id"].astype(str), df["text_raw"].astype(str))]
            if manifest.get("embedding_order_sha256"):
                current_order = compute_embedding_order_hash(build_embedding_row_keys(corpus))
                if current_order != manifest["embedding_order_sha256"]:
                    raise ValueError("corpus order/content differs from the indexed map")
            pairs = duplicate_text_pairs(corpus, max_pairs=max_pairs)
            consistency = embedding_self_consistency(embeddings, pairs)
            if not consistency.get("aligned"):
                raise ValueError(f"self-consistency failed: {consistency}")
            return {"action": "reused", "index_dir": str(index_dir), "aligned": True,
                    "self_consistency": consistency}
        except Exception as exc:
            failures.append(f"{index_dir.name}: {exc}")
            continue
    return {"action": "rebuild_needed", "aligned": False, "failures": failures}


def ensure_dense_index(
    index_dir: str,
    corpus_path: str,
    model_id: str,
    revision: str,
    device: str = "cuda:0",
    batch_size: int = 256,
    dtype: str = "float16",
    force_rebuild: bool = False,
) -> dict:
    """Idempotent gate: reuse the staged index only if aligned, else cold rebuild.

    Never mutates a misaligned index in place; rebuilds go to a temp dir with
    an atomic rename after the reload gate passes.
    """
    revision = require_pinned_revision(revision)
    if not force_rebuild:
        report = check_dense_alignment(index_dir, corpus_path)
        if report.get("aligned"):
            report["action"] = "reused"
            report["index_dir"] = index_dir
            print(f"dense index aligned, reuse: {index_dir}")
            return report
        print(f"dense index unusable ({report.get('status')}); cold rebuilding...")
    manifest = build_verified_index(corpus_path, index_dir, model_id, revision, batch_size, device, dtype)
    manifest["action"] = "rebuilt"
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Cold rebuild a verified dense index.")
    parser.add_argument("--corpus", required=True, help="legal_chunks.parquet (row order is the index order)")
    parser.add_argument("--out", required=True, help="Destination index directory")
    parser.add_argument("--model", default="runs/20260920-215402/encoder_ft_v2")
    parser.add_argument("--revision", required=True, help="Immutable 40-hex encoder commit")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", default="float16", choices=["float16", "float32"])
    parser.add_argument("--force", action="store_true", help="Replace an existing index directory")
    args = parser.parse_args()

    revision = str(args.revision).strip().lower()
    require_pinned_revision(revision)

    if Path(args.out).exists() and any(Path(args.out).iterdir()) and not args.force:
        raise SystemExit(f"refusing to overwrite existing index dir without --force: {args.out}")

    manifest = build_verified_index(
        corpus_path=args.corpus,
        out_dir=args.out,
        model_id=args.model,
        revision=revision,
        batch_size=args.batch_size,
        device=args.device,
        dtype=args.dtype,
    )
    # --force replaces any existing dir inside build_verified_index; the CLI
    # guard above keeps the explicit-overwrite intent for direct rebuilds.
    print(f"PASS: verified dense index at {manifest['index_dir']}")


if __name__ == "__main__":
    main()
