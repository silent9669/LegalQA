#!/usr/bin/env python3
"""Bounded small-scale QLoRA training probe for Kaggle Dual-T4.

Validates the real CUDA training path (data join, SFT config, optimizer
steps, VRAM stability, adapter save/strict-reload, single-query generation)
on a tight budget: 200 examples, 60 optimizer steps, held-out fold 0.
Makes NO quality or promotion claims; the artifact is a logic/VRAM proof.

Usage (Kaggle GPU notebook):
  python scripts/kaggle_train_probe.py --candidate PATH --data-dir DIR --output-dir DIR
"""

from __future__ import annotations

import argparse
import datetime
import json
import sys
from pathlib import Path
from typing import Any, Dict

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

DEFAULT_MAX_STEPS = 60
DEFAULT_MAX_EXAMPLES = 200


def validate_probe_request(candidate_path: str, data_dir: str, max_steps: int, max_examples: int) -> Dict[str, Any]:
    """Fail closed on bad probe bounds before any GPU spend."""
    if not Path(candidate_path).is_file():
        raise FileNotFoundError(f"candidate manifest not found: {candidate_path}")
    if not Path(data_dir).is_dir():
        raise FileNotFoundError(f"data directory not found: {data_dir}")
    if not 1 <= int(max_steps) <= 200:
        raise ValueError(f"max_steps must be within 1..200, got {max_steps}")
    if not 1 <= int(max_examples) <= 1000:
        raise ValueError(f"max_examples must be within 1..1000, got {max_examples}")
    return {
        "candidate_path": str(candidate_path),
        "data_dir": str(data_dir),
        "max_steps": int(max_steps),
        "max_examples": int(max_examples),
    }


def run_train_probe(
    candidate_path: str,
    data_dir: str,
    output_dir: str,
    max_steps: int = DEFAULT_MAX_STEPS,
    max_examples: int = DEFAULT_MAX_EXAMPLES,
) -> Dict[str, Any]:
    """Execute the bounded training probe and emit a verified report."""
    import torch

    from src.task2.config.loader import load_resolved_config
    from src.task2.generation.trainer import train_generator_qlora
    from src.task2.generator import QwenGenerator
    from src.task2.provenance.candidate import CandidateManifest

    req = validate_probe_request(candidate_path, data_dir, max_steps, max_examples)
    candidate = CandidateManifest.load_json(req["candidate_path"])
    out_p = Path(output_dir)
    out_p.mkdir(parents=True, exist_ok=True)

    if not torch.cuda.is_available() or torch.cuda.device_count() < 1:
        raise RuntimeError("train probe requires at least 1 CUDA GPU")

    resolved = load_resolved_config(
        REPO_ROOT / "configs" / "task2" / "algorithm.yaml",
        REPO_ROOT / "configs" / "task2" / "runtime" / "kaggle_t4x2.yaml",
        candidate_id=candidate.candidate_id,
    )
    candidate.validate_against_config(resolved)

    data_p = Path(req["data_dir"])
    train_out = str(out_p / "adapter")
    result = train_generator_qlora(
        model_name_or_path=resolved.algorithm.models.generator.id,
        qa_path=str(data_p / "qa_unique.parquet"),
        labels_path=str(data_p / "retrieval_labels.parquet"),
        chunks_path=str(data_p / "legal_chunks.parquet"),
        output_dir=train_out,
        resolved_config=resolved,
        val_fold=0,
        max_steps=req["max_steps"],
        max_train_examples=req["max_examples"],
        device=resolved.runtime.devices.get("generator", "cuda:0"),
    )
    steps_done = int(result.get("optimizer_steps", 0))
    if steps_done < req["max_steps"]:
        raise RuntimeError(f"probe incomplete: {steps_done} < {req['max_steps']} steps")

    peak_mb = 0.0
    if torch.cuda.is_available():
        peak_mb = float(torch.cuda.max_memory_allocated(0) / 1024 / 1024)

    generator = QwenGenerator.load(
        model_path=resolved.algorithm.models.generator.id,
        adapter_path=train_out,
        device=resolved.runtime.devices.get("generator", "cuda:0"),
        fail_on_fallback=True,
        final_mode=False,
        require_adapter=True,
    )
    sample = generator.generate("Mức phạt vi phạm hành chính là gì?", "Căn cứ Điều 17.", max_new_tokens=64)
    if not sample.strip():
        raise RuntimeError("probe generation returned empty text")

    report = {
        "kind": "train_probe",
        "status": "PASS",
        "candidate_id": candidate.candidate_id,
        "git_commit_sha": candidate.git_commit_sha,
        "optimizer_steps": steps_done,
        "max_examples": req["max_examples"],
        "val_fold": 0,
        "peak_allocated_mb": round(peak_mb, 1),
        "reload": "pass",
        "sample_chars": len(sample),
        "finished_at_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    }
    (out_p / "train_probe_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"[PASS] train probe: {steps_done} steps, peak {peak_mb:.0f} MB, reload ok")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Bounded Kaggle Dual-T4 training probe.")
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--output-dir", default="/kaggle/working")
    parser.add_argument("--max-steps", type=int, default=DEFAULT_MAX_STEPS)
    parser.add_argument("--max-examples", type=int, default=DEFAULT_MAX_EXAMPLES)
    args = parser.parse_args()
    run_train_probe(args.candidate, args.data_dir, args.output_dir, args.max_steps, args.max_examples)


if __name__ == "__main__":
    main()
