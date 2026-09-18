#!/usr/bin/env python3
"""Modal A100 adapter: thin launcher around the shared pipeline runner.

Requests exactly one A100-40GB; availability is never promised. The adapter
calls the same run_gpu_gate shared runner as Colab (never a forked
algorithm) with the modal_a100 runtime profile. Switching platforms requires
its own target-GPU microprobe and full timing report even for an identical
candidate commit.

Usage:
  python scripts/modal_app.py --candidate <candidate-manifest> --profile configs/task2/runtime/modal_a100.yaml
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))


def _load_profile(profile_path: Path) -> dict:
    import yaml

    with open(profile_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise ValueError(f"runtime profile must be a mapping: {profile_path}")
    return data


def run_modal_stage(
    candidate_path: str,
    profile_path: str,
    data_dir: str,
    output_dir: str,
    stage: str = "a100_micro_probe",
    parent_report_path: str | None = None,
) -> dict:
    """Run one Modal A100 gate stage through the shared runner."""
    from src.task2.config.loader import load_resolved_config
    from src.task2.dataset.validator import compute_sha256
    from src.task2.provenance.candidate import CandidateManifest
    from scripts.run_gpu_gate import build_gate_request, run_platform_stage

    t0 = time.monotonic()
    profile = _load_profile(Path(profile_path))
    if profile.get("profile_name") != "modal_a100":
        raise ValueError(f"Modal adapter requires profile_name=modal_a100, got {profile.get('profile_name')}")
    if int(profile.get("required_gpu_count", 0)) != 1 or "A100" not in str(profile.get("required_gpu_name_contains", "")):
        raise ValueError("Modal adapter requires exactly one A100-40GB")

    candidate = CandidateManifest.load_json(candidate_path)
    algo_path = REPO_ROOT / "configs" / "task2" / "algorithm.yaml"
    resolved = load_resolved_config(algo_path, profile_path, candidate_id=candidate.candidate_id)
    candidate.validate_against_config(resolved)

    parent_report: dict | None = None
    if parent_report_path:
        from src.task2.provenance.gate_report import GateReport

        parent = GateReport.load_json(parent_report_path)
        parent_report = {
            "status": parent.status,
            "candidate_sha": parent.candidate_id,
            "stage": parent.stage,
            "report_sha256": parent.compute_sha256(),
        }
    request = build_gate_request(
        {
            "candidate_sha": candidate.candidate_id,
            "algorithm_sha256": candidate.algorithm_sha256,
            "dataset_sha256": candidate.dataset.manifest_sha256,
            "scorer_sha256": compute_sha256(str(REPO_ROOT / "Scoring-Program-Task-LegalQA" / "scoring.py")),
            "runtime_profile": "modal_a100",
            "runtime_sha256": resolved.runtime_sha256,
        },
        stage,
        parent_report,
    )
    request.update(
        {
            "candidate_path": candidate_path,
            "data_dir": data_dir,
            "output_dir": output_dir,
            "parent_report_path": parent_report_path,
            "platform": "modal",
        }
    )
    report = run_platform_stage(request)
    wall_seconds = int(time.monotonic() - t0)

    receipt = {
        "platform": "modal",
        "requested_gpu": "A100-40GB",
        "candidate_id": candidate.candidate_id,
        "stage": stage,
        "status": report.get("status"),
        "wall_seconds": wall_seconds,
        "report_sha256": report.get("report_sha256", ""),
    }
    out_p = Path(output_dir)
    out_p.mkdir(parents=True, exist_ok=True)
    (out_p / "modal_a100_receipt.json").write_text(json.dumps(receipt, indent=2), encoding="utf-8")
    print(f"[Modal A100] stage={stage} status={receipt['status']} wall={wall_seconds}s")
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser(description="Modal A100 thin launcher (shared runner).")
    parser.add_argument("--candidate", required=True, help="Path to candidate_manifest.json")
    parser.add_argument("--profile", default=str(REPO_ROOT / "configs/task2/runtime/modal_a100.yaml"))
    parser.add_argument("--data-dir", default="/root/legalqa_data")
    parser.add_argument("--output-dir", default="/root/legalqa_run")
    parser.add_argument("--stage", default="a100_micro_probe",
                        choices=["kaggle_t4x2", "colab_t4", "a100_micro_probe"])
    parser.add_argument("--parent-report", default=None)
    args = parser.parse_args()
    run_modal_stage(args.candidate, args.profile, args.data_dir, args.output_dir, args.stage, args.parent_report)


try:
    import modal  # type: ignore

    app = modal.App("legalqa-a100")

    @app.function(gpu="A100-40GB", timeout=60 * 60 * 5)
    def modal_remote_stage(request: dict) -> dict:
        from scripts.run_gpu_gate import run_platform_stage as _run

        return _run(request)
except ImportError:
    modal = None
    app = None


if __name__ == "__main__":
    main()
