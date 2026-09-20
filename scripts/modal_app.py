#!/usr/bin/env python3
"""Modal A100 remote pipeline runner for LegalQA Task 2.

Idiomatic Modal structure: the local client only builds and validates the
request, then dispatches with ``.remote()``. All GPU work happens inside the
A100 container (image + secrets + volumes declared below).

Container image pins follow constraints-gpu.txt (single source of truth,
parsed locally at image definition time); torch itself comes from the CUDA
12.1 index and its resolved version is recorded in run telemetry.

Usage:
  modal run scripts/modal_app.py --stage micro_probe
  modal run scripts/modal_app.py --stage full --test-path private-official.json
  modal run scripts/modal_app.py --stage full --test-path public-official.json --candidate <id>
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

def _repo_root() -> Path:
    """Resolve the repo root without crashing on unreadable system paths.

    Path.is_dir() raises PermissionError (not False) when a parent directory
    is not listable, e.g. /root on CI runners. Fail open to the file-relative
    root in that case.
    """
    try:
        if Path("/root/LegalQA").is_dir():
            return Path("/root/LegalQA")
    except OSError:
        pass
    return Path(__file__).resolve().parent.parent


REPO_ROOT = _repo_root()
sys.path.insert(0, str(REPO_ROOT))

try:
    import modal
except ImportError:  # CPU-only hosts: request building still works.
    modal = None

APP_NAME = "legalqa-a100-pipeline"
DATA_VOLUME_NAME = "legalqa-data-vol"
RUNS_VOLUME_NAME = "legalqa-runs-vol"
SECRET_NAME = "legalqa-secrets"
DATASET_SLUG = "phucdangg/legalqa-task2-clean-data"
DENSE_MODEL_ID = "CODE4LIFEOFFICIAL/huydang-dek21-embedding-v2"
KNOWN_TEST_FILES = ("private-official.json", "public-official.json")


def read_pin_file(name: str) -> List[str]:
    """Parse a local requirements/constraints file into pip specifiers."""
    target = REPO_ROOT / name
    if not target.is_file():
        try:
            alt = Path("/root/LegalQA") / name
            if alt.is_file():
                target = alt
            else:
                return []
        except OSError:
            return []
    specs = []
    for line in target.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        specs.append(line)
    return specs


def build_modal_request(
    stage: str,
    candidate_manifest: Dict[str, Any],
    test_path: str = "private-official.json",
    parent_report: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Build a validated remote-execution request (pure, locally testable).

    The parent chain is mandatory: micro_probe requires the colab_t4 PASS
    report, full requires the a100_micro_probe PASS report, each for the
    same candidate. No bypass, no cross-candidate reuse.
    """
    if stage not in ("micro_probe", "full"):
        raise ValueError(f"unknown Modal stage: {stage}")
    if not candidate_manifest.get("candidate_id") or not candidate_manifest.get("git_commit_sha"):
        raise ValueError("candidate manifest must carry candidate_id and git_commit_sha")
    dense_revision = ((candidate_manifest.get("models") or {}).get("dense") or {}).get("revision", "")
    if not dense_revision or len(str(dense_revision)) != 40:
        raise ValueError("candidate must pin an immutable 40-hex dense revision")
    required_parent = {"micro_probe": "colab_t4", "full": "a100_micro_probe"}[stage]
    if not isinstance(parent_report, dict):
        raise ValueError(f"Modal {stage} requires the {required_parent} parent report (no bypass)")
    if parent_report.get("status") != "PASS":
        raise ValueError(f"parent {required_parent} report is not PASS")
    if parent_report.get("stage") != required_parent:
        raise ValueError(
            f"parent stage mismatch for Modal {stage}: required {required_parent}, "
            f"got {parent_report.get('stage')}"
        )
    parent_candidate = parent_report.get("candidate_id", parent_report.get("candidate_sha"))
    if parent_candidate != candidate_manifest["candidate_id"]:
        raise ValueError("parent report candidate mismatch: cross-candidate reuse refused")
    if not parent_report.get("report_sha256"):
        raise ValueError("parent report lacks report_sha256")
    return {
        "stage": stage,
        "candidate_id": candidate_manifest["candidate_id"],
        "candidate_manifest": candidate_manifest,
        "test_filename": str(test_path),
        "dense_revision": str(dense_revision),
        "parent_report": parent_report,
    }


def validate_modal_request(request: Dict[str, Any]) -> None:
    """Fail closed on malformed remote requests before any cloud spend."""
    build_modal_request(
        stage=request.get("stage", ""),
        candidate_manifest=request.get("candidate_manifest", {}),
        test_path=request.get("test_filename", ""),
        parent_report=request.get("parent_report"),
    )
    if request.get("test_filename") not in KNOWN_TEST_FILES:
        raise ValueError(f"refusing unknown test file: {request.get('test_filename')}")


def resolve_test_file(data_dir: Path, requested: str) -> Path:
    """Resolve the inference test set; fail closed on unknown/missing files."""
    if requested not in KNOWN_TEST_FILES:
        raise ValueError(f"refusing unknown test file: {requested}")
    candidate = data_dir / requested
    if candidate.is_file():
        return candidate
    fallback = data_dir / "public-official.json"
    if requested != "public-official.json" and fallback.is_file():
        print(f"Notice: {requested} absent; falling back to public-official.json")
        return fallback
    raise FileNotFoundError(f"test file not found: {candidate}")


def test_file_fingerprint(path: Path) -> Dict[str, Any]:
    """Count + hash the inference test set for the evidence record."""
    raw = path.read_bytes()
    data = json.loads(raw.decode("utf-8"))
    if not isinstance(data, dict) or not data:
        raise ValueError(f"test file must be a nonempty JSON object: {path}")
    return {
        "filename": path.name,
        "num_queries": len(data),
        "sha256": hashlib.sha256(raw).hexdigest(),
        "bytes": len(raw),
    }


#: Graceful hard-stop budget (seconds) for the full Modal attempt: the runner
#: writes INCOMPLETE and preserves checkpoints instead of hitting the hard
#: container timeout with uncommitted volume state.
MODAL_DEADLINE_BUDGET_SECONDS = 17100

#: Historical v10 generation ceiling (tokens). The shared default (384)
#: truncates the measured ~736-token answers; the full run restores 1,400.
MODAL_MAX_NEW_TOKENS = 1400


def build_remote_paths(
    data_dir: str, dense_index_dir: str, test_path: str, qwen_model_path: str = "Qwen/Qwen2.5-3B-Instruct"
) -> Dict[str, str]:
    """Runner paths for the Modal container (pure, testable)."""
    return {
        "data_dir": str(data_dir),
        "bm25_dir": str(Path(data_dir) / "indexes" / "bm25"),
        "dek21_dir": str(dense_index_dir),
        "qwen_model_path": str(qwen_model_path),
        "public_test_path": str(test_path),
        "deadline_budget_seconds": MODAL_DEADLINE_BUDGET_SECONDS,
    }


def build_remote_production_cfg() -> Any:
    """Production selection for the Modal full run (pure, testable).

    Mirrors the shared default but restores the historical 1,400-token
    generation ceiling. Governance stays with the candidate + parent chain
    (modal_a100 is not on the screen-promotion path).
    """
    import dataclasses

    from src.task2.production_config import get_default_production_selection

    return dataclasses.replace(get_default_production_selection(), max_new_tokens=MODAL_MAX_NEW_TOKENS)


# ----------------------------------------------------------------------
# Container image, volumes, secrets (declared when the SDK is present).
# ----------------------------------------------------------------------
if modal is not None:
    _constraint_specs = read_pin_file("constraints-gpu.txt")
    _base_specs = [
        spec for spec in read_pin_file("requirements.txt")
        if not spec.lower().startswith(("torch", "transformers", "peft", "trl",
                                        "bitsandbytes", "accelerate", "datasets"))
    ]
    legalqa_image = (
        modal.Image.debian_slim(python_version="3.11")
        .apt_install("git", "curl", "wget", "unzip", "build-essential")
        .pip_install("torch", index_url="https://download.pytorch.org/whl/cu121")
        .pip_install(*_constraint_specs)
        .pip_install(*_base_specs)
        .pip_install("kaggle", "kagglehub")
        .run_commands(
            "python -c \"import nltk; nltk.download('wordnet'); nltk.download('omw-1.4')\""
        )
        .add_local_dir(
            str(REPO_ROOT),
            remote_path="/root/LegalQA",
            copy=True,
            ignore=[
                ".venv*",
                ".git*",
                "dsc2026",
                "kaggle_dataset",
                "artifacts",
                "__pycache__",
                "*.parquet",
                "*.npy",
                "*.zip",
                ".playwright-mcp",
                ".pytest_cache",
                "fix",
            ],
        )
    )

    app = modal.App(APP_NAME, image=legalqa_image)
    data_volume = modal.Volume.from_name(DATA_VOLUME_NAME, create_if_missing=True)
    runs_volume = modal.Volume.from_name(RUNS_VOLUME_NAME, create_if_missing=True)

    # Attach available workspace secrets: user workspace holds kaggle-secret and huggingface-secret
    secrets = [
        modal.Secret.from_name("kaggle-secret"),
        modal.Secret.from_name("huggingface-secret"),
    ]
else:
    app = None
    data_volume = None
    runs_volume = None
    secrets = []


if modal is not None:

    def _write_candidate(run_dir: Path, candidate: Dict[str, Any]) -> str:
        path = run_dir / "candidate_manifest.json"
        path.write_text(json.dumps(candidate, indent=2), encoding="utf-8")
        return str(path)

    def _write_parent(run_dir: Path, parent: Dict[str, Any]) -> str:
        from src.task2.provenance.gate_report import GateReport

        path = run_dir / "parent_gate_report.json"
        path.write_text(json.dumps(parent), encoding="utf-8")
        if GateReport.load_json(path).compute_sha256() != parent.get("report_sha256"):
            raise ValueError("parent report sha mismatch: refusing cross-report reuse")
        return str(path)

    @app.function(
        gpu="A100-40GB",
        timeout=18000,
        volumes={"/data": data_volume, "/runs": runs_volume},
        secrets=secrets,
    )
    def run_modal_a100_remote(request: Dict[str, Any]) -> Dict[str, Any]:
        """Execute one Modal stage inside the A100 container."""
        import subprocess

        import torch

        validate_modal_request(request)
        stage = request["stage"]
        candidate = request["candidate_manifest"]
        print("=== Modal A100 container ===")
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
        print(f"torch: {torch.__version__} | cuda: {torch.version.cuda}")
        t_start = time.monotonic()

        sys.path.insert(0, "/root/LegalQA")
        os.chdir("/root/LegalQA")
        if candidate.get("git_commit_sha"):
            os.environ["GIT_COMMIT_SHA"] = candidate["git_commit_sha"]
            (Path("/root/LegalQA") / ".git_commit_sha").write_text(candidate["git_commit_sha"], encoding="utf-8")

        from scripts.rebuild_dense_index import build_verified_index, check_dense_alignment

        data_dir = Path("/data/legalqa-task2-clean-data")
        data_dir.mkdir(parents=True, exist_ok=True)
        chunks_file = data_dir / "legal_chunks.parquet"
        if not chunks_file.exists():
            # Setup ~/.kaggle/kaggle.json if environment credentials are present
            k_user = os.environ.get("KAGGLE_USERNAME") or os.environ.get("KAGGLE_USER")
            k_key = os.environ.get("KAGGLE_KEY") or os.environ.get("KAGGLE_API_TOKEN") or os.environ.get("KAGGLE_TOKEN")
            if k_user and k_key:
                k_dir = Path.home() / ".kaggle"
                k_dir.mkdir(parents=True, exist_ok=True)
                k_file = k_dir / "kaggle.json"
                if not k_file.exists():
                    k_file.write_text(json.dumps({"username": k_user, "key": k_key}))
                    k_file.chmod(0o600)
            print(f"[+] Pulling {DATASET_SLUG} from Kaggle...")
            subprocess.run(
                ["kaggle", "datasets", "download", "-d", DATASET_SLUG,
                 "-p", str(data_dir), "--unzip", "--force"],
                check=True,
            )
            data_volume.commit()
        else:
            print("[+] Dataset cached on volume.")

        staged_candidates = [
            data_dir / "indexes" / "dek21_rebuilt",
            data_dir / "indexes" / "dek21",
        ]
        dense_index_dir = None
        for candidate_dir in staged_candidates:
            if candidate_dir.exists():
                alignment = check_dense_alignment(str(candidate_dir), str(chunks_file))
                if alignment.get("aligned"):
                    print(f"[+] Dense index aligned at {candidate_dir.name}; reuse.")
                    dense_index_dir = str(candidate_dir)
                    break

        if dense_index_dir is None:
            print("[!] Dense index missing or misaligned; cold rebuild...")
            rebuilt = data_dir / "indexes" / "dek21_rebuilt"
            manifest = build_verified_index(
                corpus_path=str(chunks_file),
                out_dir=str(rebuilt),
                model_id=DENSE_MODEL_ID,
                revision=request["dense_revision"],
                batch_size=512,
                device="cuda:0",
            )
            data_volume.commit()
            dense_index_dir = str(rebuilt)
            print(f"[+] Rebuilt in {manifest.get('seconds')}s; committed to volume.")

        run_output_dir = Path(f"/runs/modal_{request['candidate_id']}_{int(time.time())}")
        run_output_dir.mkdir(parents=True, exist_ok=True)
        candidate_path = _write_candidate(run_output_dir, candidate)
        parent_path = _write_parent(run_output_dir, request["parent_report"])

        if stage == "micro_probe":
            from scripts.run_gpu_gate import run_gpu_gate

            report = run_gpu_gate(
                stage="a100_micro_probe",
                candidate_path=candidate_path,
                data_dir=str(data_dir),
                output_dir=str(run_output_dir),
                skip_gpu_assert=False,
                parent_report_path=parent_path,
                runtime_profile="modal_a100",
            )
            result = {"status": report.status, "stage": "micro_probe",
                      "report": report.to_dict(),
                      "report_sha256": report.compute_sha256()}
        else:
            from src.task2.config.loader import load_resolved_config
            from src.task2.pipeline.runner import run_pipeline
            from src.task2.pipeline.profiles import load_profile_from_yaml
            from src.task2.provenance.candidate import CandidateManifest
            from src.task2.provenance.gate_report import verify_gate_report

            manifest = CandidateManifest.load_json(candidate_path)
            verify_gate_report(parent_path, manifest, expected_stage="a100_micro_probe")
            print("[+] Microprobe parent chain verified.")
            test_path = resolve_test_file(data_dir, request["test_filename"])
            fingerprint = test_file_fingerprint(test_path)
            print(f"[+] Test set: {fingerprint}")
            resolved = load_resolved_config(
                "/root/LegalQA/configs/task2/algorithm.yaml",
                "/root/LegalQA/configs/task2/runtime/modal_a100.yaml",
                candidate_id=manifest.candidate_id,
            )
            manifest.validate_against_config(resolved)
            profile = load_profile_from_yaml("/root/LegalQA/configs/task2/runtime/modal_a100.yaml")
            outputs = run_pipeline(
                profile=profile,
                paths=build_remote_paths(
                    str(data_dir), dense_index_dir, str(test_path),
                    qwen_model_path=manifest.models.generator.id,
                ),
                production_cfg=build_remote_production_cfg(),
                resolved_config=resolved,
                gen_device="cuda:0",
                retrieval_device="cuda:0",
                output_dir=str(run_output_dir),
                seed=resolved.algorithm.seed,
                code_root="/root/LegalQA",
                allow_single_gpu=True,
            )
            sub_zip = outputs.get("stages", {}).get("submission", {}).get("submission_zip")
            sub_zip_b64 = None
            if sub_zip and Path(sub_zip).is_file():
                import base64
                sub_zip_b64 = base64.b64encode(Path(sub_zip).read_bytes()).decode("ascii")

            result = {"status": "PASS", "stage": "full",
                      "test_fingerprint": fingerprint,
                      "submission_zip": sub_zip,
                      "submission_zip_b64": sub_zip_b64,
                      "stages": sorted(outputs.get("stages", {}).keys())}

        runs_volume.commit()
        result["wall_seconds"] = int(time.monotonic() - t_start)
        result["output_dir"] = str(run_output_dir)
        print(f"[+] Done in {result['wall_seconds'] / 60:.1f} min: {result['status']}")
        return result

    @app.local_entrypoint()
    def main(
        stage: str = "full",
        test_path: str = "private-official.json",
        candidate: str = "",
        parent_report: str = "",
    ):
        manifest_path = Path(candidate) if candidate else None
        if manifest_path is None or not manifest_path.is_file():
            cands = sorted(
                (REPO_ROOT / "artifacts" / "candidates").glob("*/candidate_manifest.json"),
                key=lambda p: p.stat().st_mtime,
            )
            if not cands:
                raise SystemExit("no candidate manifest: pass --candidate <candidate_manifest.json>")
            manifest_path = cands[-1]
            print(f"using latest local candidate: {manifest_path}")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        required_parent = {"micro_probe": "colab_t4", "full": "a100_micro_probe"}.get(stage)
        parent_path = Path(parent_report) if parent_report else None
        if parent_path is None and required_parent:
            gates_dir = REPO_ROOT / "artifacts" / "gates" / manifest["candidate_id"]
            if gates_dir.is_dir():
                auto = sorted(gates_dir.glob(f"{required_parent}_report.json"),
                              key=lambda p: p.stat().st_mtime)
                if auto:
                    parent_path = auto[-1]
                    print(f"using parent report ({required_parent}): {parent_path}")
        if parent_path is None or not parent_path.is_file():
            raise SystemExit(
                f"Modal {stage} requires a PASS parent report for {required_parent} "
                f"(--parent-report <report.json>); no bypass."
            )
        parent = json.loads(parent_path.read_text(encoding="utf-8"))
        if not parent.get("report_sha256"):
            from src.task2.provenance.gate_report import GateReport

            parent = dict(parent, report_sha256=GateReport.load_json(parent_path).compute_sha256())
        request = build_modal_request(stage, manifest, test_path, parent)
        print(f"=== Dispatching to Modal A100: stage={stage} candidate={manifest['candidate_id']} ===")
        result = run_modal_a100_remote.remote(request)
        print(json.dumps({k: v for k, v in result.items() if k != "submission_zip_b64"}, indent=2))

        # Auto-register micro_probe report locally
        if stage == "micro_probe" and result.get("report"):
            out_gate = REPO_ROOT / "artifacts" / "gates" / manifest["candidate_id"] / "a100_micro_probe_report.json"
            out_gate.parent.mkdir(parents=True, exist_ok=True)
            out_gate.write_text(json.dumps(result["report"], indent=2), encoding="utf-8")
            print(f"\n[+] Parent gate report automatically saved locally to: {out_gate}")

        # Auto-download full submission locally
        if stage == "full" and result.get("submission_zip_b64"):
            import base64
            zip_bytes = base64.b64decode(result["submission_zip_b64"])
            sub_dir = REPO_ROOT / "artifacts" / "submissions" / manifest["candidate_id"]
            sub_dir.mkdir(parents=True, exist_ok=True)
            local_sub = sub_dir / "submission.json.zip"
            local_sub.write_bytes(zip_bytes)
            root_sub = REPO_ROOT / "submission.json.zip"
            root_sub.write_bytes(zip_bytes)
            print(f"\n[+] Submission ZIP automatically downloaded to: {local_sub}")
            print(f"[+] Root copy ready for submission at: {root_sub}")


if __name__ == "__main__" and modal is None:
    parser = argparse.ArgumentParser(description="Modal A100 runner (modal SDK required to dispatch).")
    parser.add_argument("--stage", default="full")
    parser.add_argument("--test-path", default="private-official.json")
    parser.add_argument("--candidate", default="")
    args = parser.parse_args()
    raise SystemExit(
        "modal SDK not installed locally (pip install modal) or no token; "
        f"validated args stage={args.stage} test-path={args.test_path}. "
        "Dispatch with: modal run scripts/modal_app.py --stage <micro_probe|full>"
    )
