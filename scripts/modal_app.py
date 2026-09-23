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
import subprocess
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
DENSE_MODEL_ID = "runs/20260920-215402/encoder_ft_v2"
HF_REPO = "dangphuc2109/legalqa-qwen2.5-3b-adapter"
KNOWN_TEST_FILES = ("private-official.json", "public-official.json")

#: R0 reuse adapter pins (repo + immutable revision + subfolder). The 5433
#: adapter is the generator behind the runs/20260920-215402 outputs and
#: the run_v16_dual_assembled manifest. File digests are NOT defaulted
#: here: requests must carry measured SHA-256 from pinned bytes
#: (see R0_ADAPTER_FILE_DIGESTS in reuse_contract for observed values).
R0_ADAPTER_REPO = "dangphuc2109/legalqa-qwen2.5-3b-adapter"
R0_ADAPTER_REVISION = "b6e86e35e20c403bb82b40b25f85690c987e1d02"
R0_ADAPTER_SUBFOLDER = "runs/run_5433e8b4787137c9_20260920_193355/final_adapter"
R0_GENERATOR_BASE_REVISION = "aa8e72537993ba99e69dfaafa59ed015b17504d1"

R0_ENCODER_SUBFOLDER = "runs/20260920-215402/encoder_ft_v2"

GENERATOR_MODES = ("reuse", "fresh")

#: Baked image source identity: "<git-sha> <clean|dirty|unknown>".
#: Computed on the DISPATCH client (which has .git) at import time and
#: written into the image by run_commands, because the image excludes
#: .git* so `git rev-parse` inside the container cannot attest the code.
#: The remote never substitutes the candidate SHA for this value.
IMAGE_SOURCE_FILE = "/root/LegalQA/.image_source_sha"


def client_source_identity(repo_root: Optional[Path] = None) -> str:
    """Measure the dispatch client's code revision ("sha state" or "unknown unknown")."""
    root = str(repo_root or REPO_ROOT)
    try:
        sha = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True, cwd=root).strip().lower()
        porcelain = subprocess.check_output(["git", "status", "--porcelain"], text=True, cwd=root)
        state = "dirty" if porcelain.strip() else "clean"
        if len(sha) != 40:
            return "unknown unknown"
        return f"{sha} {state}"
    except Exception:
        return "unknown unknown"


_CLIENT_SOURCE_IDENTITY = client_source_identity()


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
    kaggle_report: Optional[Dict[str, Any]] = None,
    colab_report: Optional[Dict[str, Any]] = None,
    skip_hf_upload: bool = True,
    skip_parent_check: bool = False,
    dense_model: str = "",
    generator_mode: str = "",
    adapter_spec: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Build a validated remote-execution request (pure, locally testable).

    The parent chain is mandatory: micro_probe requires the colab_t4 PASS
    report, full requires the a100_micro_probe PASS report, each for the
    same candidate. No bypass, no cross-candidate reuse.

    Full runs additionally require an explicit ``generator_mode``:
    ``reuse`` pins one adapter (repo + immutable revision + subfolder +
    measured file digests, verified before inference, trainer never runs)
    while ``fresh`` blocks every auto-copy source so the trainer always
    runs. Uploads stay off unless explicitly opted in.
    """
    if stage not in ("kaggle_t4x2", "colab_t4", "micro_probe", "full"):
        raise ValueError(f"unknown Modal stage: {stage}")
    if not candidate_manifest.get("candidate_id") or not candidate_manifest.get("git_commit_sha"):
        raise ValueError("candidate manifest must carry candidate_id and git_commit_sha")
    dense_revision = ((candidate_manifest.get("models") or {}).get("dense") or {}).get("revision", "")
    if not dense_revision or len(str(dense_revision)) != 40:
        raise ValueError("candidate must pin an immutable 40-hex dense revision")
    from scripts.run_gpu_gate import GATE_PARENTS

    # Accepted parents derive from the one shared DAG map. The full stage
    # additionally requires the microprobe report (a different gate stage).
    if stage == "kaggle_t4x2":
        allowed_parents = ()
    elif stage == "full":
        allowed_parents = ("a100_micro_probe",)
    else:
        gate_stage = {"colab_t4": "colab_t4", "micro_probe": "a100_micro_probe"}[stage]
        allowed_parents = GATE_PARENTS[gate_stage]

    if not allowed_parents:
        if parent_report is not None:
            raise ValueError(f"Modal {stage} takes no parent report")
        parent_policy = "none"
    elif skip_parent_check:
        # Explicit, recorded bypass only: never synthesize a PASS report.
        # The remote skips verify_gate_report and the execution record keeps
        # parent_policy=bypass_explicit.
        if parent_report is None:
            from src.task2.provenance.reuse_contract import explicit_bypass_parent_report

            parent_report = explicit_bypass_parent_report(candidate_manifest, allowed_parents[0])
        parent_policy = "bypass_explicit"
    else:
        if not isinstance(parent_report, dict):
            raise ValueError(f"Modal {stage} requires parent report in {allowed_parents} (no bypass)")
        if parent_report.get("status") != "PASS":
            raise ValueError(f"parent report for {stage} is not PASS")
        if parent_report.get("stage") not in allowed_parents:
            raise ValueError(
                f"parent stage mismatch for Modal {stage}: required one of {allowed_parents}, "
                f"got {parent_report.get('stage')}"
            )
        parent_candidate = parent_report.get("candidate_id", parent_report.get("candidate_sha"))
        if parent_candidate != candidate_manifest["candidate_id"]:
            raise ValueError("parent report candidate mismatch: cross-candidate reuse refused")
        if not parent_report.get("report_sha256"):
            raise ValueError("parent report lacks report_sha256")
        parent_policy = "verify_parent_report"

    mode = str(generator_mode or "").strip()
    normalized_adapter: Optional[Dict[str, Any]] = None
    if stage == "full":
        if mode not in GENERATOR_MODES:
            raise ValueError(
                f"Modal full requires an explicit generator_mode in {GENERATOR_MODES}, got {generator_mode!r}"
            )
        if mode == "reuse":
            if not isinstance(adapter_spec, dict):
                raise ValueError("reuse full requires an explicit adapter_spec (repo/revision/subfolder/file_digests)")
            from src.task2.provenance.reuse_contract import validate_adapter_spec

            normalized_adapter = validate_adapter_spec(adapter_spec)
    elif mode:
        if mode not in GENERATOR_MODES:
            raise ValueError(f"unknown generator_mode: {mode!r}")
        if adapter_spec is not None:
            from src.task2.provenance.reuse_contract import validate_adapter_spec

            normalized_adapter = validate_adapter_spec(adapter_spec)

    # Auto-resolve historical gate reports for stage='full' if not explicitly passed
    cid = candidate_manifest.get("candidate_id")
    if stage == "full" and cid:
        if kaggle_report is None:
            k_file = REPO_ROOT / "artifacts" / "gates" / cid / "kaggle_t4x2_report.json"
            if k_file.is_file():
                kaggle_report = json.loads(k_file.read_text(encoding="utf-8"))
        if colab_report is None:
            c_file = REPO_ROOT / "artifacts" / "gates" / cid / "colab_t4_report.json"
            if c_file.is_file():
                colab_report = json.loads(c_file.read_text(encoding="utf-8"))

    return {
        "stage": stage,
        "candidate_id": candidate_manifest["candidate_id"],
        "candidate_manifest": candidate_manifest,
        "test_filename": str(test_path),
        "dense_revision": str(dense_revision),
        "parent_report": parent_report,
        "parent_policy": parent_policy,
        "kaggle_report": kaggle_report,
        "colab_report": colab_report,
        "skip_hf_upload": bool(skip_hf_upload),
        "upload_policy": "disabled" if skip_hf_upload else "opt_in_enabled",
        "skip_parent_check": bool(skip_parent_check),
        "dense_model": str(dense_model),
        "generator_mode": mode,
        "adapter_spec": normalized_adapter,
    }


def validate_modal_request(request: Dict[str, Any]) -> None:
    """Fail closed on malformed remote requests before any cloud spend."""
    build_modal_request(
        stage=request.get("stage", ""),
        candidate_manifest=request.get("candidate_manifest", {}),
        test_path=request.get("test_filename", ""),
        parent_report=request.get("parent_report"),
        skip_hf_upload=request.get("skip_hf_upload", True),
        skip_parent_check=request.get("skip_parent_check", False),
        dense_model=request.get("dense_model", ""),
        generator_mode=request.get("generator_mode", ""),
        adapter_spec=request.get("adapter_spec"),
    )
    if request.get("test_filename") not in KNOWN_TEST_FILES:
        raise ValueError(f"refusing unknown test file: {request.get('test_filename')}")


def resolve_test_file(data_dir: Path, requested: str, allow_fallback: bool = True) -> Path:
    """Resolve the inference test set; fail closed on unknown/missing files.

    Full runs pass ``allow_fallback=False``: a missing private set raises
    instead of silently substituting the public set.
    """
    if requested not in KNOWN_TEST_FILES:
        raise ValueError(f"refusing unknown test file: {requested}")
    candidate = data_dir / requested
    if candidate.is_file():
        return candidate
    if not allow_fallback:
        raise FileNotFoundError(f"test file not found (no fallback on the full path): {candidate}")
    fallback = data_dir / "public-official.json"
    if requested != "public-official.json" and fallback.is_file():
        print(f"Notice: {requested} absent; falling back to public-official.json")
        return fallback
    raise FileNotFoundError(f"test file not found: {candidate}")


def measure_encoder_weights(encoder_dir: str | Path) -> str:
    """Measure staged encoder weights SHA-256 for the log (never blocking).

    Returns the hex digest, or "" when the bytes are absent/unreadable.
    """
    try:
        from src.common.dense import hash_encoder_weights_dir

        return hash_encoder_weights_dir(str(encoder_dir)).lower()
    except Exception:
        return ""


def fetch_pinned_snapshot(
    download_fn: Any,
    *,
    repo: str,
    revision: str,
    subfolder: str,
    target_dir: str | Path,
) -> Path:
    """Download a pinned HF subfolder snapshot and stage it (no gate).

    ``download_fn`` mirrors ``huggingface_hub.snapshot_download`` and is
    injectable so tests prove the revision pin without network. Returns
    the staged directory.
    """
    from scripts.rebuild_dense_index import require_pinned_revision

    revision = require_pinned_revision(revision)
    snapshot = Path(str(download_fn(repo_id=repo, revision=revision, allow_patterns=f"{subfolder}/*")))
    src = snapshot / subfolder
    if not src.is_dir():
        raise FileNotFoundError(f"pinned snapshot missing subfolder: {src}")
    dst = Path(str(target_dir))
    if dst.is_dir():
        import shutil as _shutil

        _shutil.rmtree(str(dst))
    dst.parent.mkdir(parents=True, exist_ok=True)
    import shutil as _shutil

    _shutil.copytree(str(src), str(dst))
    return dst


def resolve_adapter_plan(generator_mode: str) -> Dict[str, Any]:
    """Decide which adapter sources a full run may touch (pure, testable).

    Reuse may ONLY stage the pinned HF snapshot (repo + immutable revision
    + subfolder, digest-verified after download). The prior-run mtime scan,
    the generic volume adapter, and any unpinned fallback are never
    consulted. Fresh stages nothing: the trainer always runs.
    """
    mode = str(generator_mode or "").strip()
    if mode not in GENERATOR_MODES:
        raise ValueError(f"unknown generator_mode: {mode!r}")
    if mode == "fresh":
        return {"mode": "fresh", "allowed_sources": [], "trainer_runs": True}
    return {"mode": "reuse", "allowed_sources": ["pinned_hf_snapshot"], "trainer_runs": False}


def stage_verified_adapter(
    snapshot_subfolder: str | Path,
    target_dir: str | Path,
    spec: Dict[str, Any],
    expected_base_model: str,
) -> Dict[str, Any]:
    """Stage one pinned adapter snapshot into the run dir (no gate).

    Copies ``snapshot_subfolder`` (already downloaded at the pinned
    revision) to ``target_dir``. Digests are measured and logged when
    present in the spec, but staging never blocks on them.
    """
    from src.task2.provenance.checksums import compute_file_sha256
    from src.task2.provenance.reuse_contract import validate_adapter_spec

    validate_adapter_spec(spec)
    src = Path(snapshot_subfolder)
    dst = Path(target_dir)
    if not src.is_dir():
        raise FileNotFoundError(f"pinned adapter snapshot missing: {src}")
    import shutil as _shutil

    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.is_dir():
        _shutil.rmtree(str(dst))
    _shutil.copytree(str(src), str(dst))
    digests: Dict[str, str] = {}
    for rel in (spec.get("file_digests") or {}):
        candidate = dst / rel
        if candidate.is_file():
            try:
                digests[rel] = compute_file_sha256(candidate).lower()
            except Exception:
                pass
    return {"staged": True, "digests": digests}


def decide_full_compute_status(
    outputs: Dict[str, Any],
    submission_json: str | Path | None,
    submission_zip: str | Path | None,
    expected_ids: List[str],
) -> Dict[str, Any]:
    """Compute the terminal compute status for a full run (pure, testable).

    PASS only when the runner did not stop INCOMPLETE, the submission JSON
    exists and is nonempty, IDs exactly match the test set actually used,
    and ZIP inner/loose bytes and SHAs have parity. Anything else is FAIL
    (or INCOMPLETE when the runner stopped at the deadline). Never writes
    files or uploads.
    """
    from src.task2.scorer_contract import validate_prediction_payload, verify_zip_inner_matches_loose

    reasons: List[str] = []
    if isinstance(outputs, dict) and outputs.get("status") == "INCOMPLETE":
        return {"compute_status": "INCOMPLETE", "reasons": ["runner stopped INCOMPLETE at the deadline gate"]}
    stages = outputs.get("stages", {}) if isinstance(outputs, dict) else {}
    if "submission" not in stages:
        reasons.append("runner produced no submission stage")
    for label, path in (("submission_json", submission_json), ("submission_zip", submission_zip)):
        if not path or not Path(str(path)).is_file():
            reasons.append(f"{label} missing: {path}")
    if reasons:
        return {"compute_status": "FAIL", "reasons": reasons}
    try:
        submission = json.loads(Path(str(submission_json)).read_text(encoding="utf-8"))
    except Exception as exc:
        return {"compute_status": "FAIL", "reasons": [f"submission_json unreadable: {exc}"]}
    if not isinstance(submission, dict) or not submission:
        return {"compute_status": "FAIL", "reasons": ["submission_json empty or not an object"]}
    try:
        validate_prediction_payload(submission, [str(v) for v in expected_ids])
    except Exception as exc:
        return {"compute_status": "FAIL", "reasons": [f"submission ID check failed: {exc}"]}
    try:
        parity = verify_zip_inner_matches_loose(str(submission_zip), str(submission_json))
    except Exception as exc:
        return {"compute_status": "FAIL", "reasons": [f"ZIP parity failed: {exc}"]}
    return {"compute_status": "PASS", "reasons": [], "zip_report": parity}


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

#: Calibrated generation ceiling (tokens) covering 99.5% of statutory answers
#: with high-recall statutory and reasoning capacity on A100.
MODAL_MAX_NEW_TOKENS = 1536


def build_remote_paths(
    data_dir: str,
    dense_index_dir: str,
    test_path: str,
    qwen_model_path: str = "Qwen/Qwen2.5-3B-Instruct",
    predicted_inference_seconds: int = 3600,
) -> Dict[str, Any]:
    """Runner paths for the Modal container (pure, testable)."""
    return {
        "data_dir": str(data_dir),
        "bm25_dir": str(Path(data_dir) / "indexes" / "bm25"),
        "dek21_dir": str(dense_index_dir),
        "qwen_model_path": str(qwen_model_path),
        "public_test_path": str(test_path),
        "deadline_budget_seconds": MODAL_DEADLINE_BUDGET_SECONDS,
        "predicted_inference_seconds": int(predicted_inference_seconds),
    }


def build_remote_production_cfg(resolved_cfg: Optional[Any] = None) -> Any:
    """Production selection for the Modal full run, sourced from hashed runtime config."""
    import dataclasses

    from src.task2.production_config import get_default_production_selection

    max_tokens = MODAL_MAX_NEW_TOKENS
    best_cand = "dual_assembled"
    if resolved_cfg is not None and hasattr(resolved_cfg, "runtime") and hasattr(resolved_cfg.runtime, "inference"):
        inf = resolved_cfg.runtime.inference
        max_tokens = getattr(inf, "max_new_tokens", max_tokens)
        best_cand = getattr(inf, "best_fixed_candidate", best_cand)

    return dataclasses.replace(
        get_default_production_selection(),
        max_new_tokens=max_tokens,
        best_fixed_candidate=best_cand,
    )


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
        # Bake the dispatch client's code revision into the image: .git*
        # is excluded from add_local_dir below, so the container cannot
        # attest its own source via git. The remote reads this file as the
        # executed revision and never substitutes the candidate SHA.
        .run_commands(
            f"mkdir -p /root/LegalQA && printf '%s' '{_CLIENT_SOURCE_IDENTITY}' > {IMAGE_SOURCE_FILE}"
        )
        .add_local_dir(
            str(REPO_ROOT),
            remote_path="/root/LegalQA",
            copy=True,
            ignore=[
                ".venv*",
                ".git*",
                ".remember*",
                ".agents*",
                ".playwright-mcp*",
                ".pytest_cache*",
                ".superpowers*",
                "dsc2026*",
                "kaggle_dataset*",
                "artifacts*",
                "__pycache__*",
                "*.parquet",
                "*.npy",
                "*.zip",
                "*.log",
                ".DS_Store",
                "fix*",
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

    def _write_parent(run_dir: Path, parent: Optional[Dict[str, Any]], skip_check: bool = False) -> Optional[str]:
        if not parent:
            return None
        from src.task2.provenance.gate_report import GateReport

        path = run_dir / "parent_gate_report.json"
        path.write_text(json.dumps(parent, indent=2), encoding="utf-8")
        if skip_check or parent.get("status") != "PASS":
            # Explicit bypass records (BYPASSED_EXPLICIT) are filed as-is;
            # they are never parsed as gate reports nor upgraded to PASS.
            return str(path)
        declared_sha = parent.get("report_sha256")
        if declared_sha:
            actual_sha = GateReport.load_json(path).compute_sha256()
            if actual_sha != declared_sha:
                raise ValueError(f"parent report sha mismatch: expected {declared_sha}, got {actual_sha}")
        return str(path)

    @app.function(
        gpu="T4:2",
        timeout=3600,
        volumes={"/data": data_volume, "/runs": runs_volume},
        secrets=secrets,
    )
    def run_modal_kaggle_t4x2_remote(request: Dict[str, Any]) -> Dict[str, Any]:
        """Execute kaggle_t4x2 root gate stage on remote Modal Dual Tesla T4 GPUs."""
        import subprocess
        import torch

        validate_modal_request(request)
        candidate = request["candidate_manifest"]
        print("=== Modal Dual Tesla T4 container ===")
        print(f"GPUs: {torch.cuda.device_count()} x {torch.cuda.get_device_name(0)}")
        print(f"torch: {torch.__version__} | cuda: {torch.version.cuda}")

        sys.path.insert(0, "/root/LegalQA")
        os.chdir("/root/LegalQA")
        if candidate.get("git_commit_sha"):
            os.environ["GIT_COMMIT_SHA"] = candidate["git_commit_sha"]
            (Path("/root/LegalQA") / ".git_commit_sha").write_text(candidate["git_commit_sha"], encoding="utf-8")

        data_dir = Path("/data/legalqa-task2-clean-data")
        data_dir.mkdir(parents=True, exist_ok=True)
        chunks_file = data_dir / "legal_chunks.parquet"
        if not chunks_file.exists():
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

        run_output_dir = Path(f"/runs/modal_kaggle_{request['candidate_id']}_{int(time.time())}")
        run_output_dir.mkdir(parents=True, exist_ok=True)
        candidate_path = _write_candidate(run_output_dir, candidate)

        from scripts.run_gpu_gate import run_gpu_gate

        report = run_gpu_gate(
            stage="kaggle_t4x2",
            candidate_path=candidate_path,
            data_dir=str(data_dir),
            output_dir=str(run_output_dir),
            skip_gpu_assert=False,
            parent_report_path=None,
        )
        runs_volume.commit()
        return {
            "status": report.status,
            "stage": "kaggle_t4x2",
            "report": report.to_dict(),
            "report_sha256": report.compute_sha256(),
        }

    @app.function(
        gpu="T4",
        timeout=3600,
        volumes={"/data": data_volume, "/runs": runs_volume},
        secrets=secrets,
    )
    def run_modal_t4_remote(request: Dict[str, Any]) -> Dict[str, Any]:
        """Execute colab_t4 gate stage on a remote Modal Tesla T4 GPU."""
        import subprocess

        import torch

        validate_modal_request(request)
        stage = request["stage"]
        candidate = request["candidate_manifest"]
        print("=== Modal Tesla T4 container ===")
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"torch: {torch.__version__} | cuda: {torch.version.cuda}")

        sys.path.insert(0, "/root/LegalQA")
        os.chdir("/root/LegalQA")
        if candidate.get("git_commit_sha"):
            os.environ["GIT_COMMIT_SHA"] = candidate["git_commit_sha"]
            (Path("/root/LegalQA") / ".git_commit_sha").write_text(candidate["git_commit_sha"], encoding="utf-8")

        data_dir = Path("/data/legalqa-task2-clean-data")
        data_dir.mkdir(parents=True, exist_ok=True)
        chunks_file = data_dir / "legal_chunks.parquet"
        if not chunks_file.exists():
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

        run_output_dir = Path(f"/runs/modal_{request['candidate_id']}_{int(time.time())}")
        run_output_dir.mkdir(parents=True, exist_ok=True)
        candidate_path = _write_candidate(run_output_dir, candidate)
        skip_parent = bool(request.get("skip_parent_check"))
        parent_path = _write_parent(run_output_dir, request.get("parent_report"), skip_check=skip_parent)

        from scripts.run_gpu_gate import run_gpu_gate

        report = run_gpu_gate(
            stage="colab_t4",
            candidate_path=candidate_path,
            data_dir=str(data_dir),
            output_dir=str(run_output_dir),
            skip_gpu_assert=False,
            parent_report_path=parent_path,
        )
        runs_volume.commit()
        return {
            "status": report.status,
            "stage": "colab_t4",
            "report": report.to_dict(),
            "report_sha256": report.compute_sha256(),
        }

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
            # Declared candidate pin only: the ACTUAL running revision is
            # measured separately via get_executed_git_identity and stored
            # as executed_git_sha in execution_record.json (P0-D).
            os.environ["GIT_COMMIT_SHA"] = candidate["git_commit_sha"]
            (Path("/root/LegalQA") / ".git_commit_sha").write_text(candidate["git_commit_sha"], encoding="utf-8")

        from scripts.rebuild_dense_index import build_verified_index

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

        # Ensure BM25 index is present; rebuild if missing. The marker is the
        # bm25s params file inside the index directory (not a flat file).
        bm25_dir = data_dir / "indexes" / "bm25"
        if not (bm25_dir / "bm25s_index" / "params.index.json").exists():
            print("[!] BM25 index missing; cold building verified index...")
            from scripts.rebuild_bm25_index import build_verified_bm25
            build_verified_bm25(corpus_path=str(chunks_file), out_dir=str(bm25_dir))
            data_volume.commit()
            print("[+] Built verified BM25 index and committed to volume.")
        else:
            print("[+] BM25 index cached on volume.")

        run_output_dir = Path(f"/runs/modal_{request['candidate_id']}_{int(time.time())}")
        run_output_dir.mkdir(parents=True, exist_ok=True)
        candidate_path = _write_candidate(run_output_dir, candidate)
        parent_path = _write_parent(
            run_output_dir, request["parent_report"],
            skip_check=bool(request.get("skip_parent_check")),
        )

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
            dense_model_id = request.get("dense_model") or DENSE_MODEL_ID
            cand_dense = (candidate.get("models", {}).get("dense", {}) or {})
            if not request.get("dense_model") and cand_dense.get("id"):
                dense_model_id = cand_dense["id"]
            dense_revision = request["dense_revision"]

            if "encoder_ft" in str(dense_model_id) or "20260920-215402" in str(dense_model_id):
                dense_local = data_dir / "models" / "encoder_ft_v2"
                if (dense_local / "model.safetensors").is_file():
                    print(f"[+] Using cached encoder at {dense_local}.")
                else:
                    print(f"[+] Fetching encoder_ft_v2 from HF repo {HF_REPO} at pinned revision {dense_revision}...")
                    from huggingface_hub import snapshot_download
                    fetch_pinned_snapshot(
                        snapshot_download,
                        repo=HF_REPO,
                        revision=dense_revision,
                        subfolder=R0_ENCODER_SUBFOLDER,
                        target_dir=dense_local,
                    )
                    data_volume.commit()
                    print(f"[+] encoder_ft_v2 cached to data volume: {dense_local}")
                dense_model_id = str(dense_local)
                print(f"[+] Encoder weights SHA: {measure_encoder_weights(dense_local)[:16] or 'unmeasured'}...")

            # Dense/index: measure the encoder bytes for the log, then reuse
            # a staged index when it matches, else cold-rebuild side-by-side
            # (automatic, never blocking).
            from scripts.rebuild_dense_index import select_verified_dense_index
            from src.common.dense import preprocessing_fingerprint

            encoder_weights_sha = measure_encoder_weights(str(dense_model_id))
            _expected_preprocessing = preprocessing_fingerprint(
                max_seq_length=256, vietnamese_tokenized=True,
                pooling="mean", normalized=True, dtype="float16",
            )
            _dense_expected = {
                "model_id": str(dense_model_id),
                "revision": dense_revision,
                "encoder_weights_sha256": encoder_weights_sha,
                "preprocessing": _expected_preprocessing,
            }
            if "encoder_ft" in str(dense_model_id):
                staged_candidates = [data_dir / "indexes" / "encoder_ft_v2_rebuilt"]
            else:
                staged_candidates = [
                    data_dir / "indexes" / "dek21_rebuilt",
                    data_dir / "indexes" / "dek21",
                ]
            _selection = select_verified_dense_index(
                [str(p) for p in staged_candidates], str(chunks_file), _dense_expected,
            )
            dense_index_dir = None
            if _selection.get("action") == "reused":
                print(f"[+] Dense index strictly verified at {Path(str(_selection['index_dir'])).name}; reuse.")
                dense_index_dir = str(_selection["index_dir"])
            else:
                for failure in _selection.get("failures", []):
                    print(f"[!] Dense index rejected: {failure}")

            if dense_index_dir is None:
                idx_name = "encoder_ft_v2_rebuilt" if "encoder_ft" in str(dense_model_id) else "dek21_rebuilt"
                rebuilt = data_dir / "indexes" / idx_name
                print(f"[!] Dense index missing or misaligned; cold rebuild to {idx_name} using {dense_model_id}...")
                manifest = build_verified_index(
                    corpus_path=str(chunks_file),
                    out_dir=str(rebuilt),
                    model_id=dense_model_id,
                    revision=dense_revision,
                    batch_size=512,
                    device="cuda:0",
                )
                data_volume.commit()
                dense_index_dir = str(rebuilt)
                print(f"[+] Rebuilt in {manifest.get('seconds')}s; committed to volume.")

            from src.task2.config.loader import load_resolved_config
            from src.task2.pipeline.runner import run_pipeline
            from src.task2.pipeline.profiles import load_profile_from_yaml
            from src.task2.provenance.candidate import CandidateManifest
            from src.task2.provenance.gate_report import verify_gate_report

            manifest = CandidateManifest.load_json(candidate_path)
            if not request.get("skip_parent_check"):
                verify_gate_report(parent_path, manifest, expected_stage="a100_micro_probe")
                print("[+] Microprobe parent chain verified.")
            else:
                print("[*] Notice: skip_parent_check active, skipping verify_gate_report.")
            test_path = resolve_test_file(data_dir, request["test_filename"], allow_fallback=False)
            fingerprint = test_file_fingerprint(test_path)
            print(f"[+] Test set: {fingerprint}")
            with open(test_path, "r", encoding="utf-8") as _tf:
                _expected_ids = [str(k) for k in json.load(_tf).keys()]
            resolved = load_resolved_config(
                "/root/LegalQA/configs/task2/algorithm.yaml",
                "/root/LegalQA/configs/task2/runtime/modal_a100.yaml",
                candidate_id=manifest.candidate_id,
            )
            manifest.validate_against_config(resolved)
            # Explicit generator mode: reuse stages the pinned adapter
            # snapshot (repo + immutable revision + subfolder); fresh stages
            # nothing so the runner always trains.
            from src.task2.provenance.reuse_contract import validate_adapter_spec

            generator_mode = str(request.get("generator_mode") or "")
            if generator_mode not in ("reuse", "fresh"):
                raise ValueError(f"full run requires explicit generator_mode reuse|fresh, got {generator_mode!r}")
            adapter_plan = resolve_adapter_plan(generator_mode)
            print(f"[+] Generator mode: {generator_mode} (allowed sources: {adapter_plan['allowed_sources']})")
            target_adapter = run_output_dir / "checkpoints" / "generator" / "hf_adapter"
            staged_adapter_report: Optional[Dict[str, Any]] = None
            adapter_spec: Optional[Dict[str, Any]] = None
            if generator_mode == "reuse":
                raw_spec = request.get("adapter_spec")
                if not isinstance(raw_spec, dict):
                    raise ValueError("reuse full requires request adapter_spec; refusing to guess an adapter")
                adapter_spec = validate_adapter_spec(raw_spec)
                from huggingface_hub import snapshot_download

                print(f"[+] Fetching pinned adapter {adapter_spec['repo']}@{adapter_spec['revision']}:"
                      f"{adapter_spec['subfolder']} ...")
                dl_p = snapshot_download(
                    repo_id=adapter_spec["repo"],
                    revision=adapter_spec["revision"],
                    allow_patterns=f"{adapter_spec['subfolder']}/*",
                )
                src_ad = Path(dl_p) / adapter_spec["subfolder"]
                staged_adapter_report = stage_verified_adapter(
                    src_ad, target_adapter, adapter_spec, manifest.models.generator.id,
                )
                print(f"[+] Pinned adapter staged: digests={staged_adapter_report['digests']}")
            else:
                print("[+] Fresh mode: the trainer will run.")

            profile = load_profile_from_yaml("/root/LegalQA/configs/task2/runtime/modal_a100.yaml")
            outputs = run_pipeline(
                profile=profile,
                paths=build_remote_paths(
                    str(data_dir), dense_index_dir, str(test_path),
                    qwen_model_path=manifest.models.generator.id,
                ),
                production_cfg=build_remote_production_cfg(resolved),
                resolved_config=resolved,
                gen_device="cuda:0",
                retrieval_device="cuda:0",
                output_dir=str(run_output_dir),
                seed=resolved.algorithm.seed,
                code_root="/root/LegalQA",
                allow_single_gpu=True,
                generator_mode=generator_mode,
                expected_adapter=adapter_spec if generator_mode == "reuse" else None,
            )
            # Execution record: candidate SHA vs ACTUAL running SHA stay separate.
            from src.task2.provenance.reuse_contract import (
                build_source_identity,
                get_executed_git_identity,
                write_execution_record,
            )

            _exec_git = get_executed_git_identity("/root/LegalQA")
            _gen_stage = outputs.get("stages", {}).get("generator", {})
            _source_identity = build_source_identity(
                candidate_manifest,
                _exec_git["executed_git_sha"],
                _exec_git["dirty"],
                {
                    "executed_source": _exec_git.get("source", "unknown"),
                    "generator_mode": generator_mode,
                    "training_performed": bool(_gen_stage.get("training_performed", generator_mode == "fresh")),
                    "adapter_spec": adapter_spec if generator_mode == "reuse" else None,
                    "adapter_digests": (_gen_stage.get("adapter_digests")
                                        or (staged_adapter_report or {}).get("digests")),
                    "dense_revision": dense_revision,
                    "encoder_weights_sha256": encoder_weights_sha,
                    "test_fingerprint": fingerprint,
                    "parent_policy": request.get("parent_policy", "verify_parent_report"),
                    "upload_policy": request.get("upload_policy", "disabled"),
                },
            )
            write_execution_record(run_output_dir / "execution_record.json", _source_identity)

            sub_json = outputs.get("stages", {}).get("submission", {}).get("submission_json")
            sub_zip = outputs.get("stages", {}).get("submission", {}).get("submission_zip")
            # Terminal compute status: PASS only on a complete runner plus a
            # real, nonempty, ID-checked submission with ZIP parity.
            _status = decide_full_compute_status(outputs, sub_json, sub_zip, _expected_ids)
            compute_status = str(_status["compute_status"])
            if compute_status != "PASS":
                print(f"[!] Full run compute status {compute_status}: {'; '.join(_status.get('reasons', []))}")
            sub_zip_b64 = None
            if compute_status == "PASS" and sub_zip and Path(sub_zip).is_file():
                import base64
                sub_zip_b64 = base64.b64encode(Path(sub_zip).read_bytes()).decode("ascii")

            # Release is separate from compute: uploads stay OFF unless the
            # request explicitly opts in, and only a verified PASS bundles.
            # A failed upload sets release FAILED without retraining and
            # never claims an uploaded PASS.
            hf_res: Optional[Dict[str, Any]] = None
            release_status = "disabled"
            upload_opted_in = request.get("upload_policy") == "opt_in_enabled" and not request.get("skip_hf_upload")
            if upload_opted_in and compute_status == "PASS" and not os.environ.get("HF_TOKEN"):
                print("[*] Upload opted in but no HF_TOKEN in the container: release blocked, compute stays PASS.")
                upload_opted_in = False
                release_status = "blocked_no_token"
            if upload_opted_in and compute_status == "PASS":
                try:
                    import datetime
                    from src.task2.provenance.run_bundle import build_production_run_bundle
                    from src.task2.hf_uploader import upload_run_bundle_to_hf

                    timestamp_str = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%d_%H%M%S")
                    run_id = f"run_{manifest.candidate_id}_{timestamp_str}"
                    bundle_dir = run_output_dir / "bundle" / run_id
                    bundle_dir.parent.mkdir(parents=True, exist_ok=True)

                    gen_stage = outputs.get("stages", {}).get("generator", {})
                    eval_stage = outputs.get("stages", {}).get("evaluation", {})
                    # Measured trainer outputs only: missing telemetry blocks
                    # the release claim instead of falling back to estimates.
                    if "optimizer_steps" not in gen_stage or "dataset_size" not in gen_stage:
                        raise ValueError(
                            "refusing release: trainer manifest lacks measured "
                            "optimizer_steps/dataset_size"
                        )
                    opt_steps = int(gen_stage["optimizer_steps"])
                    dataset_sz = int(gen_stage["dataset_size"])
                    metrics = {
                        "selected_meteor": eval_stage.get("selected_meteor"),
                        "candidate_family_meteors": eval_stage.get("candidate_family_meteors"),
                        "dev_sample_size": eval_stage.get("sample_size"),
                        "held_out_fold": eval_stage.get("held_out_fold"),
                    }

                    cand_adapters = [
                        run_output_dir / "checkpoints" / "generator" / "hf_adapter",
                        run_output_dir / "checkpoints" / "generator",
                        run_output_dir / "hf_adapter",
                        run_output_dir,
                    ]
                    adapter_src = next((p for p in cand_adapters if (p / "adapter_model.safetensors").is_file()), run_output_dir)

                    k_rep_p = run_output_dir / "kaggle_t4x2_report.json"
                    if not k_rep_p.exists() and request.get("kaggle_report"):
                        k_rep_p.write_text(json.dumps(request["kaggle_report"], indent=2), encoding="utf-8")

                    c_rep_p = run_output_dir / "colab_t4_report.json"
                    if not c_rep_p.exists() and request.get("colab_report"):
                        c_rep_p.write_text(json.dumps(request["colab_report"], indent=2), encoding="utf-8")

                    if not k_rep_p.is_file():
                        if not request.get("skip_parent_check"):
                            raise ValueError("Missing verified kaggle_t4x2_report.json required for release packaging")
                        else:
                            print("[*] Notice: skip_parent_check enabled; continuing packaging without local kaggle_t4x2_report.json.")

                    sub_json = outputs.get("stages", {}).get("submission", {}).get("submission_json")
                    sub_zip = outputs.get("stages", {}).get("submission", {}).get("submission_zip")
                    prov_file = run_output_dir / "submission_provenance.json"
                    ds_manifest = data_dir / "dataset_manifest.json"
                    ds_report = data_dir / "validation_report.json"

                    print("\n" + "=" * 65)
                    print(" [+] Packaging Audited Production Run Bundle for Hugging Face Release ")
                    print("=" * 65)
                    build_production_run_bundle(
                        run_id=run_id,
                        candidate=manifest,
                        adapter_source_dir=adapter_src,
                        kaggle_report_path=k_rep_p,
                        colab_t4_report_path=c_rep_p if (c_rep_p and c_rep_p.is_file()) else None,
                        a100_micro_probe_report_path=parent_path,
                        train_log_path=run_output_dir / "train.log",
                        output_dir=bundle_dir,
                        metrics=metrics,
                        optimizer_steps=opt_steps,
                        training_sample_count=dataset_sz,
                        num_train_epochs=int(resolved.algorithm.generator.num_train_epochs),
                        effective_batch_size=8,
                        hf_repository="dangphuc2109/legalqa-qwen2.5-3b-adapter",
                        dataset_manifest_path=ds_manifest if ds_manifest.is_file() else None,
                        dataset_validation_report_path=ds_report if ds_report.is_file() else None,
                        runtime_profile="modal_a100",
                        submission_path=sub_json if (sub_json and Path(sub_json).is_file()) else None,
                        submission_provenance_path=prov_file if prov_file.is_file() else None,
                    )

                    print("\n" + "=" * 65)
                    print(" [+] Uploading Audited Run Bundle to Hugging Face Hub ")
                    print("=" * 65)
                    hf_res = upload_run_bundle_to_hf(
                        bundle_dir=bundle_dir,
                        repo_id="dangphuc2109/legalqa-qwen2.5-3b-adapter",
                        run_id=run_id,
                    )
                    print(f"[+] Hugging Face upload complete! Commit: {hf_res.get('commit_sha')} -> {hf_res.get('repo_url')}")
                    release_status = "succeeded"
                except Exception as e:
                    print(f"[!] Upload failed; compute stays {compute_status}, release FAILED (no retrain): {e}")
                    hf_res = {"status": "FAILED", "error": str(e)}
                    release_status = "failed"
            elif upload_opted_in and compute_status != "PASS":
                print(f"[*] Upload opted in but compute is {compute_status}: refusing to release an unverified bundle.")
                release_status = "blocked_unverified_compute"

            result = {"status": compute_status, "compute_status": compute_status,
                      "release_status": release_status, "stage": "full",
                      "generator_mode": generator_mode,
                      "test_fingerprint": fingerprint,
                      "submission_zip": sub_zip,
                      "submission_zip_b64": sub_zip_b64,
                      "huggingface": hf_res,
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
        skip_parent_check: bool = False,
        dense_model: str = "",
        generator_mode: str = "",
        adapter_spec: str = "",
        allow_hf_upload: bool = False,
    ):
        if stage == "full":
            if generator_mode not in ("reuse", "fresh"):
                raise SystemExit("full requires an explicit --generator-mode reuse|fresh")
            if generator_mode == "reuse":
                # No mtime auto-pick, no auto --skip-parent-check on the
                # reuse path: every selection must be explicit (P0-D/P1).
                if not candidate or not Path(candidate).is_file():
                    raise SystemExit("reuse full requires an explicit --candidate <candidate_manifest.json>")
                if not parent_report and not skip_parent_check:
                    raise SystemExit(
                        "reuse full requires an explicit --parent-report <report.json> "
                        "or an explicit --skip-parent-check bypass"
                    )
                if not adapter_spec or not Path(adapter_spec).is_file():
                    raise SystemExit(
                        "reuse full requires an explicit --adapter-spec <adapter_spec.json> "
                        "(repo/revision/subfolder/measured file_digests)"
                    )
        manifest_path = Path(candidate) if candidate else None
        if manifest_path is None or not manifest_path.is_file():
            cands = sorted(
                (REPO_ROOT / "artifacts" / "candidates").glob("*/candidate_manifest.json"),
                key=lambda p: p.stat().st_mtime,
            )
            if not cands:
                raise SystemExit("no candidate manifest: pass --candidate <candidate_manifest.json>")
            # Use the newest minted candidate manifest
            manifest_path = cands[-1]
            print(f"using local candidate: {manifest_path}")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        stage_parents = {
            "kaggle_t4x2": (),
            "micro_probe": ("kaggle_t4x2",),
            "full": ("a100_micro_probe",),
            "colab_t4": ("kaggle_t4x2",),
        }.get(stage, ())
        parent_path = Path(parent_report) if parent_report else None
        if parent_path is None and stage_parents:
            gates_dir = REPO_ROOT / "artifacts" / "gates" / manifest["candidate_id"]
            if gates_dir.is_dir():
                for p_stage in stage_parents:
                    auto = sorted(gates_dir.glob(f"{p_stage}_report.json"),
                                  key=lambda p: p.stat().st_mtime)
                    if auto:
                        parent_path = auto[-1]
                        print(f"using parent report ({p_stage}): {parent_path}")
                        break
        parent = None
        if stage_parents:
            if (parent_path is None or not parent_path.is_file()) and not skip_parent_check:
                raise SystemExit(
                    f"Modal {stage} requires a PASS parent report from {stage_parents} "
                    f"(--parent-report <report.json>); pass --skip-parent-check to run directly."
                )
            if parent_path and parent_path.is_file():
                parent = json.loads(parent_path.read_text(encoding="utf-8"))
                if not parent.get("report_sha256"):
                    from src.task2.provenance.gate_report import GateReport

                    parent = dict(parent, report_sha256=GateReport.load_json(parent_path).compute_sha256())

        k_rep = None
        c_rep = None
        if stage == "full":
            gates_dir = REPO_ROOT / "artifacts" / "gates" / manifest["candidate_id"]
            k_path = gates_dir / "kaggle_t4x2_report.json"
            if k_path.is_file():
                k_rep = json.loads(k_path.read_text(encoding="utf-8"))
            c_path = gates_dir / "colab_t4_report.json"
            if c_path.is_file():
                c_rep = json.loads(c_path.read_text(encoding="utf-8"))

        adapter_spec_dict: Optional[Dict[str, Any]] = None
        if adapter_spec:
            adapter_spec_dict = json.loads(Path(adapter_spec).read_text(encoding="utf-8"))

        # Uploads default OFF for experiments: only --allow-hf-upload opts in.
        request = build_modal_request(
            stage, manifest, test_path, parent,
            kaggle_report=k_rep, colab_report=c_rep,
            skip_hf_upload=not allow_hf_upload,
            skip_parent_check=skip_parent_check,
            dense_model=dense_model,
            generator_mode=generator_mode,
            adapter_spec=adapter_spec_dict,
        )

        if stage == "kaggle_t4x2":
            print(f"=== Dispatching to Modal Dual Tesla T4: stage={stage} candidate={manifest['candidate_id']} ===")
            result = run_modal_kaggle_t4x2_remote.remote(request)
            print(json.dumps(result, indent=2))
            if result.get("report"):
                out_gate = REPO_ROOT / "artifacts" / "gates" / manifest["candidate_id"] / "kaggle_t4x2_report.json"
                out_gate.parent.mkdir(parents=True, exist_ok=True)
                out_gate.write_text(json.dumps(result["report"], indent=2), encoding="utf-8")
                print(f"\n[+] Kaggle T4x2 gate report automatically saved locally to: {out_gate}")
            return result

        if stage == "colab_t4":
            print(f"=== Dispatching to Modal Tesla T4: stage={stage} candidate={manifest['candidate_id']} ===")
            result = run_modal_t4_remote.remote(request)
            print(json.dumps(result, indent=2))
            if result.get("report"):
                out_gate = REPO_ROOT / "artifacts" / "gates" / manifest["candidate_id"] / "colab_t4_report.json"
                out_gate.parent.mkdir(parents=True, exist_ok=True)
                out_gate.write_text(json.dumps(result["report"], indent=2), encoding="utf-8")
                print(f"\n[+] Colab T4 gate report automatically saved locally to: {out_gate}")
            return result

        print(f"=== Dispatching to Modal A100: stage={stage} candidate={manifest['candidate_id']} ===")
        result = run_modal_a100_remote.remote(request)
        print(json.dumps({k: v for k, v in result.items() if k != "submission_zip_b64"}, indent=2))

        # Auto-register micro_probe report locally
        if stage == "micro_probe" and result.get("report"):
            out_gate = REPO_ROOT / "artifacts" / "gates" / manifest["candidate_id"] / "a100_micro_probe_report.json"
            out_gate.parent.mkdir(parents=True, exist_ok=True)
            out_gate.write_text(json.dumps(result["report"], indent=2), encoding="utf-8")
            print(f"\n[+] Parent gate report automatically saved locally to: {out_gate}")

        # Download the verified full submission locally (compute PASS only).
        # Never writes a root submission ZIP: only the versioned path below.
        if stage == "full" and result.get("compute_status") == "PASS" and result.get("submission_zip_b64"):
            import base64
            zip_bytes = base64.b64decode(result["submission_zip_b64"])
            sub_dir = REPO_ROOT / "artifacts" / "submissions" / manifest["candidate_id"]
            sub_dir.mkdir(parents=True, exist_ok=True)
            local_sub = sub_dir / "submission.json.zip"
            local_sub.write_bytes(zip_bytes)
            print(f"\n[+] Verified submission ZIP downloaded to: {local_sub}")
        elif stage == "full":
            print(f"[*] No submission downloaded: compute_status={result.get('compute_status')}, "
                  f"release_status={result.get('release_status')}")

        # Report Hugging Face release
        if stage == "full" and result.get("huggingface"):
            hf_info = result["huggingface"]
            print(f"\n[+] Hugging Face Release Status: {hf_info.get('status')}")
            if hf_info.get("repo_url"):
                print(f"[+] Model & Proof Repository: {hf_info.get('repo_url')}")
            if hf_info.get("commit_sha"):
                print(f"[+] Remote Commit SHA: {hf_info.get('commit_sha')}")


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
