#!/usr/bin/env python3
"""Stage-Aware Google Colab Session Orchestrator for LegalQA Task 2.

Manages remote lifecycle via Colab CLI for both:
- colab-t4 (Single-T4 promotion gate)
- a100 (Full production training gated behind verified T4 reports and micro-probe)

Lifecycle:
1. Local preflight: Git, CI status, candidate manifest, gate reports, credentials.
2. VM provisioning: colab new -s <session> --gpu <T4|A100>
3. Bootstrap upload: colab upload -s <session> <local_bootstrap> /content/legalqa_bootstrap
4. Remote entry execution: colab exec -s <session> -f scripts/colab_remote_entry.py
   (NOTE: --timeout is NEVER passed to colab exec)
5. Evidence download: colab download -s <session> /content/legalqa_run <local_artifacts>
6. Local verification of downloaded gate reports.
7. Cleanup: colab stop -s <session> in finally block (unless --keep-alive).

Usage:
  python scripts/launch_colab_training.py --stage colab-t4 --candidate PATH --kaggle-report PATH
  python scripts/launch_colab_training.py --stage a100 --candidate PATH --kaggle-report PATH --colab-t4-report PATH
"""

from __future__ import annotations

import argparse
import datetime
import json
import os
import shutil
import subprocess
import sys
import tempfile
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.common.env_loader import parse_env_file
from src.task2.provenance.candidate import CandidateManifest
from src.task2.provenance.gate_report import GateReport, verify_gate_report
from scripts.verify_ci_status import verify_ci_status


def find_colab_binary(custom_path: Optional[str] = None) -> str:
    """Locate colab CLI binary."""
    if custom_path and os.path.isfile(custom_path) and os.access(custom_path, os.X_OK):
        return custom_path

    in_path = shutil.which("colab")
    if in_path:
        return in_path

    home_local = Path.home() / ".local" / "bin" / "colab"
    if home_local.is_file() and os.access(home_local, os.X_OK):
        return str(home_local)

    # For testing / mock fallback if colab isn't installed locally
    return "colab"


def build_run_request(
    stage: str,
    candidate_id: str,
    git_commit_sha: str,
    dataset_slug: str,
    dataset_version: int,
    algorithm_path: str = "configs/task2/algorithm.yaml",
    runtime_profile_path: Optional[str] = None,
    output_dir: str = "/content/legalqa_run",
    requested_gpu: Optional[str] = None,
    repository: str = "https://github.com/silent9669/LegalQA.git",
) -> Dict[str, Any]:
    """Construct JSON run request contract uploaded to /content/legalqa_bootstrap/run_request.json."""
    norm_stage = stage.replace("-", "_")
    req_gpu = requested_gpu or ("T4" if "t4" in norm_stage else "A100")
    rt_path = runtime_profile_path or f"configs/task2/runtime/{norm_stage}.yaml"

    return {
        "stage": norm_stage,
        "candidate_id": candidate_id,
        "requested_gpu": req_gpu,
        "repository": repository,
        "git_commit_sha": git_commit_sha,
        "dataset_slug": dataset_slug,
        "dataset_version": dataset_version,
        "algorithm_path": algorithm_path,
        "runtime_profile_path": rt_path,
        "output_dir": output_dir,
        "created_at_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    }


class ColabLauncher:
    """Manages full lifecycle of remote Colab execution."""

    def __init__(
        self,
        stage: str,
        candidate_path: str,
        kaggle_report_path: Optional[str] = None,
        colab_t4_report_path: Optional[str] = None,
        session_name: Optional[str] = None,
        keep_alive: bool = False,
        timeout: Optional[int] = None,
        colab_bin: Optional[str] = None,
        env_file: Optional[str] = None,
        skip_ci_check: bool = False,
        output_dir: Optional[str] = None,
    ):
        self.stage = stage.replace("-", "_")
        self.candidate_path = Path(candidate_path)
        self.kaggle_report_path = Path(kaggle_report_path) if kaggle_report_path else None
        self.colab_t4_report_path = Path(colab_t4_report_path) if colab_t4_report_path else None
        self.session_name = session_name or f"legalqa-{self.stage}-{uuid.uuid4().hex[:8]}"
        self.keep_alive = keep_alive
        self.timeout = timeout
        self.colab_bin = find_colab_binary(colab_bin)
        self.env_file = Path(env_file) if env_file else (REPO_ROOT / ".env")
        self.skip_ci_check = skip_ci_check
        self.output_dir = Path(output_dir) if output_dir else (REPO_ROOT / "artifacts" / "gates")
        self.candidate: Optional[CandidateManifest] = None
        if self.candidate_path.is_file():
            try:
                self.candidate = CandidateManifest.load_json(self.candidate_path)
            except Exception:
                pass

    def _preflight_checks(self) -> None:
        """Validate credentials, candidate manifest, and prior gate reports before GPU allocation."""
        print(f"\n[*] Running Preflight Checks for Stage: {self.stage.upper()}...")

        # 1. Credentials
        if not self.env_file.is_file():
            print(f"  Notice: {self.env_file} not found; proceeding if environment variables exist.")
        else:
            env_vars = parse_env_file(self.env_file)
            if not env_vars.get("HF_TOKEN") and not os.environ.get("HF_TOKEN"):
                raise ValueError("HF_TOKEN missing from .env and environment.")
            if not env_vars.get("KAGGLE_KEY") and not os.environ.get("KAGGLE_KEY"):
                raise ValueError("KAGGLE_KEY missing from .env and environment.")
            print("  OK: Credentials verified (HF_TOKEN & KAGGLE_KEY).")

        # 2. Candidate Manifest
        if not self.candidate_path.is_file():
            raise FileNotFoundError(f"Candidate manifest not found at: {self.candidate_path}")
        self.candidate = CandidateManifest.load_json(self.candidate_path)
        print(f"  OK: Candidate loaded: {self.candidate.candidate_id} (commit={self.candidate.git_commit_sha})")

        # 3. GitHub CI Status Check
        if not self.skip_ci_check:
            verify_ci_status(self.candidate.git_commit_sha, allow_offline=True)

        # 4. Gate Reports Chaining
        if self.stage in ("colab_t4", "a100"):
            if not self.kaggle_report_path or not self.kaggle_report_path.is_file():
                raise FileNotFoundError(f"Required Kaggle T4x2 gate report missing at: {self.kaggle_report_path}")
            k_rep = verify_gate_report(self.kaggle_report_path, self.candidate, expected_stage="kaggle_t4x2")
            print(f"  OK: Prior Kaggle T4x2 report verified (SHA={k_rep.compute_sha256()[:16]}...).")

        if self.stage == "a100":
            if not self.colab_t4_report_path or not self.colab_t4_report_path.is_file():
                raise FileNotFoundError(f"Required Colab T4 gate report missing at: {self.colab_t4_report_path}")
            k_sha = GateReport.load_json(self.kaggle_report_path).compute_sha256()
            c_rep = verify_gate_report(
                self.colab_t4_report_path,
                self.candidate,
                expected_stage="colab_t4",
                required_parent_sha256=k_sha,
            )
            print(f"  OK: Prior Colab T4 report verified (SHA={c_rep.compute_sha256()[:16]}...).")

        print("  [+] Preflight checks PASSED. Ready to provision compute.\n")

    def _prepare_bootstrap_bundle(self, staging_dir: Path) -> Path:
        """Stage bootstrap bundle files for remote upload."""
        staging_dir.mkdir(parents=True, exist_ok=True)

        req_data = build_run_request(
            stage=self.stage,
            candidate_id=self.candidate.candidate_id,
            git_commit_sha=self.candidate.git_commit_sha,
            dataset_slug=self.candidate.dataset.slug,
            dataset_version=self.candidate.dataset.version,
        )
        (staging_dir / "run_request.json").write_text(json.dumps(req_data, indent=2), encoding="utf-8")

        shutil.copy(str(self.candidate_path), str(staging_dir / "candidate_manifest.json"))

        if self.kaggle_report_path and self.kaggle_report_path.is_file():
            shutil.copy(str(self.kaggle_report_path), str(staging_dir / "kaggle_t4x2_report.json"))

        if self.colab_t4_report_path and self.colab_t4_report_path.is_file():
            shutil.copy(str(self.colab_t4_report_path), str(staging_dir / "colab_t4_report.json"))

        if self.env_file.is_file():
            shutil.copy(str(self.env_file), str(staging_dir / ".env"))

        return staging_dir

    def _verify_downloaded_artifacts(self, download_dir: Path) -> None:
        """Verify report files downloaded from remote run."""
        expected_report_name = f"{self.stage}_report.json"
        if self.stage == "a100":
            expected_report_name = "a100_micro_probe_report.json"

        report_file = download_dir / expected_report_name
        if not report_file.is_file():
            raise FileNotFoundError(f"Expected gate report {expected_report_name} not found in downloaded artifacts.")

        rep = GateReport.load_json(report_file)
        if rep.status != "PASS":
            raise RuntimeError(f"Downloaded gate report status is not PASS: {rep.status}")
        print(f"\n[+] Successfully verified downloaded report: {expected_report_name} (Status: PASS)")

    def launch(self, preflight_only: bool = False) -> None:
        """Run complete provision, upload, exec, download, and stop sequence."""
        self._preflight_checks()
        if preflight_only:
            print("[+] Preflight checks completed successfully (--preflight-only specified). Exiting without provisioning VM.")
            return

        requested_gpu = "A100" if self.stage == "a100" else "T4"
        print(f"[*] Provisioning Colab GPU instance: session={self.session_name} | GPU={requested_gpu}...")

        session_created = False
        with tempfile.TemporaryDirectory() as tmpdir:
            staging_dir = Path(tmpdir) / "bootstrap"
            staging_dir.mkdir(parents=True, exist_ok=True)
            res_staging = self._prepare_bootstrap_bundle(staging_dir)
            if res_staging and Path(res_staging).is_dir():
                staging_dir = Path(res_staging)

            try:
                # 1. colab new
                new_cmd = [self.colab_bin, "new", "-s", self.session_name, "--gpu", requested_gpu]
                subprocess.run(new_cmd, check=True)
                session_created = True

                # 2. colab upload (upload each staged file to /content)
                for file_p in sorted(staging_dir.iterdir()):
                    if file_p.is_file():
                        remote_target = f"/content/{file_p.name}"
                        upload_cmd = [self.colab_bin, "upload", "-s", self.session_name, str(file_p), remote_target]
                        subprocess.run(upload_cmd, check=True)

                # 3. colab exec (NEVER pass --timeout to colab exec)
                exec_cmd = [self.colab_bin, "exec", "-s", self.session_name, "-f", "scripts/colab_remote_entry.py"]
                # Enforce timeout in local subprocess if configured and prevent 30s kernel client timeout via REQUEST_TIMEOUT
                sub_env = os.environ.copy()
                sub_env["REQUEST_TIMEOUT"] = str(int(self.timeout or 86400))
                subprocess.run(exec_cmd, check=True, timeout=self.timeout, env=sub_env)

                # 4. colab download (download key control artifacts individually)
                cand_id = self.candidate.candidate_id if self.candidate else "candidate"
                target_gate_dir = self.output_dir / cand_id
                target_gate_dir.mkdir(parents=True, exist_ok=True)

                remote_files = [f"{self.stage}_report.json", f"{self.stage}.log", f"{self.stage}_telemetry.json"]
                if self.stage in ("a100", "colab_a100"):
                    remote_files = [
                        "a100_micro_probe_report.json",
                        "a100_micro_probe.log",
                        "telemetry.json",
                        "production_run_manifest.json",
                        "metrics.json",
                        "checksums.sha256",
                    ]

                for rf in remote_files:
                    remote_file_path = f"/content/legalqa_run/{rf}"
                    local_dest = str(target_gate_dir / rf)
                    dl_cmd = [self.colab_bin, "download", "-s", self.session_name, remote_file_path, local_dest]
                    try:
                        subprocess.run(dl_cmd, check=True)
                        print(f"  Downloaded: {rf}")
                    except Exception as e:
                        print(f"  Notice: Download of {rf} skipped or not produced: {e}")

                self._verify_downloaded_artifacts(target_gate_dir)

            finally:
                if session_created and not self.keep_alive:
                    print(f"[*] Stopping Colab session {self.session_name} in finally block...")
                    stop_cmd = [self.colab_bin, "stop", "-s", self.session_name]
                    subprocess.run(stop_cmd, check=False)
                elif self.keep_alive:
                    print(f"[!] Warning: Preserving Colab session {self.session_name} (--keep-alive active).")


def find_latest_verified_candidate() -> Optional[Path]:
    """Find latest candidate manifest that has complete verified gate reports."""
    candidates_dir = REPO_ROOT / "artifacts" / "candidates"
    if not candidates_dir.is_dir():
        return None
    cands = sorted(candidates_dir.glob("*/candidate_manifest.json"), key=lambda p: p.stat().st_mtime, reverse=True)
    # Prefer candidate with both kaggle and colab_t4 reports
    for c in cands:
        cand_id = c.parent.name
        gates_dir = REPO_ROOT / "artifacts" / "gates" / cand_id
        if (gates_dir / "kaggle_t4x2_report.json").is_file() and (gates_dir / "colab_t4_report.json").is_file():
            return c
    # Fallback to candidate with kaggle report
    for c in cands:
        cand_id = c.parent.name
        gates_dir = REPO_ROOT / "artifacts" / "gates" / cand_id
        if (gates_dir / "kaggle_t4x2_report.json").is_file() or (gates_dir / "kaggle_smoke_report.json").is_file():
            return c
    return cands[0] if cands else None


def main():
    parser = argparse.ArgumentParser(description="Launch Colab training or promotion gate.")
    parser.add_argument("--stage", choices=["colab-t4", "a100"], default="a100", help="Stage to execute (default: a100)")
    parser.add_argument("--candidate", default=None, help="Path to candidate_manifest.json (auto-detected if omitted)")
    parser.add_argument("--kaggle-report", default=None, help="Path to verified kaggle_t4x2_report.json (auto-detected if omitted)")
    parser.add_argument("--colab-t4-report", default=None, help="Path to verified colab_t4_report.json (for a100, auto-detected if omitted)")
    parser.add_argument("--session-name", default=None, help="Explicit Colab session identifier")
    parser.add_argument("--keep-alive", action="store_true", help="Do not stop VM upon completion")
    parser.add_argument("--timeout", type=int, default=None, help="Local subprocess timeout in seconds")
    parser.add_argument("--colab-bin", default=None, help="Path to colab executable")
    parser.add_argument("--env-file", default=None, help="Path to .env file")
    parser.add_argument("--skip-ci-check", action="store_true", help="Skip GitHub CI status check")
    parser.add_argument("--preflight-only", action="store_true", help="Perform only preflight checks and exit without provisioning VM")
    args = parser.parse_args()

    candidate_path = args.candidate
    if not candidate_path:
        cand_p = find_latest_verified_candidate()
        if not cand_p:
            parser.error("No candidate manifest provided and none found in artifacts/candidates/")
        candidate_path = str(cand_p)
        print(f"[*] Auto-detected latest candidate manifest: {candidate_path}")

    cand_id = Path(candidate_path).parent.name
    cand_gates_dir = REPO_ROOT / "artifacts" / "gates" / cand_id

    kaggle_report = args.kaggle_report
    if not kaggle_report:
        if (cand_gates_dir / "kaggle_t4x2_report.json").is_file():
            kaggle_report = str(cand_gates_dir / "kaggle_t4x2_report.json")
        elif (cand_gates_dir / "kaggle_smoke_report.json").is_file():
            kaggle_report = str(cand_gates_dir / "kaggle_smoke_report.json")
        elif (REPO_ROOT / "kaggle_smoke_report.json").is_file():
            kaggle_report = str(REPO_ROOT / "kaggle_smoke_report.json")
        if kaggle_report:
            print(f"[*] Auto-detected Kaggle gate report: {kaggle_report}")

    colab_t4_report = args.colab_t4_report
    if not colab_t4_report and args.stage == "a100":
        if (cand_gates_dir / "colab_t4_report.json").is_file():
            colab_t4_report = str(cand_gates_dir / "colab_t4_report.json")
            print(f"[*] Auto-detected Colab T4 gate report: {colab_t4_report}")

    launcher = ColabLauncher(
        stage=args.stage,
        candidate_path=candidate_path,
        kaggle_report_path=kaggle_report,
        colab_t4_report_path=colab_t4_report,
        session_name=args.session_name,
        keep_alive=args.keep_alive,
        timeout=args.timeout,
        colab_bin=args.colab_bin,
        env_file=args.env_file,
        skip_ci_check=args.skip_ci_check,
    )
    launcher.launch(preflight_only=args.preflight_only)


if __name__ == "__main__":
    main()
