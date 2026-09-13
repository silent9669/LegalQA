#!/usr/bin/env python3
"""Verify GitHub Actions CI status for a specific commit SHA before GPU allocation.

Usage:
  python scripts/verify_ci_status.py --sha <commit_sha> [--allow-offline]
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import urllib.request
from typing import Dict, List, Optional

REQUIRED_CI_JOBS = [
    "local-contract",
    "python-tests-3.10",
    "python-tests-3.12",
    "exact-gpu-userspace-stack",
    "security-and-provenance",
]

DEFAULT_REPO = "silent9669/LegalQA"


def fetch_github_check_runs(commit_sha: str, repo: str = DEFAULT_REPO, token: Optional[str] = None) -> List[Dict[str, Any]]:
    """Query GitHub API for check runs associated with a commit SHA."""
    url = f"https://api.github.com/repos/{repo}/commits/{commit_sha}/check-runs"
    req = urllib.request.Request(url)
    req.add_header("Accept", "application/vnd.github.v3+json")
    req.add_header("User-Agent", "LegalQA-Gatekeeper")

    gh_token = token or os.environ.get("GITHUB_TOKEN") or os.environ.get("GH_TOKEN")
    if not gh_token:
        try:
            gh_out = subprocess.check_output(["gh", "auth", "token"], text=True).strip()
            if gh_out:
                gh_token = gh_out
        except Exception:
            pass

    if gh_token:
        req.add_header("Authorization", f"Bearer {gh_token}")

    with urllib.request.urlopen(req, timeout=15) as resp:
        data = json.loads(resp.read().decode("utf-8"))
    return data.get("check_runs", [])


def verify_ci_status(
    commit_sha: str,
    repo: str = DEFAULT_REPO,
    allow_offline: bool = False,
    token: Optional[str] = None,
) -> bool:
    """Verify that all required GitHub Actions jobs succeeded for commit_sha."""
    print(f"[*] Verifying GitHub CI status for commit {commit_sha} on {repo}...")

    try:
        check_runs = fetch_github_check_runs(commit_sha, repo, token)
    except Exception as e:
        if allow_offline:
            print(f"  [Notice] GitHub API query failed ({e}); bypassed with --allow-offline.")
            return True
        raise RuntimeError(
            f"Failed to query GitHub CI status for {commit_sha}: {e}\n"
            "Ensure commit is pushed to GitHub or use --allow-offline for local testing."
        ) from e

    completed_runs = {}
    for cr in check_runs:
        name = cr.get("name")
        status = cr.get("status")
        conclusion = cr.get("conclusion")
        completed_runs[name] = {"status": status, "conclusion": conclusion}

    missing_jobs = []
    failed_jobs = []

    for job in REQUIRED_CI_JOBS:
        if job not in completed_runs:
            missing_jobs.append(job)
        else:
            run_info = completed_runs[job]
            if run_info["status"] != "completed" or run_info["conclusion"] != "success":
                failed_jobs.append(f"{job}: {run_info['status']}/{run_info['conclusion']}")

    if missing_jobs or failed_jobs:
        err_msg = f"GitHub CI verification failed for {commit_sha}:\n"
        if missing_jobs:
            err_msg += f"  Missing required jobs: {missing_jobs}\n"
        if failed_jobs:
            err_msg += f"  Failed/incomplete jobs: {failed_jobs}\n"
        if allow_offline:
            print(f"  [Notice] {err_msg} (bypassed with --allow-offline)")
            return True
        raise RuntimeError(err_msg)

    print(f"  [+] All {len(REQUIRED_CI_JOBS)} required CI jobs SUCCESS for commit {commit_sha}.")
    return True


def main():
    parser = argparse.ArgumentParser(description="Verify GitHub CI status before GPU provisioning.")
    parser.add_argument("--sha", required=True, help="Exact Git commit SHA to verify")
    parser.add_argument("--repo", default=DEFAULT_REPO, help="GitHub repository (owner/repo)")
    parser.add_argument("--allow-offline", action="store_true", help="Allow bypass if offline or for local testing")
    args = parser.parse_args()

    try:
        verify_ci_status(commit_sha=args.sha, repo=args.repo, allow_offline=args.allow_offline)
        sys.exit(0)
    except Exception as e:
        print(f"[X] {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
