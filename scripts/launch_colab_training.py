#!/usr/bin/env python3
"""
Automated Google Colab Session Orchestrator for LegalQA Task 2.

Manages the complete remote lifecycle via the Colab CLI:
1. Preflight check (local .env, verified kaggle_smoke_report.json, Colab CLI).
2. Provisions a GPU session (A100 for production, T4 for smoke test).
3. Injects local environment credentials and smoke verification report.
4. Executes the training notebook with streaming logs.
5. Captures generated artifacts and terminates the VM to prevent credit leakage.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import uuid
from pathlib import Path
from typing import List, Optional

# Ensure repository root is on sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.common.env_loader import parse_env_file
from src.task2.provenance.freeze_tuple import verify_smoke_pass


def find_colab_binary(custom_path: Optional[str] = None) -> str:
    """Locate the colab CLI executable."""
    if custom_path and os.path.isfile(custom_path) and os.access(custom_path, os.X_OK):
        return custom_path

    in_path = shutil.which("colab")
    if in_path:
        return in_path

    home_local = Path.home() / ".local" / "bin" / "colab"
    if home_local.is_file() and os.access(home_local, os.X_OK):
        return str(home_local)

    raise FileNotFoundError(
        "Colab CLI ('colab') not found in PATH or ~/.local/bin/colab. "
        "Please install it or specify --colab-bin."
    )


def preflight_checks(
    env_file: Path,
    smoke_report_file: Path,
    strict_smoke: bool = True,
) -> None:
    """Validate that required local credentials and smoke gate reports exist."""
    print("=== Running Local Preflight Checks ===")

    # 1. Check .env file
    if not env_file.is_file():
        raise FileNotFoundError(
            f"Environment credential file not found at: {env_file}. "
            f"Please create .env with HF_TOKEN, KAGGLE_USERNAME, and KAGGLE_KEY."
        )

    parsed_env = parse_env_file(env_file)
    if not parsed_env.get("HF_TOKEN"):
        raise ValueError(f"HF_TOKEN is missing from {env_file}.")
    if not parsed_env.get("KAGGLE_KEY"):
        raise ValueError(f"KAGGLE_KEY is missing from {env_file}.")

    print(f" [+] Credentials verified in {env_file} (HF_TOKEN & KAGGLE_KEY present)")

    # 2. Check Kaggle smoke pass gate
    if smoke_report_file.is_file():
        if not verify_smoke_pass(str(smoke_report_file)):
            if strict_smoke:
                raise RuntimeError(
                    f"Kaggle smoke gate verification at {smoke_report_file} does NOT report PASS. "
                    f"Refusing to allocate billable Colab GPU."
                )
            else:
                print(f" [!] Warning: Smoke report did not report PASS, continuing due to non-strict mode.")
        else:
            print(f" [+] Kaggle Dual-T4 smoke gate PASS verified from: {smoke_report_file}")
    else:
        if strict_smoke:
            raise FileNotFoundError(
                f"Kaggle smoke report not found at: {smoke_report_file}. "
                f"Production A100 training strictly requires verified Kaggle smoke PASS."
            )
        else:
            print(f" [!] Warning: No smoke report file found at {smoke_report_file}.")


def list_active_server_sessions(colab_bin: str) -> List[Dict[str, str]]:
    """Query Colab CLI for active server-side GPU/TPU session assignments."""
    try:
        out = subprocess.check_output([colab_bin, "sessions"], stderr=subprocess.STDOUT).decode()
        active = []
        for line in out.splitlines():
            line = line.strip()
            if line.startswith("[") and "]" in line:
                name = line[1:line.index("]")].strip()
                active.append({"name": name, "raw": line})
        return active
    except Exception:
        return []


def run_colab_command(cmd: List[str], desc: str) -> None:
    """Execute a Colab CLI command and stream output."""
    print(f"\n[*] {desc}...")
    print(f"    Executing: {' '.join(cmd)}")
    ret = subprocess.run(cmd)
    if ret.returncode != 0:
        raise RuntimeError(f"Colab command failed with exit code {ret.returncode}: {' '.join(cmd)}")


def main():
    parser = argparse.ArgumentParser(description="Automated Google Colab Training Launcher")
    parser.add_argument("--gpu", choices=["A100", "T4", "L4"], default="A100", help="GPU accelerator type")
    parser.add_argument("-s", "--session-name", default=None, help="Custom Colab session name")
    parser.add_argument("--attach", default=None, help="Attach to an existing active Colab session by name")
    parser.add_argument("--reuse-existing", action="store_true", help="Automatically reuse matching active server session if present")
    parser.add_argument("--notebook", default="notebooks/colab_a100_train.ipynb", help="Notebook path to execute")
    parser.add_argument("--env-file", default=".env", help="Path to local .env file")
    parser.add_argument("--smoke-report", default="kaggle_smoke_report.json", help="Path to kaggle_smoke_report.json")
    parser.add_argument("--timeout", type=float, default=None, help="Timeout in seconds for code execution")
    parser.add_argument("--non-strict-smoke", action="store_true", help="Allow running without strict smoke report PASS")
    parser.add_argument("--keep-alive", action="store_true", help="Keep Colab session running instead of stopping on exit")
    parser.add_argument("--stop-on-finish", action="store_true", help="Force stop the session even if it was previously attached")
    parser.add_argument("--colab-bin", default=None, help="Explicit path to colab binary")

    args = parser.parse_args()

    colab_bin = find_colab_binary(args.colab_bin)
    env_path = (REPO_ROOT / args.env_file).resolve()
    smoke_path = (REPO_ROOT / args.smoke_report).resolve()
    notebook_path = (REPO_ROOT / args.notebook).resolve()

    if not notebook_path.is_file():
        raise FileNotFoundError(f"Notebook file not found: {notebook_path}")

    # Set appropriate execution timeout
    default_timeout = 7200.0 if args.gpu == "A100" else 900.0
    timeout_sec = args.timeout or default_timeout

    # Preflight verification
    preflight_checks(
        env_file=env_path,
        smoke_report_file=smoke_path,
        strict_smoke=not args.non_strict_smoke,
    )

    session_name = args.attach
    session_created = False

    if not session_name and args.reuse_existing:
        active_sessions = list_active_server_sessions(colab_bin)
        for s in active_sessions:
            if args.gpu in s["raw"]:
                session_name = s["name"]
                print(f"Reusing active server session: {session_name} ({s['raw']})")
                break

    if not session_name:
        session_name = args.session_name or f"colab-{args.gpu.lower()}-{uuid.uuid4().hex[:6]}"
        print(f"\nTarget Session Name: {session_name}")
        print(f"Target Accelerator:  {args.gpu}")
        print(f"Execution Timeout:   {timeout_sec:.0f} seconds")

        try:
            # 1. Provision Colab Session
            run_colab_command(
                [colab_bin, "new", "-s", session_name, "--gpu", args.gpu],
                desc=f"Provisioning Colab session '{session_name}' with {args.gpu} GPU",
            )
            session_created = True
        except RuntimeError as e:
            active_sessions = list_active_server_sessions(colab_bin)
            matching = [s for s in active_sessions if args.gpu in s["raw"]]
            if matching:
                session_name = matching[0]["name"]
                print(f"\n[Notice] Single-GPU account quota active. Attaching to existing {args.gpu} session: '{session_name}'")
            else:
                raise e
    else:
        print(f"\nAttaching to Active Session: {session_name}")
        print(f"Execution Timeout:         {timeout_sec:.0f} seconds")

    try:
        # 2. Upload Credentials (.env)
        run_colab_command(
            [colab_bin, "upload", str(env_path), "/content/.env", "-s", session_name],
            desc="Uploading local .env credentials to Colab VM",
        )

        # 3. Upload Smoke Report if available
        if smoke_path.is_file():
            run_colab_command(
                [colab_bin, "upload", str(smoke_path), "/content/kaggle_smoke_report.json", "-s", session_name],
                desc="Uploading verified kaggle_smoke_report.json to Colab VM",
            )

        # 4. Execute Notebook
        run_colab_command(
            [colab_bin, "exec", "-s", session_name, "-f", str(notebook_path), "--timeout", str(timeout_sec)],
            desc=f"Executing notebook {notebook_path.name} on remote VM",
        )

        print("\n" + "=" * 60)
        print(" [SUCCESS] Colab training execution completed successfully!")
        print("=" * 60)

    except Exception as e:
        print(f"\n[ERROR] Colab execution encountered an error: {e}", file=sys.stderr)
        raise
    finally:
        # 5. Clean up billable resources
        should_stop = (session_created and not args.keep_alive) or args.stop_on_finish
        if should_stop:
            print(f"\n[*] Releasing Colab VM session '{session_name}' to prevent credit consumption...")
            try:
                subprocess.run([colab_bin, "stop", "-s", session_name], check=True)
                print(f"[+] Session '{session_name}' successfully stopped.")
            except Exception as stop_err:
                print(f"Warning: Failed to stop session {session_name}: {stop_err}", file=sys.stderr)
        else:
            print(f"\nNotice: Session '{session_name}' left running.")


if __name__ == "__main__":
    main()
