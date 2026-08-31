"""Monitor a remote Kaggle kernel execution until completion, then fetch verified outputs."""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import time


def monitor_kernel(
    slug: str = "phucdangg/legalqa-top-2-training-gpu",
    poll_interval_sec: int = 20,
    max_wait_minutes: int = 120,
) -> int:
    kaggle_bin = shutil.which("kaggle") or os.path.abspath(".venv-ml/bin/kaggle")
    python_bin = sys.executable or os.path.abspath(".venv-ml/bin/python")

    print(f"Monitoring Kaggle kernel '{slug}' (polling every {poll_interval_sec}s, max wait {max_wait_minutes} min)...", flush=True)

    start_time = time.time()
    max_wait_seconds = max_wait_minutes * 60

    while True:
        res = subprocess.run([kaggle_bin, "kernels", "status", slug], capture_output=True, text=True)
        status_line = res.stdout.strip()
        timestamp = time.strftime("%H:%M:%S")
        elapsed_min = (time.time() - start_time) / 60
        print(f"[{timestamp} | +{elapsed_min:.1f}m] {status_line}", flush=True)

        status_upper = status_line.upper()

        if "COMPLETE" in status_upper:
            print("\nKernel completed successfully! Fetching submission output...", flush=True)
            fetch_script = os.path.abspath("scripts/fetch_kaggle_submission.py")
            fetch_res = subprocess.run([python_bin, fetch_script, "--slug", slug])
            return fetch_res.returncode

        if "ERROR" in status_upper or "FAILED" in status_upper:
            print(f"\nKernel execution failed with status: {status_line}", file=sys.stderr)
            return 1

        if "CANCELLED" in status_upper:
            print(f"\nKernel was cancelled.", file=sys.stderr)
            return 2

        if time.time() - start_time > max_wait_seconds:
            print(f"\nTimeout exceeded ({max_wait_minutes} minutes).", file=sys.stderr)
            return 3

        time.sleep(poll_interval_sec)


def main():
    parser = argparse.ArgumentParser(description="Monitor a running Kaggle kernel.")
    parser.add_argument("--slug", default="phucdangg/legalqa-top-2-training-gpu", help="Kaggle kernel slug")
    parser.add_argument("--interval", type=int, default=20, help="Polling interval in seconds")
    parser.add_argument("--timeout", type=int, default=120, help="Max wait timeout in minutes")
    args = parser.parse_args()

    rc = monitor_kernel(slug=args.slug, poll_interval_sec=args.interval, max_wait_minutes=args.timeout)
    sys.exit(rc)


if __name__ == "__main__":
    main()
