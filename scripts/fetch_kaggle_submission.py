"""Fetch and strictly validate Kaggle submission outputs from a completed kernel run."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path


def validate_submission_dict(data: dict, expected_count: int = 1000) -> bool:
    if not isinstance(data, dict):
        print("Validation Error: Submission root is not a dictionary.", file=sys.stderr)
        return False
    if len(data) != expected_count:
        print(f"Validation Error: Expected {expected_count} items, got {len(data)}.", file=sys.stderr)
        return False

    empty_count = 0
    for qid, val in data.items():
        if not isinstance(val, dict) or "answer" not in val:
            print(f"Validation Error: Item '{qid}' missing 'answer' field.", file=sys.stderr)
            return False
        ans = str(val["answer"]).strip()
        if not ans:
            empty_count += 1

    if empty_count > 0:
        print(f"Validation Error: Found {empty_count} empty answers in submission.", file=sys.stderr)
        return False

    return True


def fetch_and_validate(
    kernel_slug: str = "phucdangg/legalqa-top-2-training-gpu",
    target_dir: str = "artifacts/task2/submissions",
) -> bool:
    os.makedirs(target_dir, exist_ok=True)
    kaggle_bin = shutil.which("kaggle") or os.path.abspath(".venv-ml/bin/kaggle")

    print(f"Checking status for Kaggle kernel: {kernel_slug}...")
    status_cmd = [kaggle_bin, "kernels", "status", kernel_slug]
    res = subprocess.run(status_cmd, capture_output=True, text=True)
    status_text = res.stdout.strip()
    print("Status:", status_text)

    if "COMPLETE" not in status_text.upper():
        print(f"Kernel is not in COMPLETE state. Current status: {status_text}", file=sys.stderr)
        return False

    with tempfile.TemporaryDirectory(prefix="kaggle_fetch_") as temp_dir:
        print(f"Downloading output files from Kaggle to {temp_dir}...")
        dl_cmd = [kaggle_bin, "kernels", "output", kernel_slug, "-p", temp_dir]
        dl_res = subprocess.run(dl_cmd, capture_output=True, text=True)
        if dl_res.returncode != 0:
            print(f"Kaggle download failed:\n{dl_res.stderr}", file=sys.stderr)
            return False

        sub_json = os.path.join(temp_dir, "submission.json")
        sub_zip = os.path.join(temp_dir, "submission.json.zip")

        if not os.path.exists(sub_json) and os.path.exists(sub_zip):
            with zipfile.ZipFile(sub_zip, "r") as z:
                z.extract("submission.json", path=temp_dir)

        if not os.path.exists(sub_json):
            print("Error: No submission.json found in Kaggle output artifacts.", file=sys.stderr)
            return False

        with open(sub_json, "r", encoding="utf-8") as f:
            try:
                sub_data = json.load(f)
            except Exception as e:
                print(f"Error parsing submission.json: {e}", file=sys.stderr)
                return False

        if not validate_submission_dict(sub_data, expected_count=1000):
            print("Validation failed. Aborting submission overwrite.", file=sys.stderr)
            return False

        dest_json = os.path.join(target_dir, "submission.json")
        dest_zip = os.path.join(target_dir, "submission.json.zip")

        shutil.copy2(sub_json, dest_json)
        with zipfile.ZipFile(dest_zip, "w", zipfile.ZIP_DEFLATED) as z:
            z.write(dest_json, arcname="submission.json")

        print(f"Validated and saved final submission to {dest_zip} ({os.path.getsize(dest_zip)/1024:.1f} KB).")
        return True


def main():
    parser = argparse.ArgumentParser(description="Fetch and validate Kaggle submission outputs.")
    parser.add_argument("--slug", default="phucdangg/legalqa-top-2-training-gpu", help="Kaggle kernel slug")
    parser.add_argument("--target_dir", default="artifacts/task2/submissions", help="Target directory for submission")
    args = parser.parse_args()

    success = fetch_and_validate(kernel_slug=args.slug, target_dir=args.target_dir)
    if not success:
        sys.exit(1)


if __name__ == "__main__":
    main()
