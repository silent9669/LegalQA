import subprocess
import time
import sys
import os

def main():
    slug = "phucdangg/legalqa-top-2-training-gpu"
    kaggle_bin = os.path.abspath(".venv-ml/bin/kaggle")
    python_bin = os.path.abspath(".venv-ml/bin/python")

    print(f"Monitoring Kaggle kernel: {slug}...", flush=True)

    while True:
        res = subprocess.run([kaggle_bin, "kernels", "status", slug], capture_output=True, text=True)
        status_line = res.stdout.strip()
        timestamp = time.strftime("%H:%M:%S")
        print(f"[{timestamp}] {status_line}", flush=True)

        if "COMPLETE" in status_line or "ERROR" in status_line or "CANCELLED" in status_line:
            break

        time.sleep(20)

    print("\nKernel status finished. Fetching submission output...", flush=True)
    fetch_script = os.path.abspath("scripts/fetch_kaggle_submission.py")
    fetch_res = subprocess.run([python_bin, fetch_script], capture_output=True, text=True)
    print(fetch_res.stdout, flush=True)
    if fetch_res.stderr:
        print("STDERR:", fetch_res.stderr, flush=True)

if __name__ == "__main__":
    main()
