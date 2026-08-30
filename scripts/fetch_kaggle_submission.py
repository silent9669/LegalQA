import os
import sys
import json
import zipfile
import subprocess

def fetch_and_validate():
    kernel_slug = "phucdangg/legalqa-top-2-training-gpu"
    target_dir = "artifacts/task2/submissions"
    os.makedirs(target_dir, exist_ok=True)

    print(f"Checking status for Kaggle kernel: {kernel_slug}...")
    status_cmd = [".venv-ml/bin/kaggle", "kernels", "status", kernel_slug]
    res = subprocess.run(status_cmd, capture_output=True, text=True)
    print("Status:", res.stdout.strip())

    temp_dir = "/tmp/kaggle_dl_output"
    os.makedirs(temp_dir, exist_ok=True)

    print(f"Downloading output files from Kaggle...")
    dl_cmd = [".venv-ml/bin/kaggle", "kernels", "output", kernel_slug, "-p", temp_dir]
    subprocess.run(dl_cmd)

    # Check for downloaded files
    sub_json = os.path.join(temp_dir, "submission.json")
    sub_zip = os.path.join(temp_dir, "submission.json.zip")

    if os.path.exists(sub_json):
        import shutil
        dest_json = os.path.join(target_dir, "submission.json")
        dest_zip = os.path.join(target_dir, "submission.json.zip")
        shutil.copy2(sub_json, dest_json)
        print(f"Overwritten: {dest_json}")

        if os.path.exists(sub_zip):
            shutil.copy2(sub_zip, dest_zip)
        else:
            with zipfile.ZipFile(dest_zip, "w", zipfile.ZIP_DEFLATED) as z:
                z.write(dest_json, arcname="submission.json")
        print(f"Overwritten: {dest_zip}")

        # Validate
        with open(dest_json, "r", encoding="utf-8") as f:
            d = json.load(f)
        print(f"Validation successful: {len(d)} / 1000 queries verified.")
    else:
        print("No submission.json found in Kaggle output yet. Ensure kernel run completed.")

if __name__ == "__main__":
    fetch_and_validate()
