#!/usr/bin/env python3
"""Preload and verify models & dataset on Modal Volume using CPU (Zero GPU)."""

import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

try:
    import modal
except ImportError:
    print("modal package required: pip install modal")
    sys.exit(1)

app = modal.App("legalqa-volume-preloader")
data_volume = modal.Volume.from_name("legalqa-data-vol")
runs_volume = modal.Volume.from_name("legalqa-runs-vol")
hf_secret = modal.Secret.from_name("huggingface-secret")

HF_REPO = "dangphuc2109/legalqa-qwen2.5-3b-adapter"

image = modal.Image.debian_slim(python_version="3.11").pip_install(
    "huggingface_hub",
    "safetensors",
    "requests",
)

@app.function(
    image=image,
    volumes={"/data": data_volume, "/runs": runs_volume},
    secrets=[hf_secret],
    timeout=600,
    cpu=2,
)
def preload_assets():
    from huggingface_hub import snapshot_download
    import shutil

    print("=== Modal CPU Volume Preloader (Zero GPU) ===")
    data_dir = Path("/data/legalqa-task2-clean-data")
    data_dir.mkdir(parents=True, exist_ok=True)
    models_dir = data_dir / "models"
    models_dir.mkdir(parents=True, exist_ok=True)

    # 1. Check / Download encoder_ft_v2
    ft_target = models_dir / "encoder_ft_v2"
    if not (ft_target / "model.safetensors").is_file():
        print(f"[+] Downloading encoder_ft_v2 from {HF_REPO}...")
        dl_p = snapshot_download(
            repo_id=HF_REPO,
            allow_patterns="runs/20260920-215402/encoder_ft_v2/*",
        )
        src_ft = Path(dl_p) / "runs/20260920-215402/encoder_ft_v2"
        shutil.copytree(str(src_ft), str(ft_target), dirs_exist_ok=True)
        data_volume.commit()
        print(f"[+] encoder_ft_v2 saved and committed: {ft_target}")
    else:
        print(f"[+] encoder_ft_v2 already cached on volume: {ft_target}")

    # 2. Check / Copy Generator Adapter
    gen_target = data_dir / "models" / "qwen_adapter"
    if not (gen_target / "adapter_model.safetensors").is_file():
        # Check if already in /runs
        prior_runs = sorted(Path("/runs").glob("modal_d2618710d9d0b6de_*/checkpoints/generator/hf_adapter"))
        found = False
        for pr in reversed(prior_runs):
            if (pr / "adapter_model.safetensors").is_file():
                print(f"[+] Copying Qwen adapter from existing volume run: {pr}")
                shutil.copytree(str(pr), str(gen_target), dirs_exist_ok=True)
                found = True
                break
        if not found:
            print(f"[+] Fetching Qwen adapter from HF {HF_REPO}...")
            dl_p = snapshot_download(
                repo_id=HF_REPO,
                allow_patterns="runs/run_d2618710d9d0b6de_20260921_154231/final_adapter/*",
            )
            src_ad = Path(dl_p) / "runs/run_d2618710d9d0b6de_20260921_154231/final_adapter"
            shutil.copytree(str(src_ad), str(gen_target), dirs_exist_ok=True)

        data_volume.commit()
        print(f"[+] Qwen adapter saved and committed: {gen_target}")
    else:
        print(f"[+] Qwen adapter already cached on volume: {gen_target}")

    # 3. Inventory summary
    data_files = [str(f.relative_to(data_dir)) for f in data_dir.rglob("*") if f.is_file() and not f.name.startswith(".")]
    print(f"\n[+] Total files on /data volume: {len(data_files)}")
    return {
        "encoder_ft_v2": [str(f.name) for f in ft_target.glob("*") if f.is_file()],
        "qwen_adapter": [str(f.name) for f in gen_target.glob("*") if f.is_file()],
        "data_sample": data_files[:15],
    }

@app.local_entrypoint()
def main():
    res = preload_assets.remote()
    print("\nPreload Inventory Result:")
    import json
    print(json.dumps(res, indent=2, ensure_ascii=False))

if __name__ == "__main__":
    main()
