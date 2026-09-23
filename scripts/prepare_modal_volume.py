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
data_volume = modal.Volume.from_name("legalqa-data-vol", create_if_missing=True)
runs_volume = modal.Volume.from_name("legalqa-runs-vol", create_if_missing=True)
hf_secret = modal.Secret.from_name("huggingface-secret")

HF_REPO = "dangphuc2109/legalqa-qwen2.5-3b-adapter"
#: Immutable pins: encoder from candidate 94aed, R0 reuse adapter 5433
#: (generator behind runs/20260920-215402). Never float these.
HF_ENCODER_REVISION = "6a2721e34a083eae202dbb80e4fe529707ec4097"
HF_ADAPTER_REVISION = "b6e86e35e20c403bb82b40b25f85690c987e1d02"
HF_ADAPTER_SUBFOLDER = "runs/run_5433e8b4787137c9_20260920_193355/final_adapter"

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

    # 1. Encoder_ft_v2 (pinned revision download; present cache is kept)
    ft_target = models_dir / "encoder_ft_v2"
    if (ft_target / "model.safetensors").is_file():
        print(f"[+] encoder_ft_v2 already cached on volume: {ft_target}")
    else:
        print(f"[+] Downloading encoder_ft_v2 from {HF_REPO}@{HF_ENCODER_REVISION}...")
        dl_p = snapshot_download(
            repo_id=HF_REPO,
            revision=HF_ENCODER_REVISION,
            allow_patterns="runs/20260920-215402/encoder_ft_v2/*",
        )
        src_ft = Path(dl_p) / "runs/20260920-215402/encoder_ft_v2"
        shutil.copytree(str(src_ft), str(ft_target), dirs_exist_ok=True)
        data_volume.commit()
        print(f"[+] encoder_ft_v2 saved and committed: {ft_target}")

    # 2. Generator adapter (pinned R0 5433 download; present cache is kept)
    gen_target = data_dir / "models" / "qwen_adapter"
    if (gen_target / "adapter_model.safetensors").is_file():
        print(f"[+] Qwen adapter already cached on volume: {gen_target}")
    else:
        prior_runs = sorted(Path("/runs").glob("modal_*_*/checkpoints/generator/hf_adapter"))
        found = next(
            (str(p) for p in reversed(prior_runs)
             if (p / "adapter_model.safetensors").is_file()),
            None,
        )
        if found:
            print(f"[+] Copying Qwen adapter from existing volume run: {found}")
            shutil.copytree(found, str(gen_target), dirs_exist_ok=True)
        else:
            print(f"[+] Fetching Qwen adapter {HF_ADAPTER_SUBFOLDER} from HF {HF_REPO}...")
            dl_p = snapshot_download(
                repo_id=HF_REPO,
                revision=HF_ADAPTER_REVISION,
                allow_patterns=f"{HF_ADAPTER_SUBFOLDER}/*",
            )
            src_ad = Path(dl_p) / HF_ADAPTER_SUBFOLDER
            shutil.copytree(str(src_ad), str(gen_target), dirs_exist_ok=True)

        data_volume.commit()
        print(f"[+] Qwen adapter saved and committed: {gen_target}")

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
