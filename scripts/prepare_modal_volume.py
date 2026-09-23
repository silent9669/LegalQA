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

#: Independent expected bytes (mirrors the R0 pins in
#: src/task2/provenance/reuse_contract.py; duplicated here because this
#: preloader image ships no repo code). Sources: release checksums at the
#: pinned commit (encoder) and measured pinned-revision bytes (adapter).
EXPECTED_ENCODER_WEIGHTS_SHA256 = "15a895b69a3f974d230771b021ca53ece647551a4c29f03bf8229468eebb1a46"
EXPECTED_ADAPTER_DIGESTS = {
    "adapter_model.safetensors": "dd5af2848f23234e7bd7cb7987b0b199c6832a4f5fde5dbe59a3e21ee379484e",
    "adapter_config.json": "301cac83325dbc5e6a60d5bcf86c0ebb509e46f8f7e7c2a2837031690dc1256b",
    "generator_manifest.json": "1f46415bb5db5633e7bb6b7d4b0da27a3adc7d22b3e22ea775060978725e36a5",
}
EXPECTED_ADAPTER_BASE_MODEL = "Qwen/Qwen2.5-3B-Instruct"
EXPECTED_ADAPTER_SCOPE = "all_allowed_task2_data"


def _sha256_file(path):
    import hashlib

    h = hashlib.sha256()
    with open(str(path), "rb") as f:
        while chunk := f.read(8 * 1024 * 1024):
            h.update(chunk)
    return h.hexdigest()


def _verify_encoder_weights(encoder_dir, expected_sha256):
    """Fail closed when staged encoder bytes differ from the release SHA."""
    import hashlib

    target = Path(encoder_dir)
    weights = sorted(
        p for p in target.iterdir() if p.is_file() and (
            p.name in ("model.safetensors", "pytorch_model.bin")
            or (p.name.startswith("model-") and p.name.endswith(".safetensors"))
        )
    ) if target.is_dir() else []
    if not weights:
        raise FileNotFoundError(f"encoder weights bytes missing at {encoder_dir}")
    h = hashlib.sha256()
    for w in weights:
        with open(w, "rb") as f:
            while chunk := f.read(8 * 1024 * 1024):
                h.update(chunk)
    observed = h.hexdigest()
    if observed.lower() != str(expected_sha256).lower():
        raise ValueError(
            f"refusing cached encoder at {encoder_dir}: weights SHA {observed} "
            f"!= expected {expected_sha256} (stale cache is never adopted)"
        )
    return observed


def _verify_adapter_dir(adapter_dir):
    """Verify a staged adapter against the R0 pins (digests + contract)."""
    import json

    target = Path(adapter_dir)
    if not target.is_dir():
        raise FileNotFoundError(f"adapter directory missing: {target}")
    for rel, expected in EXPECTED_ADAPTER_DIGESTS.items():
        candidate = target / rel
        if not candidate.is_file():
            raise FileNotFoundError(f"adapter file missing: {rel} in {target}")
        actual = _sha256_file(candidate)
        if actual.lower() != expected.lower():
            raise ValueError(f"adapter file digest mismatch for {rel}")
    manifest = json.loads((target / "generator_manifest.json").read_text(encoding="utf-8"))
    if manifest.get("is_final_checkpoint") is not True:
        raise ValueError("adapter is NOT marked is_final_checkpoint=true")
    if manifest.get("smoke_only") is True:
        raise ValueError("adapter is a smoke checkpoint; refusing")
    if manifest.get("training_scope") != EXPECTED_ADAPTER_SCOPE:
        raise ValueError("adapter training_scope mismatch")
    if manifest.get("val_fold_excluded", manifest.get("val_fold")) is not None:
        raise ValueError("adapter trained with held-out val_fold")
    base_m = manifest.get("base_model_id") or manifest.get("base_model") or manifest.get("base_model_name_or_path")
    if base_m and str(base_m) != EXPECTED_ADAPTER_BASE_MODEL:
        raise ValueError(f"adapter base model mismatch: {base_m!r}")
    return True


def _find_verified_preload_source(candidate_dirs):
    """First pre-existing adapter that verifies, else (None, rejections)."""
    rejections = []
    for candidate in candidate_dirs:
        try:
            _verify_adapter_dir(candidate)
            return str(candidate), rejections
        except Exception as exc:
            rejections.append(f"{candidate}: {exc}")
    return None, rejections

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

    # 1. Check / Verify encoder_ft_v2 (pinned revision; a present cache
    # verifies against the independent release SHA, never its own bytes)
    ft_target = models_dir / "encoder_ft_v2"
    if (ft_target / "model.safetensors").is_file():
        try:
            _verify_encoder_weights(ft_target, EXPECTED_ENCODER_WEIGHTS_SHA256)
            print(f"[+] encoder_ft_v2 cache already verified on volume: {ft_target}")
        except Exception as exc:
            print(f"[!] Encoder cache rejected ({exc}); re-downloading pinned revision.")
            shutil.rmtree(str(ft_target), ignore_errors=True)
    if not (ft_target / "model.safetensors").is_file():
        print(f"[+] Downloading encoder_ft_v2 from {HF_REPO}@{HF_ENCODER_REVISION}...")
        dl_p = snapshot_download(
            repo_id=HF_REPO,
            revision=HF_ENCODER_REVISION,
            allow_patterns="runs/20260920-215402/encoder_ft_v2/*",
        )
        src_ft = Path(dl_p) / "runs/20260920-215402/encoder_ft_v2"
        shutil.copytree(str(src_ft), str(ft_target), dirs_exist_ok=True)
        _verify_encoder_weights(ft_target, EXPECTED_ENCODER_WEIGHTS_SHA256)
        data_volume.commit()
        print(f"[+] encoder_ft_v2 saved, verified and committed: {ft_target}")

    # 2. Check / Verify Generator Adapter (R0 5433 only; every source AND
    # the existing cache verify against the pinned repo/revision/subfolder,
    # MEASURED digests and checkpoint contract before commit. Unverified
    # bytes are never copied or kept under the R0 name.)
    gen_target = data_dir / "models" / "qwen_adapter"
    cache_ok = False
    if (gen_target / "adapter_model.safetensors").is_file():
        try:
            _verify_adapter_dir(gen_target)
            print(f"[+] Qwen adapter cache already verified on volume: {gen_target}")
            cache_ok = True
        except Exception as exc:
            print(f"[!] Existing adapter cache rejected ({exc}); replacing with pinned download.")
            shutil.rmtree(str(gen_target), ignore_errors=True)
    if not cache_ok:
        prior_runs = sorted(Path("/runs").glob("modal_*_*/checkpoints/generator/hf_adapter"))
        source_dir, rejections = _find_verified_preload_source(
            [str(p) for p in reversed(prior_runs)],
        )
        if source_dir:
            print(f"[+] Copying verified Qwen adapter from existing volume run: {source_dir}")
            shutil.copytree(str(source_dir), str(gen_target), dirs_exist_ok=True)
        else:
            for rejection in rejections[:5]:
                print(f"[!] Prior adapter rejected: {rejection}")
            print(f"[+] Fetching Qwen adapter {HF_ADAPTER_SUBFOLDER} from HF {HF_REPO}...")
            dl_p = snapshot_download(
                repo_id=HF_REPO,
                revision=HF_ADAPTER_REVISION,
                allow_patterns=f"{HF_ADAPTER_SUBFOLDER}/*",
            )
            src_ad = Path(dl_p) / HF_ADAPTER_SUBFOLDER
            shutil.copytree(str(src_ad), str(gen_target), dirs_exist_ok=True)
            _verify_adapter_dir(gen_target)
            print("[+] Pinned adapter downloaded and verified.")

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
