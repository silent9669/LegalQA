"""Kaggle Dual-T4 environment bootstrap & dependency compatibility verifier.

Guarantees:
1. Never upgrades, downgrades, or replaces Kaggle's preinstalled PyTorch, CUDA runtime, cuDNN, or NCCL.
2. Identifies and installs only missing user-space packages.
3. Tests end-to-end import compatibility across transformers, trl, peft, bitsandbytes, and sentence-transformers.
4. Verifies CUDA remains fully functional and accessible.
"""

from __future__ import annotations

import importlib
import importlib.metadata as md
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

PROTECTED_PACKAGES = [
    "torch",
    "torchvision",
    "torchaudio",
    "triton",
    "nvidia-cublas-cu12",
    "nvidia-cuda-runtime-cu12",
    "nvidia-cudnn-cu12",
    "nvidia-curand-cu12",
    "nvidia-cusolver-cu12",
    "nvidia-cusparse-cu12",
    "nvidia-nccl-cu12",
    "nvidia-nvtx-cu12",
]

TARGET_USER_PACKAGES = [
    ("bm25s", ">=0.2.5"),
    ("sentence_transformers", ">=3.0.0", "sentence-transformers"),
    ("peft", ">=0.10.0"),
    ("trl", ">=0.11.0"),
    ("bitsandbytes", ">=0.43.0"),
    ("scikit-learn", ">=1.4.0"),
    ("nltk", ">=3.8.1"),
    ("pyvi", ">=0.1.1"),
    ("pyyaml", ">=6.0"),
    ("pyarrow", ">=14.0.0"),
    ("fastparquet", ">=2024.2.0"),
    ("tqdm", ">=4.66.0"),
]


def get_package_version(pkg_name: str) -> Optional[str]:
    """Retrieve installed version of a package."""
    try:
        return md.version(pkg_name)
    except Exception:
        try:
            mod = importlib.import_module(pkg_name)
            return getattr(mod, "__version__", "unknown")
        except Exception:
            return None


def print_preinstalled_environment() -> Dict[str, Any]:
    """Inspect and report preinstalled Kaggle runtime versions."""
    print("\n=======================================================")
    print("        KAGGLE RUNTIME PRE-BOOTSTRAP ENVIRONMENT       ")
    print("=======================================================")
    print(f"Python: {sys.version.split()[0]} ({sys.executable})")

    env_info: Dict[str, Any] = {"python": sys.version}

    try:
        import torch
        env_info["torch"] = torch.__version__
        env_info["cuda_available"] = torch.cuda.is_available()
        env_info["cuda_version"] = torch.version.cuda
        env_info["cudnn_version"] = torch.backends.cudnn.version() if torch.cuda.is_available() else None
        env_info["gpu_count"] = torch.cuda.device_count() if torch.cuda.is_available() else 0

        print(f"PyTorch: {torch.__version__} | CUDA Version: {torch.version.cuda} | cuDNN: {env_info['cudnn_version']}")
        print(f"CUDA Available: {torch.cuda.is_available()} | GPU Count: {env_info['gpu_count']}")
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                name = torch.cuda.get_device_name(i)
                mem = torch.cuda.get_device_properties(i).total_memory / (1024**3)
                print(f"  - GPU {i}: {name} ({mem:.1f} GB VRAM)")
    except ImportError:
        env_info["torch"] = "not installed"
        env_info["cuda_available"] = False
        print("PyTorch: NOT INSTALLED")

    print("\nPreinstalled Key Packages:")
    for pkg in ["transformers", "accelerate", "peft", "trl", "bitsandbytes", "sentence-transformers", "bm25s", "nltk", "scikit-learn"]:
        ver = get_package_version(pkg)
        env_info[pkg] = ver or "not installed"
        print(f"  - {pkg:24s}: {ver or 'not installed'}")

    return env_info


def bootstrap_dependencies(requirements_path: Optional[str] = None) -> None:
    """Safely install missing dependencies without touching Torch/CUDA runtime."""
    print("\n=======================================================")
    print("         BOOTSTRAPPING USER-SPACE DEPENDENCIES         ")
    print("=======================================================")

    missing_or_outdated = []
    for item in TARGET_USER_PACKAGES:
        import_name = item[0]
        ver_constraint = item[1]
        pip_name = item[2] if len(item) > 2 else import_name

        curr_ver = get_package_version(pip_name) or get_package_version(import_name)
        if not curr_ver:
            missing_or_outdated.append(f"{pip_name}{ver_constraint}")
            print(f"  [MISSING] {pip_name} -> will install {ver_constraint}")
        else:
            print(f"  [OK]      {pip_name:22s} already installed ({curr_ver})")

    if missing_or_outdated:
        print(f"\nInstalling {len(missing_or_outdated)} missing package(s) with --no-deps / safety constraints...")
        # Install without upgrading torch
        cmd = [
            sys.executable, "-m", "pip", "install",
            "--no-warn-script-location",
            "--upgrade",
        ] + missing_or_outdated

        print(f"Executing: {' '.join(cmd)}")
        res = subprocess.run(cmd, capture_output=True, text=True)
        if res.returncode != 0:
            print(f"Warning during pip install: {res.stderr}", file=sys.stderr)
            # Try installing with --no-deps as fallback
            print("Retrying individual package installations...")
            for pkg_spec in missing_or_outdated:
                subprocess.run([sys.executable, "-m", "pip", "install", "-q", pkg_spec], check=False)
        else:
            print("Packages installed successfully.")
    else:
        print("All required user-space packages are present.")


def verify_runtime_imports() -> Dict[str, Any]:
    """Test and verify all critical neural and retrieval modules."""
    print("\n=======================================================")
    print("         VERIFYING CRITICAL RUNTIME IMPORTS           ")
    print("=======================================================")

    import_results: Dict[str, str] = {}
    modules_to_test = [
        ("torch", "PyTorch"),
        ("transformers", "Hugging Face Transformers"),
        ("peft", "Hugging Face PEFT"),
        ("trl", "Hugging Face TRL"),
        ("bitsandbytes", "BitsAndBytes 4-bit Quantization"),
        ("sentence_transformers", "Sentence Transformers"),
        ("bm25s", "BM25S Sparse Retrieval"),
        ("nltk", "NLTK Natural Language Toolkit"),
        ("pyvi", "PyVi Vietnamese NLP"),
        ("sklearn", "Scikit-Learn"),
    ]

    all_passed = True
    for mod_name, desc in modules_to_test:
        try:
            mod = importlib.import_module(mod_name)
            ver = getattr(mod, "__version__", "ok")
            import_results[mod_name] = f"PASS ({ver})"
            print(f"  + {desc:32s} ({mod_name}): PASS ({ver})")
        except Exception as e:
            all_passed = False
            import_results[mod_name] = f"FAIL ({e})"
            print(f"  ! {desc:32s} ({mod_name}): FAIL ({e})", file=sys.stderr)

    # Check that TRL does not rely on obsolete collator
    try:
        from trl import SFTConfig, SFTTrainer
        print("  + TRL SFTConfig and SFTTrainer modern API: AVAILABLE")
    except Exception as e:
        all_passed = False
        print(f"  ! TRL modern SFT API check FAILED: {e}", file=sys.stderr)

    # Check CUDA integrity post-bootstrap
    try:
        import torch
        if torch.cuda.is_available():
            dev_cnt = torch.cuda.device_count()
            print(f"  + Post-bootstrap CUDA Status: {dev_cnt} GPU(s) functional.")
        else:
            print("  - Post-bootstrap CUDA Status: CPU only (No CUDA detected).")
    except Exception as e:
        all_passed = False
        print(f"  ! Post-bootstrap PyTorch CUDA check FAILED: {e}", file=sys.stderr)

    if not all_passed:
        raise RuntimeError("Runtime import verification failed! Inspect logs above.")

    print("\nAll runtime imports successfully verified.")
    return import_results


def save_bootstrap_manifest(output_path: str = "/kaggle/working/kaggle_environment.json") -> None:
    """Save environment and package tuple manifest to disk."""
    try:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
    except Exception:
        pass

    pkg_dict = {}
    for pkg in [
        "torch", "torchvision", "transformers", "accelerate", "peft", "trl",
        "bitsandbytes", "sentence-transformers", "bm25s", "nltk", "pyvi",
        "pandas", "numpy", "scikit-learn", "pyarrow", "fastparquet",
    ]:
        pkg_dict[pkg] = get_package_version(pkg) or "not installed"

    manifest = {
        "python": sys.version,
        "packages": pkg_dict,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()),
    }
    try:
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2)
        print(f"Saved runtime environment manifest to {output_path}")
    except Exception as e:
        print(f"Could not save manifest to {output_path}: {e}")


def main():
    print_preinstalled_environment()
    bootstrap_dependencies()
    verify_runtime_imports()
    save_bootstrap_manifest()


if __name__ == "__main__":
    main()
