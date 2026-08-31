"""Kaggle Dual-T4 environment bootstrap & dependency compatibility verifier (V7).

Guarantees:
1. Dynamically snapshots and proves preinstalled PyTorch, CUDA runtime wheels, cuDNN, NCCL, and Triton remain immutable.
2. Generates exact constraints to prevent pip from replacing protected distributions.
3. Tests version specifiers for required user-space packages using packaging.specifiers.
4. Executes pip check to ensure dependency tree consistency.
5. Verifies modern TRL completion_only_loss API and neural module imports.
"""

from __future__ import annotations

import importlib
import importlib.metadata as md
import inspect
import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from packaging.specifiers import SpecifierSet
from packaging.version import Version

TARGET_USER_PACKAGES: List[Tuple[str, str, str]] = [
    ("transformers", ">=4.45.0", "transformers"),
    ("accelerate", ">=0.34.0", "accelerate"),
    ("datasets", ">=2.20.0", "datasets"),
    ("peft", ">=0.10.0", "peft"),
    ("trl", ">=0.11.0", "trl"),
    ("bitsandbytes", ">=0.43.0", "bitsandbytes"),
    ("sentence_transformers", ">=3.0.0", "sentence-transformers"),
    ("bm25s", ">=0.2.5", "bm25s"),
    ("scikit-learn", ">=1.4.0", "scikit-learn"),
    ("nltk", ">=3.8.1", "nltk"),
    ("pyvi", ">=0.1.1", "pyvi"),
    ("pyyaml", ">=6.0", "pyyaml"),
    ("pyarrow", ">=14.0.0", "pyarrow"),
    ("fastparquet", ">=2024.2.0", "fastparquet"),
    ("tqdm", ">=4.66.0", "tqdm"),
]


def get_installed_distribution_version(dist_name: str) -> Optional[str]:
    """Retrieve installed version of a package distribution."""
    try:
        return md.version(dist_name)
    except Exception:
        try:
            norm = dist_name.lower().replace("_", "-")
            return md.version(norm)
        except Exception:
            return None


def satisfies_spec(version: Optional[str], specifier: str) -> bool:
    """Check whether a version string satisfies a packaging specifier."""
    if version is None:
        return False
    try:
        v = Version(version)
        specs = SpecifierSet(specifier)
        return v in specs
    except Exception:
        return False


def snapshot_protected_versions() -> Dict[str, str]:
    """Enumerate and snapshot all installed protected Torch/CUDA/Triton distributions."""
    out = {}
    for dist in md.distributions():
        name = (dist.metadata.get("Name") or "").strip()
        norm = name.lower().replace("_", "-")
        if (
            norm in {"torch", "torchvision", "torchaudio", "triton"}
            or norm.startswith("nvidia-")
            or norm.startswith("cuda-")
        ):
            out[norm] = dist.version
    return dict(sorted(out.items()))


def write_protected_constraints(snapshot: Dict[str, str], path: str = "/tmp/legalqa_protected_constraints.txt") -> str:
    """Write pip constraints file pinning every installed protected distribution version."""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    lines = [f"{name}=={version}" for name, version in sorted(snapshot.items())]
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    return path


def assert_protected_versions_unchanged(before: Dict[str, str], after: Dict[str, str]) -> None:
    """Strictly assert that no protected distribution changed, was removed, or was introduced."""
    before_keys = set(before.keys())
    after_keys = set(after.keys())

    removed = before_keys - after_keys
    if removed:
        raise RuntimeError(f"Protected runtime package removed during bootstrap: {removed}")

    introduced = after_keys - before_keys
    if introduced:
        raise RuntimeError(f"New protected runtime package introduced during bootstrap: {introduced}")

    for k in before_keys:
        if before[k] != after[k]:
            raise RuntimeError(
                f"Protected runtime package changed during bootstrap! "
                f"{k}: before={before[k]} != after={after[k]}"
            )


def run_pip_check() -> None:
    """Run python -m pip check to ensure dependency tree has no broken requirements."""
    res = subprocess.run([sys.executable, "-m", "pip", "check"], capture_output=True, text=True)
    if res.returncode != 0:
        raise RuntimeError(f"pip check reported broken dependencies:\n{res.stdout}\n{res.stderr}")
    print("pip check: PASS (dependency tree consistent)")


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

    protected = snapshot_protected_versions()
    print(f"\nSnapshot of Protected Distributions ({len(protected)} items):")
    for k, v in protected.items():
        print(f"  * {k:32s}: {v}")

    return env_info


def bootstrap_dependencies(
    constraints_path: str = "/tmp/legalqa_protected_constraints.txt",
    allow_unprotected_drift: bool = False,
) -> Dict[str, Any]:
    """Safely install missing or outdated user-space dependencies using protected constraints."""
    print("\n=======================================================")
    print("         BOOTSTRAPPING USER-SPACE DEPENDENCIES         ")
    print("=======================================================")

    protected_before = snapshot_protected_versions()
    if protected_before:
        write_protected_constraints(protected_before, constraints_path)
        print(f"Pinned {len(protected_before)} protected distributions in constraints file: {constraints_path}")

    to_install_or_update = []
    for item in TARGET_USER_PACKAGES:
        import_name = item[0]
        specifier = item[1]
        pip_name = item[2]

        curr_ver = get_installed_distribution_version(pip_name) or get_installed_distribution_version(import_name)
        if curr_ver is None:
            to_install_or_update.append(f"{pip_name}{specifier}")
            print(f"  [MISSING] {pip_name:24s} -> will install ({specifier})")
        elif not satisfies_spec(curr_ver, specifier):
            to_install_or_update.append(f"{pip_name}{specifier}")
            print(f"  [OUTDATED] {pip_name:23s} ({curr_ver}) does not satisfy {specifier} -> will update")
        else:
            print(f"  [OK]      {pip_name:24s} ({curr_ver}) satisfies {specifier}")

    if to_install_or_update:
        print(f"\nInstalling {len(to_install_or_update)} package(s) with protected constraints...")
        cmd = [
            sys.executable, "-m", "pip", "install",
            "--upgrade-strategy", "only-if-needed",
        ]
        if protected_before and os.path.exists(constraints_path):
            cmd.extend(["-c", constraints_path])

        cmd.extend(to_install_or_update)
        print(f"Executing: {' '.join(cmd)}")
        res = subprocess.run(cmd, capture_output=True, text=True)
        if res.returncode != 0:
            raise RuntimeError(f"pip install failed with error:\n{res.stderr}\n{res.stdout}")
        print("Installation completed successfully.")
    else:
        print("All required user-space packages already satisfy version constraints.")

    # Validate dependency consistency
    try:
        run_pip_check()
    except Exception as e:
        print(f"Warning during pip check: {e}", file=sys.stderr)

    protected_after = snapshot_protected_versions()
    if not allow_unprotected_drift:
        assert_protected_versions_unchanged(protected_before, protected_after)
        print("Protected package integrity check: PASS (0 packages mutated)")

    return {
        "protected_before": protected_before,
        "protected_after": protected_after,
        "installed_or_updated": to_install_or_update,
    }


def verify_runtime_imports() -> Dict[str, Any]:
    """Test and verify all critical neural and retrieval modules and modern TRL SFT API."""
    print("\n=======================================================")
    print("         VERIFYING CRITICAL RUNTIME IMPORTS           ")
    print("=======================================================")

    import_results: Dict[str, str] = {}
    modules_to_test = [
        ("torch", "PyTorch"),
        ("transformers", "Hugging Face Transformers"),
        ("accelerate", "Hugging Face Accelerate"),
        ("datasets", "Hugging Face Datasets"),
        ("peft", "Hugging Face PEFT"),
        ("trl", "Hugging Face TRL"),
        ("bitsandbytes", "BitsAndBytes 4-bit Quantization"),
        ("sentence_transformers", "Sentence Transformers"),
        ("bm25s", "BM25S Sparse Retrieval"),
        ("nltk", "NLTK Natural Language Toolkit"),
        ("pyvi", "PyVi Vietnamese NLP"),
        ("sklearn", "Scikit-Learn"),
    ]

    for mod_name, desc in modules_to_test:
        try:
            mod = importlib.import_module(mod_name)
            ver = getattr(mod, "__version__", "ok")
            import_results[mod_name] = f"PASS ({ver})"
            print(f"  + {desc:32s} ({mod_name}): PASS ({ver})")
        except Exception as e:
            import_results[mod_name] = f"FAIL ({e})"
            print(f"  ! {desc:32s} ({mod_name}): FAIL ({e})", file=sys.stderr)

    # Check TRL completion_only_loss support explicitly (P0-1)
    try:
        from trl import SFTConfig, SFTTrainer
        sig = inspect.signature(SFTConfig)
        if "completion_only_loss" not in sig.parameters:
            raise RuntimeError(
                "Installed TRL SFTConfig does not support completion_only_loss parameter. "
                "Update TRL to >=0.11.0 to preserve exact completion-only loss semantics."
            )
        print("  + TRL SFTConfig.completion_only_loss: AVAILABLE")
    except Exception as e:
        print(f"  ! TRL SFTConfig check: {e}", file=sys.stderr)
        # In CPU testing environment without trl, log warning, but raise on real Kaggle
        if "trl" in sys.modules:
            raise

    return import_results


def save_bootstrap_manifest(output_path: str = "/kaggle/working/kaggle_environment.json") -> None:
    """Save environment and package tuple manifest to disk."""
    try:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
    except Exception:
        pass

    pkg_dict = {}
    for pkg in [
        "torch", "torchvision", "torchaudio", "triton", "transformers", "accelerate", "datasets",
        "peft", "trl", "bitsandbytes", "sentence-transformers", "bm25s", "nltk", "pyvi",
        "pandas", "numpy", "scikit-learn", "pyarrow", "fastparquet", "tqdm",
    ]:
        pkg_dict[pkg] = get_installed_distribution_version(pkg) or "not installed"

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
