from pathlib import Path
import pytest
from scripts.bootstrap_kaggle_env import (
    TARGET_USER_PACKAGES,
    PROTECTED_RUNTIME_PREFIXES,
    assert_no_new_pip_conflicts,
)


def test_requirements_kaggle_contains_required_pins():
    paths = [Path("requirements/kaggle.txt"), Path("requirements-kaggle.txt")]
    for p in paths:
        assert p.exists(), f"{p} must exist"
        content = p.read_text(encoding="utf-8")
        lines = [line.strip() for line in content.splitlines() if line.strip() and not line.strip().startswith("#")]

        # Must strictly pin trl and liger-kernel
        assert "trl==1.12.0" in lines, f"{p} must contain trl==1.12.0"
        assert "liger-kernel==0.8.2" in lines, f"{p} must contain liger-kernel==0.8.2"


def test_requirements_kaggle_forbids_torch_cuda_triton_replacement():
    forbidden_prefixes = ("torch", "torchvision", "torchaudio", "triton", "cuda", "nvidia")
    paths = [Path("requirements/kaggle.txt"), Path("requirements-kaggle.txt")]
    for p in paths:
        content = p.read_text(encoding="utf-8")
        for line in content.splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            pkg = line.split("==")[0].split(">=")[0].split("<=")[0].split("<")[0].split(">")[0].strip()
            for forbidden in forbidden_prefixes:
                assert pkg != forbidden and not pkg.startswith(f"{forbidden}-"), (
                    f"{p} must not contain protected package {pkg} (matches {forbidden})"
                )


def test_requirements_base_and_gpu_test_exist():
    base_p = Path("requirements/base.txt")
    gpu_p = Path("requirements/gpu-test.txt")

    assert base_p.exists()
    assert gpu_p.exists()

    base_content = base_p.read_text(encoding="utf-8")
    assert "numpy" in base_content
    assert "bm25s" in base_content

    gpu_content = gpu_p.read_text(encoding="utf-8")
    assert "kaggle.txt" in gpu_content


def test_bootstrap_target_packages_has_liger_and_trl():
    pkg_map = {item[2]: item[1] for item in TARGET_USER_PACKAGES}
    assert "trl" in pkg_map, "TARGET_USER_PACKAGES must contain trl"
    assert pkg_map["trl"] == "==1.12.0", f"trl must be ==1.12.0, got {pkg_map['trl']}"

    assert "liger-kernel" in pkg_map, "TARGET_USER_PACKAGES must contain liger-kernel"
    assert pkg_map["liger-kernel"] == "==0.8.2", f"liger-kernel must be ==0.8.2, got {pkg_map['liger-kernel']}"


def test_bootstrap_protected_prefixes_protect_torch_triton_cuda():
    expected_protected = {"torch", "triton", "cuda", "nvidia"}
    prefixes = set(PROTECTED_RUNTIME_PREFIXES)
    for exp in expected_protected:
        assert any(p.startswith(exp) or exp.startswith(p) for p in prefixes), (
            f"PROTECTED_RUNTIME_PREFIXES must protect {exp}"
        )


def test_pip_check_baseline_aware_regression_guard_intact():
    before = ["pkg-a 1.0 has requirement pkg-b>2.0, but you have pkg-b 1.5."]
    after_same = list(before)
    after_new = before + ["new-pkg 2.0 has requirement other<1.0, but you have other 1.5."]

    # Unchanged baseline conflicts must not raise
    assert assert_no_new_pip_conflicts(before, after_same) == []

    # New conflicts introduced must raise RuntimeError
    with pytest.raises(RuntimeError, match="LegalQA bootstrap introduced new dependency conflict"):
        assert_no_new_pip_conflicts(before, after_new)
