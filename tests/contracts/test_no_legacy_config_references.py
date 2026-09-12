import re
from pathlib import Path

LEGACY_CONFIG_PATTERNS = [
    re.compile(r"configs/models\.yaml"),
    re.compile(r"configs/production_selection\.yaml"),
    re.compile(r"configs/pipeline\.yaml"),
    re.compile(r"configs/runtime_api\.yaml"),
    re.compile(r"configs/task2\.yaml"),
]

SCAN_DIRS = ["src", "scripts"]


def test_no_legacy_config_references_in_production_code():
    """Ensure no production code in src/ or scripts/ references deleted legacy configs."""
    repo_root = Path(__file__).resolve().parent.parent.parent
    violations = []

    for dir_name in SCAN_DIRS:
        target_dir = repo_root / dir_name
        if not target_dir.exists():
            continue

        for py_file in target_dir.rglob("*.py"):
            text = py_file.read_text(encoding="utf-8")
            for pattern in LEGACY_CONFIG_PATTERNS:
                matches = pattern.findall(text)
                if matches:
                    rel_path = py_file.relative_to(repo_root)
                    violations.append(f"{rel_path}: matches {pattern.pattern}")

    assert not violations, (
        f"Found {len(violations)} references to deleted legacy configs:\n"
        + "\n".join(f"  - {v}" for v in violations)
    )
