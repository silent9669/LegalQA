"""Security and preflight checks for LegalQA pipeline.

Scans files, directories, and notebooks for leaked credentials or secrets.
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Dict, List, Sequence

# Common secret patterns
SECRET_PATTERNS = {
    "huggingface_token": re.compile(r"\bhf_[A-Za-z0-9]{20,}\b"),
    "aws_access_key": re.compile(r"\bAKIA[0-9A-Z]{16}\b"),
    "github_token": re.compile(r"\b(ghp|gho|ghu|ghs|ghr)_[A-Za-z0-9_]{36,}\b"),
    "generic_api_key": re.compile(r"(?i)(api[_-]?key|secret[_-]?key|access[_-]?token)\s*[:=]\s*['\"][A-Za-z0-9_\-]{24,}['\"]"),
}

EXCLUDED_EXTENSIONS = {
    ".parquet",
    ".safetensors",
    ".bin",
    ".pt",
    ".pth",
    ".npy",
    ".npz",
    ".zip",
    ".tar",
    ".gz",
    ".pyc",
}

EXCLUDED_DIRS = {
    ".git",
    ".venv",
    ".venv-ml",
    "__pycache__",
    ".pytest_cache",
    ".playwright-mcp",
    "node_modules",
}


def scan_text_for_secrets(text: str) -> List[Dict[str, str]]:
    """Scan raw string content for known secret patterns."""
    findings = []
    for name, pattern in SECRET_PATTERNS.items():
        for match in pattern.finditer(text):
            secret_matched = match.group(0)
            masked = secret_matched[:4] + "..." + secret_matched[-4:] if len(secret_matched) > 8 else "***"
            findings.append({
                "type": name,
                "masked_preview": masked,
                "start": match.start(),
                "end": match.end(),
            })
    return findings


def scan_file_for_secrets(file_path: str | Path) -> List[Dict[str, str]]:
    """Scan a single file (including .ipynb notebooks) for secrets."""
    path = Path(file_path)
    if not path.is_file():
        return []
    if path.suffix.lower() in EXCLUDED_EXTENSIONS:
        return []

    try:
        if path.suffix.lower() == ".ipynb":
            with open(path, "r", encoding="utf-8", errors="ignore") as f:
                nb_data = json.load(f)
            text_blocks = []
            for cell in nb_data.get("cells", []):
                src = cell.get("source", [])
                if isinstance(src, list):
                    text_blocks.append("".join(src))
                elif isinstance(src, str):
                    text_blocks.append(src)
            content = "\n".join(text_blocks)
        else:
            with open(path, "r", encoding="utf-8", errors="ignore") as f:
                content = f.read()

        findings = scan_text_for_secrets(content)
        for f in findings:
            f["file"] = str(path)
        return findings
    except Exception:
        return []


def scan_directory_for_secrets(
    root_dir: str | Path,
    include_extensions: Sequence[str] = (".py", ".ipynb", ".json", ".yaml", ".yml", ".sh", ".md"),
    exclude_dirs: Sequence[str] = None,
) -> List[Dict[str, str]]:
    """Recursively scan a directory for secrets in text-like files."""
    root = Path(root_dir)
    findings = []
    ext_set = {e.lower() for e in include_extensions}
    ex_dirs = EXCLUDED_DIRS.union(set(exclude_dirs or []))

    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d not in ex_dirs]

        for filename in filenames:
            file_path = Path(dirpath) / filename
            if file_path.suffix.lower() in ext_set:
                file_findings = scan_file_for_secrets(file_path)
                findings.extend(file_findings)

    return findings


def assert_no_secrets_in_workspace(root_dir: str | Path, exclude_tests: bool = True) -> None:
    """Preflight check that raises RuntimeError if any secrets are detected in workspace code."""
    ex_dirs = ["tests"] if exclude_tests else []
    findings = scan_directory_for_secrets(root_dir, exclude_dirs=ex_dirs)
    if findings:
        report = "\n".join(f"- {f['file']} matched {f['type']} ({f['masked_preview']})" for f in findings)
        raise RuntimeError(f"CRITICAL: Secret scanner detected credentials in workspace:\n{report}")
