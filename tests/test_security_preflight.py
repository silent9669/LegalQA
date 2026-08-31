from pathlib import Path
import pytest
from src.common.security import (
    scan_text_for_secrets,
    scan_file_for_secrets,
    scan_directory_for_secrets,
    assert_no_secrets_in_workspace,
)


def test_scan_text_for_secrets():
    clean_text = "MODEL_PATH = 'Qwen/Qwen2.5-3B-Instruct'\nHF_TOKEN = os.environ.get('HF_TOKEN')"
    assert scan_text_for_secrets(clean_text) == []

    dirty_hf = "token = '" + "hf_" + "1234567890abcdef1234567890" + "'"
    findings = scan_text_for_secrets(dirty_hf)
    assert len(findings) == 1
    assert findings[0]["type"] == "huggingface_token"
    assert findings[0]["masked_preview"].startswith("hf_1")

    dirty_gh = "gh_token = '" + "ghp_" + "1234567890abcdefghijklmnopqrstuvwxyz12" + "'"
    assert len(scan_text_for_secrets(dirty_gh)) == 1


def test_scan_notebook_file(tmp_path: Path):
    nb_clean = tmp_path / "clean.ipynb"
    nb_clean.write_text('{"cells": [{"source": ["print(\'hello\')"]}]}', encoding="utf-8")
    assert scan_file_for_secrets(nb_clean) == []

    nb_dirty = tmp_path / "dirty.ipynb"
    dirty_token = "hf_" + "0123456789abcdef0123456789"
    nb_dirty.write_text(f'{{"cells": [{{"source": ["HF_TOKEN = \'{dirty_token}\'"]}}]}}', encoding="utf-8")
    findings = scan_file_for_secrets(nb_dirty)
    assert len(findings) == 1
    assert findings[0]["type"] == "huggingface_token"


def test_workspace_has_no_secrets():
    workspace_root = Path(__file__).resolve().parent.parent
    assert_no_secrets_in_workspace(workspace_root, exclude_tests=True)
