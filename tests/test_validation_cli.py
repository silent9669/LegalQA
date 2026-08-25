import pytest
import os
import sys
import subprocess

def test_validation_cli_help():
    result = subprocess.run(
        ["/Users/phucdang/Downloads/.venv/bin/python3", "validation.py", "--help"],
        capture_output=True,
        text=True
    )
    assert result.returncode == 0
    assert "CodaBench 5-Fold OOF Validation" in result.stdout
