from pathlib import Path
from src.common.security import scan_directory_for_secrets

def test_no_secrets_in_repository():
    """Fail-closed security test ensuring zero API keys, tokens, or credentials exist in workspace."""
    repo_root = Path(__file__).resolve().parent.parent.parent
    findings = scan_directory_for_secrets(
        repo_root,
        exclude_dirs=[".git", ".venv", ".venv-ml", ".venv311", ".pytest_cache", ".playwright-mcp", "artifacts/raw", "tests"]
    )
    assert len(findings) == 0, f"Found leaked secrets in workspace: {findings}"
