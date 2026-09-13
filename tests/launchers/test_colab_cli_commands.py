from pathlib import Path
from unittest.mock import MagicMock, patch
import pytest

from scripts.launch_colab_training import ColabLauncher, build_run_request


def test_build_run_request():
    req = build_run_request(
        stage="colab_t4",
        candidate_id="cand_123456",
        git_commit_sha="abcdef123456",
        dataset_slug="phucdangg/legalqa-task2-clean-data",
        dataset_version=1,
    )
    assert req["stage"] == "colab_t4"
    assert req["requested_gpu"] == "T4"
    assert req["candidate_id"] == "cand_123456"
    assert req["git_commit_sha"] == "abcdef123456"


def test_colab_cli_command_sequence(tmp_path):
    """Test standard Colab CLI execution sequence and assert NO --timeout in colab exec."""
    executed_commands = []

    def mock_subprocess_run(cmd, *args, **kwargs):
        executed_commands.append(list(cmd))
        mock_res = MagicMock()
        mock_res.returncode = 0
        mock_res.stdout = "OK"
        return mock_res

    launcher = ColabLauncher(
        stage="colab-t4",
        candidate_path="artifacts/candidates/test/candidate_manifest.json",
        kaggle_report_path="artifacts/gates/test/kaggle_t4x2_report.json",
        session_name="legalqa-test-session",
        keep_alive=False,
    )

    (tmp_path / "run_request.json").write_text("{}")

    with patch("subprocess.run", side_effect=mock_subprocess_run), \
         patch("subprocess.check_output", return_value=b"colab version 0.1.0"), \
         patch.object(launcher, "_preflight_checks", return_value=None), \
         patch.object(launcher, "_prepare_bootstrap_bundle", return_value=tmp_path), \
         patch.object(launcher, "_verify_downloaded_artifacts", return_value=None):

        launcher.launch()

    # Flatten command strings
    cmd_strs = [" ".join(c) for c in executed_commands]

    # 1. colab new must specify GPU
    assert any("colab new -s legalqa-test-session --gpu T4" in s for s in cmd_strs)

    # 2. colab upload must upload bootstrap
    assert any("colab upload -s legalqa-test-session" in s for s in cmd_strs)

    # 3. colab exec must run colab_remote_entry.py WITHOUT --timeout
    exec_cmds = [c for c in executed_commands if any("colab" in token for token in c) and "exec" in c]
    assert len(exec_cmds) > 0, "No colab exec command found"
    for c in exec_cmds:
        assert "--timeout" not in c, f"Found forbidden '--timeout' in colab exec arguments: {c}"
        assert "-f" in c or "scripts/colab_remote_entry.py" in " ".join(c)

    # 4. colab download must download results
    assert any("colab download -s legalqa-test-session" in s for s in cmd_strs)

    # 5. colab stop must be called at the end
    assert any("colab stop -s legalqa-test-session" in s for s in cmd_strs)
