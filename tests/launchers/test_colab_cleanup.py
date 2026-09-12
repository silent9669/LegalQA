from unittest.mock import MagicMock, patch
import pytest

from scripts.launch_colab_training import ColabLauncher


def test_colab_cleanup_runs_on_exec_failure(tmp_path):
    """Ensure colab stop is called in finally block even when colab exec fails."""
    executed_commands = []

    def mock_subprocess_run(cmd, *args, **kwargs):
        executed_commands.append(list(cmd))
        if "exec" in cmd:
            raise RuntimeError("Remote execution failed with exit code 1")
        mock_res = MagicMock()
        mock_res.returncode = 0
        return mock_res

    launcher = ColabLauncher(
        stage="colab-t4",
        candidate_path="artifacts/candidates/test/candidate_manifest.json",
        kaggle_report_path="artifacts/gates/test/kaggle_t4x2_report.json",
        session_name="legalqa-cleanup-test",
        keep_alive=False,
    )

    with patch("subprocess.run", side_effect=mock_subprocess_run), \
         patch("subprocess.check_output", return_value=b"colab version 0.1.0"), \
         patch.object(launcher, "_preflight_checks", return_value=None), \
         patch.object(launcher, "_prepare_bootstrap_bundle", return_value=tmp_path):

        with pytest.raises(RuntimeError, match="Remote execution failed"):
            launcher.launch()

    cmd_strs = [" ".join(c) for c in executed_commands]
    assert any("colab stop -s legalqa-cleanup-test" in s for s in cmd_strs), (
        "Expected 'colab stop' to be called in finally block upon failure"
    )


def test_colab_keep_alive_skips_stop(tmp_path):
    """Ensure colab stop is skipped when keep_alive=True."""
    executed_commands = []

    def mock_subprocess_run(cmd, *args, **kwargs):
        executed_commands.append(list(cmd))
        mock_res = MagicMock()
        mock_res.returncode = 0
        return mock_res

    launcher = ColabLauncher(
        stage="colab-t4",
        candidate_path="artifacts/candidates/test/candidate_manifest.json",
        kaggle_report_path="artifacts/gates/test/kaggle_t4x2_report.json",
        session_name="legalqa-keepalive-test",
        keep_alive=True,
    )

    with patch("subprocess.run", side_effect=mock_subprocess_run), \
         patch("subprocess.check_output", return_value=b"colab version 0.1.0"), \
         patch.object(launcher, "_preflight_checks", return_value=None), \
         patch.object(launcher, "_prepare_bootstrap_bundle", return_value=tmp_path), \
         patch.object(launcher, "_verify_downloaded_artifacts", return_value=None):

        launcher.launch()

    cmd_strs = [" ".join(c) for c in executed_commands]
    assert not any("colab stop -s legalqa-keepalive-test" in s for s in cmd_strs)
