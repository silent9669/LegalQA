"""Train-probe request validation regressions (execution needs CUDA)."""

from __future__ import annotations

import pytest

from scripts.kaggle_train_probe import DEFAULT_MAX_EXAMPLES, DEFAULT_MAX_STEPS, validate_probe_request


def test_probe_bounds_reject_absurd_values(tmp_path):
    cand = tmp_path / "cand.json"
    cand.write_text("{}", encoding="utf-8")
    data = tmp_path / "data"
    data.mkdir()
    with pytest.raises(ValueError, match="max_steps"):
        validate_probe_request(str(cand), str(data), 0, 200)
    with pytest.raises(ValueError, match="max_steps"):
        validate_probe_request(str(cand), str(data), 10000, 200)
    with pytest.raises(ValueError, match="max_examples"):
        validate_probe_request(str(cand), str(data), 60, 0)
    with pytest.raises(FileNotFoundError, match="candidate"):
        validate_probe_request(str(tmp_path / "missing.json"), str(data), 60, 200)
    with pytest.raises(FileNotFoundError, match="data directory"):
        validate_probe_request(str(cand), str(tmp_path / "missing"), 60, 200)
    req = validate_probe_request(str(cand), str(data), 60, 200)
    assert req["max_steps"] == 60 and req["max_examples"] == 200
    assert DEFAULT_MAX_STEPS == 60 and DEFAULT_MAX_EXAMPLES == 200
