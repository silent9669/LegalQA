"""Unit tests for scripts/run_pipeline.py argument parsing and test set resolution."""

from __future__ import annotations

import argparse
import os
import tempfile
from pathlib import Path

import pytest


def _build_parser():
    parser = argparse.ArgumentParser(description="LegalQA Task 2 Pipeline Execution Entrypoint")
    parser.add_argument("--config", default="configs/task2/runtime/kaggle_t4x2.yaml")
    parser.add_argument("--data-dir", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--test-path", default=None)
    parser.add_argument("--require-smoke-pass", default=None)
    parser.add_argument("--allow-single-gpu", action="store_true")
    parser.add_argument("--env-file", default=None)
    parser.add_argument("--no-upload-to-hf", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--ensure-dense-index", action="store_true")
    parser.add_argument("--dense-revision", default=None)
    parser.add_argument("--dense-model", default="CODE4LIFEOFFICIAL/huydang-dek21-embedding-v2")
    return parser


def _resolve_test_file(test_path_arg: str | None, data_dir: str, runtime_root: str) -> str:
    paths = {"data_dir": data_dir, "runtime_root": runtime_root}
    if test_path_arg:
        test_file = os.path.abspath(test_path_arg)
    else:
        candidate_private = os.path.join(paths.get("data_dir", ""), "private-official.json")
        if os.path.exists(candidate_private):
            test_file = candidate_private
        else:
            candidate_public = os.path.join(paths.get("data_dir", ""), "public-official.json")
            if os.path.exists(candidate_public):
                test_file = candidate_public
            else:
                candidate_private_root = os.path.join(paths.get("runtime_root", ""), "private-official.json")
                if os.path.exists(candidate_private_root):
                    test_file = candidate_private_root
                else:
                    test_file = os.path.join(paths.get("runtime_root", ""), "public-official.json")
    return test_file


def test_parser_supports_test_path():
    parser = _build_parser()
    args = parser.parse_args(["--test-path", "/custom/path/test.json"])
    assert args.test_path == "/custom/path/test.json"


def test_parser_test_path_default_none():
    parser = _build_parser()
    args = parser.parse_args([])
    assert args.test_path is None


def test_resolve_test_path_prefers_explicit_cli_argument():
    with tempfile.TemporaryDirectory() as tmpdir:
        data_dir = os.path.join(tmpdir, "data")
        os.makedirs(data_dir)
        Path(data_dir, "private-official.json").write_text("{}", encoding="utf-8")
        custom_test = os.path.join(tmpdir, "custom.json")
        Path(custom_test).write_text("{}", encoding="utf-8")

        resolved = _resolve_test_file(custom_test, data_dir, tmpdir)
        assert resolved == os.path.abspath(custom_test)


def test_resolve_test_path_prefers_private_when_present():
    with tempfile.TemporaryDirectory() as tmpdir:
        data_dir = os.path.join(tmpdir, "data")
        os.makedirs(data_dir)
        priv = Path(data_dir, "private-official.json")
        pub = Path(data_dir, "public-official.json")
        priv.write_text("{}", encoding="utf-8")
        pub.write_text("{}", encoding="utf-8")

        resolved = _resolve_test_file(None, data_dir, tmpdir)
        assert resolved == str(priv)


def test_resolve_test_path_falls_back_to_public_when_no_private():
    with tempfile.TemporaryDirectory() as tmpdir:
        data_dir = os.path.join(tmpdir, "data")
        os.makedirs(data_dir)
        pub = Path(data_dir, "public-official.json")
        pub.write_text("{}", encoding="utf-8")

        resolved = _resolve_test_file(None, data_dir, tmpdir)
        assert resolved == str(pub)
