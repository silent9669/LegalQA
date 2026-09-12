import os
import glob
import hashlib
import json
from typing import Dict, Any, List, Optional
import yaml
import pandas as pd

def compute_sha256(file_path: str) -> str:
    h = hashlib.sha256()
    with open(file_path, "rb") as f:
        while chunk := f.read(1024 * 1024):
            h.update(chunk)
    return h.hexdigest()

def validate_dataset(data_dir: str, schema_path: str = "configs/dataset_schema.yaml") -> Dict[str, Any]:
    """
    Validates canonical dataset artifacts in data_dir against schema_path.
    Guarantees:
    - Required files exist and match schema columns
    - Referential integrity between citations and legal chunks
    - No code is bundled inside the dataset directory
    - SHA256 checksums match dataset_manifest.json if present
    """
    report: Dict[str, Any] = {
        "status": "PASS",
        "data_dir": os.path.abspath(data_dir),
        "schema_path": schema_path,
        "errors": [],
        "warnings": [],
        "tables_validated": {},
        "manifest_verified": False,
        "detected_directories": [],
    }

    if not os.path.exists(data_dir):
        report["status"] = "FAIL"
        report["errors"].append(f"Data directory does not exist: {data_dir}")
        return report

    # 1. Zero-code check: ensure no code directories or source scripts exist
    subdirs = [d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))]
    report["detected_directories"] = subdirs

    if "code" in subdirs or "src" in subdirs:
        report["status"] = "FAIL"
        report["errors"].append(
            f"Code bundle detected inside dataset directory ({subdirs}). "
            "New architecture strictly requires dataset releases to be pure data."
        )

    code_files = glob.glob(os.path.join(data_dir, "**/*.py"), recursive=True) + glob.glob(os.path.join(data_dir, "**/*.sh"), recursive=True)
    if code_files:
        report["status"] = "FAIL"
        report["errors"].append(f"Executable code files found inside dataset directory: {code_files}")

    # 2. Schema check
    if not os.path.exists(schema_path):
        report["status"] = "FAIL"
        report["errors"].append(f"Schema configuration not found at {schema_path}")
        return report

    with open(schema_path, "r") as f:
        schema = yaml.safe_load(f)

    tables_cfg = schema.get("tables", {})
    loaded_tables: Dict[str, pd.DataFrame] = {}

    for table_name, tcfg in tables_cfg.items():
        req_file = tcfg.get("required_file")
        if not req_file:
            continue

        file_path = os.path.join(data_dir, req_file)
        if not os.path.exists(file_path):
            report["status"] = "FAIL"
            report["errors"].append(f"Required dataset table file missing: {req_file}")
            continue

        if req_file.endswith(".parquet"):
            try:
                df = pd.read_parquet(file_path)
                loaded_tables[table_name] = df
                report["tables_validated"][table_name] = {
                    "file": req_file,
                    "rows": len(df),
                    "columns": list(df.columns),
                }

                # Column presence check
                req_cols = tcfg.get("required_columns", {})
                for col in req_cols.keys():
                    if col not in df.columns:
                        report["status"] = "FAIL"
                        report["errors"].append(f"Table '{table_name}' missing required column '{col}'")

                # Primary key check
                pk = tcfg.get("primary_key")
                if pk and pk in df.columns:
                    null_count = df[pk].isnull().sum()
                    if null_count > 0:
                        report["status"] = "FAIL"
                        report["errors"].append(f"Table '{table_name}' has {null_count} nulls in primary key '{pk}'")

            except Exception as e:
                report["status"] = "FAIL"
                report["errors"].append(f"Failed to read parquet table '{req_file}': {e}")

        elif req_file.endswith(".json"):
            try:
                with open(file_path, "r", encoding="utf-8") as jf:
                    jdata = json.load(jf)
                report["tables_validated"][table_name] = {
                    "file": req_file,
                    "type": "json",
                    "count": len(jdata) if isinstance(jdata, (list, dict)) else 1,
                }
            except Exception as e:
                report["status"] = "FAIL"
                report["errors"].append(f"Failed to read json file '{req_file}': {e}")

    # 3. Referential integrity
    if "qa_unique" in loaded_tables:
        valid_qa_ids = set(loaded_tables["qa_unique"]["qa_id"].dropna().unique())
        for ref_table in ["qa_citations", "retrieval_labels", "fold_assignments"]:
            if ref_table in loaded_tables and "qa_id" in loaded_tables[ref_table].columns:
                ref_qa_ids = set(loaded_tables[ref_table]["qa_id"].dropna().unique())
                dangling_qa = ref_qa_ids - valid_qa_ids
                if dangling_qa:
                    report["status"] = "FAIL"
                    report["errors"].append(
                        f"Referential integrity failure: {len(dangling_qa)} qa_ids in '{ref_table}' not found in qa_unique"
                    )

    if "retrieval_labels" in loaded_tables and "legal_chunks" in loaded_tables:
        valid_chunk_ids = set(loaded_tables["legal_chunks"]["chunk_id"].dropna().unique())
        pos_chunks = set(loaded_tables["retrieval_labels"]["positive_chunk_id"].dropna().unique())
        missing_chunks = pos_chunks - valid_chunk_ids
        if missing_chunks:
            report["status"] = "FAIL"
            report["errors"].append(
                f"Referential integrity failure: {len(missing_chunks)} positive_chunk_ids in retrieval_labels not found in legal_chunks"
            )

    # 4. Manifest verification
    manifest_path = os.path.join(data_dir, "dataset_manifest.json")
    if os.path.exists(manifest_path):
        try:
            with open(manifest_path, "r", encoding="utf-8") as mf:
                manifest = json.load(mf)
            files_meta = manifest.get("files", {})
            if not files_meta:
                report["status"] = "FAIL"
                report["errors"].append("dataset_manifest.json contains empty or missing 'files' dictionary")
            else:
                missing_files = []
                mismatch_files = []
                for fname, meta in files_meta.items():
                    expected_sha = meta.get("sha256")
                    fpath = os.path.join(data_dir, fname)
                    if not os.path.exists(fpath):
                        missing_files.append(fname)
                    elif expected_sha:
                        actual_sha = compute_sha256(fpath)
                        if actual_sha != expected_sha:
                            mismatch_files.append((fname, expected_sha, actual_sha))

                if missing_files or mismatch_files:
                    report["status"] = "FAIL"
                    report["manifest_verified"] = False
                    if missing_files:
                        report["errors"].append(f"Manifest-listed files missing in dataset directory: {missing_files}")
                    if mismatch_files:
                        report["errors"].append(f"Manifest SHA256 mismatch for files: {mismatch_files}")
                else:
                    report["manifest_verified"] = True
        except Exception as e:
            report["status"] = "FAIL"
            report["errors"].append(f"Could not parse or verify dataset_manifest.json: {e}")

    return report
