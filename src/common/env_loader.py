"""Environment and credential loader supporting local .env files, Colab secrets, and Kaggle/HF auth."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional


def parse_env_file(path: str | Path) -> Dict[str, str]:
    """Parse a .env file into key-value pairs without requiring third-party libraries."""
    env_vars: Dict[str, str] = {}
    p = Path(path)
    if not p.is_file():
        return env_vars

    with open(p, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if "=" in line:
                key, val = line.split("=", 1)
                key = key.strip()
                val = val.strip().strip("'").strip('"')
                if key:
                    env_vars[key] = val
    return env_vars


def load_environment(
    env_file: Optional[str | Path] = None,
    override: bool = False,
) -> Dict[str, Any]:
    """Discover, load, and configure environment variables across local and Colab environments.

    Search order:
    1. Explicit env_file path if supplied
    2. ./.env, ./LegalQA/.env
    3. Parent directory ../.env
    4. /content/.env, /content/LegalQA/.env
    5. google.colab.userdata (Colab secrets)
    6. Existing os.environ variables
    """
    candidates: List[Path] = []
    if env_file:
        candidates.append(Path(env_file))

    cwd = Path.cwd()
    candidates.extend([
        cwd / ".env",
        cwd / "LegalQA" / ".env",
        cwd.parent / ".env",
        Path("/content/.env"),
        Path("/content/LegalQA/.env"),
        Path("/kaggle/working/.env"),
        Path("/kaggle/working/LegalQA/.env"),
        Path("/kaggle/input/.env"),
    ])
    # Search for mounted dataset .env on Kaggle if present
    kaggle_input = Path("/kaggle/input")
    if kaggle_input.is_dir():
        try:
            candidates.extend(list(kaggle_input.glob("*/.env")))
        except Exception:
            pass

    loaded_from: Optional[str] = None
    for cand in candidates:
        if cand.is_file():
            parsed = parse_env_file(cand)
            for k, v in parsed.items():
                if override or k not in os.environ:
                    os.environ[k] = v
            loaded_from = str(cand)
            break

    # Colab userdata fallback
    try:
        from google.colab import userdata  # type: ignore

        if "HF_TOKEN" not in os.environ:
            tok = userdata.get("HF_TOKEN") or userdata.get("HUGGINGFACE_TOKEN") or userdata.get("HF_TOKEN_WRITE")
            if tok:
                os.environ["HF_TOKEN"] = str(tok)

        if "KAGGLE_KEY" not in os.environ:
            kkey = userdata.get("KAGGLE_KEY") or userdata.get("KAGGLE_API_TOKEN")
            if kkey:
                os.environ["KAGGLE_KEY"] = str(kkey)

        if "KAGGLE_USERNAME" not in os.environ:
            kuser = userdata.get("KAGGLE_USERNAME")
            if kuser:
                os.environ["KAGGLE_USERNAME"] = str(kuser)
    except Exception:
        pass

    # Kaggle secrets fallback
    try:
        from kaggle_secrets import UserSecretsClient  # type: ignore
        secrets_client = UserSecretsClient()
        if "HF_TOKEN" not in os.environ:
            tok = (
                secrets_client.get_secret("HF_TOKEN")
                or secrets_client.get_secret("HUGGINGFACE_TOKEN")
                or secrets_client.get_secret("huggingface_token")
                or secrets_client.get_secret("HF_TOKEN_WRITE")
            )
            if tok:
                os.environ["HF_TOKEN"] = str(tok)

        if "KAGGLE_KEY" not in os.environ:
            kkey = secrets_client.get_secret("KAGGLE_KEY") or secrets_client.get_secret("KAGGLE_API_TOKEN")
            if kkey:
                os.environ["KAGGLE_KEY"] = str(kkey)

        if "KAGGLE_USERNAME" not in os.environ:
            kuser = secrets_client.get_secret("KAGGLE_USERNAME")
            if kuser:
                os.environ["KAGGLE_USERNAME"] = str(kuser)
    except Exception:
        pass

    # Synchronize HF token aliases
    hf_token = (
        os.environ.get("HF_TOKEN")
        or os.environ.get("HF_TOKEN_WRITE")
        or os.environ.get("HUGGING_FACE_HUB_TOKEN")
        or os.environ.get("HF_TOKEN_READ")
    )
    if hf_token:
        os.environ["HF_TOKEN"] = hf_token
        os.environ["HUGGING_FACE_HUB_TOKEN"] = hf_token
        try:
            import huggingface_hub
            huggingface_hub.login(token=hf_token, add_to_git_credential=False)
        except Exception:
            pass

    # Synchronize Kaggle key aliases
    k_key = os.environ.get("KAGGLE_KEY") or os.environ.get("KAGGLE_API_TOKEN")
    if k_key:
        os.environ["KAGGLE_KEY"] = k_key
        os.environ["KAGGLE_API_TOKEN"] = k_key

    # Configure Kaggle credentials file if KAGGLE_USERNAME and KAGGLE_KEY exist
    kaggle_configured = False
    k_user = os.environ.get("KAGGLE_USERNAME")
    k_key = os.environ.get("KAGGLE_KEY")
    if k_user and k_key:
        try:
            k_dir = Path.home() / ".kaggle"
            k_dir.mkdir(parents=True, exist_ok=True)
            k_json = k_dir / "kaggle.json"
            with open(k_json, "w", encoding="utf-8") as f:
                json.dump({"username": k_user, "key": k_key}, f, indent=2)
            k_json.chmod(0o600)
            kaggle_configured = True
        except Exception:
            kaggle_configured = False

    def _mask(val: Optional[str]) -> str:
        if not val:
            return "not set"
        return val[:4] + "..." + val[-4:] if len(val) > 8 else "***"

    return {
        "loaded_from_file": loaded_from,
        "hf_token_configured": bool(hf_token),
        "hf_token_masked": _mask(hf_token),
        "kaggle_configured": kaggle_configured or (Path.home() / ".kaggle" / "kaggle.json").exists(),
        "kaggle_user": k_user or "unknown",
    }
