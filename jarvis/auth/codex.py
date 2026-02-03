"""Codex CLI auth integration."""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path
from typing import Optional, Dict

from jarvis.assistant import load_config, save_config


CODEX_AUTH_PATH = Path.home() / ".codex" / "auth.json"


def read_codex_auth() -> Optional[Dict[str, str]]:
    if not CODEX_AUTH_PATH.exists():
        return None

    try:
        data = json.loads(CODEX_AUTH_PATH.read_text())
    except Exception:
        return None

    api_key = data.get("OPENAI_API_KEY")
    tokens = data.get("tokens") or {}
    access_token = tokens.get("access_token")
    refresh_token = tokens.get("refresh_token")

    return {
        "api_key": api_key,
        "access_token": access_token,
        "refresh_token": refresh_token,
        "source": "codex_cli",
    }


def import_codex_auth() -> dict:
    """Import Codex CLI credentials into settings.yaml."""
    creds = read_codex_auth() or {}
    config = load_config()

    providers = config.setdefault("providers", {})
    openai_cfg = providers.setdefault("openai", {})

    if creds.get("api_key"):
        openai_cfg["api_key"] = creds["api_key"]
    if creds.get("access_token"):
        openai_cfg["access_token"] = creds["access_token"]

    if creds:
        openai_cfg["auth_source"] = "codex_cli"

    save_config(config)
    return {
        "imported": bool(creds),
        "has_api_key": bool(openai_cfg.get("api_key")),
        "has_access_token": bool(openai_cfg.get("access_token")),
        "source": openai_cfg.get("auth_source"),
    }


def codex_device_login(timeout: int = 300) -> dict:
    """Run Codex CLI device auth and import credentials."""
    try:
        result = subprocess.run(
            ["codex", "login", "--device-auth"],
            capture_output=True,
            text=True,
            timeout=timeout
        )
        output = (result.stdout or "") + (("\n" + result.stderr) if result.stderr else "")
        imported = import_codex_auth()
        return {
            "ok": result.returncode == 0,
            "output": output.strip(),
            "import": imported,
        }
    except FileNotFoundError:
        return {"ok": False, "output": "codex CLI not found", "import": {"imported": False}}
    except subprocess.TimeoutExpired:
        return {"ok": False, "output": "codex login timed out", "import": {"imported": False}}
