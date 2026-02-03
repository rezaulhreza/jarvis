"""Claude CLI auth integration (best-effort)."""

from __future__ import annotations

import os
import json
import subprocess
from pathlib import Path
from typing import Optional, Dict

from jarvis.assistant import load_config, save_config
from jarvis import get_data_dir


OPENCLAW_CRED_DIR = Path.home() / ".openclaw" / "credentials"
JARVIS_CRED_DIR = get_data_dir() / "credentials"


def _find_token_in_json(path: Path) -> Optional[Dict[str, str]]:
    try:
        data = json.loads(path.read_text())
    except Exception:
        return None

    # Common token keys
    candidates = {}
    for key in ["access_token", "refresh_token", "api_key", "token", "id_token"]:
        val = data.get(key)
        if isinstance(val, str) and len(val) > 20:
            candidates[key] = val

    return candidates or None


def _scan_openclaw_credentials() -> Optional[Dict[str, str]]:
    if not OPENCLAW_CRED_DIR.exists():
        return None

    for path in OPENCLAW_CRED_DIR.glob("*.json"):
        # Only scan credential JSON files, skip known unrelated ones
        if path.name.startswith("telegram-"):
            continue
        found = _find_token_in_json(path)
        if found:
            found["source_file"] = str(path)
            return found
    return None


def import_anthropic_key_from_env() -> dict:
    """Import ANTHROPIC_API_KEY from environment into settings.yaml."""
    key = os.getenv("ANTHROPIC_API_KEY") or os.getenv("CLAUDE_API_KEY")
    config = load_config()
    providers = config.setdefault("providers", {})
    anthropic_cfg = providers.setdefault("anthropic", {})

    if key:
        anthropic_cfg["api_key"] = key
        anthropic_cfg["auth_source"] = "env"
        save_config(config)
        return {"imported": True, "source": "env"}

    return {"imported": False, "source": None}


def import_claude_access_token() -> dict:
    """Import Claude access token from OpenClaw credentials into settings.yaml."""
    found = _scan_openclaw_credentials()
    if not found:
        return {"imported": False, "source": None}

    config = load_config()
    providers = config.setdefault("providers", {})
    anthropic_cfg = providers.setdefault("anthropic", {})

    if found.get("access_token"):
        anthropic_cfg["access_token"] = found["access_token"]
        anthropic_cfg["auth_source"] = "openclaw_credentials"
        save_config(config)
        return {"imported": True, "source": "openclaw_credentials"}

    return {"imported": False, "source": None}


def store_access_token_locally(access_token: str) -> dict:
    """Store a Claude access token in Jarvis credentials dir and config."""
    if not access_token or len(access_token) < 20:
        return {"stored": False}

    JARVIS_CRED_DIR.mkdir(parents=True, exist_ok=True)
    cred_path = JARVIS_CRED_DIR / "claude.json"
    try:
        cred_path.write_text(json.dumps({"access_token": access_token}, indent=2))
    except Exception:
        return {"stored": False}

    config = load_config()
    providers = config.setdefault("providers", {})
    anthropic_cfg = providers.setdefault("anthropic", {})
    anthropic_cfg["access_token"] = access_token
    anthropic_cfg["auth_source"] = "jarvis_credentials"
    save_config(config)
    return {"stored": True, "path": str(cred_path)}


def claude_device_login(timeout: int = 300) -> dict:
    """
    Run Claude CLI login. This is best-effort: the CLI manages its own auth store.
    """
    try:
        result = subprocess.run(
            ["claude", "setup-token"],
            capture_output=True,
            text=True,
            timeout=timeout
        )
        output = (result.stdout or "") + (("\n" + result.stderr) if result.stderr else "")
        imported = import_claude_access_token()
        if not imported.get("imported"):
            imported = import_anthropic_key_from_env()
        return {
            "ok": result.returncode == 0,
            "output": output.strip(),
            "import": imported,
        }
    except FileNotFoundError:
        return {"ok": False, "output": "claude CLI not found", "import": {"imported": False}}
    except subprocess.TimeoutExpired:
        return {"ok": False, "output": "claude login timed out", "import": {"imported": False}}
