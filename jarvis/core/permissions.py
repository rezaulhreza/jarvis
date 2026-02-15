"""
Permission system for tool execution.

Risk-based tool gating with interactive prompts and YAML persistence.
"""

import yaml
from pathlib import Path
from typing import Optional


# Risk categories for all tools
TOOL_RISK_LEVELS = {
    # SAFE - auto-approve (read-only, informational)
    "read_file": "safe",
    "list_files": "safe",
    "search_files": "safe",
    "glob_files": "safe",
    "grep": "safe",
    "get_project_structure": "safe",
    "get_project_overview": "safe",
    "web_search": "safe",
    "web_fetch": "safe",
    "get_weather": "safe",
    "get_current_time": "safe",
    "get_current_news": "safe",
    "get_gold_price": "safe",
    "calculate": "safe",
    "recall_memory": "safe",
    "git_status": "safe",
    "git_diff": "safe",
    "git_log": "safe",
    "task_list": "safe",
    "task_get": "safe",
    "github_search": "safe",
    "find_definition": "safe",
    "find_references": "safe",
    # MODERATE - ask once, can set always-allow
    "write_file": "moderate",
    "edit_file": "moderate",
    "git_add": "moderate",
    "git_commit": "moderate",
    "git_branch": "moderate",
    "save_memory": "moderate",
    "create_pr": "moderate",
    "task_create": "moderate",
    "task_update": "moderate",
    # DANGEROUS - always ask
    "run_command": "dangerous",
    "apply_patch": "dangerous",
    "git_stash": "dangerous",
}


class PermissionManager:
    """Manages tool execution permissions with risk-based gating."""

    def __init__(self, config_dir: Path = None):
        self._config_dir = Path(config_dir) if config_dir else Path.home() / ".jarvis" / "config"
        self._config_dir.mkdir(parents=True, exist_ok=True)
        self._config_path = self._config_dir / "permissions.yaml"
        self._overrides: dict = {}  # tool_name -> "allow" | "deny"
        self._session_cache: dict = {}  # tool_name -> "allow" (cleared on restart)
        self._load()

    def _load(self):
        """Load saved permission overrides from YAML."""
        if self._config_path.exists():
            try:
                with open(self._config_path) as f:
                    data = yaml.safe_load(f) or {}
                self._overrides = data.get("overrides", {})
            except Exception:
                self._overrides = {}

    def _save(self):
        """Save permission overrides to YAML."""
        try:
            data = {"overrides": self._overrides}
            with open(self._config_path, 'w') as f:
                yaml.dump(data, f, default_flow_style=False)
        except Exception:
            pass

    def get_risk_level(self, tool_name: str) -> str:
        """Get the risk level for a tool."""
        return TOOL_RISK_LEVELS.get(tool_name, "moderate")

    def check(self, tool_name: str, args: dict, ui=None) -> bool:
        """Check if a tool is allowed to execute.

        Returns True if allowed, False if denied.
        """
        level = self.get_risk_level(tool_name)

        # Safe tools always pass
        if level == "safe":
            return True

        # Check persistent overrides
        override = self._overrides.get(tool_name)
        if override == "allow":
            return True
        if override == "deny":
            return False

        # Check session cache (for moderate tools after first approval)
        if tool_name in self._session_cache:
            return self._session_cache[tool_name] == "allow"

        # Dangerous tools always prompt (no caching)
        # Moderate tools prompt once then cache for session
        if ui and hasattr(ui, 'prompt_tool_permission'):
            response = ui.prompt_tool_permission(tool_name, args, level)
            if response in ('y', 'yes'):
                if level == "moderate":
                    self._session_cache[tool_name] = "allow"
                return True
            elif response == 'a':
                self._overrides[tool_name] = "allow"
                self._save()
                return True
            elif response == 'd':
                self._overrides[tool_name] = "deny"
                self._save()
                return False
            else:
                return False

        # No UI available - allow safe/moderate, deny dangerous
        return level != "dangerous"

    def set_always_allow(self, tool_name: str):
        """Set a tool to always be allowed."""
        self._overrides[tool_name] = "allow"
        self._save()

    def set_always_deny(self, tool_name: str):
        """Set a tool to always be denied."""
        self._overrides[tool_name] = "deny"
        self._save()

    def reset(self):
        """Reset all permission overrides."""
        self._overrides = {}
        self._session_cache = {}
        if self._config_path.exists():
            self._config_path.unlink()

    def get_all_levels(self) -> dict:
        """Get risk levels for all known tools."""
        return dict(TOOL_RISK_LEVELS)

    def get_setting(self, tool_name: str) -> str:
        """Get current setting for a tool: 'default', 'allow', or 'deny'."""
        return self._overrides.get(tool_name, "default")
