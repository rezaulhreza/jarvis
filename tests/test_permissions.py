"""Tests for jarvis/core/permissions.py — PermissionManager."""

import pytest
import tempfile
import yaml
from pathlib import Path
from unittest.mock import MagicMock

from jarvis.core.permissions import PermissionManager, TOOL_RISK_LEVELS


# ──────────────────────────────────────────────
# Fixtures
# ──────────────────────────────────────────────

@pytest.fixture
def tmp_config(tmp_path):
    """Return a temp config directory for PermissionManager."""
    return tmp_path / "config"


@pytest.fixture
def pm(tmp_config):
    """Fresh PermissionManager with temp config."""
    return PermissionManager(config_dir=tmp_config)


@pytest.fixture
def mock_ui():
    """Mock terminal UI with prompt_tool_permission."""
    ui = MagicMock()
    return ui


# ──────────────────────────────────────────────
# Risk level classification
# ──────────────────────────────────────────────

class TestRiskLevels:
    """Verify every tool has the right risk classification."""

    SAFE_TOOLS = [
        "read_file", "list_files", "search_files", "glob_files", "grep",
        "get_project_structure", "get_project_overview", "web_search",
        "web_fetch", "get_weather", "get_current_time", "get_current_news",
        "get_gold_price", "calculate", "recall_memory", "git_status",
        "git_diff", "git_log", "task_list", "task_get", "github_search",
        "find_definition", "find_references",
    ]

    MODERATE_TOOLS = [
        "write_file", "edit_file", "git_add", "git_commit", "git_branch",
        "save_memory", "create_pr", "task_create", "task_update",
    ]

    DANGEROUS_TOOLS = [
        "run_command", "apply_patch", "git_stash",
    ]

    @pytest.mark.parametrize("tool", SAFE_TOOLS)
    def test_safe_tools(self, pm, tool):
        assert pm.get_risk_level(tool) == "safe"

    @pytest.mark.parametrize("tool", MODERATE_TOOLS)
    def test_moderate_tools(self, pm, tool):
        assert pm.get_risk_level(tool) == "moderate"

    @pytest.mark.parametrize("tool", DANGEROUS_TOOLS)
    def test_dangerous_tools(self, pm, tool):
        assert pm.get_risk_level(tool) == "dangerous"

    def test_unknown_tool_defaults_moderate(self, pm):
        assert pm.get_risk_level("nonexistent_tool") == "moderate"

    def test_all_tools_classified(self):
        """Every tool in TOOL_RISK_LEVELS is safe, moderate, or dangerous."""
        for tool, level in TOOL_RISK_LEVELS.items():
            assert level in ("safe", "moderate", "dangerous"), f"{tool} has invalid level: {level}"


# ──────────────────────────────────────────────
# Auto-approve for safe tools
# ──────────────────────────────────────────────

class TestSafeAutoApprove:

    @pytest.mark.parametrize("tool", TestRiskLevels.SAFE_TOOLS)
    def test_safe_tools_auto_approve(self, pm, tool):
        """Safe tools pass without any UI prompt."""
        assert pm.check(tool, {}) is True

    def test_safe_tool_no_ui_needed(self, pm):
        """Safe tools pass even without a UI reference."""
        assert pm.check("read_file", {"path": "foo.py"}, ui=None) is True


# ──────────────────────────────────────────────
# Interactive prompts (moderate / dangerous)
# ──────────────────────────────────────────────

class TestInteractivePrompts:

    def test_moderate_prompts_user_approve(self, pm, mock_ui):
        """Moderate tool approved on 'y' → returns True."""
        mock_ui.prompt_tool_permission.return_value = "y"
        assert pm.check("write_file", {"path": "x"}, ui=mock_ui) is True
        mock_ui.prompt_tool_permission.assert_called_once()

    def test_moderate_prompts_user_deny(self, pm, mock_ui):
        """Moderate tool denied on 'n' → returns False."""
        mock_ui.prompt_tool_permission.return_value = "n"
        assert pm.check("write_file", {"path": "x"}, ui=mock_ui) is False

    def test_dangerous_always_prompts(self, pm, mock_ui):
        """Dangerous tool always prompts, even after previous approval."""
        mock_ui.prompt_tool_permission.return_value = "y"
        assert pm.check("run_command", {"command": "ls"}, ui=mock_ui) is True
        # Second call should still prompt (dangerous = no session cache)
        assert pm.check("run_command", {"command": "ls"}, ui=mock_ui) is True
        assert mock_ui.prompt_tool_permission.call_count == 2

    def test_moderate_cached_after_first_approve(self, pm, mock_ui):
        """Moderate tool caches approval for session after first 'y'."""
        mock_ui.prompt_tool_permission.return_value = "y"
        assert pm.check("edit_file", {"path": "x"}, ui=mock_ui) is True
        # Second call should NOT prompt again
        assert pm.check("edit_file", {"path": "x"}, ui=mock_ui) is True
        assert mock_ui.prompt_tool_permission.call_count == 1

    def test_no_ui_moderate_allows(self, pm):
        """Moderate tools pass when no UI is available (headless/web)."""
        assert pm.check("write_file", {"path": "x"}, ui=None) is True

    def test_no_ui_dangerous_denies(self, pm):
        """Dangerous tools are denied when no UI is available."""
        assert pm.check("run_command", {"command": "rm -rf /"}, ui=None) is False


# ──────────────────────────────────────────────
# Always-allow / always-deny persistent overrides
# ──────────────────────────────────────────────

class TestPersistentOverrides:

    def test_always_allow(self, pm, mock_ui):
        """User picks 'a' (always allow) → persists, no future prompts."""
        mock_ui.prompt_tool_permission.return_value = "a"
        assert pm.check("write_file", {"path": "x"}, ui=mock_ui) is True
        # Second call should NOT prompt
        assert pm.check("write_file", {"path": "x"}, ui=mock_ui) is True
        assert mock_ui.prompt_tool_permission.call_count == 1

    def test_always_deny(self, pm, mock_ui):
        """User picks 'd' (always deny) → persists, auto-denied."""
        mock_ui.prompt_tool_permission.return_value = "d"
        assert pm.check("write_file", {"path": "x"}, ui=mock_ui) is False
        # Second call should NOT prompt — auto-denied
        assert pm.check("write_file", {"path": "x"}, ui=mock_ui) is False
        assert mock_ui.prompt_tool_permission.call_count == 1

    def test_set_always_allow_api(self, pm):
        """set_always_allow() bypasses prompts."""
        pm.set_always_allow("run_command")
        assert pm.check("run_command", {"command": "ls"}) is True

    def test_set_always_deny_api(self, pm, mock_ui):
        """set_always_deny() blocks even with UI."""
        pm.set_always_deny("write_file")
        assert pm.check("write_file", {"path": "x"}, ui=mock_ui) is False
        mock_ui.prompt_tool_permission.assert_not_called()

    def test_get_setting_default(self, pm):
        assert pm.get_setting("read_file") == "default"

    def test_get_setting_after_override(self, pm):
        pm.set_always_allow("write_file")
        assert pm.get_setting("write_file") == "allow"

    def test_get_all_levels(self, pm):
        levels = pm.get_all_levels()
        assert len(levels) > 30
        assert levels["read_file"] == "safe"
        assert levels["run_command"] == "dangerous"


# ──────────────────────────────────────────────
# YAML persistence
# ──────────────────────────────────────────────

class TestYAMLPersistence:

    def test_save_and_reload(self, tmp_config):
        """Overrides survive PermissionManager re-creation."""
        pm1 = PermissionManager(config_dir=tmp_config)
        pm1.set_always_allow("write_file")
        pm1.set_always_deny("run_command")

        # Create a new instance pointing at same dir
        pm2 = PermissionManager(config_dir=tmp_config)
        assert pm2.get_setting("write_file") == "allow"
        assert pm2.get_setting("run_command") == "deny"

    def test_yaml_file_created(self, tmp_config):
        pm = PermissionManager(config_dir=tmp_config)
        pm.set_always_allow("edit_file")
        assert (tmp_config / "permissions.yaml").exists()

        content = yaml.safe_load((tmp_config / "permissions.yaml").read_text())
        assert content["overrides"]["edit_file"] == "allow"

    def test_reset_clears_overrides(self, tmp_config):
        pm = PermissionManager(config_dir=tmp_config)
        pm.set_always_allow("write_file")
        pm.reset()
        assert pm.get_setting("write_file") == "default"
        assert not (tmp_config / "permissions.yaml").exists()

    def test_reset_clears_session_cache(self, tmp_config):
        pm = PermissionManager(config_dir=tmp_config)
        pm._session_cache["edit_file"] = "allow"
        pm.reset()
        assert "edit_file" not in pm._session_cache
