"""Tests for plan mode tool gating and permission checks in Agent."""

import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch

from jarvis.core.agent import Agent
from jarvis.core.permissions import PermissionManager


# ──────────────────────────────────────────────
# Fixtures
# ──────────────────────────────────────────────

@pytest.fixture
def tmp_project(tmp_path):
    """Create a minimal project directory."""
    (tmp_path / "test.py").write_text("print('hello')")
    return tmp_path


@pytest.fixture
def mock_provider():
    provider = MagicMock()
    provider.name = "mock"
    provider.model = "mock-model"
    provider.supports_tools = False
    return provider


@pytest.fixture
def pm(tmp_path):
    return PermissionManager(config_dir=tmp_path / "config")


@pytest.fixture
def agent(mock_provider, tmp_project, pm):
    return Agent(
        provider=mock_provider,
        project_root=tmp_project,
        ui=MagicMock(),
        config={},
        permissions=pm,
    )


# ──────────────────────────────────────────────
# Plan mode tool gating
# ──────────────────────────────────────────────

class TestPlanModeGating:

    BLOCKED_TOOLS = [
        "write_file", "edit_file", "run_command", "apply_patch",
        "git_commit", "git_add", "git_stash", "git_branch",
        "save_memory", "create_pr",
    ]

    ALLOWED_TOOLS = [
        "read_file", "list_files", "search_files", "glob_files", "grep",
        "web_search", "web_fetch", "git_status", "git_diff", "git_log",
        "get_project_structure", "find_definition", "find_references",
        "calculate", "recall_memory", "task_list", "task_get",
    ]

    @pytest.mark.parametrize("tool", BLOCKED_TOOLS)
    def test_plan_mode_blocks_writes(self, agent, tool):
        """Plan mode blocks all write/destructive tools."""
        agent.plan_mode = True
        result = agent._execute_tool(tool, {})
        assert "[Plan mode]" in result
        assert "blocked" in result.lower()

    @pytest.mark.parametrize("tool", ALLOWED_TOOLS)
    def test_plan_mode_allows_reads(self, agent, tool):
        """Plan mode allows all read-only tools to execute."""
        agent.plan_mode = True
        result = agent._execute_tool(tool, {})
        # Should NOT contain the plan mode block message
        assert "[Plan mode]" not in result

    def test_normal_mode_no_blocking(self, agent):
        """Normal mode allows write tools."""
        agent.plan_mode = False
        # write_file needs a valid path, so we just check it doesn't hit plan gate
        result = agent._execute_tool("write_file", {"path": "test.py", "content": "x"})
        assert "[Plan mode]" not in result

    def test_plan_blocked_tools_constant(self, agent):
        """PLAN_BLOCKED_TOOLS set matches expected tools."""
        expected = {
            "write_file", "edit_file", "run_command", "apply_patch",
            "git_commit", "git_add", "git_stash", "git_branch",
            "save_memory", "create_pr", "update_pr",
        }
        assert agent.PLAN_BLOCKED_TOOLS == expected


# ──────────────────────────────────────────────
# Permission gating in agent
# ──────────────────────────────────────────────

class TestAgentPermissionGating:

    def test_safe_tool_no_prompt(self, agent):
        """Safe tools execute without permission prompt."""
        result = agent._execute_tool("git_status", {})
        agent.ui.prompt_tool_permission.assert_not_called()

    def test_denied_tool_returns_message(self, agent):
        """Denied tool returns denial message."""
        agent.permissions.set_always_deny("write_file")
        result = agent._execute_tool("write_file", {"path": "x", "content": "x"})
        assert "denied" in result.lower()

    def test_allowed_override_skips_prompt(self, agent):
        """Always-allow override skips the UI prompt."""
        agent.permissions.set_always_allow("run_command")
        result = agent._execute_tool("run_command", {"command": "echo hi"})
        agent.ui.prompt_tool_permission.assert_not_called()
        assert "denied" not in result.lower()

    def test_no_permissions_object(self, mock_provider, tmp_project):
        """Agent works without a permissions object (backward compat)."""
        agent = Agent(
            provider=mock_provider,
            project_root=tmp_project,
            ui=MagicMock(),
            config={},
            permissions=None,
        )
        # Should not raise
        result = agent._execute_tool("git_status", {})
        assert result  # Some output


# ──────────────────────────────────────────────
# Plan mode + permissions interaction
# ──────────────────────────────────────────────

class TestPlanModePermissionInteraction:

    def test_plan_mode_checked_before_permissions(self, agent):
        """Plan mode blocks before permission check even runs."""
        agent.plan_mode = True
        agent.permissions.set_always_allow("write_file")
        result = agent._execute_tool("write_file", {"path": "x", "content": "x"})
        assert "[Plan mode]" in result
        # Permission prompt should NOT have been called
        agent.ui.prompt_tool_permission.assert_not_called()
