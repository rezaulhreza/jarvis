"""Tests for create_pr tool function and its registration."""

import pytest
from unittest.mock import patch, MagicMock
from pathlib import Path

from jarvis.core.tools import (
    create_pr, ALL_TOOLS, TOOL_REGISTRY,
    _TOOL_FUNC_TO_NAME, _CATEGORY_GROUPS,
)


# ──────────────────────────────────────────────
# Registration
# ──────────────────────────────────────────────

class TestCreatePRRegistration:

    def test_in_all_tools(self):
        """create_pr is in ALL_TOOLS list."""
        assert create_pr in ALL_TOOLS

    def test_in_tool_registry(self):
        """create_pr is in TOOL_REGISTRY."""
        assert "create_pr" in TOOL_REGISTRY
        meta = TOOL_REGISTRY["create_pr"]
        assert meta["category"] == "git"
        assert "git" in meta["intents"]
        assert "pr" in meta["keywords"] or "pull request" in meta["keywords"]

    def test_in_func_to_name(self):
        """create_pr is in _TOOL_FUNC_TO_NAME."""
        assert create_pr in _TOOL_FUNC_TO_NAME
        assert _TOOL_FUNC_TO_NAME[create_pr] == "create_pr"

    def test_in_git_category_group(self):
        """create_pr is in the git category group."""
        assert "create_pr" in _CATEGORY_GROUPS["git"]


# ──────────────────────────────────────────────
# Agent integration
# ──────────────────────────────────────────────

class TestCreatePRInAgent:

    def test_in_known_tool_names(self):
        from jarvis.core.agent import Agent
        assert "create_pr" in Agent._KNOWN_TOOL_NAMES

    def test_format_tool_display(self):
        from jarvis.core.agent import Agent
        agent = MagicMock(spec=Agent)
        display = Agent._format_tool_display(agent, "create_pr", {"title": "Fix bug", "draft": False})
        assert "Create PR" in display
        assert "Fix bug" in display

    def test_format_tool_display_draft(self):
        from jarvis.core.agent import Agent
        agent = MagicMock(spec=Agent)
        display = Agent._format_tool_display(agent, "create_pr", {"title": "WIP", "draft": True})
        assert "(draft)" in display

    def test_validate_tool_call_requires_title(self):
        from jarvis.core.agent import Agent
        agent = MagicMock(spec=Agent)
        # Missing title
        valid, error = Agent._validate_tool_call(agent, "create_pr", {})
        assert valid is False
        assert "title" in error.lower()

    def test_validate_tool_call_valid(self):
        from jarvis.core.agent import Agent
        agent = MagicMock(spec=Agent)
        valid, error = Agent._validate_tool_call(agent, "create_pr", {"title": "Fix stuff"})
        assert valid is True


# ──────────────────────────────────────────────
# Function execution
# ──────────────────────────────────────────────

class TestCreatePRExecution:

    @patch("jarvis.core.tools.subprocess.run")
    def test_gh_not_installed(self, mock_run):
        mock_run.side_effect = FileNotFoundError("gh not found")
        result = create_pr("Test PR")
        assert "not installed" in result.lower()

    @patch("jarvis.core.tools.subprocess.run")
    def test_not_authenticated(self, mock_run):
        def side_effect(cmd, **kwargs):
            r = MagicMock()
            if "auth" in cmd:
                r.returncode = 1
                r.stderr = "not logged in"
            else:
                r.returncode = 0
                r.stdout = "gh version 2.0"
            return r

        mock_run.side_effect = side_effect
        result = create_pr("Test PR")
        assert "authenticated" in result.lower() or "auth" in result.lower()

    @patch("jarvis.core.tools.subprocess.run")
    def test_on_main_branch(self, mock_run):
        def side_effect(cmd, **kwargs):
            r = MagicMock()
            r.returncode = 0
            if "branch" in cmd and "--show-current" in cmd:
                r.stdout = "main"
            elif "auth" in cmd:
                r.returncode = 0
            elif "--version" in cmd:
                r.stdout = "gh version 2.0"
            return r

        mock_run.side_effect = side_effect
        result = create_pr("Test PR")
        assert "cannot create pr" in result.lower() or "feature branch" in result.lower()

    @patch("jarvis.core.tools.subprocess.run")
    def test_successful_pr_creation(self, mock_run):
        call_count = [0]

        def side_effect(cmd, **kwargs):
            call_count[0] += 1
            r = MagicMock()
            r.returncode = 0
            if "--version" in cmd:
                r.stdout = "gh version 2.0"
            elif "auth" in cmd:
                pass  # returncode 0
            elif "branch" in cmd and "--show-current" in cmd:
                r.stdout = "feature/my-branch"
            elif "pr" in cmd and "create" in cmd:
                r.stdout = "https://github.com/user/repo/pull/42"
            return r

        mock_run.side_effect = side_effect
        result = create_pr("Fix the thing", body="Description", base="main")
        assert "https://github.com" in result
        assert "42" in result

    @patch("jarvis.core.tools.subprocess.run")
    def test_draft_flag_passed(self, mock_run):
        captured_cmds = []

        def side_effect(cmd, **kwargs):
            captured_cmds.append(cmd)
            r = MagicMock()
            r.returncode = 0
            if "--version" in cmd:
                r.stdout = "gh version 2.0"
            elif "branch" in cmd:
                r.stdout = "feature/x"
            elif "pr" in cmd:
                r.stdout = "https://github.com/user/repo/pull/1"
            return r

        mock_run.side_effect = side_effect
        create_pr("WIP", draft=True)

        # Find the pr create command and check for --draft
        pr_cmd = [c for c in captured_cmds if "pr" in c and "create" in c]
        assert pr_cmd, "gh pr create was not called"
        assert "--draft" in pr_cmd[0]

    @patch("jarvis.core.tools.subprocess.run")
    def test_pr_already_exists(self, mock_run):
        def side_effect(cmd, **kwargs):
            r = MagicMock()
            r.returncode = 0
            if "--version" in cmd:
                r.stdout = "gh version 2.0"
            elif "branch" in cmd:
                r.stdout = "feature/x"
            elif "pr" in cmd and "create" in cmd:
                r.returncode = 1
                r.stderr = "a pull request for branch 'feature/x' already exists"
            return r

        mock_run.side_effect = side_effect
        result = create_pr("Dupe")
        assert "already exists" in result.lower()
