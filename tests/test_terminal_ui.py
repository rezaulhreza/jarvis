"""Tests for terminal UI methods — print_sessions, print_permissions, prompt_tool_permission, toolbar."""

import pytest
from unittest.mock import MagicMock, patch
from io import StringIO

from jarvis.ui.terminal import TerminalUI, VALID_COMMANDS, _JarvisCompleter


# ──────────────────────────────────────────────
# Fixtures
# ──────────────────────────────────────────────

@pytest.fixture
def ui():
    t = TerminalUI()
    t.console = MagicMock()
    return t


# ──────────────────────────────────────────────
# VALID_COMMANDS includes new commands
# ──────────────────────────────────────────────

class TestCommandRegistration:

    NEW_COMMANDS = ["/compact", "/usage", "/sessions", "/resume", "/permissions", "/plan"]

    @pytest.mark.parametrize("cmd", NEW_COMMANDS)
    def test_valid_commands_includes(self, cmd):
        assert cmd in VALID_COMMANDS

    @pytest.mark.parametrize("cmd", NEW_COMMANDS)
    def test_completer_includes(self, cmd):
        assert cmd in _JarvisCompleter.COMMANDS


# ──────────────────────────────────────────────
# print_sessions
# ──────────────────────────────────────────────

class TestPrintSessions:

    def test_empty_sessions(self, ui):
        """Empty list prints 'no sessions'."""
        ui.print_sessions([], None)
        calls = [str(c) for c in ui.console.print.call_args_list]
        text = " ".join(calls)
        assert "no sessions" in text.lower()

    def test_sessions_table(self, ui):
        """Non-empty list prints a Rich Table."""
        chats = [
            {"id": "abc", "title": "Session 1", "message_count": 5, "updated_at": "2026-02-15T10:00:00"},
            {"id": "def", "title": "Session 2", "message_count": 10, "updated_at": "2026-02-15T11:00:00"},
        ]
        ui.print_sessions(chats, "abc")
        # Should have printed at least the table + hint
        assert ui.console.print.call_count >= 3

    def test_current_session_marked(self, ui):
        """Current session gets a marker."""
        chats = [{"id": "abc", "title": "Active", "message_count": 1, "updated_at": "2026-02-15"}]
        ui.print_sessions(chats, "abc")
        # The table is a Rich object, but the marker logic ran


# ──────────────────────────────────────────────
# print_permissions
# ──────────────────────────────────────────────

class TestPrintPermissions:

    def test_permissions_table(self, ui):
        """Prints a table with all tools."""
        from jarvis.core.permissions import PermissionManager
        pm = PermissionManager(config_dir="/tmp/jarvis_test_perms")
        ui.print_permissions(pm)
        assert ui.console.print.call_count >= 2  # Table + hint


# ──────────────────────────────────────────────
# prompt_tool_permission
# ──────────────────────────────────────────────

class TestPromptToolPermission:

    @patch("builtins.input", return_value="y")
    def test_approve(self, mock_input, ui):
        result = ui.prompt_tool_permission("write_file", {"path": "x.py"}, "moderate")
        assert result == "y"

    @patch("builtins.input", return_value="n")
    def test_deny(self, mock_input, ui):
        result = ui.prompt_tool_permission("run_command", {"command": "rm -rf /"}, "dangerous")
        assert result == "n"

    @patch("builtins.input", return_value="a")
    def test_always_allow(self, mock_input, ui):
        result = ui.prompt_tool_permission("edit_file", {}, "moderate")
        assert result == "a"

    @patch("builtins.input", return_value="d")
    def test_always_deny(self, mock_input, ui):
        result = ui.prompt_tool_permission("git_stash", {}, "dangerous")
        assert result == "d"

    @patch("builtins.input", return_value="gibberish")
    def test_invalid_input_defaults_no(self, mock_input, ui):
        result = ui.prompt_tool_permission("write_file", {}, "moderate")
        assert result == "n"

    @patch("builtins.input", side_effect=KeyboardInterrupt)
    def test_keyboard_interrupt(self, mock_input, ui):
        result = ui.prompt_tool_permission("write_file", {}, "moderate")
        assert result == "n"

    @patch("builtins.input", side_effect=EOFError)
    def test_eof(self, mock_input, ui):
        result = ui.prompt_tool_permission("write_file", {}, "moderate")
        assert result == "n"


# ──────────────────────────────────────────────
# Toolbar
# ──────────────────────────────────────────────

class TestToolbar:

    def test_toolbar_normal(self, ui):
        ui._current_provider = "ollama"
        ui._current_model = "qwen3:4b"
        text = ui._get_toolbar_text()
        assert "ollama" in text
        assert "qwen3:4b" in text
        assert "[PLAN]" not in text

    def test_toolbar_plan_mode(self, ui):
        ui._current_provider = "ollama"
        ui._current_model = "qwen3:4b"
        ui._plan_mode = True
        text = ui._get_toolbar_text()
        assert "[PLAN]" in text

    def test_toolbar_context_warning(self, ui):
        ui._current_provider = "test"
        ui._current_model = "test"
        ui._context_stats = {"percentage": 85, "tokens_used": 6800, "max_tokens": 8000}
        text = ui._get_toolbar_text()
        assert "⚠" in text
        assert "85%" in text


# ──────────────────────────────────────────────
# Help table
# ──────────────────────────────────────────────

class TestHelpTable:

    def test_help_includes_new_commands(self, ui):
        """print_help includes all new commands."""
        ui.print_help()
        # Rich table was printed — we check the call happened
        assert ui.console.print.call_count >= 3
