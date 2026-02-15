"""Tests for CLI slash commands — /compact, /usage, /sessions, /resume, /plan, /permissions.

Uses mocking to avoid needing a real LLM provider or Ollama running.
"""

import pytest
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch, PropertyMock
from io import StringIO

from jarvis.core.permissions import PermissionManager


# ──────────────────────────────────────────────
# Helpers & Fixtures
# ──────────────────────────────────────────────

def _make_jarvis(tmp_path):
    """Build a Jarvis instance with mocked provider, avoiding real Ollama."""
    from jarvis.assistant import Jarvis, CONFIG_DIR
    from jarvis.ui.terminal import TerminalUI

    # Mock provider
    provider = MagicMock()
    provider.name = "mock"
    provider.model = "mock-model"
    provider.is_configured.return_value = True
    provider.list_models.return_value = ["mock-model"]
    provider.get_default_model.return_value = "mock-model"
    provider.get_context_length.return_value = 8192

    ui = TerminalUI()
    ui.console = MagicMock()  # Suppress all Rich output
    # Mock UI methods so we can assert on them
    ui.print_info = MagicMock()
    ui.print_warning = MagicMock()
    ui.print_success = MagicMock()
    ui.print_error = MagicMock()
    ui.print_help = MagicMock()
    ui.print_sessions = MagicMock()
    ui.print_permissions = MagicMock()
    ui.prompt_tool_permission = MagicMock(return_value="y")

    # Build Jarvis with patched provider init
    with patch("jarvis.assistant.get_provider", return_value=provider), \
         patch("jarvis.assistant.load_config", return_value={"provider": "mock"}), \
         patch("jarvis.assistant.ProjectContext") as MockPC, \
         patch("jarvis.assistant.get_rag_engine", return_value=None):

        MockPC.return_value.project_root = tmp_path
        MockPC.return_value.project_name = "test-project"
        MockPC.return_value.project_type = "Python"
        MockPC.return_value.git_branch = "main"
        MockPC.return_value.soul = ""
        MockPC.return_value.agents = {}
        MockPC.return_value.assistant_name = None

        jarvis = Jarvis(ui=ui, working_dir=tmp_path)

    return jarvis


@pytest.fixture
def jarvis(tmp_path):
    return _make_jarvis(tmp_path)


# ──────────────────────────────────────────────
# /compact
# ──────────────────────────────────────────────

class TestCompactCommand:

    def test_compact_with_few_messages(self, jarvis):
        """Compact on empty context prints info message."""
        result = jarvis._handle_command("/compact")
        jarvis.ui.print_info.assert_called_once()
        assert "too few" in jarvis.ui.print_info.call_args[0][0].lower()

    def test_compact_with_messages(self, jarvis):
        """Compact with messages prints before/after stats."""
        # Add enough messages to trigger compaction
        for i in range(10):
            jarvis.context.add_message("user", f"message {i} " * 50)
            jarvis.context.add_message("assistant", f"response {i} " * 50)

        result = jarvis._handle_command("/compact")
        jarvis.ui.print_success.assert_called_once()
        msg = jarvis.ui.print_success.call_args[0][0]
        assert "Compacted" in msg
        assert "→" in msg  # Shows before → after


# ──────────────────────────────────────────────
# /usage
# ──────────────────────────────────────────────

class TestUsageCommand:

    def test_usage_initial_zeros(self, jarvis):
        """Usage starts at zero tokens."""
        result = jarvis._handle_command("/usage")
        console_calls = [str(c) for c in jarvis.ui.console.print.call_args_list]
        text = " ".join(console_calls)
        assert "Session Token Usage" in text

    def test_usage_tracks_input(self, jarvis):
        """Session tokens track input."""
        jarvis.session_tokens['input'] = 1500
        jarvis.session_tokens['output'] = 3000
        result = jarvis._handle_command("/usage")
        console_calls = [str(c) for c in jarvis.ui.console.print.call_args_list]
        text = " ".join(console_calls)
        assert "1,500" in text
        assert "3,000" in text
        assert "4,500" in text  # total


# ──────────────────────────────────────────────
# /sessions and /resume
# ──────────────────────────────────────────────

class TestSessionCommands:

    def test_sessions_shows_table(self, jarvis):
        """Sessions command calls print_sessions."""
        jarvis.context.create_chat("Test Session 1")
        jarvis.context.create_chat("Test Session 2")
        result = jarvis._handle_command("/sessions")
        jarvis.ui.print_sessions.assert_called_once()
        chats = jarvis.ui.print_sessions.call_args[0][0]
        assert len(chats) >= 2

    def test_resume_no_args(self, jarvis):
        """Resume without args shows warning."""
        result = jarvis._handle_command("/resume")
        jarvis.ui.print_warning.assert_called()
        assert "Usage" in jarvis.ui.print_warning.call_args[0][0]

    def test_resume_valid_index(self, jarvis):
        """Resume with valid index switches chat."""
        chat_id = jarvis.context.create_chat("Resumable Session")
        jarvis.context.add_message("user", "hello from old session")
        jarvis.context.create_chat("Current")  # Switch away

        result = jarvis._handle_command("/resume 1")
        jarvis.ui.print_success.assert_called()
        assert "Resumed" in jarvis.ui.print_success.call_args[0][0]

    def test_resume_invalid_index(self, jarvis):
        """Resume with out-of-range index shows warning."""
        result = jarvis._handle_command("/resume 999")
        jarvis.ui.print_warning.assert_called()

    def test_resume_non_numeric(self, jarvis):
        """Resume with non-numeric arg shows warning."""
        result = jarvis._handle_command("/resume abc")
        jarvis.ui.print_warning.assert_called()


# ──────────────────────────────────────────────
# /plan
# ──────────────────────────────────────────────

class TestPlanModeCommand:

    def test_plan_toggle_on(self, jarvis):
        """Plan command toggles plan mode on."""
        assert jarvis.plan_mode is False
        jarvis._handle_command("/plan")
        assert jarvis.plan_mode is True
        jarvis.ui.print_info.assert_called()
        assert "PLAN MODE" in jarvis.ui.print_info.call_args[0][0]

    def test_plan_toggle_off(self, jarvis):
        """Plan command toggles plan mode off."""
        jarvis.plan_mode = True
        jarvis._handle_command("/plan")
        assert jarvis.plan_mode is False

    def test_plan_mode_syncs_to_agent(self, jarvis):
        """Plan mode state is synced to agent before runs."""
        jarvis.plan_mode = True
        # Mock agent.run to just return
        jarvis.agent.run = MagicMock(return_value="plan output")
        jarvis.agent.last_streamed = False
        jarvis.context.add_message("user", "test")
        jarvis._run_agent("make a plan")
        assert jarvis.agent.plan_mode is True


# ──────────────────────────────────────────────
# /permissions
# ──────────────────────────────────────────────

class TestPermissionsCommand:

    def test_permissions_no_args_shows_table(self, jarvis):
        """Permissions with no args calls print_permissions."""
        jarvis._handle_command("/permissions")
        jarvis.ui.print_permissions.assert_called_once_with(jarvis.permissions)

    def test_permissions_allow(self, jarvis):
        """Permissions allow <tool> sets override."""
        jarvis._handle_command("/permissions allow run_command")
        assert jarvis.permissions.get_setting("run_command") == "allow"
        jarvis.ui.print_success.assert_called()

    def test_permissions_deny(self, jarvis):
        """Permissions deny <tool> sets override."""
        jarvis._handle_command("/permissions deny write_file")
        assert jarvis.permissions.get_setting("write_file") == "deny"

    def test_permissions_reset(self, jarvis):
        """Permissions reset clears all overrides."""
        jarvis.permissions.set_always_allow("run_command")
        jarvis._handle_command("/permissions reset")
        assert jarvis.permissions.get_setting("run_command") == "default"


# ──────────────────────────────────────────────
# Existing commands still work
# ──────────────────────────────────────────────

class TestExistingCommands:

    def test_help(self, jarvis):
        """Help command works and includes new commands."""
        jarvis._handle_command("/help")
        # print_help was called
        jarvis.ui.print_help.assert_called_once()

    def test_context(self, jarvis):
        """Context command works."""
        jarvis._handle_command("/context")
        console_calls = [str(c) for c in jarvis.ui.console.print.call_args_list]
        text = " ".join(console_calls)
        assert "Context Usage" in text

    def test_clear(self, jarvis):
        """Clear command works."""
        jarvis.context.add_message("user", "hello")
        jarvis._handle_command("/clear")
        assert len(jarvis.context.get_messages()) == 0

    def test_unknown_command(self, jarvis):
        """Unknown command shows warning."""
        jarvis._handle_command("/foobar")
        jarvis.ui.print_warning.assert_called()
