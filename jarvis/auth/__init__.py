"""Auth helpers for external CLIs, providers, and user authentication."""

from .codex import import_codex_auth, codex_device_login, read_codex_auth
from .claude import (
    import_anthropic_key_from_env,
    import_claude_access_token,
    store_access_token_locally,
    claude_device_login,
)

__all__ = [
    "import_codex_auth",
    "codex_device_login",
    "read_codex_auth",
    "import_anthropic_key_from_env",
    "import_claude_access_token",
    "store_access_token_locally",
    "claude_device_login",
]


def init_auth():
    """Initialize auth tables and cleanup expired data. Call on app startup."""
    from .db import init_auth_tables, cleanup_expired
    init_auth_tables()
    cleanup_expired()
