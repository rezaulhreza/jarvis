"""
Jarvis UI components
"""

# Terminal UI is always available
from .terminal import TerminalUI

# Web UI is optional (requires fastapi)
try:
    from .app import create_app
    __all__ = ["TerminalUI", "create_app"]
except ImportError:
    __all__ = ["TerminalUI"]
