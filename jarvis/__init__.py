"""
Jarvis - Local-first Personal AI Assistant

A multi-model AI assistant powered by Ollama with:
- Multiple personas
- Tool/skill execution
- Persistent memory
- Voice input support
- Web UI option
"""

__version__ = "0.1.0"
__author__ = "Boss"

from pathlib import Path

# Package paths
PACKAGE_DIR = Path(__file__).parent
PROJECT_ROOT = PACKAGE_DIR.parent

# User data directory (for config, memory, etc.)
def get_data_dir() -> Path:
    """Get the user data directory for Jarvis."""
    import os

    # Check for custom data dir
    custom_dir = os.environ.get("JARVIS_DATA_DIR")
    if custom_dir:
        return Path(custom_dir)

    # Default to ~/.jarvis
    return Path.home() / ".jarvis"


def ensure_data_dir() -> Path:
    """Ensure the data directory exists with required structure."""
    data_dir = get_data_dir()

    # Create subdirectories
    (data_dir / "config" / "personas").mkdir(parents=True, exist_ok=True)
    (data_dir / "memory").mkdir(parents=True, exist_ok=True)
    (data_dir / "knowledge" / "documents").mkdir(parents=True, exist_ok=True)
    (data_dir / "knowledge" / "notes").mkdir(parents=True, exist_ok=True)
    (data_dir / "logs").mkdir(parents=True, exist_ok=True)

    return data_dir
