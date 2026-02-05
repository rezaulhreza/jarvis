"""
LLM Providers - Abstract interface for different AI backends

Supported:
- Ollama (local)
- Ollama Cloud (remote Ollama)
- Chutes (recommended - comprehensive AI platform)
"""

from .base import BaseProvider, Message
from .ollama import OllamaProvider
from .ollama_cloud import OllamaCloudProvider
from .chutes import ChutesProvider

PROVIDERS = {
    "ollama": OllamaProvider,
    "ollama_cloud": OllamaCloudProvider,
    "chutes": ChutesProvider,
}


def get_provider(name: str, **kwargs) -> BaseProvider:
    """Get a provider instance by name."""
    if name not in PROVIDERS:
        raise ValueError(f"Unknown provider: {name}. Available: {list(PROVIDERS.keys())}")
    return PROVIDERS[name](**kwargs)


def list_providers() -> list[str]:
    """List available provider names."""
    return list(PROVIDERS.keys())


__all__ = [
    "BaseProvider",
    "Message",
    "OllamaProvider",
    "OllamaCloudProvider",
    "ChutesProvider",
    "get_provider",
    "list_providers",
]
