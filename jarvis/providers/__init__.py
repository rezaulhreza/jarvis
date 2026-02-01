"""
LLM Providers - Abstract interface for different AI backends

Supported:
- Ollama (local)
- Anthropic (Claude)
- OpenAI (GPT)
- Google (Gemini)
"""

from .base import BaseProvider, Message
from .ollama import OllamaProvider
from .anthropic import AnthropicProvider
from .openai import OpenAIProvider
from .gemini import GeminiProvider

PROVIDERS = {
    "ollama": OllamaProvider,
    "anthropic": AnthropicProvider,
    "openai": OpenAIProvider,
    "gemini": GeminiProvider,
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
    "AnthropicProvider",
    "OpenAIProvider",
    "GeminiProvider",
    "get_provider",
    "list_providers",
]
