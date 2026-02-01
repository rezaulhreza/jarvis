"""Base provider interface for LLM backends."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Generator, Optional, List


@dataclass
class Message:
    """A chat message."""
    role: str  # "user", "assistant", "system"
    content: str


class BaseProvider(ABC):
    """Abstract base class for LLM providers."""

    name: str = "base"
    supports_streaming: bool = True
    supports_vision: bool = False
    supports_tools: bool = False

    def __init__(self, model: str = None, api_key: str = None, **kwargs):
        self.model = model
        self.api_key = api_key
        self.kwargs = kwargs
        self._stop_flag = False

    def stop(self):
        """Signal to stop current generation."""
        self._stop_flag = True

    def reset_stop(self):
        """Reset the stop flag."""
        self._stop_flag = False

    def chat_with_tools(self, messages: List, system: str = None, tools: List = None):
        """Non-streaming chat with tool calling support.

        Override in subclasses to implement provider-specific tool calling.
        Default implementation raises NotImplementedError.
        """
        raise NotImplementedError(f"{self.name} does not support tool calling")

    @abstractmethod
    def chat(
        self,
        messages: List[Message],
        system: str = None,
        stream: bool = True
    ) -> Generator[str, None, None] | str:
        """
        Send a chat request.

        Args:
            messages: List of Message objects
            system: System prompt
            stream: Whether to stream response

        Yields/Returns:
            Response text (streamed or complete)
        """
        pass

    @abstractmethod
    def list_models(self) -> List[str]:
        """List available models for this provider."""
        pass

    def vision(self, image_path: str, prompt: str) -> str:
        """Analyze an image (if supported)."""
        raise NotImplementedError(f"{self.name} does not support vision")

    def is_configured(self) -> bool:
        """Check if provider is properly configured."""
        return True

    def get_config_help(self) -> str:
        """Get help text for configuring this provider."""
        return f"{self.name} provider"
