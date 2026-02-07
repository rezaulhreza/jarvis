"""Base provider interface for LLM backends."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Generator, Optional, List, Dict, Any


@dataclass
class Message:
    """A chat message."""
    role: str  # "user", "assistant", "system"
    content: str


@dataclass
class ModelInfo:
    """Information about an available model."""
    id: str
    name: Optional[str] = None
    capabilities: List[str] = field(default_factory=list)
    context_length: Optional[int] = None
    description: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def supports(self, capability: str) -> bool:
        """Check if model supports a capability."""
        return capability.lower() in [c.lower() for c in self.capabilities]


class BaseProvider(ABC):
    """Abstract base class for LLM providers."""

    name: str = "base"
    supports_streaming: bool = True
    supports_vision: bool = False
    supports_tools: bool = False

    # Task to model mapping - override in subclasses
    TASK_MODELS: Dict[str, str] = {}

    def __init__(self, model: str = None, api_key: str = None, **kwargs):
        self.model = model
        self.api_key = api_key
        self.kwargs = kwargs
        self._stop_flag = False
        self._discovered_models: Optional[List[ModelInfo]] = None
        self._config = kwargs.get("config", {})

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
        stream: bool = True,
        **kwargs
    ) -> Generator[str, None, None] | str:
        """
        Send a chat request.

        Args:
            messages: List of Message objects
            system: System prompt
            stream: Whether to stream response
            **kwargs: Provider-specific options (ignored by default)

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

    def get_context_length(self, model: str = None) -> int:
        """Get the context window size for the current model in tokens.

        Subclasses should override this to query the actual model.
        Returns a safe default of 8192 if unknown.
        """
        return 8192

    # === Dynamic Model Discovery ===

    async def discover_models(self) -> List[ModelInfo]:
        """
        Discover available models from the provider API.

        Override in subclasses to implement provider-specific discovery.
        Returns cached results if already discovered.
        """
        if self._discovered_models is not None:
            return self._discovered_models

        # Default: convert list_models to ModelInfo
        try:
            models = self.list_models()
            self._discovered_models = [
                ModelInfo(id=m, name=m)
                for m in models
            ]
        except Exception:
            self._discovered_models = []

        return self._discovered_models

    def discover_models_sync(self) -> List[ModelInfo]:
        """Synchronous version of discover_models."""
        import asyncio
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                # Can't run async in existing loop, return cached or empty
                return self._discovered_models or []
            return loop.run_until_complete(self.discover_models())
        except RuntimeError:
            return asyncio.run(self.discover_models())

    def get_model_for_task(self, task: str) -> str:
        """
        Get the best model for a specific task type.

        Checks:
        1. Provider-specific config in settings
        2. Provider's TASK_MODELS mapping
        3. Falls back to current model

        Args:
            task: One of "default", "fast", "reasoning", "vision", "code", etc.

        Returns:
            Model name/ID
        """
        # Check provider config from settings
        provider_cfg = self._config.get("providers", {}).get(self.name, {})
        task_models = provider_cfg.get("task_models", {}) or provider_cfg.get("models", {})
        if task in task_models:
            return task_models[task]

        # Check class-level mapping
        if task in self.TASK_MODELS:
            return self.TASK_MODELS[task]

        # Fall back to current model
        return self.model

    def set_model_for_task(self, task: str, model: str):
        """Set the model to use for a specific task."""
        self.TASK_MODELS[task] = model

    def get_available_tasks(self) -> List[str]:
        """Get list of task types this provider can handle."""
        tasks = set(self.TASK_MODELS.keys())

        # Add from config
        provider_cfg = self._config.get("providers", {}).get(self.name, {})
        task_models = provider_cfg.get("task_models", {}) or provider_cfg.get("models", {})
        tasks.update(task_models.keys())

        return sorted(tasks)
