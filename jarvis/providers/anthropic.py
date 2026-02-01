"""Anthropic Claude provider with tool calling support."""

import os
import inspect
from typing import Generator, List, Callable
from .base import BaseProvider, Message


class AnthropicProvider(BaseProvider):
    """Anthropic Claude API provider with native tool calling."""

    name = "anthropic"
    supports_streaming = True
    supports_vision = True
    supports_tools = True

    MODELS = [
        "claude-opus-4-5-20251101",
        "claude-opus-4-5",
        "claude-sonnet-4-5-20250929",
        "claude-sonnet-4",
        "claude-opus-4-1-20250805",
        "claude-opus-4",
        "claude-3-5-sonnet-20241022",
        "claude-3-5-haiku-20241022",
    ]

    def __init__(self, model: str = "claude-opus-4-5", api_key: str = None, **kwargs):
        super().__init__(model=model, api_key=api_key, **kwargs)

        self.api_key = api_key or os.getenv("ANTHROPIC_API_KEY")

        if self.api_key:
            try:
                import anthropic
                self.client = anthropic.Anthropic(api_key=self.api_key)
            except ImportError:
                raise ImportError("anthropic package required: pip install anthropic")
        else:
            self.client = None

    def _convert_tools_to_anthropic(self, tools: List[Callable]) -> List[dict]:
        """Convert Python functions to Anthropic tool format."""
        anthropic_tools = []
        for func in tools:
            # Get function signature
            sig = inspect.signature(func)
            params = {}
            required = []

            for name, param in sig.parameters.items():
                param_type = "string"  # Default to string
                if param.annotation != inspect.Parameter.empty:
                    if param.annotation == int:
                        param_type = "integer"
                    elif param.annotation == bool:
                        param_type = "boolean"
                    elif param.annotation == float:
                        param_type = "number"

                params[name] = {"type": param_type, "description": f"The {name} parameter"}

                if param.default == inspect.Parameter.empty:
                    required.append(name)

            # Get docstring for description
            doc = func.__doc__ or f"Function {func.__name__}"
            description = doc.split("\n\n")[0].strip()

            anthropic_tools.append({
                "name": func.__name__,
                "description": description,
                "input_schema": {
                    "type": "object",
                    "properties": params,
                    "required": required
                }
            })

        return anthropic_tools

    def chat(
        self,
        messages: List[Message],
        system: str = None,
        stream: bool = True
    ) -> Generator[str, None, None] | str:
        """Send chat request to Claude."""
        if not self.client:
            raise ValueError("Anthropic API key not configured. Set ANTHROPIC_API_KEY or run /config")

        self.reset_stop()

        # Convert messages
        msg_list = [{"role": m.role, "content": m.content} for m in messages]

        if stream:
            with self.client.messages.stream(
                model=self.model,
                max_tokens=4096,
                system=system or "You are a helpful assistant.",
                messages=msg_list,
            ) as stream:
                for text in stream.text_stream:
                    if self._stop_flag:
                        break
                    yield text
        else:
            response = self.client.messages.create(
                model=self.model,
                max_tokens=4096,
                system=system or "You are a helpful assistant.",
                messages=msg_list,
            )
            return response.content[0].text

    def list_models(self) -> List[str]:
        """List available Claude models."""
        return self.MODELS

    def vision(self, image_path: str, prompt: str) -> str:
        """Analyze image with Claude vision."""
        if not self.client:
            raise ValueError("Anthropic API key not configured")

        import base64
        import mimetypes

        # Read and encode image
        mime_type, _ = mimetypes.guess_type(image_path)
        with open(image_path, "rb") as f:
            image_data = base64.standard_b64encode(f.read()).decode("utf-8")

        response = self.client.messages.create(
            model=self.model,
            max_tokens=1024,
            messages=[{
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": mime_type or "image/jpeg",
                            "data": image_data
                        }
                    },
                    {"type": "text", "text": prompt}
                ]
            }]
        )
        return response.content[0].text

    def is_configured(self) -> bool:
        """Check if API key is set."""
        return bool(self.api_key)

    def chat_with_tools(
        self,
        messages: List[dict],
        system: str = None,
        tools: List[Callable] = None
    ):
        """Non-streaming chat that returns tool calls."""
        if not self.client:
            raise ValueError("Anthropic API key not configured")

        # Convert tools to Anthropic format
        anthropic_tools = self._convert_tools_to_anthropic(tools) if tools else None

        # Build messages list
        msg_list = []
        for m in messages:
            if isinstance(m, dict):
                msg_list.append(m)
            else:
                msg_list.append({"role": m.role, "content": m.content})

        kwargs = {
            "model": self.model,
            "max_tokens": 4096,
            "messages": msg_list,
        }

        if system:
            kwargs["system"] = system

        if anthropic_tools:
            kwargs["tools"] = anthropic_tools

        response = self.client.messages.create(**kwargs)

        # Convert Anthropic response to a consistent format
        result = {
            "message": {
                "content": "",
                "tool_calls": []
            }
        }

        for block in response.content:
            if block.type == "text":
                result["message"]["content"] = block.text
            elif block.type == "tool_use":
                result["message"]["tool_calls"].append({
                    "function": {
                        "name": block.name,
                        "arguments": block.input
                    }
                })

        # If no tool calls, set to None for consistency
        if not result["message"]["tool_calls"]:
            result["message"]["tool_calls"] = None

        return type('Response', (), result)()

    def get_config_help(self) -> str:
        return """Anthropic Claude

1. Get API key: https://console.anthropic.com/
2. Set environment variable:
   export ANTHROPIC_API_KEY=sk-ant-...

Or add to ~/.jarvis/.env:
   ANTHROPIC_API_KEY=sk-ant-..."""
