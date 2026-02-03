"""OpenAI GPT provider with tool calling support."""

import os
import json
import inspect
from typing import Generator, List, Callable
from .base import BaseProvider, Message


class OpenAIProvider(BaseProvider):
    """OpenAI GPT API provider with native tool calling."""

    name = "openai"
    supports_streaming = True
    supports_vision = True
    supports_tools = True

    MODELS = [
        "gpt-5.2-codex",
        "gpt-5.2",
        "gpt-5.1-codex",
        "gpt-5.1",
        "gpt-4.1",
        "gpt-4.1-mini",
        "gpt-4.1-nano",
        "gpt-4o",
        "gpt-4o-mini",
        "o3",
        "o3-pro",
        "o4-mini",
        "o3-deep-research",
    ]

    def __init__(self, model: str = "gpt-5.2-codex", api_key: str = None, **kwargs):
        super().__init__(model=model, api_key=api_key, **kwargs)

        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.base_url = kwargs.get("base_url")  # For OpenAI-compatible APIs

        if self.api_key:
            try:
                from openai import OpenAI
                self.client = OpenAI(
                    api_key=self.api_key,
                    base_url=self.base_url
                )
            except ImportError:
                raise ImportError("openai package required: pip install openai")
        else:
            self.client = None

    def _convert_tools_to_openai(self, tools: List[Callable]) -> List[dict]:
        """Convert Python functions to OpenAI tool format."""
        openai_tools = []
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

            openai_tools.append({
                "type": "function",
                "function": {
                    "name": func.__name__,
                    "description": description,
                    "parameters": {
                        "type": "object",
                        "properties": params,
                        "required": required
                    }
                }
            })

        return openai_tools

    def chat(
        self,
        messages: List[Message],
        system: str = None,
        stream: bool = True
    ) -> Generator[str, None, None] | str:
        """Send chat request to OpenAI."""
        if not self.client:
            raise ValueError("OpenAI API key not configured. Set OPENAI_API_KEY or run /config")

        self.reset_stop()

        # Build messages
        msg_list = []
        if system:
            msg_list.append({"role": "system", "content": system})
        msg_list.extend([{"role": m.role, "content": m.content} for m in messages])

        if stream:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=msg_list,
                stream=True
            )
            for chunk in response:
                if self._stop_flag:
                    break
                if chunk.choices[0].delta.content:
                    yield chunk.choices[0].delta.content
        else:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=msg_list,
                stream=False
            )
            return response.choices[0].message.content

    def list_models(self) -> List[str]:
        """List available models."""
        if not self.client:
            return self.MODELS

        try:
            response = self.client.models.list()
            gpt_models = [m.id for m in response.data if 'gpt' in m.id or 'o1' in m.id]
            return sorted(gpt_models) if gpt_models else self.MODELS
        except Exception:
            return self.MODELS

    def vision(self, image_path: str, prompt: str) -> str:
        """Analyze image with GPT-4 Vision."""
        if not self.client:
            raise ValueError("OpenAI API key not configured")

        import base64

        with open(image_path, "rb") as f:
            image_data = base64.b64encode(f.read()).decode("utf-8")

        response = self.client.chat.completions.create(
            model="gpt-4o",  # Vision model
            messages=[{
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{image_data}"}
                    }
                ]
            }],
            max_tokens=1024
        )
        return response.choices[0].message.content

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
            raise ValueError("OpenAI API key not configured")

        # Convert tools to OpenAI format
        openai_tools = self._convert_tools_to_openai(tools) if tools else None

        # Build messages list
        msg_list = []
        if system:
            msg_list.append({"role": "system", "content": system})

        for m in messages:
            if isinstance(m, dict):
                msg_list.append(m)
            else:
                msg_list.append({"role": m.role, "content": m.content})

        kwargs = {
            "model": self.model,
            "messages": msg_list,
        }

        if openai_tools:
            kwargs["tools"] = openai_tools

        response = self.client.chat.completions.create(**kwargs)
        msg = response.choices[0].message

        # Convert OpenAI response to a consistent format
        result = {
            "message": {
                "content": msg.content or "",
                "tool_calls": []
            }
        }

        if msg.tool_calls:
            for tc in msg.tool_calls:
                args = tc.function.arguments
                if isinstance(args, str):
                    args = json.loads(args)
                result["message"]["tool_calls"].append({
                    "id": tc.id,
                    "function": {
                        "name": tc.function.name,
                        "arguments": args
                    }
                })

        # If no tool calls, set to None for consistency
        if not result["message"]["tool_calls"]:
            result["message"]["tool_calls"] = None

        return type('Response', (), result)()

    def get_config_help(self) -> str:
        return """OpenAI GPT

1. Get API key: https://platform.openai.com/api-keys
2. Set environment variable:
   export OPENAI_API_KEY=sk-...

Or add to ~/.jarvis/.env:
   OPENAI_API_KEY=sk-..."""
