"""Chutes AI provider - OpenAI-compatible API with reliable tool calling.

Chutes provides access to various models through an OpenAI-compatible API.
Recommended models by task:
- default: Qwen/Qwen3-32B (general purpose)
- reasoning: deepseek-ai/DeepSeek-V3 (complex reasoning)
- vision: Qwen/Qwen2.5-VL-72B-Instruct (image analysis)
- code: Qwen/Qwen2.5-Coder-32B-Instruct (code generation)
- fast: unsloth/gemma-3-4b-it (quick responses)
"""

import os
import json
import inspect
from typing import Generator, List, Callable
from .base import BaseProvider, Message


class ChutesProvider(BaseProvider):
    """Chutes AI provider using OpenAI-compatible API."""

    name = "chutes"
    supports_streaming = True
    supports_vision = True
    supports_tools = True

    BASE_URL = "https://llm.chutes.ai/v1"

    # Recommended models by task type
    MODELS = {
        "default": "Qwen/Qwen3-32B",
        "reasoning": "deepseek-ai/DeepSeek-V3",
        "vision": "Qwen/Qwen2.5-VL-72B-Instruct",
        "code": "Qwen/Qwen2.5-Coder-32B-Instruct",
        "fast": "unsloth/gemma-3-4b-it",
    }

    # Full list of available models
    AVAILABLE_MODELS = [
        "Qwen/Qwen3-32B",
        "Qwen/Qwen3-235B-A22B",
        "deepseek-ai/DeepSeek-V3",
        "deepseek-ai/DeepSeek-R1",
        "Qwen/Qwen2.5-VL-72B-Instruct",
        "Qwen/Qwen2.5-Coder-32B-Instruct",
        "unsloth/gemma-3-4b-it",
        "unsloth/Llama-3.3-70B-Instruct",
        "meta-llama/Llama-3.1-405B-Instruct",
    ]

    def __init__(self, model: str = None, api_key: str = None, **kwargs):
        super().__init__(model=model, api_key=api_key, **kwargs)

        self.api_key = api_key or os.getenv("CHUTES_API_KEY")
        self.base_url = kwargs.get("base_url") or os.getenv("CHUTES_BASE_URL", self.BASE_URL)

        # Use default model if none specified
        if not self.model:
            self.model = self.MODELS["default"]

        if self.api_key:
            try:
                from openai import OpenAI
                self.client = OpenAI(api_key=self.api_key, base_url=self.base_url)
            except ImportError:
                raise ImportError("openai package required: pip install openai")
        else:
            self.client = None

    def _convert_tools_to_openai(self, tools: List[Callable]) -> List[dict]:
        """Convert Python functions to OpenAI tool format."""
        openai_tools = []
        for func in tools:
            sig = inspect.signature(func)
            params = {}
            required = []

            for name, param in sig.parameters.items():
                param_type = "string"
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
        stream: bool = True,
        **kwargs
    ) -> Generator[str, None, None] | str:
        """Send chat request to Chutes API."""
        if not self.client:
            raise ValueError("Chutes API key not configured. Set CHUTES_API_KEY or run /config")

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

    def chat_with_tools(
        self,
        messages: List[dict],
        system: str = None,
        tools: List[Callable] = None
    ):
        """Non-streaming chat that returns tool calls."""
        if not self.client:
            raise ValueError("Chutes API key not configured")

        openai_tools = self._convert_tools_to_openai(tools) if tools else None

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

        if not result["message"]["tool_calls"]:
            result["message"]["tool_calls"] = None

        return type('Response', (), result)()

    def vision(self, image_path: str, prompt: str) -> str:
        """Analyze image with Chutes vision model."""
        if not self.client:
            raise ValueError("Chutes API key not configured")

        import base64

        with open(image_path, "rb") as f:
            image_data = base64.b64encode(f.read()).decode("utf-8")

        # Use vision model
        vision_model = self.MODELS.get("vision", self.model)

        response = self.client.chat.completions.create(
            model=vision_model,
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

    def list_models(self) -> List[str]:
        """List available Chutes models."""
        # Chutes doesn't have a list models endpoint, return known models
        return self.AVAILABLE_MODELS

    def is_configured(self) -> bool:
        """Check if API key is set."""
        return bool(self.api_key)

    def get_default_model(self) -> str:
        """Get the default model."""
        return self.MODELS["default"]

    def get_model_for_task(self, task: str) -> str:
        """Get the recommended model for a specific task type.

        Args:
            task: One of "default", "reasoning", "vision", "code", "fast"

        Returns:
            Model name for the task
        """
        return self.MODELS.get(task, self.MODELS["default"])

    def get_config_help(self) -> str:
        return """Chutes AI

1. Get API key: https://chutes.ai
2. Set environment variable:
   export CHUTES_API_KEY=your-api-key

Or add to ~/.jarvis/.env:
   CHUTES_API_KEY=your-api-key

Recommended models:
- Default: Qwen/Qwen3-32B (general purpose)
- Reasoning: deepseek-ai/DeepSeek-V3
- Vision: Qwen/Qwen2.5-VL-72B-Instruct
- Code: Qwen/Qwen2.5-Coder-32B-Instruct
- Fast: unsloth/gemma-3-4b-it"""
