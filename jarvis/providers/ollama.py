"""Ollama provider with native tool calling support."""

import inspect
from typing import Generator, List, Callable, Optional
from .base import BaseProvider, Message


class OllamaProvider(BaseProvider):
    """Ollama local LLM provider with native tool calling."""

    name = "ollama"
    supports_streaming = True
    supports_vision = True
    supports_tools = True

    # Models that support tool calling well
    TOOL_MODELS = [
        "qwen3", "qwen2.5", "llama3", "mistral", "functiongemma",
        "gpt-oss", "granite4", "glm-4", "glm4",
    ]

    # Reasoning models that need longer timeouts
    REASONING_MODELS = ["gpt-oss", "deepseek-r1", "qwq", "o1", "o3"]

    def __init__(self, model: str = None, **kwargs):
        # Default model will be auto-detected if None
        super().__init__(model=model or "pending", **kwargs)
        self.base_url = kwargs.get("base_url", "http://localhost:11434")
        self._model_auto = model is None  # Track if we need to auto-detect

        # Determine timeout based on model type
        model_for_timeout = model or "default"
        is_reasoning = any(r in model_for_timeout.lower() for r in self.REASONING_MODELS)
        timeout = 600.0 if is_reasoning else 120.0  # 10 min for reasoning, 2 min default

        try:
            import ollama
            import httpx
            self.ollama = ollama
            # Create client with extended timeout for reasoning models
            self.client = ollama.Client(
                host=self.base_url,
                timeout=httpx.Timeout(timeout, connect=30.0)
            )
        except ImportError:
            raise ImportError("ollama package required: pip install ollama")

    def _convert_tools_to_ollama(self, tools: List[Callable]) -> List[dict]:
        """Convert Python functions to Ollama tool format."""
        ollama_tools = []
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

                params[name] = {
                    "type": param_type,
                    "description": f"The {name} parameter"
                }

                if param.default == inspect.Parameter.empty:
                    required.append(name)

            # Get docstring for description
            doc = func.__doc__ or f"Function {func.__name__}"
            # Get first line/paragraph of docstring
            description = doc.split("\n\n")[0].strip().split("\n")[0]

            ollama_tools.append({
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

        return ollama_tools

    def chat(
        self,
        messages: List[Message],
        system: str = None,
        stream: bool = True,
        tools: List[Callable] = None,
        options: dict = None
    ) -> Generator[str, None, None]:
        """Send chat request with optional tool calling."""
        self.reset_stop()

        # Convert messages
        msg_list = []
        if system:
            msg_list.append({"role": "system", "content": system})
        for m in messages:
            if isinstance(m, Message):
                msg_list.append({"role": m.role, "content": m.content})
            else:
                msg_list.append(m)

        # Build request kwargs
        kwargs = {
            "model": self.model,
            "messages": msg_list,
            "stream": stream,
        }

        # Add options if provided (num_predict, temperature, etc.)
        if options:
            kwargs["options"] = options

        # Convert and add tools if provided
        if tools:
            kwargs["tools"] = self._convert_tools_to_ollama(tools)

        if stream:
            response = self.client.chat(**kwargs)
            for chunk in response:
                if self._stop_flag:
                    break

                content = ""
                if hasattr(chunk, 'message'):
                    content = getattr(chunk.message, 'content', '') or ''
                    if isinstance(chunk.message, dict):
                        content = chunk.message.get('content', '')
                elif isinstance(chunk, dict) and 'message' in chunk:
                    msg = chunk['message']
                    content = msg.get('content', '') if isinstance(msg, dict) else getattr(msg, 'content', '')

                if content:
                    yield content
        else:
            response = self.client.chat(**kwargs)
            return response

    def chat_with_tools(
        self,
        messages: List[dict],
        system: str = None,
        tools: List[Callable] = None
    ):
        """Non-streaming chat that returns tool calls."""
        msg_list = []
        if system:
            msg_list.append({"role": "system", "content": system})
        msg_list.extend(messages)

        kwargs = {
            "model": self.model,
            "messages": msg_list,
            "stream": False,
        }

        # Convert and add tools if provided
        if tools:
            kwargs["tools"] = self._convert_tools_to_ollama(tools)

        return self.client.chat(**kwargs)

    def list_models(self) -> List[str]:
        try:
            response = self.client.list()
            if hasattr(response, 'models'):
                return [m.model if hasattr(m, 'model') else m.get('model', '') for m in response.models]
            elif isinstance(response, dict) and 'models' in response:
                return [m.get('model', m.get('name', '')) for m in response['models']]
            return []
        except Exception:
            return []

    def vision(self, image_path: str, prompt: str, model: str = "llava") -> str:
        response = self.client.chat(
            model=model,
            messages=[{
                "role": "user",
                "content": prompt,
                "images": [image_path]
            }],
            stream=False
        )
        if hasattr(response, 'message'):
            return getattr(response.message, 'content', '')
        return response['message']['content']

    def is_configured(self) -> bool:
        try:
            self.client.list()
            return True
        except Exception:
            return False

    def get_default_model(self) -> Optional[str]:
        """Get first available model, preferring known good ones for tool calling."""
        models = self.list_models()
        if not models:
            return None

        # Prefer these models in order (good for tool calling)
        preferred = ["qwen3", "llama3.2", "llama3.1", "mistral", "qwen2.5", "llama3"]
        for pref in preferred:
            for model in models:
                if pref in model.lower():
                    return model

        # Return first available model
        return models[0]

    def get_config_help(self) -> str:
        return """Ollama (Local)

1. Install Ollama: https://ollama.ai
2. Start server: ollama serve
3. Pull a model: ollama pull qwen3:4b

For tool calling, use: qwen3:4b, llama3.2, llama3.1, or mistral"""
