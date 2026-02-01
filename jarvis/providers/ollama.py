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
    TOOL_MODELS = ["qwen3", "qwen2.5", "llama3", "mistral", "functiongemma"]

    def __init__(self, model: str = "llama3.2:latest", **kwargs):
        super().__init__(model=model, **kwargs)
        self.base_url = kwargs.get("base_url", "http://localhost:11434")

        try:
            import ollama
            self.ollama = ollama
            self.client = ollama.Client(host=self.base_url)
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
        tools: List[Callable] = None
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

    def get_config_help(self) -> str:
        return """Ollama (Local)

1. Install Ollama: https://ollama.ai
2. Start server: ollama serve
3. Pull a model: ollama pull llama3.2

For tool calling, use: qwen3:4b, llama3.2, or mistral"""
