"""Ollama provider with native tool calling support."""

from typing import Generator, List, Callable, Optional
from .base import BaseProvider, Message


class OllamaProvider(BaseProvider):
    """Ollama local LLM provider with native tool calling."""

    name = "ollama"
    supports_streaming = True
    supports_vision = True
    supports_tools = True

    # Model capability mapping for tool calling reliability
    # format: "native" = use Ollama's native tool calling
    #         "prompt" = use prompt-based JSON tool calling (more reliable for some models)
    TOOL_CAPABLE_MODELS = {
        # High reliability - native tools work well
        "qwen3": {"reliability": "high", "format": "native"},
        "qwen2.5": {"reliability": "high", "format": "native"},
        "llama3.2": {"reliability": "high", "format": "native"},
        "llama3.1": {"reliability": "medium", "format": "native"},
        "mistral": {"reliability": "medium", "format": "native"},
        "functiongemma": {"reliability": "high", "format": "native"},
        "granite4": {"reliability": "medium", "format": "native"},
        "glm-4": {"reliability": "medium", "format": "native"},
        "glm4": {"reliability": "medium", "format": "native"},

        # Low reliability - prefer prompt-based
        "deepseek-r1": {"reliability": "low", "format": "prompt"},
        "qwq": {"reliability": "low", "format": "prompt"},
        "gpt-oss": {"reliability": "medium", "format": "native"},

        # Vision models - typically don't support tools well
        "llava": {"reliability": "low", "format": "prompt"},
    }

    # Reasoning models that need longer timeouts
    REASONING_MODELS = ["gpt-oss", "deepseek-r1", "qwq", "o1", "o3"]

    # Vision-capable models (ordered by preference)
    VISION_MODELS = [
        "granite3.2-vision",  # Fast, small (2.4GB)
        "minicpm-v",          # Good balance
        "moondream",          # Fast/small
        "llava",              # Classic
        "llava-llama3",       # Good quality
        "llama3.2-vision",    # Best quality but slow
        "bakllava",           # Alternative
    ]

    def __init__(self, model: str = None, **kwargs):
        # Default model will be auto-detected if None
        super().__init__(model=model or "pending", **kwargs)
        self.base_url = kwargs.get("base_url", "http://localhost:11434")
        self.headers = kwargs.get("headers")
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
                timeout=httpx.Timeout(timeout, connect=30.0),
                headers=self.headers
            )
        except ImportError:
            raise ImportError("ollama package required: pip install ollama")

    def _should_use_native_tools(self) -> bool:
        """Check if current model reliably supports native tool calling."""
        if not self.model:
            return False

        model_base = self.model.split(":")[0].lower()

        # Check against known models
        for name, info in self.TOOL_CAPABLE_MODELS.items():
            if name in model_base:
                # Use native tools only if format is "native" and reliability isn't "low"
                return info["format"] == "native" and info["reliability"] != "low"

        # Default: try native tools for unknown models (let Ollama handle it)
        return True

    def get_tool_reliability(self) -> str:
        """Get the tool calling reliability rating for the current model."""
        if not self.model:
            return "unknown"

        model_base = self.model.split(":")[0].lower()

        for name, info in self.TOOL_CAPABLE_MODELS.items():
            if name in model_base:
                return info["reliability"]

        return "unknown"

    def get_context_length(self, model: str = None) -> int:
        """Get the context window size by querying Ollama's model info."""
        target = model or self.model
        if not target or target == "pending":
            return 8192

        try:
            response = self.client.show(target)

            # Check modelinfo (Ollama uses 'modelinfo' not 'model_info')
            # Keys are like 'qwen3.context_length', 'llama.context_length', etc.
            modelinfo = getattr(response, 'modelinfo', None) or getattr(response, 'model_info', None)
            if modelinfo and isinstance(modelinfo, dict):
                for key, val in modelinfo.items():
                    if 'context_length' in key.lower():
                        return int(val)

            # Check parameters string for user-set num_ctx override
            params = getattr(response, 'parameters', None)
            if params and isinstance(params, str):
                for line in params.split('\n'):
                    if 'num_ctx' in line:
                        try:
                            return int(line.split()[-1])
                        except (ValueError, IndexError):
                            pass

            # Check modelfile for PARAMETER num_ctx
            modelfile = getattr(response, 'modelfile', None)
            if modelfile and isinstance(modelfile, str):
                for line in modelfile.split('\n'):
                    if 'num_ctx' in line.lower():
                        parts = line.split()
                        for i, part in enumerate(parts):
                            if part.lower() == 'num_ctx' and i + 1 < len(parts):
                                try:
                                    return int(parts[i + 1])
                                except (ValueError, IndexError):
                                    pass

            # Dict access fallback (older Ollama versions)
            if isinstance(response, dict):
                for info_key in ('modelinfo', 'model_info'):
                    if info_key in response:
                        for key, val in response[info_key].items():
                            if 'context_length' in key.lower():
                                return int(val)
        except Exception:
            pass

        return 8192  # Safe default

    def _convert_tools_to_ollama(self, tools: List[Callable]) -> List[dict]:
        """Convert Python functions to Ollama tool format (delegates to base)."""
        return self.convert_tools_to_schema(tools)

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

    def get_vision_model(self) -> str | None:
        """Find the best available vision model."""
        available = [m.lower() for m in self.list_models()]
        print(f"[Ollama] Looking for vision model. Available: {len(available)} models")
        for vision_model in self.VISION_MODELS:
            for avail in available:
                if vision_model in avail:
                    # Return the actual model name (with tag)
                    for m in self.list_models():
                        if vision_model in m.lower():
                            print(f"[Ollama] Found vision model: {m}")
                            return m
        print(f"[Ollama] No vision model found. Install with: ollama pull llava")
        return None

    def vision(self, image_path: str, prompt: str, model: str = None, timeout: float = None) -> str:
        """Analyze an image using a vision model.

        Args:
            image_path: Path to the image file
            prompt: Question or instruction about the image
            model: Vision model to use (auto-detected if None)
            timeout: Request timeout in seconds (default: 300s for vision)

        Returns:
            Text analysis of the image
        """
        import time

        # Auto-detect vision model if not specified
        if not model:
            model = self.get_vision_model()
            if not model:
                raise ValueError("No vision model available. Install one with: ollama pull llava")

        print(f"[Ollama Vision] Using model: {model}")
        print(f"[Ollama Vision] Image: {image_path}")
        print(f"[Ollama Vision] Prompt: {prompt[:100]}{'...' if len(prompt) > 100 else ''}")

        # Create a client with longer timeout for vision (models can be slow)
        import httpx
        vision_timeout = timeout or 60.0  # 1 minute default for vision
        vision_client = self.ollama.Client(
            host=self.base_url,
            timeout=httpx.Timeout(vision_timeout, connect=30.0),
            headers=self.headers
        )

        start_time = time.time()
        print(f"[Ollama Vision] Starting analysis (timeout: {vision_timeout}s)...")

        response = vision_client.chat(
            model=model,
            messages=[{
                "role": "user",
                "content": prompt,
                "images": [image_path]
            }],
            stream=False
        )

        elapsed = time.time() - start_time
        print(f"[Ollama Vision] Completed in {elapsed:.1f}s")

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
