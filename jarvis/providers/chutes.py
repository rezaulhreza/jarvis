"""Chutes AI provider - Comprehensive API for LLM, TTS, STT, Image, Video, Music.

Chutes provides access to various AI models through an OpenAI-compatible API:

LLM Models (text generation):
- Qwen/Qwen3-32B (general purpose, fast)
- deepseek-ai/DeepSeek-V3 (complex reasoning)
- Qwen/Qwen2.5-Coder-32B-Instruct (code generation)
- Qwen/Qwen2.5-VL-72B-Instruct (vision/multimodal)
- unsloth/gemma-3-4b-it (quick/cheap)
- deepseek-ai/DeepSeek-R1 (deep reasoning)

TTS Models (text-to-speech):
- kokoro (high quality, natural)
- csm-1b (conversational speech model)

STT Models (speech-to-text):
- whisper-large-v3 (fast & accurate)

Image Generation:
- FLUX.1-schnell (fast, high quality)
- FLUX.1-dev (12B params)
- stable-diffusion-xl-base-1.0
- hidream (artistic)
- JuggernautXL (photorealistic)

Video Generation:
- Wan2.1-14B (text/image to video)

Music Generation:
- DiffRhythm (text to music)
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

    # API endpoints
    BASE_URL = "https://llm.chutes.ai/v1"
    TTS_URL = "https://chutes.ai/api/tts"
    STT_URL = "https://chutes.ai/api/stt"
    IMAGE_URL = "https://chutes.ai/api/image"
    VIDEO_URL = "https://chutes.ai/api/video"

    # Task to model mapping
    TASK_MODELS = {
        "default": "Qwen/Qwen3-32B",
        "reasoning": "deepseek-ai/DeepSeek-V3",
        "deep": "deepseek-ai/DeepSeek-R1-TEE",
        "vision": "Qwen/Qwen2.5-VL-72B-Instruct-TEE",
        "code": "Qwen/Qwen2.5-Coder-32B-Instruct",
        "fast": "unsloth/gemma-3-4b-it",
        "balanced": "Qwen/Qwen3-32B",
        "chat": "unsloth/gemma-3-4b-it",
        "thinking": "Qwen/Qwen3-235B-A22B-Thinking-2507",
    }

    # TTS models
    TTS_MODELS = {
        "kokoro": "kokoro",           # High quality, natural voices
        "csm-1b": "csm-1b",           # Conversational speech model
    }

    # STT models
    STT_MODELS = {
        "whisper": "whisper-large-v3",
        "whisper-fast": "whisper-large-v3-turbo",
    }

    # Image generation models (sorted by popularity/quality)
    IMAGE_MODELS = {
        "flux-schnell": "FLUX.1-schnell",        # Fast, high quality
        "flux-dev": "black-forest-labs/FLUX.1-dev",
        "sdxl": "stabilityai/stable-diffusion-xl-base-1.0",
        "hidream": "hidream",                     # Artistic
        "juggernaut": "JuggernautXL-Ragnarok",   # Photorealistic
        "dreamshaper": "Lykon/dreamshaper-xl-1-0",
        "animij": "Animij",                       # Anime style
        "qwen-image": "qwen-image",               # Qwen image gen
        "hunyuan": "hunyuan-image-3",             # Chinese style
        "chroma": "chroma",                       # Color-focused
    }

    # Video generation models
    VIDEO_MODELS = {
        "wan": "Wan2.1-14B",          # Text/image to video
    }

    # Music generation models
    MUSIC_MODELS = {
        "diffrhythm": "ASLP-lab/DiffRhythm",
    }

    # Backward compatibility alias
    MODELS = TASK_MODELS

    # Known LLM models (fallback when API discovery fails)
    KNOWN_MODELS = [
        # === TOP PICKS ===
        "Qwen/Qwen3-32B",
        "deepseek-ai/DeepSeek-V3",
        "Qwen/Qwen2.5-Coder-32B-Instruct",
        "Qwen/Qwen2.5-VL-72B-Instruct-TEE",
        "unsloth/gemma-3-4b-it",
        # === Qwen Family ===
        "Qwen/Qwen3-235B-A22B",
        "Qwen/Qwen3-235B-A22B-Instruct-2507-TEE",
        "Qwen/Qwen3-235B-A22B-Thinking-2507",
        "Qwen/Qwen3-30B-A3B",
        "Qwen/Qwen3-30B-A3B-Instruct-2507",
        "Qwen/Qwen3-14B",
        "Qwen/Qwen3-Coder-480B-A35B-Instruct-FP8-TEE",
        "Qwen/Qwen3-Coder-Next",
        "Qwen/Qwen3-Next-80B-A3B-Instruct",
        "Qwen/Qwen3-VL-235B-A22B-Instruct",
        "Qwen/Qwen2.5-72B-Instruct",
        "Qwen/Qwen2.5-VL-32B-Instruct",
        # === DeepSeek ===
        "deepseek-ai/DeepSeek-V3.2-TEE",
        "deepseek-ai/DeepSeek-V3.1-TEE",
        "deepseek-ai/DeepSeek-V3.1-Terminus-TEE",
        "deepseek-ai/DeepSeek-V3-0324-TEE",
        "deepseek-ai/DeepSeek-R1-TEE",
        "deepseek-ai/DeepSeek-R1-0528-TEE",
        "deepseek-ai/DeepSeek-R1-Distill-Llama-70B",
        # === Kimi / Moonshot ===
        "moonshotai/Kimi-K2.5-TEE",
        "moonshotai/Kimi-K2-Thinking-TEE",
        "moonshotai/Kimi-K2-Instruct-0905",
        # === Hermes / Nous ===
        "NousResearch/Hermes-4-405B-FP8-TEE",
        "NousResearch/Hermes-4-70B",
        "NousResearch/Hermes-4.3-36B",
        "NousResearch/Hermes-4-14B",
        "NousResearch/DeepHermes-3-Mistral-24B-Preview",
        # === Mistral ===
        "mistralai/Devstral-2-123B-Instruct-2512-TEE",
        "chutesai/Mistral-Small-3.2-24B-Instruct-2506",
        "chutesai/Mistral-Small-3.1-24B-Instruct-2503",
        "unsloth/Mistral-Small-24B-Instruct-2501",
        "unsloth/Mistral-Nemo-Instruct-2407",
        # === Gemma ===
        "unsloth/gemma-3-27b-it",
        "unsloth/gemma-3-12b-it",
        # === Llama ===
        "unsloth/Llama-3.2-3B-Instruct",
        "unsloth/Llama-3.2-1B-Instruct",
        # === GLM / ZAI ===
        "zai-org/GLM-4.7-TEE",
        "zai-org/GLM-4.7-Flash",
        "zai-org/GLM-4.6-TEE",
        "zai-org/GLM-4.6V",
        "zai-org/GLM-4.5-TEE",
        # === Other Notable ===
        "OpenGVLab/InternVL3-78B-TEE",
        "openai/gpt-oss-120b-TEE",
        "openai/gpt-oss-20b",
        "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16",
        "MiniMaxAI/MiniMax-M2.1-TEE",
        "XiaomiMiMo/MiMo-V2-Flash",
        "tngtech/DeepSeek-TNG-R1T2-Chimera",
    ]

    # Backward compatibility
    AVAILABLE_MODELS = KNOWN_MODELS

    def __init__(self, model: str = None, api_key: str = None, **kwargs):
        super().__init__(model=model, api_key=api_key, **kwargs)

        # Load API key: param > credentials.json > env
        if not api_key:
            try:
                from jarvis.auth.credentials import get_credential
                api_key = get_credential("chutes", "api_key")
            except ImportError:
                pass
        self.api_key = api_key or os.getenv("CHUTES_API_KEY")
        self.base_url = kwargs.get("base_url") or os.getenv("CHUTES_BASE_URL", self.BASE_URL)

        # Load task models from config if provided
        config = kwargs.get("config", {})
        provider_cfg = config.get("providers", {}).get("chutes", {})
        models_cfg = provider_cfg.get("models", {}) or provider_cfg.get("task_models", {})
        if models_cfg:
            self.TASK_MODELS.update(models_cfg)
            self.MODELS = self.TASK_MODELS

        # Use default model if none specified
        if not self.model:
            self.model = self.TASK_MODELS.get("default", "Qwen/Qwen3-32B")

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
            return self._chat_streaming(msg_list)
        else:
            return self._chat_non_streaming(msg_list)

    def _chat_streaming(self, msg_list: List[dict]) -> Generator[str, None, None]:
        """Streaming chat implementation."""
        response = self.client.chat.completions.create(
            model=self.model,
            messages=msg_list,
            stream=True
        )
        for chunk in response:
            if self._stop_flag:
                break
            # Some chunks have empty choices array - skip those
            if chunk.choices and chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content

    def _chat_non_streaming(self, msg_list: List[dict]) -> str:
        """Non-streaming chat implementation."""
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
        if self._discovered_models:
            return [m.id for m in self._discovered_models]
        return self.KNOWN_MODELS

    async def discover_models(self) -> List:
        """Discover available models from Chutes API."""
        from .base import ModelInfo

        if self._discovered_models is not None:
            return self._discovered_models

        if not self.client:
            self._discovered_models = [
                ModelInfo(id=m, name=m.split("/")[-1])
                for m in self.KNOWN_MODELS
            ]
            return self._discovered_models

        try:
            response = self.client.models.list()
            self._discovered_models = []

            for model in response.data:
                capabilities = []
                model_lower = model.id.lower()
                if "vl" in model_lower or "vision" in model_lower:
                    capabilities.append("vision")
                if "coder" in model_lower or "code" in model_lower:
                    capabilities.append("code")
                if "r1" in model_lower or "reasoning" in model_lower or "thinking" in model_lower:
                    capabilities.append("reasoning")

                self._discovered_models.append(ModelInfo(
                    id=model.id,
                    name=model.id.split("/")[-1] if "/" in model.id else model.id,
                    capabilities=capabilities,
                    context_length=getattr(model, "context_window", None),
                    metadata={"owned_by": getattr(model, "owned_by", None)}
                ))

            print(f"[Chutes] Discovered {len(self._discovered_models)} models")
            return self._discovered_models

        except Exception as e:
            print(f"[Chutes] Model discovery failed: {e}, using known models")
            self._discovered_models = [
                ModelInfo(id=m, name=m.split("/")[-1])
                for m in self.KNOWN_MODELS
            ]
            return self._discovered_models

    def is_configured(self) -> bool:
        """Check if API key is set."""
        return bool(self.api_key)

    def get_default_model(self) -> str:
        """Get the default model."""
        return self.MODELS["default"]

    def get_model_for_task(self, task: str) -> str:
        """Get the recommended model for a specific task type."""
        return self.MODELS.get(task, self.MODELS["default"])

    def get_tts_models(self) -> dict:
        """Get available TTS models."""
        return self.TTS_MODELS

    def get_stt_models(self) -> dict:
        """Get available STT models."""
        return self.STT_MODELS

    def get_image_models(self) -> dict:
        """Get available image generation models."""
        return self.IMAGE_MODELS

    def get_video_models(self) -> dict:
        """Get available video generation models."""
        return self.VIDEO_MODELS

    def get_music_models(self) -> dict:
        """Get available music generation models."""
        return self.MUSIC_MODELS

    def get_all_services(self) -> dict:
        """Get all available Chutes services and their models."""
        return {
            "llm": {
                "models": self.TASK_MODELS,
                "endpoint": self.BASE_URL,
            },
            "tts": {
                "models": self.TTS_MODELS,
                "recommended": "kokoro",
            },
            "stt": {
                "models": self.STT_MODELS,
                "recommended": "whisper-large-v3",
            },
            "image": {
                "models": self.IMAGE_MODELS,
                "recommended": "flux-schnell",
            },
            "video": {
                "models": self.VIDEO_MODELS,
                "recommended": "wan",
            },
            "music": {
                "models": self.MUSIC_MODELS,
                "recommended": "diffrhythm",
            },
        }

    def get_context_length(self, model: str = None) -> int | None:
        """Get context length for a model.

        Returns context length in tokens, or None if unknown.
        Sources: HuggingFace model cards, official docs, API documentation
        """
        model = model or self.model

        # Check discovered models first
        if self._discovered_models:
            for m in self._discovered_models:
                if m.id == model and m.context_length:
                    return m.context_length

        # Known context lengths for popular models (verified from official sources)
        KNOWN_CONTEXT = {
            # Kimi / Moonshot - https://huggingface.co/moonshotai/Kimi-K2.5
            "kimi-k2.5": 262144,   # 256k
            "kimi-k2": 131072,    # 128k
            "k2.5": 262144,
            "k2-": 131072,
            "moonshot": 131072,
            # Qwen family - https://huggingface.co/Qwen/Qwen3-32B
            "qwen3-235b": 131072,  # 128k with YaRN
            "qwen3-32b": 32768,    # 32k native, 128k with YaRN
            "qwen3-14b": 32768,
            "qwen3-30b": 32768,
            "qwen2.5-72b": 131072,
            "qwen2.5-coder": 131072,
            "qwen2.5-vl": 32768,
            "qwen2.5": 32768,
            "qwen3": 32768,
            # DeepSeek - https://huggingface.co/deepseek-ai/DeepSeek-V3
            "deepseek-v3": 128000,  # 128k
            "deepseek-r1": 128000,  # 128k
            "deepseek": 128000,
            # Gemma 3 - https://ai.google.dev/gemma/docs/core (4B/12B/27B = 128k)
            "gemma-3-27b": 128000,
            "gemma-3-12b": 128000,
            "gemma-3-4b": 128000,
            "gemma-3-1b": 32768,
            "gemma-3": 128000,
            # Llama 3 - https://llama.meta.com
            "llama-3.2": 128000,
            "llama-3.1": 128000,
            "llama-3": 8192,
            # Mistral - https://docs.mistral.ai
            "mistral-small": 32768,
            "mistral-nemo": 128000,
            "mistral": 32768,
            # GLM
            "glm-4": 128000,
            # Hermes
            "hermes-4": 131072,
            "hermes-3": 131072,
        }

        model_lower = model.lower()
        for name, ctx in KNOWN_CONTEXT.items():
            if name in model_lower:
                return ctx

        # Default for unknown models
        return 32768  # Safe default

    def get_config_help(self) -> str:
        return """Chutes AI - Comprehensive AI Platform

1. Get API key: https://chutes.ai
2. Set environment variable:
   export CHUTES_API_KEY=your-api-key

Or add to ~/.jarvis/.env:
   CHUTES_API_KEY=your-api-key

=== Available Services ===

LLM (Text Generation):
- Qwen/Qwen3-32B (general purpose)
- deepseek-ai/DeepSeek-V3 (reasoning)
- Qwen/Qwen2.5-Coder-32B-Instruct (code)
- Qwen/Qwen2.5-VL-72B-Instruct (vision)
- unsloth/gemma-3-4b-it (fast/cheap)

TTS (Text-to-Speech):
- kokoro (high quality, natural)
- csm-1b (conversational)

STT (Speech-to-Text):
- whisper-large-v3 (fast & accurate)

Image Generation:
- FLUX.1-schnell (fast, high quality)
- stable-diffusion-xl-base-1.0
- JuggernautXL (photorealistic)

Video Generation:
- Wan2.1-14B (text/image to video)

Music Generation:
- DiffRhythm (text to music)"""
