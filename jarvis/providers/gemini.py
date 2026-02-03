"""Google Gemini provider with tool calling support."""

import os
import inspect
from typing import Generator, List, Callable
from .base import BaseProvider, Message


class GeminiProvider(BaseProvider):
    """Google Gemini API provider with native tool calling."""

    name = "gemini"
    supports_streaming = True
    supports_vision = True
    supports_tools = True

    MODELS = [
        "gemini-2.5-pro",
        "gemini-2.5-flash",
        "gemini-2.5-flash-lite",
        "gemini-2.0-flash",
        "gemini-2.0-flash-lite",
        "gemini-1.5-pro",
        "gemini-1.5-flash",
    ]

    def __init__(self, model: str = "gemini-2.5-flash", api_key: str = None, **kwargs):
        super().__init__(model=model, api_key=api_key, **kwargs)

        self.api_key = api_key or os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")

        if self.api_key:
            try:
                import google.generativeai as genai
                genai.configure(api_key=self.api_key)
                self.genai = genai
                self.client = genai.GenerativeModel(self.model)
            except ImportError:
                raise ImportError("google-generativeai package required: pip install google-generativeai")
        else:
            self.genai = None
            self.client = None

    def _convert_tools_to_gemini(self, tools: List[Callable]) -> List:
        """Convert Python functions to Gemini tool format."""
        gemini_tools = []
        for func in tools:
            # Get function signature
            sig = inspect.signature(func)
            params = {}
            required = []

            for name, param in sig.parameters.items():
                param_type = "STRING"  # Gemini uses uppercase
                if param.annotation != inspect.Parameter.empty:
                    if param.annotation == int:
                        param_type = "INTEGER"
                    elif param.annotation == bool:
                        param_type = "BOOLEAN"
                    elif param.annotation == float:
                        param_type = "NUMBER"

                params[name] = {
                    "type": param_type,
                    "description": f"The {name} parameter"
                }

                if param.default == inspect.Parameter.empty:
                    required.append(name)

            # Get docstring for description
            doc = func.__doc__ or f"Function {func.__name__}"
            description = doc.split("\n\n")[0].strip()

            gemini_tools.append({
                "name": func.__name__,
                "description": description,
                "parameters": {
                    "type": "OBJECT",
                    "properties": params,
                    "required": required
                }
            })

        return gemini_tools

    def chat(
        self,
        messages: List[Message],
        system: str = None,
        stream: bool = True,
        **kwargs
    ) -> Generator[str, None, None] | str:
        """Send chat request to Gemini."""
        if not self.client:
            raise ValueError("Gemini API key not configured. Set GOOGLE_API_KEY or GEMINI_API_KEY")

        self.reset_stop()

        # Rebuild client if model changed
        if self.client.model_name != f"models/{self.model}":
            self.client = self.genai.GenerativeModel(self.model)

        # Convert messages to Gemini format
        history = []
        for m in messages[:-1]:  # All but last
            role = "user" if m.role == "user" else "model"
            history.append({"role": role, "parts": [m.content]})

        # Start chat with history
        chat = self.client.start_chat(history=history)

        # Last message is the current one
        last_msg = messages[-1].content if messages else ""
        if system:
            last_msg = f"{system}\n\n{last_msg}"

        if stream:
            response = chat.send_message(last_msg, stream=True)
            for chunk in response:
                if self._stop_flag:
                    break
                if chunk.text:
                    yield chunk.text
        else:
            response = chat.send_message(last_msg)
            return response.text

    def list_models(self) -> List[str]:
        """List available Gemini models."""
        if not self.genai:
            return self.MODELS

        try:
            models = []
            for m in self.genai.list_models():
                if "generateContent" in m.supported_generation_methods:
                    name = m.name.replace("models/", "")
                    if "gemini" in name:
                        models.append(name)
            return sorted(models, reverse=True) if models else self.MODELS
        except Exception:
            return self.MODELS

    def vision(self, image_path: str, prompt: str) -> str:
        """Analyze image with Gemini vision."""
        if not self.client:
            raise ValueError("Gemini API key not configured")

        import PIL.Image

        img = PIL.Image.open(image_path)

        # Use vision-capable model
        vision_model = self.genai.GenerativeModel("gemini-2.5-flash")
        response = vision_model.generate_content([prompt, img])
        return response.text

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
            raise ValueError("Gemini API key not configured")

        # Convert tools to Gemini format
        gemini_tools = None
        if tools:
            tool_declarations = self._convert_tools_to_gemini(tools)
            gemini_tools = [self.genai.protos.Tool(
                function_declarations=[
                    self.genai.protos.FunctionDeclaration(**t) for t in tool_declarations
                ]
            )]

        # Rebuild client with tools
        model = self.genai.GenerativeModel(
            self.model,
            tools=gemini_tools
        )

        # Convert messages to Gemini format
        history = []
        for m in messages[:-1]:  # All but last
            role = "user" if m.get("role") == "user" else "model"
            content = m.get("content", "")
            history.append({"role": role, "parts": [content]})

        # Start chat with history
        chat = model.start_chat(history=history)

        # Last message is the current one
        last_msg = messages[-1].get("content", "") if messages else ""
        if system:
            last_msg = f"{system}\n\n{last_msg}"

        response = chat.send_message(last_msg)

        # Convert Gemini response to a consistent format
        result = {
            "message": {
                "content": "",
                "tool_calls": []
            }
        }

        for part in response.parts:
            if hasattr(part, 'text') and part.text:
                result["message"]["content"] = part.text
            elif hasattr(part, 'function_call'):
                fc = part.function_call
                result["message"]["tool_calls"].append({
                    "function": {
                        "name": fc.name,
                        "arguments": dict(fc.args)
                    }
                })

        # If no tool calls, set to None for consistency
        if not result["message"]["tool_calls"]:
            result["message"]["tool_calls"] = None

        return type('Response', (), result)()

    def get_config_help(self) -> str:
        return """Google Gemini

1. Get API key: https://aistudio.google.com/apikey
2. Set environment variable:
   export GOOGLE_API_KEY=...

Or add to ~/.jarvis/.env:
   GOOGLE_API_KEY=..."""
