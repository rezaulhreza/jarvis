"""Ollama client wrapper for model interactions"""

import ollama
from typing import Generator


class OllamaClient:
    """Wrapper around Ollama for easy model switching and streaming."""

    def __init__(self, default_model: str = "qwen3:4b"):
        self.default_model = default_model
        self.client = ollama.Client()

    def chat(
        self,
        messages: list[dict],
        model: str = None,
        stream: bool = True,
        system: str = None
    ) -> Generator[str, None, None] | str:
        """Send a chat request to Ollama."""
        model = model or self.default_model

        # Prepend system message if provided
        if system:
            messages = [{"role": "system", "content": system}] + messages

        if stream:
            response = self.client.chat(
                model=model,
                messages=messages,
                stream=True
            )
            for chunk in response:
                # Handle both dict and object access patterns
                if hasattr(chunk, 'message'):
                    content = chunk.message.content if hasattr(chunk.message, 'content') else chunk.message.get('content', '')
                elif isinstance(chunk, dict) and 'message' in chunk:
                    msg = chunk['message']
                    content = msg.get('content', '') if isinstance(msg, dict) else getattr(msg, 'content', '')
                else:
                    content = ''

                if content:
                    yield content
        else:
            response = self.client.chat(
                model=model,
                messages=messages,
                stream=False
            )
            # Handle both dict and object access
            if hasattr(response, 'message'):
                return response.message.content if hasattr(response.message, 'content') else response.message.get('content', '')
            return response['message']['content']

    def generate(self, prompt: str, model: str = None, stream: bool = True):
        """Simple text generation without chat format."""
        model = model or self.default_model

        if stream:
            response = self.client.generate(model=model, prompt=prompt, stream=True)
            for chunk in response:
                content = chunk.get('response', '') if isinstance(chunk, dict) else getattr(chunk, 'response', '')
                if content:
                    yield content
        else:
            response = self.client.generate(model=model, prompt=prompt, stream=False)
            return response.get('response', '') if isinstance(response, dict) else getattr(response, 'response', '')

    def embed(self, text: str, model: str = "nomic-embed-text") -> list[float]:
        """Generate embeddings for text."""
        response = self.client.embeddings(model=model, prompt=text)
        return response.get('embedding', []) if isinstance(response, dict) else getattr(response, 'embedding', [])

    def list_models(self) -> list[str]:
        """List available models."""
        response = self.client.list()

        # Handle new Ollama API format (returns ListResponse with Model objects)
        if hasattr(response, 'models'):
            models = response.models
            return [m.model if hasattr(m, 'model') else m.get('model', m.get('name', str(m))) for m in models]
        elif isinstance(response, dict) and 'models' in response:
            return [m.get('model', m.get('name', '')) for m in response['models']]

        return []

    def vision(
        self,
        image_path: str,
        prompt: str = "Describe this image",
        model: str = "llava"
    ) -> str:
        """Analyze an image with a vision model."""
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
            return response.message.content if hasattr(response.message, 'content') else response.message.get('content', '')
        return response['message']['content']
