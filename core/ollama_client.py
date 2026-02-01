"""Ollama client wrapper for model interactions"""

import ollama
from typing import Generator
import json


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
        """
        Send a chat request to Ollama.

        Args:
            messages: List of message dicts with 'role' and 'content'
            model: Model to use (defaults to self.default_model)
            stream: Whether to stream the response
            system: Optional system prompt to prepend

        Yields/Returns:
            Streamed chunks or complete response
        """
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
                if 'message' in chunk and 'content' in chunk['message']:
                    yield chunk['message']['content']
        else:
            response = self.client.chat(
                model=model,
                messages=messages,
                stream=False
            )
            return response['message']['content']

    def generate(self, prompt: str, model: str = None, stream: bool = True):
        """Simple text generation without chat format."""
        model = model or self.default_model

        if stream:
            response = self.client.generate(model=model, prompt=prompt, stream=True)
            for chunk in response:
                if 'response' in chunk:
                    yield chunk['response']
        else:
            response = self.client.generate(model=model, prompt=prompt, stream=False)
            return response['response']

    def embed(self, text: str, model: str = "nomic-embed-text") -> list[float]:
        """Generate embeddings for text."""
        response = self.client.embeddings(model=model, prompt=text)
        return response['embedding']

    def list_models(self) -> list[str]:
        """List available models."""
        response = self.client.list()
        return [m['name'] for m in response['models']]

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
        return response['message']['content']
