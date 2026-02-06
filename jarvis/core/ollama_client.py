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

    def get_model_info(self, model: str = None) -> dict:
        """Get model information including context length.

        Returns dict with keys like 'context_length', 'parameter_size', etc.
        """
        model = model or self.default_model
        try:
            response = self.client.show(model)

            info = {}

            # Extract from model_info (newer Ollama versions)
            if hasattr(response, 'model_info') and response.model_info:
                model_info = response.model_info
                # Context length is often in various keys
                for key in model_info:
                    if 'context' in key.lower():
                        info['context_length'] = model_info[key]
                        break

            # Extract from parameters string (e.g., "num_ctx 4096")
            if hasattr(response, 'parameters') and response.parameters:
                params = response.parameters
                if isinstance(params, str):
                    for line in params.split('\n'):
                        if 'num_ctx' in line:
                            try:
                                info['context_length'] = int(line.split()[-1])
                            except (ValueError, IndexError):
                                pass

            # Extract from modelfile
            if hasattr(response, 'modelfile') and response.modelfile:
                for line in response.modelfile.split('\n'):
                    if 'num_ctx' in line.lower():
                        try:
                            # Parse "PARAMETER num_ctx 131072" format
                            parts = line.split()
                            for i, part in enumerate(parts):
                                if part.lower() == 'num_ctx' and i + 1 < len(parts):
                                    info['context_length'] = int(parts[i + 1])
                                    break
                        except (ValueError, IndexError):
                            pass

            # Also try dict access for older Ollama versions
            if isinstance(response, dict):
                if 'model_info' in response:
                    for key, val in response['model_info'].items():
                        if 'context' in key.lower():
                            info['context_length'] = val
                            break
                if 'parameters' in response and isinstance(response['parameters'], str):
                    for line in response['parameters'].split('\n'):
                        if 'num_ctx' in line:
                            try:
                                info['context_length'] = int(line.split()[-1])
                            except (ValueError, IndexError):
                                pass

            return info
        except Exception as e:
            return {'error': str(e)}

    def get_context_length(self, model: str = None) -> int | None:
        """Get the context length for a model.

        Returns context length in tokens, or None if unknown.
        """
        info = self.get_model_info(model)
        return info.get('context_length')
