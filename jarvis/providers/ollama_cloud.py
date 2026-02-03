"""Ollama Cloud provider (remote Ollama API)."""

import os
from typing import Optional

from .ollama import OllamaProvider


class OllamaCloudProvider(OllamaProvider):
    """Ollama Cloud provider using remote base URL and optional auth."""

    name = "ollama_cloud"

    def __init__(
        self,
        model: str = None,
        api_key: str = None,
        base_url: str = None,
        **kwargs
    ):
        self.api_key = api_key or os.getenv("OLLAMA_CLOUD_API_KEY")
        self.base_url = base_url or os.getenv("OLLAMA_CLOUD_BASE_URL")

        headers = kwargs.get("headers") or {}
        if self.api_key:
            headers = {**headers, "Authorization": f"Bearer {self.api_key}"}

        super().__init__(
            model=model,
            base_url=self.base_url or "http://localhost:11434",
            headers=headers if headers else None,
            **kwargs
        )

    def is_configured(self) -> bool:
        return bool(self.base_url)
