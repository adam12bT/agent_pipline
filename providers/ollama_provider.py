"""
OllamaProvider — direct calls to a local (or self-hosted) Ollama server's
/api/chat endpoint. No API key. The repo already runs Ollama for
embeddings (see .env.example: OLLAMA_BASE_URL, EMBEDDING=ollama:...) —
this reuses the same server for chat completions on the generation side.

Env vars:
    OLLAMA_BASE_URL  optional, default 'http://localhost:11434'
    OLLAMA_MODEL     optional, default 'llama3.1'
                      (must already be pulled on the server: `ollama pull llama3.1`)
"""

import os
from typing import Optional

import requests

from providers.base import LLMProvider, LLMProviderError

DEFAULT_MODEL = "llama3.1"
DEFAULT_BASE_URL = "http://localhost:11434"


class OllamaProvider(LLMProvider):
    def __init__(self, model: Optional[str] = None, base_url: Optional[str] = None):
        self._model = model or os.environ.get("OLLAMA_MODEL", DEFAULT_MODEL)
        self._base_url = (
            base_url or os.environ.get("OLLAMA_BASE_URL", DEFAULT_BASE_URL)
        ).rstrip("/")

    @property
    def name(self) -> str:
        return "ollama"

    def complete(
        self,
        prompt: str,
        *,
        system: Optional[str] = None,
        temperature: float = 0.2,
        max_tokens: int = 4096,
        **kwargs,
    ) -> str:
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        try:
            resp = requests.post(
                f"{self._base_url}/api/chat",
                json={
                    "model": self._model,
                    "messages": messages,
                    "stream": False,
                    "options": {
                        "temperature": temperature,
                        "num_predict": max_tokens,
                    },
                },
                # Local inference can be slow — especially the first call
                # after the server has to load the model into memory.
                timeout=300,
            )
            resp.raise_for_status()
            data = resp.json()
            return data.get("message", {}).get("content", "")
        except requests.exceptions.ConnectionError as e:
            raise LLMProviderError(
                f"Could not reach Ollama at {self._base_url} — is the "
                f"server running (`ollama serve`)? Original error: {e}"
            ) from e
        except Exception as e:
            raise LLMProviderError(f"Ollama completion failed: {e}") from e
