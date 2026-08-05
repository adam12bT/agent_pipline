"""
GroqProvider — direct calls to Groq's Chat Completions API (OpenAI-
compatible wire format, Groq's own hosted endpoint). Cloud-hosted, fast
inference — the counterpart to OllamaProvider's local inference.

The repo already depends on GROQ_API_KEY for gpt-researcher (see
.env.example / research_agent.py) — this reuses the same key for direct
chat completions outside of gpt-researcher.

Env vars:
    GROQ_API_KEY   required
    GROQ_MODEL     optional, default 'llama-3.3-70b-versatile'
    GROQ_BASE_URL  optional, default 'https://api.groq.com/openai/v1'
"""

import os
from typing import Optional

import requests

from providers.base import LLMProvider, LLMProviderError

DEFAULT_MODEL = "llama-3.3-70b-versatile"
DEFAULT_BASE_URL = "https://api.groq.com/openai/v1"


class GroqProvider(LLMProvider):
    def __init__(
        self,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        base_url: Optional[str] = None,
    ):
        self._api_key = api_key or os.environ.get("GROQ_API_KEY")
        if not self._api_key:
            raise LLMProviderError("GROQ_API_KEY is not set.")
        self._model = model or os.environ.get("GROQ_MODEL", DEFAULT_MODEL)
        self._base_url = (
            base_url or os.environ.get("GROQ_BASE_URL", DEFAULT_BASE_URL)
        ).rstrip("/")

    @property
    def name(self) -> str:
        return "groq"

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
                f"{self._base_url}/chat/completions",
                headers={"Authorization": f"Bearer {self._api_key}"},
                json={
                    "model": self._model,
                    "messages": messages,
                    "temperature": temperature,
                    "max_tokens": max_tokens,
                },
                timeout=120,
            )
            resp.raise_for_status()
            data = resp.json()
            return data["choices"][0]["message"]["content"]
        except Exception as e:
            raise LLMProviderError(f"Groq completion failed: {e}") from e
