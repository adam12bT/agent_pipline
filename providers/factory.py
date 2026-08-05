"""
Provider selection.

Usage in an agent:

    from providers import get_provider

    provider = get_provider()  # reads LLM_PROVIDER env var
    text = provider.complete(EXTRACTION_PROMPT)

Set LLM_PROVIDER in .env to switch backends without touching any agent
code: LLM_PROVIDER=ollama (default, local/free) | groq (cloud, fast).
"""

import os
from typing import Dict, Optional, Type

from providers.base import LLMProvider, LLMProviderError
from providers.groq_provider import GroqProvider
from providers.ollama_provider import OllamaProvider

_REGISTRY: Dict[str, Type[LLMProvider]] = {
    "ollama": OllamaProvider,
    "groq": GroqProvider,
}

# Instances are cached per provider name — constructing one validates its
# API key (Groq) or resolves its base URL (Ollama), so we don't want to
# redo that on every single agent call.
_cache: Dict[str, LLMProvider] = {}


def get_provider(name: Optional[str] = None) -> LLMProvider:
    """Return the active LLMProvider instance.

    Resolution order: explicit `name` arg > LLM_PROVIDER env var >
    'ollama' (local, no API key required — the friendliest default).
    """
    key = (name or os.environ.get("LLM_PROVIDER", "ollama")).strip().lower()

    if key not in _REGISTRY:
        raise LLMProviderError(
            f"Unknown LLM provider '{key}'. Available: {sorted(_REGISTRY)}. "
            f"Set LLM_PROVIDER to one of these, or call register_provider() "
            f"first if you're adding a custom backend."
        )

    if key not in _cache:
        _cache[key] = _REGISTRY[key]()

    return _cache[key]


def register_provider(name: str, cls: Type[LLMProvider]) -> None:
    """Register a custom LLMProvider subclass under a new name — e.g. for
    OpenAI, Anthropic, or a different self-hosted backend. Call this
    (before get_provider()) from wherever your app does startup wiring."""
    _REGISTRY[name.strip().lower()] = cls


def reset_cache() -> None:
    """Drop cached provider instances. Mainly useful in tests, or after
    changing LLM_PROVIDER / an API key at runtime."""
    _cache.clear()
