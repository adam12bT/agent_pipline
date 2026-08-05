"""
Abstract interface every LLM provider backend must implement.

This exists so agents/*.py can call `provider.complete(prompt)` without
caring whether the active backend is AnythingLLM's RAG-grounded chat, a
direct OpenAI call, direct Anthropic, direct Groq, or something added
later (a house-hosted vLLM/Ollama endpoint, etc). Swapping providers is
then a one-line env var change (LLM_PROVIDER=...) instead of touching
every call site — see providers/factory.py.
"""

from abc import ABC, abstractmethod
from typing import Optional


class LLMProviderError(RuntimeError):
    """Raised when a provider fails to produce a completion, or is
    misconfigured (missing API key, unknown model, etc). Callers can
    catch this one type regardless of which backend is active, instead
    of needing to know each provider's underlying exception types."""


class LLMProvider(ABC):
    """Common contract for a chat/completion backend.

    `complete()` is deliberately the only required method: any backend,
    RAG-grounded or not, reduces to "take a prompt (+ optional system
    instructions), return text". Anything provider-specific — workspace
    slugs, model names, retrieval mode, temperature, max tokens — is
    threaded through **kwargs and/or the provider's own __init__, so
    this ABC never has to change just because one backend needs a new
    knob that the others don't have.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Short identifier for logging/error messages, e.g. 'openai'."""
        raise NotImplementedError

    @abstractmethod
    def complete(self, prompt: str, *, system: Optional[str] = None, **kwargs) -> str:
        """Send `prompt` to the underlying LLM and return its text response.

        Args:
            prompt: the user-turn content.
            system: optional system/instruction text. Providers without a
                native system-prompt slot (e.g. AnythingLLM's chat
                endpoint) should fold it into the prompt rather than
                silently dropping it.
            **kwargs: provider-specific options (workspace_slug, mode,
                temperature, max_tokens, model override, ...). A provider
                must ignore kwargs it doesn't understand rather than
                raising, so callers can pass a common superset of options
                without knowing which backend is active.

        Returns:
            The model's text response.

        Raises:
            LLMProviderError: on any failure — auth, network, malformed
                response, or a required kwarg missing (e.g. AnythingLLM
                requires workspace_slug).
        """
        raise NotImplementedError

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__} name={self.name!r}>"
