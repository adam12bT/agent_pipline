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
    GROQ_MAX_RETRIES                  retry attempts after the first request
    GROQ_RETRY_BASE_SECONDS           exponential-backoff starting delay
    GROQ_RETRY_MAX_SECONDS            maximum delay between attempts
    GROQ_RETRY_JITTER_SECONDS         random jitter added to retry delays
    GROQ_MIN_INTERVAL_SECONDS         minimum delay between successful requests
    GROQ_TIMEOUT_SECONDS              HTTP request timeout
"""

import logging
import os
import random
import threading
import time
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from typing import Optional

import requests

from providers.base import LLMProvider, LLMProviderError

logger = logging.getLogger(__name__)

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
        self._max_retries = max(0, int(os.environ.get("GROQ_MAX_RETRIES", "1")))
        self._retry_base_seconds = max(
            0.1, float(os.environ.get("GROQ_RETRY_BASE_SECONDS", "2"))
        )
        self._retry_max_seconds = max(
            self._retry_base_seconds,
            float(os.environ.get("GROQ_RETRY_MAX_SECONDS", "60")),
        )
        self._retry_jitter_seconds = max(
            0.0, float(os.environ.get("GROQ_RETRY_JITTER_SECONDS", "1"))
        )
        self._max_retry_after_seconds = max(
            0.0, float(os.environ.get("GROQ_MAX_RETRY_AFTER_SECONDS", "60"))
        )
        self._min_interval_seconds = max(
            0.0, float(os.environ.get("GROQ_MIN_INTERVAL_SECONDS", "30"))
        )
        self._timeout_seconds = max(
            1.0, float(os.environ.get("GROQ_TIMEOUT_SECONDS", "120"))
        )
        # get_provider() caches one instance, so this gate coordinates the
        # parallel Extraction/Research helper calls within this process.
        self._request_gate = threading.Lock()
        self._next_request_at = 0.0

    @property
    def name(self) -> str:
        return "groq"

    @staticmethod
    def _retry_after_seconds(response: requests.Response) -> float | None:
        value = response.headers.get("Retry-After")
        if not value:
            return None
        try:
            return max(0.0, float(value))
        except ValueError:
            try:
                retry_at = parsedate_to_datetime(value)
                if retry_at.tzinfo is None:
                    retry_at = retry_at.replace(tzinfo=timezone.utc)
                return max(0.0, (retry_at - datetime.now(timezone.utc)).total_seconds())
            except (TypeError, ValueError, OverflowError):
                logger.warning("Groq returned an invalid Retry-After header: %r", value)
                return None

    def _reserve_retry_window(self, delay: float) -> None:
        self._next_request_at = max(self._next_request_at, time.monotonic() + delay)

    def _wait_for_retry_window(self) -> None:
        delay = self._next_request_at - time.monotonic()
        if delay > 0:
            logger.info("Waiting %.1fs for the shared Groq rate-limit window", delay)
            time.sleep(delay)

    def _backoff_delay(
        self, attempt: int, response: requests.Response | None
    ) -> float | None:
        retry_after = self._retry_after_seconds(response) if response is not None else None
        if retry_after is not None:
            if retry_after > self._max_retry_after_seconds:
                logger.warning(
                    "Groq requested a %.1fs Retry-After window, above the %.1fs "
                    "configured maximum; failing fast",
                    retry_after,
                    self._max_retry_after_seconds,
                )
                return None
            return retry_after + random.uniform(
                0.0, self._retry_jitter_seconds
            )
        base_delay = min(
            self._retry_max_seconds,
            self._retry_base_seconds * (2**attempt),
        )
        return min(
            self._retry_max_seconds,
            base_delay + random.uniform(0.0, self._retry_jitter_seconds),
        )

    def complete(
        self,
        prompt: str,
        *,
        system: Optional[str] = None,
        temperature: float = 0.2,
        max_tokens: int = 4096,
        **kwargs,
    ) -> str:
        request_model = kwargs.get("model") or self._model
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        last_error: Exception | None = None
        total_attempts = self._max_retries + 1

        for attempt in range(total_attempts):
            response = None
            retry_delay = None
            retry_allowed = True
            try:
                # Serialize direct calls and respect any Retry-After window set
                # by a previous parallel agent call.
                with self._request_gate:
                    self._wait_for_retry_window()
                    logger.debug(
                        "POST Groq chat/completions model=%r attempt=%d/%d (%d char prompt)",
                        request_model,
                        attempt + 1,
                        total_attempts,
                        len(prompt),
                    )
                    response = requests.post(
                        f"{self._base_url}/chat/completions",
                        headers={"Authorization": f"Bearer {self._api_key}"},
                        json={
                            "model": request_model,
                            "messages": messages,
                            "temperature": temperature,
                            "max_tokens": max_tokens,
                        },
                        timeout=self._timeout_seconds,
                    )

                    if response.ok:
                        data = response.json()
                        self._reserve_retry_window(self._min_interval_seconds)
                        return data["choices"][0]["message"]["content"]

                    if (
                        (
                            response.status_code == 429
                            or response.status_code in {408, 500, 502, 503, 504}
                        )
                        and attempt < self._max_retries
                    ):
                        # Reserve while still holding the shared gate so a
                        # parallel caller cannot slip in before this retry
                        # window becomes visible.
                        retry_delay = self._backoff_delay(attempt, response)
                        if retry_delay is None:
                            retry_allowed = False
                        else:
                            self._reserve_retry_window(retry_delay)
                    response.raise_for_status()
            except requests.HTTPError as exc:
                last_error = exc
                status = response.status_code if response is not None else None
                retryable = status == 429 or status in {408, 500, 502, 503, 504}
                if not retryable or not retry_allowed or attempt >= self._max_retries:
                    break
            except (requests.Timeout, requests.ConnectionError) as exc:
                last_error = exc
                if attempt >= self._max_retries:
                    break
            except (KeyError, IndexError, TypeError, ValueError) as exc:
                last_error = exc
                break
            except requests.RequestException as exc:
                last_error = exc
                break

            delay = retry_delay or self._backoff_delay(attempt, response)
            if delay is None:
                break
            if retry_delay is None:
                with self._request_gate:
                    self._reserve_retry_window(delay)
            logger.warning(
                "Groq completion attempt %d/%d failed%s; retrying in %.1fs: %s",
                attempt + 1,
                total_attempts,
                f" with HTTP {response.status_code}" if response is not None else "",
                delay,
                last_error,
            )

        detail = str(last_error) if last_error else "unknown error"
        if response is not None and response.text:
            detail = f"HTTP {response.status_code}: {response.text[:500]}"
        logger.error(
            "Groq completion failed after %d attempt(s) (model=%r): %s",
            min(total_attempts, attempt + 1),
            request_model,
            detail,
        )
        raise LLMProviderError(f"Groq completion failed: {detail}") from last_error
