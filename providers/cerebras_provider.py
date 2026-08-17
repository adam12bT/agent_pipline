"""Cerebras cloud provider for the pipeline's direct LLM calls.

The Cerebras API is OpenAI-compatible, but uses ``max_completion_tokens``
for its token reservation. Keeping this provider behind the common
``LLMProvider`` interface lets extraction, research helpers, generation, and
quality evaluation switch to Cerebras through ``LLM_PROVIDER=cerebras``.

Environment variables:
    CEREBRAS_API_KEY                    required
    CEREBRAS_MODEL                      default ``gpt-oss-120b``
    CEREBRAS_BASE_URL                   default ``https://api.cerebras.ai/v1``
    CEREBRAS_REASONING_EFFORT           default ``low``
    CEREBRAS_MAX_RETRIES                retries after the first request
    CEREBRAS_RETRY_BASE_SECONDS         exponential-backoff starting delay
    CEREBRAS_RETRY_MAX_SECONDS          maximum delay between attempts
    CEREBRAS_MAX_RETRY_AFTER_SECONDS    fail fast above this server delay
    CEREBRAS_RETRY_JITTER_SECONDS       random jitter added to delays
    CEREBRAS_MIN_INTERVAL_SECONDS       minimum spacing between requests
    CEREBRAS_TIMEOUT_SECONDS            HTTP request timeout
"""

import logging
import os
import random
import re
import threading
import time
from typing import Optional

import requests

from providers.base import LLMProvider, LLMProviderError

logger = logging.getLogger(__name__)

DEFAULT_MODEL = "gpt-oss-120b"
DEFAULT_BASE_URL = "https://api.cerebras.ai/v1"
_RETRYABLE_STATUSES = {408, 429, 500, 502, 503, 504}


class CerebrasProvider(LLMProvider):
    def __init__(
        self,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        base_url: Optional[str] = None,
    ):
        self._api_key = api_key or os.environ.get("CEREBRAS_API_KEY")
        if not self._api_key:
            raise LLMProviderError("CEREBRAS_API_KEY is not set.")

        self._model = model or os.environ.get("CEREBRAS_MODEL", DEFAULT_MODEL)
        self._base_url = (
            base_url or os.environ.get("CEREBRAS_BASE_URL", DEFAULT_BASE_URL)
        ).rstrip("/")
        self._reasoning_effort = (
            os.environ.get("CEREBRAS_REASONING_EFFORT", "low").strip().lower()
        )
        self._max_retries = max(
            0, int(os.environ.get("CEREBRAS_MAX_RETRIES", "2"))
        )
        self._retry_base_seconds = max(
            0.1, float(os.environ.get("CEREBRAS_RETRY_BASE_SECONDS", "2"))
        )
        self._retry_max_seconds = max(
            self._retry_base_seconds,
            float(os.environ.get("CEREBRAS_RETRY_MAX_SECONDS", "60")),
        )
        self._max_retry_after_seconds = max(
            0.0,
            float(os.environ.get("CEREBRAS_MAX_RETRY_AFTER_SECONDS", "60")),
        )
        self._retry_jitter_seconds = max(
            0.0, float(os.environ.get("CEREBRAS_RETRY_JITTER_SECONDS", "1"))
        )
        self._min_interval_seconds = max(
            0.0, float(os.environ.get("CEREBRAS_MIN_INTERVAL_SECONDS", "2"))
        )
        self._timeout_seconds = max(
            1.0, float(os.environ.get("CEREBRAS_TIMEOUT_SECONDS", "120"))
        )
        self._request_gate = threading.Lock()
        self._next_request_at = 0.0

    @property
    def name(self) -> str:
        return "cerebras"

    @staticmethod
    def _seconds(value: str | None) -> float | None:
        if not value:
            return None
        match = re.search(r"([0-9]+(?:\.[0-9]+)?)", str(value))
        return max(0.0, float(match.group(1))) if match else None

    def _server_retry_delay(self, response: requests.Response) -> float | None:
        for header in (
            "Retry-After",
            "x-ratelimit-reset-tokens-minute",
        ):
            delay = self._seconds(response.headers.get(header))
            if delay is not None:
                return delay

        match = re.search(
            r"try again in\s+([0-9]+(?:\.[0-9]+)?)s",
            response.text or "",
            re.IGNORECASE,
        )
        return float(match.group(1)) if match else None

    def _retry_delay(
        self, attempt: int, response: requests.Response | None
    ) -> float | None:
        server_delay = (
            self._server_retry_delay(response) if response is not None else None
        )
        if server_delay is not None:
            if server_delay > self._max_retry_after_seconds:
                logger.warning(
                    "Cerebras requested a %.1fs retry window, above the %.1fs "
                    "configured maximum; failing fast",
                    server_delay,
                    self._max_retry_after_seconds,
                )
                return None
            base_delay = server_delay
        else:
            base_delay = min(
                self._retry_max_seconds,
                self._retry_base_seconds * (2**attempt),
            )
        return min(
            self._retry_max_seconds,
            base_delay + random.uniform(0.0, self._retry_jitter_seconds),
        )

    def _wait_for_request_window(self) -> None:
        delay = self._next_request_at - time.monotonic()
        if delay > 0:
            logger.info("Waiting %.1fs for the Cerebras request window", delay)
            time.sleep(delay)

    def _reserve_request_window(self, delay: float) -> None:
        self._next_request_at = max(
            self._next_request_at,
            time.monotonic() + delay,
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

        payload = {
            "model": request_model,
            "messages": messages,
            "temperature": temperature,
            "max_completion_tokens": max_tokens,
            "stream": False,
        }
        response_format = kwargs.get("response_format")
        if response_format is not None:
            payload["response_format"] = response_format
        reasoning_effort = kwargs.get("reasoning_effort", self._reasoning_effort)
        if reasoning_effort:
            payload["reasoning_effort"] = reasoning_effort

        total_attempts = self._max_retries + 1
        last_error: Exception | None = None
        response = None

        for attempt in range(total_attempts):
            try:
                with self._request_gate:
                    self._wait_for_request_window()
                    response = requests.post(
                        f"{self._base_url}/chat/completions",
                        headers={
                            "Authorization": f"Bearer {self._api_key}",
                            "Content-Type": "application/json",
                        },
                        json=payload,
                        timeout=self._timeout_seconds,
                    )
                    if response.ok:
                        data = response.json()
                        content = data["choices"][0]["message"].get("content")
                        if not content:
                            raise ValueError("Cerebras returned an empty completion")
                        self._reserve_request_window(self._min_interval_seconds)
                        return content

                response.raise_for_status()
            except (requests.Timeout, requests.ConnectionError) as exc:
                last_error = exc
            except requests.HTTPError as exc:
                last_error = exc
                if response is None or response.status_code not in _RETRYABLE_STATUSES:
                    break
            except (KeyError, IndexError, TypeError, ValueError) as exc:
                last_error = exc
                break
            except requests.RequestException as exc:
                last_error = exc
                break

            if attempt >= self._max_retries:
                break
            delay = self._retry_delay(attempt, response)
            if delay is None:
                break
            with self._request_gate:
                self._reserve_request_window(delay)
            logger.warning(
                "Cerebras completion attempt %d/%d failed%s; retrying in %.1fs",
                attempt + 1,
                total_attempts,
                f" with HTTP {response.status_code}" if response is not None else "",
                delay,
            )

        detail = str(last_error) if last_error else "unknown error"
        if response is not None and response.text:
            detail = f"HTTP {response.status_code}: {response.text[:500]}"
        logger.error(
            "Cerebras completion failed after %d attempt(s) (model=%r): %s",
            min(total_attempts, attempt + 1),
            request_model,
            detail,
        )
        raise LLMProviderError(f"Cerebras completion failed: {detail}") from last_error
