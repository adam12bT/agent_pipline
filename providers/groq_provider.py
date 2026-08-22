"""
GroqProvider — direct calls to Groq's Chat Completions API (OpenAI-
compatible wire format, Groq's own hosted endpoint). Cloud-hosted, fast
inference — the counterpart to OllamaProvider's local inference.

The repo already depends on GROQ_API_KEY for gpt-researcher (see
.env.example / research_agent.py) — this reuses the same key for direct
chat completions outside of gpt-researcher.

Env vars:
    PIPELINE_GROQ_API_KEY preferred for direct pipeline calls
    GROQ_API_KEY          backward-compatible fallback
    GROQ_MODEL     optional, default 'llama-3.3-70b-versatile'
    GROQ_BASE_URL  optional, default 'https://api.groq.com/openai/v1'
    GROQ_MAX_RETRIES                  retry attempts after the first request
    GROQ_RETRY_BASE_SECONDS           exponential-backoff starting delay
    GROQ_RETRY_MAX_SECONDS            maximum delay between attempts
    GROQ_RETRY_JITTER_SECONDS         random jitter added to retry delays
    GROQ_RETRY_SAFETY_SECONDS         extra wait after server Retry-After
    GROQ_MIN_INTERVAL_SECONDS         minimum delay between successful requests
    GROQ_TIMEOUT_SECONDS              HTTP request timeout
"""

import logging
import os
import random
import re
import threading
import time
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from typing import Optional

import requests

from providers.base import LLMProvider, LLMProviderError
from providers.telemetry import record_llm_call

logger = logging.getLogger(__name__)

DEFAULT_MODEL = "llama-3.3-70b-versatile"
DEFAULT_BASE_URL = "https://api.groq.com/openai/v1"
_MODEL_ALIASES = {
    "gpt-oss-120b": "openai/gpt-oss-120b",
    "gpt-oss-20b": "openai/gpt-oss-20b",
}


class GroqProvider(LLMProvider):
    def __init__(
        self,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        base_url: Optional[str] = None,
    ):
        self._api_key = (
            api_key
            or os.environ.get("PIPELINE_GROQ_API_KEY")
            or os.environ.get("GROQ_API_KEY")
        )
        if not self._api_key:
            raise LLMProviderError(
                "PIPELINE_GROQ_API_KEY (or legacy GROQ_API_KEY) is not set."
            )
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
        self._retry_safety_seconds = max(
            0.0, float(os.environ.get("GROQ_RETRY_SAFETY_SECONDS", "5"))
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
            # Groq sometimes provides the precise delay only in the JSON
            # error message (for example, "Please try again in 1.0725s").
            # Recover it so a short TPM window is retried instead of failed.
            match = re.search(
                r"try again in\s+([0-9]+(?:\.[0-9]+)?)s",
                response.text or "",
                re.IGNORECASE,
            )
            return max(0.0, float(match.group(1))) if match else None
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
            return retry_after + self._retry_safety_seconds + random.uniform(
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
        completion_started = time.perf_counter()
        configured_model = kwargs.get("model") or self._model
        request_model = _MODEL_ALIASES.get(configured_model, configured_model)
        response_format = kwargs.get("response_format")
        request_label = str(kwargs.get("request_label") or "unlabelled")
        reasoning_effort = kwargs.get("reasoning_effort")
        include_reasoning = kwargs.get("include_reasoning")
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
                    logger.info(
                        "Groq request label=%s model=%s attempt=%d/%d "
                        "prompt_chars=%d estimated_input_tokens=%d max_output_tokens=%d",
                        request_label,
                        request_model,
                        attempt + 1,
                        total_attempts,
                        len(prompt),
                        max(1, len(prompt) // 4),
                        max_tokens,
                    )
                    payload = {
                        "model": request_model,
                        "messages": messages,
                        "temperature": temperature,
                        "max_tokens": max_tokens,
                    }
                    if response_format is not None:
                        payload["response_format"] = response_format
                    is_gpt_oss = request_model.startswith("openai/gpt-oss-")
                    if is_gpt_oss and reasoning_effort is not None:
                        payload["reasoning_effort"] = reasoning_effort
                    if is_gpt_oss and include_reasoning is not None:
                        payload["include_reasoning"] = include_reasoning
                    response = requests.post(
                        f"{self._base_url}/chat/completions",
                        headers={"Authorization": f"Bearer {self._api_key}"},
                        json=payload,
                        timeout=self._timeout_seconds,
                    )

                    if response.ok:
                        data = response.json()
                        self._reserve_retry_window(self._min_interval_seconds)
                        choice = data["choices"][0]
                        message = choice["message"]
                        completion_metadata = kwargs.get("completion_metadata")
                        if isinstance(completion_metadata, dict):
                            completion_metadata.update(
                                {
                                    "provider": self.name,
                                    "model": request_model,
                                    "finish_reason": choice.get("finish_reason"),
                                }
                            )
                        content = message.get("content")
                        if not content:
                            raise ValueError(
                                "Groq returned an empty completion "
                                f"(finish_reason={choice.get('finish_reason')!r}, "
                                f"reasoning_chars={len(message.get('reasoning') or '')}, "
                                f"max_output_tokens={max_tokens}); the reasoning model "
                                "likely exhausted its completion allowance before "
                                "writing the answer"
                            )
                        usage = data.get("usage") or {}
                        completion_details = usage.get("completion_tokens_details") or {}
                        logger.info(
                            "Groq success label=%s model=%s prompt_tokens=%s "
                            "completion_tokens=%s reasoning_tokens=%s total_tokens=%s "
                            "finish_reason=%s output_chars=%d "
                            "remaining_tpm=%s reset_tpm=%s",
                            request_label,
                            request_model,
                            usage.get("prompt_tokens", "unknown"),
                            usage.get("completion_tokens", "unknown"),
                            completion_details.get("reasoning_tokens", "unknown"),
                            usage.get("total_tokens", "unknown"),
                            choice.get("finish_reason", "unknown"),
                            len(content),
                            response.headers.get(
                                "x-ratelimit-remaining-tokens", "unknown"
                            ),
                            response.headers.get("x-ratelimit-reset-tokens", "unknown"),
                        )
                        record_llm_call(
                            provider=self.name,
                            model=request_model,
                            duration_seconds=time.perf_counter() - completion_started,
                            request_count=attempt + 1,
                            prompt_tokens=usage.get("prompt_tokens"),
                            completion_tokens=usage.get("completion_tokens"),
                            total_tokens=usage.get("total_tokens"),
                            success=True,
                        )
                        return content

                    if response.status_code == 429:
                        try:
                            error_message = response.json().get("error", {}).get(
                                "message", response.text
                            )
                        except (TypeError, ValueError):
                            error_message = response.text
                        logger.warning(
                            "Groq 429 label=%s model=%s prompt_chars=%d "
                            "max_output_tokens=%d retry_after=%s remaining_tpm=%s "
                            "reset_tpm=%s error=%s",
                            request_label,
                            request_model,
                            len(prompt),
                            max_tokens,
                            response.headers.get("Retry-After", "body/backoff"),
                            response.headers.get(
                                "x-ratelimit-remaining-tokens", "unknown"
                            ),
                            response.headers.get("x-ratelimit-reset-tokens", "unknown"),
                            str(error_message)[:500],
                        )

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
                "Groq completion label=%s attempt %d/%d failed%s; "
                "retrying in %.1fs: %s",
                request_label,
                attempt + 1,
                total_attempts,
                f" with HTTP {response.status_code}" if response is not None else "",
                delay,
                last_error,
            )

        detail = str(last_error) if last_error else "unknown error"
        if response is not None and not response.ok and response.text:
            detail = f"HTTP {response.status_code}: {response.text[:500]}"
        logger.error(
            "Groq completion failed label=%s after %d attempt(s) (model=%r): %s",
            request_label,
            min(total_attempts, attempt + 1),
            request_model,
            detail,
        )
        record_llm_call(
            provider=self.name,
            model=request_model,
            duration_seconds=time.perf_counter() - completion_started,
            request_count=min(total_attempts, attempt + 1),
            success=False,
        )
        raise LLMProviderError(f"Groq completion failed: {detail}") from last_error
