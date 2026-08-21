"""Client adapter for the standalone structured-document extraction service."""

from __future__ import annotations

import logging
import os
import random
import time
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from typing import Any

import requests

logger = logging.getLogger(__name__)


class ExtractorServiceError(RuntimeError):
    """Raised when extraction or downstream AnythingLLM indexing fails."""


class ExtractorClient:
    def __init__(
        self,
        base_url: str | None = None,
        api_key: str | None = None,
        timeout_seconds: float | None = None,
    ):
        self.base_url = (
            base_url
            or os.environ.get("EXTRACTOR_BASE_URL")
            or "http://localhost:8007"
        ).rstrip("/")
        self.api_key = api_key or os.environ.get("EXTRACTOR_API_KEY", "")
        self.timeout_seconds = timeout_seconds or float(
            os.environ.get("EXTRACTOR_TIMEOUT_SECONDS", "900")
        )
        self.max_retries = max(1, int(os.environ.get("EXTRACTOR_MAX_RETRIES", "6")))
        self.retry_backoff_seconds = max(
            0.0, float(os.environ.get("EXTRACTOR_RETRY_BACKOFF_SECONDS", "5"))
        )
        self.retry_max_seconds = max(
            self.retry_backoff_seconds,
            float(os.environ.get("EXTRACTOR_RETRY_MAX_SECONDS", "60")),
        )
        self.retry_jitter_seconds = max(
            0.0, float(os.environ.get("EXTRACTOR_RETRY_JITTER_SECONDS", "1"))
        )

    def _headers(self) -> dict[str, str]:
        return (
            {"Authorization": f"Bearer {self.api_key}"}
            if self.api_key
            else {}
        )

    @staticmethod
    def _error_message(response: requests.Response, body: Any = None) -> str:
        if isinstance(body, dict):
            error = body.get("error")
            if isinstance(error, dict):
                return error.get("message") or error.get("code") or str(error)
            if error:
                return str(error)
            if body.get("detail"):
                return str(body["detail"])
        return response.text[:500] or response.reason or "unknown extractor error"

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
                return max(
                    0.0,
                    (retry_at - datetime.now(timezone.utc)).total_seconds(),
                )
            except (TypeError, ValueError, OverflowError):
                logger.warning(
                    "Extractor returned an invalid Retry-After header: %r", value
                )
                return None

    def process_and_index(self, file_path: str, workspace_slug: str) -> dict:
        """Extract a file and index its structured content into a workspace."""
        url = f"{self.base_url}/v1/extract-and-index"
        last_error: Exception | None = None

        for attempt in range(self.max_retries):
            response = None
            try:
                # Reopen on every attempt; a consumed multipart stream cannot
                # safely be reused after a network failure.
                with open(file_path, "rb") as file_obj:
                    response = requests.post(
                        url,
                        files={
                            "file": (
                                os.path.basename(file_path),
                                file_obj,
                                "application/octet-stream",
                            )
                        },
                        data={"workspace_slug": workspace_slug},
                        headers=self._headers(),
                        timeout=self.timeout_seconds,
                    )

                try:
                    body = response.json()
                except ValueError:
                    body = None

                if response.ok:
                    if not isinstance(body, dict):
                        raise ExtractorServiceError("Extractor returned a non-JSON response.")
                    if not body.get("success", False):
                        raise ExtractorServiceError(self._error_message(response, body))
                    logger.info(
                        "Extractor indexed %r into workspace %r",
                        os.path.basename(file_path),
                        workspace_slug,
                    )
                    return body

                error = ExtractorServiceError(
                    f"Extractor returned HTTP {response.status_code}: "
                    f"{self._error_message(response, body)}"
                )
                if response.status_code not in {408, 429, 500, 502, 503, 504}:
                    raise error
                last_error = error
            except (requests.Timeout, requests.ConnectionError) as exc:
                last_error = ExtractorServiceError(
                    f"Could not reach extractor at {self.base_url}: {exc}"
                )
            except ExtractorServiceError:
                raise
            except requests.RequestException as exc:
                last_error = ExtractorServiceError(f"Extractor request failed: {exc}")

            if attempt + 1 < self.max_retries:
                retry_after = (
                    self._retry_after_seconds(response)
                    if response is not None
                    else None
                )
                delay = retry_after if retry_after is not None else min(
                    self.retry_max_seconds,
                    self.retry_backoff_seconds * (2**attempt),
                )
                delay += random.uniform(0.0, self.retry_jitter_seconds)
                logger.warning(
                    "Extractor attempt %d/%d failed; retrying in %.1fs: %s",
                    attempt + 1,
                    self.max_retries,
                    delay,
                    last_error,
                )
                time.sleep(delay)

        raise last_error or ExtractorServiceError("Extractor request failed.")


def summarize_extractor_response(response: dict) -> dict:
    """Keep UI/audit fields while dropping large block and document arrays."""
    document = response.get("document") or {}
    index_result = response.get("index_result") or {}
    return {
        "success": bool(response.get("success")),
        "document": {
            "filename": document.get("filename"),
            "metadata": document.get("metadata") or {},
            "pages": document.get("pages") or [],
            "warnings": document.get("warnings") or [],
        },
        "index_result": {
            "success": bool(index_result.get("success")),
            "workspace_slug": index_result.get("workspace_slug"),
            "blocks_sent": index_result.get("blocks_sent", 0),
            "skipped_existing": index_result.get("skipped_existing", 0),
            "rolled_back": index_result.get("rolled_back", 0),
            "error": index_result.get("error"),
        },
        "error": response.get("error"),
    }
