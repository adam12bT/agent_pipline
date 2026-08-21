"""
Thin Python client for the stripped-down AnythingLLM server (the Node.js
backend in ../anything-llm-lightweight/server).

Routes and request/response shapes below were confirmed directly against
that server's source code (server/endpoints/api/workspace/index.js and
server/endpoints/api/document/index.js) — not guessed.

NOTE: API key auth was disabled on that server fork (see validApiKey.js),
so no Authorization header is required here. If you re-enable auth on
the server later, add the Bearer header back in _headers().
"""

import logging
import os
import random
import threading
import time
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime

import requests

logger = logging.getLogger(__name__)

ANYTHINGLLM_BASE_URL = os.environ.get("ANYTHINGLLM_BASE_URL", "http://localhost:3001/api")

# Prevent concurrent pipeline nodes from overwhelming a hosted AnythingLLM.
_VECTOR_REQUEST_GATE = threading.Lock()
_VECTOR_NEXT_REQUEST_AT = 0.0


def _env_int(name: str, default: int) -> int:
    try:
        return max(0, int(os.environ.get(name, str(default))))
    except (TypeError, ValueError):
        logger.warning("Invalid %s; using %d", name, default)
        return default


def _env_float(name: str, default: float) -> float:
    try:
        return max(0.0, float(os.environ.get(name, str(default))))
    except (TypeError, ValueError):
        logger.warning("Invalid %s; using %.2f", name, default)
        return default


class AnythingLLMClient:
    def __init__(self, base_url: str | None = None):
        # Resolve this at instance creation because local CLIs load .env after
        # modules have already been imported.
        self.base_url = (
            base_url
            or os.environ.get("ANYTHINGLLM_BASE_URL")
            or ANYTHINGLLM_BASE_URL
        ).rstrip("/")
        self.timeout_seconds = _env_float("ANYTHINGLLM_TIMEOUT_SECONDS", 30.0)
        self.max_retries = _env_int("ANYTHINGLLM_MAX_RETRIES", 3)
        self.retry_base_seconds = _env_float("ANYTHINGLLM_RETRY_BASE_SECONDS", 2.0)
        self.retry_max_seconds = _env_float("ANYTHINGLLM_RETRY_MAX_SECONDS", 60.0)
        self.retry_jitter_seconds = _env_float("ANYTHINGLLM_RETRY_JITTER_SECONDS", 1.0)
        self.vector_min_interval_seconds = _env_float(
            "ANYTHINGLLM_VECTOR_MIN_INTERVAL_SECONDS", 1.0
        )

    def _headers(self):
        # Auth disabled on the server fork — nothing to add here for now.
        return {}

    def _raise_for_status(self, resp: requests.Response, action: str) -> None:
        """Log HTTP failures at the point they actually happen, with the
        method/URL/status that caused them, before re-raising. Every
        caller in this file already wraps its request in raise_for_status()
        — this is a shared choke point so that context isn't lost by the
        time a broad `except Exception` further up the call stack catches
        it (agents log the *fact* of failure, not always the HTTP detail)."""
        try:
            resp.raise_for_status()
        except requests.HTTPError:
            logger.warning(
                "%s failed: %s %s -> %d %s",
                action, resp.request.method, resp.request.url,
                resp.status_code, resp.text[:300],
            )
            raise

    def _request(self, method: str, url: str, action: str, **kwargs) -> requests.Response:
        """Send a retryable metadata/chat request without duplicating work.

        Uploads are intentionally excluded because replaying a multipart upload
        can duplicate indexing unless the remote endpoint supplies an idempotency
        key. The extractor owns idempotent document indexing for pipeline files.
        """
        retryable_statuses = {408, 429, 500, 502, 503, 504}
        last_error = None
        kwargs.setdefault("headers", self._headers())
        kwargs.setdefault("timeout", self.timeout_seconds)

        for attempt in range(self.max_retries + 1):
            response = None
            try:
                response = requests.request(method, url, **kwargs)
                if response.status_code not in retryable_statuses:
                    self._raise_for_status(response, action)
                    return response
                last_error = requests.HTTPError(
                    f"HTTP {response.status_code}", response=response
                )
            except (requests.Timeout, requests.ConnectionError) as exc:
                last_error = exc
            except requests.RequestException:
                raise

            if attempt >= self.max_retries:
                if response is not None:
                    self._raise_for_status(response, action)
                raise last_error

            retry_after = (
                response.headers.get("Retry-After") if response is not None else None
            )
            delay = self._retry_delay(attempt, retry_after)
            logger.warning(
                "%s failed%s; retrying in %.2fs (%d/%d)",
                action,
                f" with HTTP {response.status_code}" if response is not None else "",
                delay,
                attempt + 1,
                self.max_retries,
            )
            time.sleep(delay)

        raise RuntimeError(f"{action} retry loop exited unexpectedly")

    def get_workspace(self, slug: str) -> dict | None:
        """
        GET /v1/workspace/:slug — NOTE: despite the name, this returns a
        LIST under the "workspace" key (confirmed from the server source),
        not a single object. Empty list = doesn't exist. Returns the first
        match dict, or None if not found.
        """
        logger.debug("GET workspace %r", slug)
        resp = self._request(
            "GET",
            f"{self.base_url}/v1/workspace/{slug}",
            f"get_workspace({slug!r})",
        )
        matches = resp.json().get("workspace", [])
        return matches[0] if matches else None

    def get_or_create_workspace(self, name: str) -> dict:
        """
        Idempotent workspace creation. IMPORTANT: AnythingLLM's Workspace.new()
        does NOT let you set a custom slug — it always derives the slug from
        `name` via slugify(name, {lower:true}), and if that slug is already
        taken it silently appends a random suffix instead of reusing it
        (confirmed in models/workspace.js). So calling create_workspace()
        twice with the same name creates two DIFFERENT workspaces, not one.

        To make repeated runs safe (e.g. for the shared "company knowledge"
        workspaces that should persist across every tender), always use a
        `name` that is ALREADY a valid slug (lowercase, hyphens, no spaces —
        e.g. "company-past-proposals") so slugify(name) == name, then check
        for an existing workspace at that exact slug before creating.
        """
        existing = self.get_workspace(name)
        if existing:
            return {"workspace": existing, "created": False}

        created = self.create_workspace(name)
        return {"workspace": created["workspace"], "created": True}

    def create_workspace(self, name: str) -> dict:
        """POST /v1/workspace/new -> returns the created workspace, including its slug."""
        logger.debug("POST create workspace %r", name)
        resp = self._request(
            "POST",
            f"{self.base_url}/v1/workspace/new",
            f"create_workspace({name!r})",
            json={"name": name},
        )
        logger.info("Created workspace %r", name)
        return resp.json()

    def upload_document(self, file_path: str, workspace_slug: str) -> dict:
        """
        POST /v1/document/upload (multipart) — uploads a file and, via
        addToWorkspaces, embeds it into the given workspace in one call.
        Returns the document metadata list, including each doc's `location`
        (needed if you ever want to add/remove it from other workspaces later).
        """
        logger.debug("POST upload document %r -> workspace %r", file_path, workspace_slug)
        with open(file_path, "rb") as f:
            files = {"file": (os.path.basename(file_path), f)}
            data = {"addToWorkspaces": workspace_slug}
            resp = requests.post(
                f"{self.base_url}/v1/document/upload",
                files=files,
                data=data,
                headers=self._headers(),
                timeout=120,
            )
        self._raise_for_status(resp, f"upload_document({file_path!r}, {workspace_slug!r})")
        logger.info("Uploaded and embedded %r into workspace %r", os.path.basename(file_path), workspace_slug)
        return resp.json()

    def vector_search(self, workspace_slug: str, query: str, top_n: int = 4,
                       score_threshold: float = 0.5) -> list[dict]:
        """
        POST /v1/workspace/:slug/vector-search — direct similarity search,
        no LLM call. Returns a list of {text, metadata, score, ...} chunks.
        Use this when you just need raw relevant passages (e.g. Extraction agent).
        """
        logger.debug("POST vector-search workspace=%r query=%r", workspace_slug, query)
        url = f"{self.base_url}/v1/workspace/{workspace_slug}/vector-search"
        payload = {"query": query, "topN": top_n, "scoreThreshold": score_threshold}
        retryable_statuses = {408, 429, 500, 502, 503, 504}

        global _VECTOR_NEXT_REQUEST_AT
        with _VECTOR_REQUEST_GATE:
            for attempt in range(self.max_retries + 1):
                wait_for_slot = _VECTOR_NEXT_REQUEST_AT - time.monotonic()
                if wait_for_slot > 0:
                    time.sleep(wait_for_slot)

                try:
                    resp = requests.post(
                        url,
                        json=payload,
                        headers=self._headers(),
                        timeout=self.timeout_seconds,
                    )
                except (requests.Timeout, requests.ConnectionError) as exc:
                    if attempt >= self.max_retries:
                        raise
                    delay = self._retry_delay(attempt, None)
                    _VECTOR_NEXT_REQUEST_AT = time.monotonic() + delay
                    logger.warning(
                        "AnythingLLM vector search transport error for %r (%s); "
                        "retrying in %.2fs (%d/%d)",
                        workspace_slug, type(exc).__name__, delay,
                        attempt + 1, self.max_retries,
                    )
                    continue

                if resp.status_code not in retryable_statuses:
                    _VECTOR_NEXT_REQUEST_AT = time.monotonic() + self.vector_min_interval_seconds
                    self._raise_for_status(resp, f"vector_search({workspace_slug!r})")
                    return resp.json().get("results", [])

                if attempt >= self.max_retries:
                    self._raise_for_status(resp, f"vector_search({workspace_slug!r})")

                delay = self._retry_delay(attempt, resp.headers.get("Retry-After"))
                _VECTOR_NEXT_REQUEST_AT = time.monotonic() + delay
                logger.warning(
                    "AnythingLLM vector search for %r returned HTTP %d; "
                    "retrying in %.2fs (%d/%d)",
                    workspace_slug, resp.status_code, delay,
                    attempt + 1, self.max_retries,
                )

        raise RuntimeError("AnythingLLM vector-search retry loop exited unexpectedly")

    def _retry_delay(self, attempt: int, retry_after: str | None) -> float:
        """Honor Retry-After, otherwise use capped exponential backoff and jitter."""
        if retry_after:
            try:
                return min(self.retry_max_seconds, max(0.0, float(retry_after)))
            except ValueError:
                try:
                    retry_at = parsedate_to_datetime(retry_after)
                    if retry_at.tzinfo is None:
                        retry_at = retry_at.replace(tzinfo=timezone.utc)
                    seconds = (retry_at - datetime.now(timezone.utc)).total_seconds()
                    return min(self.retry_max_seconds, max(0.0, seconds))
                except (TypeError, ValueError, OverflowError):
                    logger.debug("Ignoring invalid Retry-After value %r", retry_after)

        exponential = self.retry_base_seconds * (2 ** attempt)
        jitter = random.uniform(0.0, self.retry_jitter_seconds)
        return min(self.retry_max_seconds, exponential + jitter)

    def chat(self, workspace_slug: str, message: str, mode: str = "query",
              session_id: str = "rfp-pipeline") -> str:
        """
        POST /v1/workspace/:slug/chat — runs the message through the LLM,
        grounded in the workspace's embedded documents (RAG).
        mode="query": only answers using retrieved doc chunks, no chit-chat.
        mode="chat": general LLM knowledge + doc context + rolling history.
        Returns just the text response.
        """
        logger.debug("POST chat workspace=%r mode=%r (%d char message)", workspace_slug, mode, len(message))
        resp = self._request(
            "POST",
            f"{self.base_url}/v1/workspace/{workspace_slug}/chat",
            f"chat({workspace_slug!r}, mode={mode!r})",
            json={"message": message, "mode": mode, "sessionId": session_id},
            timeout=120,
        )
        data = resp.json()
        return data.get("textResponse", "")
