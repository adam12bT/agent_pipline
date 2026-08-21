"""Context-local LLM usage collection for one agent-node execution."""

from contextlib import contextmanager
from contextvars import ContextVar
from typing import Any, Iterator


_active_collector: ContextVar[dict[str, Any] | None] = ContextVar(
    "llm_telemetry_collector", default=None
)


def _token_count(value: Any) -> int:
    try:
        return max(0, int(value or 0))
    except (TypeError, ValueError, OverflowError):
        return 0


def _empty_usage() -> dict[str, Any]:
    return {
        "request_count": 0,
        "successful_calls": 0,
        "failed_calls": 0,
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "total_tokens": 0,
        "duration_seconds": 0.0,
        "providers": {},
        "calls": [],
    }


@contextmanager
def collect_llm_usage() -> Iterator[dict[str, Any]]:
    """Collect provider usage emitted in the current execution context."""

    collector = _empty_usage()
    token = _active_collector.set(collector)
    try:
        yield collector
    finally:
        _active_collector.reset(token)


def record_llm_call(
    *,
    provider: str,
    model: str,
    duration_seconds: float,
    request_count: int = 1,
    prompt_tokens: int | None = None,
    completion_tokens: int | None = None,
    total_tokens: int | None = None,
    success: bool,
) -> None:
    """Record one logical completion without changing provider return types."""

    collector = _active_collector.get()
    if collector is None:
        return
    prompt = _token_count(prompt_tokens)
    completion = _token_count(completion_tokens)
    total = _token_count(total_tokens) or (prompt + completion)
    requests = max(1, int(request_count))
    call = {
        "provider": provider,
        "model": model,
        "success": success,
        "request_count": requests,
        "prompt_tokens": prompt,
        "completion_tokens": completion,
        "total_tokens": total,
        "duration_seconds": max(0.0, float(duration_seconds)),
    }
    collector["calls"].append(call)
    collector["request_count"] += requests
    collector["successful_calls" if success else "failed_calls"] += 1
    collector["prompt_tokens"] += prompt
    collector["completion_tokens"] += completion
    collector["total_tokens"] += total
    collector["duration_seconds"] += call["duration_seconds"]
    provider_totals = collector["providers"].setdefault(
        provider,
        {
            "request_count": 0,
            "successful_calls": 0,
            "failed_calls": 0,
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
            "duration_seconds": 0.0,
            "models": [],
        },
    )
    provider_totals["request_count"] += requests
    provider_totals["successful_calls" if success else "failed_calls"] += 1
    provider_totals["prompt_tokens"] += prompt
    provider_totals["completion_tokens"] += completion
    provider_totals["total_tokens"] += total
    provider_totals["duration_seconds"] += call["duration_seconds"]
    if model not in provider_totals["models"]:
        provider_totals["models"].append(model)


def record_llm_usage_aggregate(
    *,
    provider: str,
    model: str,
    prompt_tokens: int | None,
    completion_tokens: int | None,
    total_tokens: int | None,
    source: str,
) -> None:
    """Add exact token totals when a library omits individual call counts."""

    collector = _active_collector.get()
    if collector is None:
        return
    prompt = _token_count(prompt_tokens)
    completion = _token_count(completion_tokens)
    total = _token_count(total_tokens) or (prompt + completion)
    collector["prompt_tokens"] += prompt
    collector["completion_tokens"] += completion
    collector["total_tokens"] += total
    collector["calls"].append(
        {
            "provider": provider,
            "model": model,
            "source": source,
            "usage_aggregate": True,
            "request_count": None,
            "prompt_tokens": prompt,
            "completion_tokens": completion,
            "total_tokens": total,
            "duration_seconds": None,
        }
    )
    provider_totals = collector["providers"].setdefault(
        provider,
        {
            "request_count": 0,
            "successful_calls": 0,
            "failed_calls": 0,
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
            "duration_seconds": 0.0,
            "models": [],
        },
    )
    provider_totals["prompt_tokens"] += prompt
    provider_totals["completion_tokens"] += completion
    provider_totals["total_tokens"] += total
    if model not in provider_totals["models"]:
        provider_totals["models"].append(model)
