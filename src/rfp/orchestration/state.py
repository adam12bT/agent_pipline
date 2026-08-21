"""Namespaced internal state and the stable public API projection."""

import operator
import time
from typing import Annotated, Any, TypedDict


class PipelineState(TypedDict, total=False):
    request: dict[str, Any]
    verifier: dict[str, Any]
    extraction: dict[str, Any]
    research: dict[str, Any]
    generation: dict[str, Any]
    security: dict[str, Any]
    quality: dict[str, Any]
    telemetry: dict[str, Any]
    control: dict[str, Any]
    errors: Annotated[list[str], operator.add]


def initial_pipeline_state(
    tender_file_path: str,
    response_template_file_path: str | None = None,
    *,
    run_id: str | None = None,
) -> PipelineState:
    started_at = time.time()
    return {
        "request": {
            "run_id": run_id,
            "tender_file_path": tender_file_path,
            "response_template_file_path": response_template_file_path,
        },
        "control": {"status": "running"},
        "telemetry": {
            "started_at_epoch": started_at,
            "updated_at_epoch": started_at,
            "total_duration_seconds": 0.0,
            "agents": {},
            "llm_usage": {
                "request_count": 0,
                "successful_calls": 0,
                "failed_calls": 0,
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_tokens": 0,
                "duration_seconds": 0.0,
                "providers": {},
                "calls": [],
            },
            "notes": [
                "Token counts include only providers that return usage metadata.",
                "GPT Researcher token totals are captured through LangChain when "
                "its active model integration emits usage metadata.",
            ],
        },
        "errors": [],
    }


def flatten_pipeline_state(state: PipelineState) -> dict[str, Any]:
    """Preserve the original frontend/API response while internals are isolated."""
    flat: dict[str, Any] = {}
    for namespace in (
        "request",
        "verifier",
        "extraction",
        "research",
        "generation",
        "security",
        "quality",
    ):
        flat.update(state.get(namespace) or {})
    flat["status"] = (state.get("control") or {}).get("status", "running")
    flat["errors"] = list(state.get("errors") or [])
    return flat
