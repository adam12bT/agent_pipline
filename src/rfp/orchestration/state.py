"""Namespaced internal state and the stable public API projection."""

import operator
from typing import Annotated, Any, TypedDict


class PipelineState(TypedDict, total=False):
    request: dict[str, Any]
    verifier: dict[str, Any]
    extraction: dict[str, Any]
    research: dict[str, Any]
    generation: dict[str, Any]
    security: dict[str, Any]
    quality: dict[str, Any]
    control: dict[str, Any]
    errors: Annotated[list[str], operator.add]


def initial_pipeline_state(
    tender_file_path: str,
    response_template_file_path: str,
    *,
    run_id: str | None = None,
) -> PipelineState:
    return {
        "request": {
            "run_id": run_id,
            "tender_file_path": tender_file_path,
            "response_template_file_path": response_template_file_path,
        },
        "control": {"status": "running"},
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
