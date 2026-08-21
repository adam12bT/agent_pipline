"""Compatibility helpers for recorded pre-migration pipeline responses."""

from typing import Any

from rfp.orchestration.state import flatten_pipeline_state


_NAMESPACE_FIELDS = {
    "request": (
        "run_id",
        "tender_file_path",
        "response_template_file_path",
    ),
    "verifier": (
        "is_verified",
        "verification_errors",
        "workspace_slug",
        "response_template_workspace_slug",
        "document_processing",
        "response_template_processing",
    ),
    "extraction": ("requirements",),
    "research": ("research_summary",),
    "generation": (
        "draft_proposal",
        "generation_evidence",
        "generation_attempts",
    ),
    "security": ("security_passed", "security_report"),
    "quality": ("quality_passed", "quality_report"),
}


def namespace_legacy_state(flat_state: dict[str, Any]) -> dict[str, Any]:
    """Map a recorded flat state to the new internal namespaced representation."""
    namespaced = {
        namespace: {
            field: flat_state[field]
            for field in fields
            if field in flat_state
        }
        for namespace, fields in _NAMESPACE_FIELDS.items()
    }
    namespaced["control"] = {"status": flat_state.get("status", "running")}
    namespaced["errors"] = list(flat_state.get("errors") or [])
    return namespaced


def replay_public_state(flat_state: dict[str, Any]) -> dict[str, Any]:
    """Project a legacy response through the new API boundary without data loss."""
    return flatten_pipeline_state(namespace_legacy_state(flat_state))
