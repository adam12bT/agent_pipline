"""Canonical fallback proposal template used when no client template is uploaded."""

from copy import deepcopy
from typing import Any


DEFAULT_TEMPLATE_NAME = "General tender response"
DEFAULT_TEMPLATE_VERSION = "1.0"

_SECTIONS = [
    "Executive Summary",
    "Understanding of the Tender",
    "Requirements and Compliance",
    "Proposed Approach and Deliverables",
    "Implementation Plan and Governance",
    "Qualifications and Supporting Evidence",
    "Risks, Quality and Compliance Controls",
    "Commercial Response",
    "Appendices",
]

_DEFAULT_RESPONSE_TEMPLATE: dict[str, Any] = {
    "name": DEFAULT_TEMPLATE_NAME,
    "version": DEFAULT_TEMPLATE_VERSION,
    "template_source": "default",
    "language": "auto",
    "required_sections": _SECTIONS,
    "section_order": _SECTIONS,
    "instructions": [
        "Follow the tender's terminology and include every extracted mandatory requirement.",
        "Use tender evidence for procurement facts and company knowledge for bidder-specific claims.",
        "Use an explicit confirmation placeholder when required information is unavailable.",
    ],
    "formatting_requirements": [],
    "outline_source": "built_in_default",
}


def default_response_template() -> dict[str, Any]:
    """Return an isolated copy so one run cannot mutate the shared default."""
    return deepcopy(_DEFAULT_RESPONSE_TEMPLATE)


def resolve_response_template(requirements: dict[str, Any] | None) -> dict[str, Any]:
    """Return a valid uploaded template definition or the canonical fallback."""
    raw = (requirements or {}).get("response_template")
    if isinstance(raw, dict):
        sections = raw.get("section_order") or raw.get("required_sections")
        if isinstance(sections, list) and any(str(item).strip() for item in sections):
            resolved = deepcopy(raw)
            resolved.setdefault("template_source", "uploaded")
            resolved.setdefault("outline_source", "uploaded_template")
            return resolved
    return default_response_template()
