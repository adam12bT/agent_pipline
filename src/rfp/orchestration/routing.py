"""Branch and retry policy for namespaced pipeline state."""

import os

from langgraph.graph import END


def max_generation_attempts() -> int:
    return max(1, int(os.environ.get("MAX_GENERATION_ATTEMPTS", "1")))


def quality_status(output: dict, attempts: int) -> str:
    if output.get("quality_passed"):
        return "done"
    if (output.get("quality_report") or {}).get("evaluation_available") is False:
        return "failed"
    return "retry_generation" if attempts < max_generation_attempts() else "failed"


def after_verifier(state: dict):
    return "dispatch" if (state.get("verifier") or {}).get("is_verified") else END


def after_generation(state: dict):
    draft = (state.get("generation") or {}).get("draft_proposal", "")
    return "security" if draft.strip() else END


def after_security(state: dict):
    passed = (state.get("security") or {}).get("security_passed", True)
    return "quality" if passed else END


def after_quality(state: dict):
    status = (state.get("control") or {}).get("status")
    return "generation" if status == "retry_generation" else END
