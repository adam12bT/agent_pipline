"""Convert namespaced pipeline state into each explicit input contract."""

from rfp.agents.extraction.contract import Input as ExtractionInput
from rfp.agents.generation.contract import Input as GenerationInput
from rfp.agents.quality.contract import Input as QualityInput
from rfp.agents.research.contract import Input as ResearchInput
from rfp.agents.security.contract import Input as SecurityInput
from rfp.agents.verifier.contract import Input as VerifierInput


def verifier_input(state: dict) -> VerifierInput:
    return VerifierInput(**(state.get("request") or {}))


def extraction_input(state: dict) -> ExtractionInput:
    request, verified = state.get("request") or {}, state.get("verifier") or {}
    return ExtractionInput(
        is_verified=verified.get("is_verified", False),
        workspace_slug=verified.get("workspace_slug", ""),
        response_template_workspace_slug=verified.get(
            "response_template_workspace_slug"
        ),
        response_template_file_path=request.get("response_template_file_path"),
    )


def research_input(state: dict) -> ResearchInput:
    verified = state.get("verifier") or {}
    requirements = (state.get("extraction") or {}).get("requirements") or {}
    return ResearchInput(
        is_verified=verified.get("is_verified", False),
        scope_summary=str(requirements.get("scope_summary") or ""),
        budget=str(requirements.get("budget") or "none stated"),
        selection_method=requirements.get("selection_method"),
        deliverables=list(requirements.get("deliverables") or []),
        technical_constraints=list(requirements.get("technical_constraints") or []),
        mandatory_requirements=list(requirements.get("mandatory_requirements") or []),
    )


def generation_input(state: dict) -> GenerationInput:
    request = state.get("request") or {}
    verified = state.get("verifier") or {}
    previous = state.get("generation") or {}
    return GenerationInput(
        run_id=request.get("run_id"),
        is_verified=verified.get("is_verified", False),
        workspace_slug=verified.get("workspace_slug", ""),
        response_template_workspace_slug=verified.get(
            "response_template_workspace_slug"
        ),
        requirements=(state.get("extraction") or {}).get("requirements") or {},
        research_summary=(state.get("research") or {}).get("research_summary", ""),
        previous_draft=previous.get("draft_proposal", ""),
        previous_generation_evidence=previous.get("generation_evidence") or {},
        generation_attempts=previous.get("generation_attempts", 0),
        quality_report=(state.get("quality") or {}).get("quality_report") or {},
    )


def security_input(state: dict) -> SecurityInput:
    verified, generated = state.get("verifier") or {}, state.get("generation") or {}
    return SecurityInput(
        is_verified=verified.get("is_verified", False),
        draft_proposal=generated.get("draft_proposal", ""),
    )


def quality_input(state: dict) -> QualityInput:
    verified = state.get("verifier") or {}
    generated = state.get("generation") or {}
    secured = state.get("security") or {}
    return QualityInput(
        is_verified=verified.get("is_verified", False),
        security_passed=secured.get("security_passed", True),
        draft_proposal=generated.get("draft_proposal", ""),
        generation_evidence=generated.get("generation_evidence") or {},
        generation_attempts=generated.get("generation_attempts", 0),
        requirements=(state.get("extraction") or {}).get("requirements") or {},
    )
