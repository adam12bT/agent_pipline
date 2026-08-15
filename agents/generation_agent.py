import logging

from anythingllm_client import AnythingLLMClient
from company_knowledge import PROPOSALS_WORKSPACE, CVS_WORKSPACE, REFERENCES_WORKSPACE
from .prompts import GENERATION_PROMPT_TEMPLATE
from providers import get_provider
from retrieval import get_relevant_chunks

logger = logging.getLogger(__name__)

_DEFAULT_PROPOSAL_SECTIONS = [
    "Executive Summary",
    "Understanding of the Requirements",
    "Proposed Approach & Methodology",
    "Indicative Work Plan / Timeline",
    "Risk Management & Quality Assurance",
    "Proposed Team (Profils Proposés)",
    "Why Us",
]


def _proposal_structure(response_template_rules: dict) -> str:
    """Turn extracted template rules into an explicit Markdown outline.

    The old prompt always included a detailed default outline, which competed
    with an uploaded client template. Defaults are now used only when the
    template genuinely contains no section structure.
    """
    rules = response_template_rules if isinstance(response_template_rules, dict) else {}
    raw_sections = rules.get("section_order") or rules.get("required_sections") or []
    sections = [str(section).strip() for section in raw_sections if str(section).strip()]
    using_client_template = bool(sections)
    if not sections:
        sections = _DEFAULT_PROPOSAL_SECTIONS

    lines = [
        "CLIENT TEMPLATE — USE THESE EXACT HEADINGS AND THIS EXACT ORDER:"
        if using_client_template
        else "NO CLIENT SECTION OUTLINE WAS FOUND — USE THESE DEFAULT HEADINGS:",
        *[f"## {section}" for section in sections],
    ]

    instructions = rules.get("instructions") or rules.get("template_instructions") or []
    formatting = rules.get("formatting_requirements") or []
    if isinstance(instructions, str):
        instructions = [instructions]
    if isinstance(formatting, str):
        formatting = [formatting]
    if instructions:
        lines.extend(["", "Template instructions:"])
        lines.extend(f"- {item}" for item in instructions if str(item).strip())
    if formatting:
        lines.extend(["", "Formatting requirements:"])
        lines.extend(f"- {item}" for item in formatting if str(item).strip())

    return "\n".join(lines)


def _search_company_knowledge(client: AnythingLLMClient, workspace_slug: str, query: str,
                                top_n: int = 3) -> str:
    """Search one company knowledge workspace and format results as readable text.
    Returns a clear "none found" message instead of an empty string, so the LLM
    prompt reads naturally either way."""
    try:
        results = client.vector_search(workspace_slug, query, top_n=top_n)
    except Exception as e:
        logger.warning("Company knowledge search failed for workspace %r: %s", workspace_slug, e)
        results = []

    if not results:
        return "(none found in the company knowledge base for this query)"

    formatted = []
    for r in results:
        title = r.get("metadata", {}).get("title", "unknown source")
        text = r.get("text", "").strip()
        formatted.append(f"- From [{title}]: {text}")
    return "\n".join(formatted)


def generation_agent(state: dict) -> dict:
    if not state.get("is_verified"):
        return {}

    client = AnythingLLMClient()
    workspace_slug = state["workspace_slug"]
    template_workspace_slug = state["response_template_workspace_slug"]
    requirements = state.get("requirements", {})
    search_query = requirements.get("scope_summary") or "technical proposal requirements"

    project_references = _search_company_knowledge(client, REFERENCES_WORKSPACE, search_query)
    cv_excerpts = _search_company_knowledge(
        client, CVS_WORKSPACE, f"consultant profile relevant to: {search_query}"
    )
    past_proposals = _search_company_knowledge(
        client, PROPOSALS_WORKSPACE, f"past proposal similar to: {search_query}"
    )

    tender_excerpts = get_relevant_chunks(client, workspace_slug, search_query, top_n=6)
    response_template_excerpts = get_relevant_chunks(
        client,
        template_workspace_slug,
        "required response structure, headings, section instructions, tables and formatting",
        top_n=10,
    )
    response_template_rules = requirements.get("response_template", {})
    proposal_structure = _proposal_structure(response_template_rules)
    revision_feedback = state.get("quality_report") or "(first generation attempt)"

    generation_evidence = {
        "tender_excerpts": tender_excerpts,
        "response_template_excerpts": response_template_excerpts,
        "requirements": requirements,
        "research_summary": state.get("research_summary", "(no research available)"),
        "project_references": project_references,
        "cv_excerpts": cv_excerpts,
        "past_proposals": past_proposals,
    }

    prompt = GENERATION_PROMPT_TEMPLATE.format(
        tender_excerpts=tender_excerpts,
        response_template_excerpts=response_template_excerpts,
        response_template_rules=response_template_rules,
        proposal_structure=proposal_structure,
        revision_feedback=revision_feedback,
        requirements=requirements,
        research_summary=state.get("research_summary", "(no research available)"),
        project_references=project_references,
        cv_excerpts=cv_excerpts,
        past_proposals=past_proposals,
    )

    attempt_number = state.get("generation_attempts", 0) + 1
    logger.info("Generation attempt %d for workspace %r", attempt_number, workspace_slug)

    try:
        draft = get_provider().complete(prompt, max_tokens=8192)
    except Exception as e:
        error_msg = f"Generation agent failed: {e}"
        logger.error(
            "Generation attempt %d failed for workspace %r: %s",
            attempt_number, workspace_slug, e, exc_info=True,
        )
        return {
            "draft_proposal": "",
            "generation_evidence": generation_evidence,
            "generation_attempts": attempt_number,
            "errors": [error_msg],
        }

    return {
        "draft_proposal": draft,
        "generation_evidence": generation_evidence,
        "generation_attempts": attempt_number,
    }
