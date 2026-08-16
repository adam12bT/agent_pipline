import logging
import os

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


def _proposal_sections(response_template_rules: dict) -> list[str]:
    rules = response_template_rules if isinstance(response_template_rules, dict) else {}
    raw_sections = rules.get("section_order") or rules.get("required_sections") or []
    sections = [str(section).strip() for section in raw_sections if str(section).strip()]
    return sections or list(_DEFAULT_PROPOSAL_SECTIONS)


def _section_batches(
    response_template_rules: dict, batch_size: int = 3
) -> list[list[str]]:
    size = max(1, batch_size)
    sections = _proposal_sections(response_template_rules)
    return [sections[index : index + size] for index in range(0, len(sections), size)]


def _proposal_structure(
    response_template_rules: dict, sections: list[str] | None = None
) -> str:
    """Turn extracted template rules into an explicit Markdown outline.

    The old prompt always included a detailed default outline, which competed
    with an uploaded client template. Defaults are now used only when the
    template genuinely contains no section structure.
    """
    rules = response_template_rules if isinstance(response_template_rules, dict) else {}
    raw_sections = rules.get("section_order") or rules.get("required_sections") or []
    using_client_template = bool(raw_sections)
    selected_sections = sections or _proposal_sections(rules)

    lines = [
        "CLIENT TEMPLATE — USE THESE EXACT HEADINGS AND THIS EXACT ORDER:"
        if using_client_template
        else "NO CLIENT SECTION OUTLINE WAS FOUND — USE THESE DEFAULT HEADINGS:",
        *[f"## {section}" for section in selected_sections],
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


def _clip(value, limit: int) -> str:
    text = str(value)
    if len(text) <= limit:
        return text
    return f"{text[:limit]}\n[context truncated to {limit} characters]"


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

    response_template_rules = requirements.get("response_template", {})
    revision_feedback = state.get("quality_report") or "(first generation attempt)"
    batch_size = max(1, int(os.environ.get("GENERATION_BATCH_SIZE", "3")))
    batch_max_tokens = max(
        512, int(os.environ.get("GENERATION_BATCH_MAX_TOKENS", "3072"))
    )
    context_limit = max(
        2000, int(os.environ.get("GENERATION_CONTEXT_MAX_CHARS", "10000"))
    )
    batches = _section_batches(response_template_rules, batch_size=batch_size)

    generation_evidence = {
        "section_batches": [],
        "requirements": requirements,
        "research_summary": state.get("research_summary", "(no research available)"),
        "project_references": project_references,
        "cv_excerpts": cv_excerpts,
        "past_proposals": past_proposals,
    }

    attempt_number = state.get("generation_attempts", 0) + 1
    logger.info(
        "Generation attempt %d for workspace %r using %d dynamic batch(es)",
        attempt_number, workspace_slug, len(batches),
    )

    draft_parts = []
    for batch_number, sections in enumerate(batches, start=1):
        section_names = "; ".join(sections)
        batch_query = (
            f"{search_query}; facts, constraints, evidence and instructions for sections: "
            f"{section_names}"
        )
        tender_excerpts = get_relevant_chunks(
            client, workspace_slug, batch_query, top_n=4
        )
        response_template_excerpts = get_relevant_chunks(
            client,
            template_workspace_slug,
            f"content instructions, tables and formatting for sections: {section_names}",
            top_n=4,
        )
        batch_evidence = {
            "sections": sections,
            "tender_excerpts": tender_excerpts,
            "response_template_excerpts": response_template_excerpts,
        }
        generation_evidence["section_batches"].append(batch_evidence)
        prompt = GENERATION_PROMPT_TEMPLATE.format(
            batch_number=batch_number,
            batch_count=len(batches),
            tender_excerpts=_clip(tender_excerpts, context_limit),
            response_template_excerpts=_clip(response_template_excerpts, context_limit),
            response_template_rules=response_template_rules,
            proposal_structure=_proposal_structure(response_template_rules, sections),
            revision_feedback=_clip(revision_feedback, context_limit),
            requirements=_clip(requirements, context_limit),
            research_summary=_clip(
                state.get("research_summary", "(no research available)"), context_limit
            ),
            project_references=_clip(project_references, context_limit),
            cv_excerpts=_clip(cv_excerpts, context_limit),
            past_proposals=_clip(past_proposals, context_limit),
        )

        try:
            batch_draft = get_provider().complete(
                prompt, max_tokens=batch_max_tokens
            ).strip()
            if not batch_draft:
                raise ValueError("the model returned an empty section batch")
            batch_evidence["draft"] = batch_draft
            draft_parts.append(batch_draft)
            logger.info(
                "Generation attempt %d completed batch %d/%d (%s)",
                attempt_number, batch_number, len(batches), section_names,
            )
        except Exception as e:
            error_msg = (
                f"Generation batch {batch_number}/{len(batches)} failed "
                f"for sections {sections}: {e}"
            )
            logger.error(
                "Generation attempt %d failed in batch %d/%d for workspace %r: %s",
                attempt_number, batch_number, len(batches), workspace_slug, e,
                exc_info=True,
            )
            return {
                "draft_proposal": "",
                "generation_evidence": generation_evidence,
                "generation_attempts": attempt_number,
                "errors": [error_msg],
                "status": "failed",
            }

    draft = "\n\n".join(draft_parts)

    return {
        "draft_proposal": draft,
        "generation_evidence": generation_evidence,
        "generation_attempts": attempt_number,
    }
