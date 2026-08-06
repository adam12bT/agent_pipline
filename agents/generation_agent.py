from anythingllm_client import AnythingLLMClient
from company_knowledge import PROPOSALS_WORKSPACE, CVS_WORKSPACE, REFERENCES_WORKSPACE
from .prompts import GENERATION_PROMPT_TEMPLATE
from providers import get_provider
from retrieval import get_relevant_chunks


def _search_company_knowledge(client: AnythingLLMClient, workspace_slug: str, query: str,
                                top_n: int = 3) -> str:
    """Search one company knowledge workspace and format results as readable text.
    Returns a clear "none found" message instead of an empty string, so the LLM
    prompt reads naturally either way."""
    try:
        results = client.vector_search(workspace_slug, query, top_n=top_n)
    except Exception:
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

    prompt = GENERATION_PROMPT_TEMPLATE.format(
        tender_excerpts=tender_excerpts,
        requirements=requirements,
        research_summary=state.get("research_summary", "(no research available)"),
        project_references=project_references,
        cv_excerpts=cv_excerpts,
        past_proposals=past_proposals,
    )

    try:
        draft = get_provider().complete(prompt, max_tokens=8192)
    except Exception as e:
        error_msg = f"Generation agent failed: {e}"
        return {
            "draft_proposal": "",
            "generation_attempts": state.get("generation_attempts", 0) + 1,
            "errors": [error_msg],
        }

    return {
        "draft_proposal": draft,
        "generation_attempts": state.get("generation_attempts", 0) + 1,
    }