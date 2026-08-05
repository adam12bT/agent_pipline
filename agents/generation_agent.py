from anythingllm_client import AnythingLLMClient
from company_knowledge import PROPOSALS_WORKSPACE, CVS_WORKSPACE, REFERENCES_WORKSPACE
from providers import get_provider
from retrieval import get_relevant_chunks

GENERATION_PROMPT_TEMPLATE = """You are a senior bid writer producing a FULL technical proposal \
report in response to the tender document below. This is a formal deliverable that \
will be submitted to the client for evaluation — not a cover letter and not an executive summary. \
Each section below must be written in complete, substantive paragraphs with real analysis and \
detail grounded in the material provided. Shallow, generic filler is worse than a shorter section \
that is actually specific to this tender.

RELEVANT TENDER DOCUMENT EXCERPTS:
{tender_excerpts}

EXTRACTED REQUIREMENTS:
{requirements}

MARKET / COMPETITOR RESEARCH:
{research_summary}

RELEVANT PAST PROJECT REFERENCES (from our company's own project history — use these for \
the "Why Us" / track record section):
{project_references}

RELEVANT CONSULTANT CVs (use these — and ONLY these — to write the "Proposed Team / Profils \
Proposés" section; do not invent names, titles, or years of experience that aren't in this list):
{cv_excerpts}

RELEVANT PAST PROPOSALS (for tone/structure reference only — do not copy text verbatim, \
just match the general style and level of detail):
{past_proposals}

Write the proposal in Markdown with these sections, each meeting the stated minimum depth. \
Treat the minimums as a floor: expand further wherever the tender's requirements give you real \
material to work with.

1. **Executive Summary** (~150-250 words) — who we are, what we're proposing, and the single \
strongest reason we're the right fit. No sub-bullets here; flowing prose.

2. **Understanding of the Requirements** (~300-400 words) — restate the scope, deliverables, \
deadlines, budget, and evaluation criteria in your own words to demonstrate genuine \
comprehension. Explicitly call out any ambiguities or risks you notice in the tender itself.

3. **Proposed Approach & Methodology** (~500-700 words) — break this into named phases or \
workstreams (e.g. Discovery & Requirements, Design & Architecture, Build, Data Migration, \
Testing & UAT, Training, Deployment, Support) that map onto the actual deliverables listed \
above. For each phase, describe concretely what will be done, the key activities, and what \
"done" looks like. Do not just name the phases — explain the reasoning behind the approach.

4. **Indicative Work Plan / Timeline** (~200-300 words plus a Markdown table) — provide a table \
with columns Phase | Duration | Key Milestones, consistent with the project duration stated in \
the requirements. Follow the table with a short paragraph on sequencing and dependencies.

5. **Risk Management & Quality Assurance** (~200-300 words) — identify 3-5 concrete risks \
specific to this tender (e.g. data migration integrity, staff adoption across many sites, \
integration with external systems) and the mitigation for each, plus how quality will be \
assured throughout (testing strategy, review gates, acceptance criteria).

6. **Proposed Team (Profils Proposés)** — based ONLY on the CV excerpts above. If no CV \
excerpts were provided, write exactly: "[TEAM PROFILES TO BE COMPLETED — no matching CVs found \
in the company knowledge base]" instead of inventing anyone. If excerpts were provided, give \
each person a short paragraph: role on this project, relevant background, and why they fit \
this specific tender.

7. **Why Us** (~250-350 words) — reference the past project references above if any were found, \
and the market research for competitive positioning against likely competitors. If no past \
references were found, keep this section general rather than inventing specific past projects, \
but still make a substantive case (methodology strengths, team depth, understanding of the \
sector) rather than a one-line platitude.

Do not invent specific figures, dates, project names, or consultant names that are not present \
in the material above or the tender document itself — leave a clear placeholder like \
[TO BE CONFIRMED] instead of making something up. Write in a professional, confident register \
appropriate for a formal procurement submission."""


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
    # Generation used to rely on AnythingLLM's mode="chat" auto-grounding
    # the whole reply in the tender workspace's embedded doc. Now that
    # the actual text generation goes through a swappable LLMProvider
    # (Groq/Ollama) instead of AnythingLLM's own chat endpoint, that
    # grounding has to be made explicit — pull the tender's own most
    # relevant excerpts the same way company-knowledge context is pulled.
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