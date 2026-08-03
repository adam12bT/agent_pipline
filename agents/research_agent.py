"""
Research Agent
---------------
Runs in PARALLEL with the Extraction agent — both fan out from the
Verifier and join at Generation. Runs autonomous web research on the
market/competitor context for this tender, using the `gpt-researcher`
pip package directly (confirmed as a real, actively maintained library —
`pip install gpt-researcher`, class GPTResearcher, see their PyPI page).

This does NOT modify GPT Researcher's own code — it's used purely as a
library, imported and called like any other dependency.

Requires GPT Researcher's own environment variables to be set (an LLM
provider key and a search engine key, e.g. TAVILY_API_KEY) — see
agents_pipeline/.env.example.

--------------------------------------------------------------------
FIX (see graph.py / state.py docstrings for the parallel-fanout model):
--------------------------------------------------------------------
This agent used to build its research query from `state["requirements"]`,
which is written by the Extraction agent. Because Extraction and Research
run in the SAME LangGraph superstep, every node in that step only ever
sees the state as of the START of the step — before either branch has
written anything. That made `requirements` unconditionally empty here,
every single run, silently degrading every query to a generic fallback
("...a project involving: the scope of this tender...") with no link to
the actual tender content.

The fix: instead of depending on Extraction's *output*, Research now
reads the same *input* Extraction reads — the tender doc embedded in
AnythingLLM — directly, via its own small, fast query. This keeps the
two branches genuinely independent (true parallelism, no ordering
requirement) while giving GPT Researcher a real, tender-specific scope
to work from instead of a generic placeholder.
"""

import asyncio
import os

from gpt_researcher import GPTResearcher

from anythingllm_client import AnythingLLMClient

# Env vars GPT Researcher needs by default. If you've configured a
# different retriever/LLM provider, adjust this list — it's only used
# to give a more useful error message, not to enforce anything.
_EXPECTED_ENV_VARS = ["TAVILY_API_KEY"]

# Deliberately tiny and specific — this is NOT the full extraction prompt.
# We only need one or two sentences to anchor the research query in the
# right domain; asking for more would just slow this branch down for no
# benefit (Extraction already produces the full structured requirements
# for Generation to use later, once both branches have joined).
_SCOPE_PROMPT = """Based ONLY on the tender document provided, answer in ONE short \
sentence (max ~30 words, no markdown, no preamble): what specific product, service, \
or work is being procured, and in what sector/domain? Be concrete (e.g. name the \
type of system, industry, or deliverable) rather than generic."""

_BUDGET_PROMPT = """Based ONLY on the tender document provided, state the total \
budget or price ceiling if one is mentioned (include currency and amount only, \
e.g. "USD 380,000"). If no budget or price range is stated anywhere in the \
document, respond with exactly: none stated"""

_FALLBACK_SCOPE = "the scope of this tender"
_FALLBACK_BUDGET = "none stated"


def _get_scope_from_tender(workspace_slug: str) -> str:
    """Independent, lightweight read of the embedded tender doc — does NOT
    rely on the Extraction agent's output, so this stays safe to run in
    parallel with it. Falls back to a generic scope string (old behavior)
    if the workspace isn't ready yet or the call fails for any reason,
    rather than blowing up the whole Research branch over it."""
    try:
        client = AnythingLLMClient()
        response_text = client.chat(workspace_slug, _SCOPE_PROMPT, mode="query")
        scope = response_text.strip().strip('"')
        return scope if scope else _FALLBACK_SCOPE
    except Exception:
        # Non-fatal — Research can still run, just with the generic
        # fallback query instead of a tender-specific one.
        return _FALLBACK_SCOPE


def _get_budget_from_tender(workspace_slug: str) -> str:
    """Independent, lightweight read of the embedded tender doc for the
    stated budget/price ceiling — same pattern as _get_scope_from_tender.
    Used to steer GPT Researcher toward firms actually sized to compete
    for this contract, rather than category-leading enterprise vendors
    a much bigger budget would attract."""
    try:
        client = AnythingLLMClient()
        response_text = client.chat(workspace_slug, _BUDGET_PROMPT, mode="query")
        budget = response_text.strip().strip('"')
        return budget if budget else _FALLBACK_BUDGET
    except Exception:
        return _FALLBACK_BUDGET


def _build_query(scope: str, budget: str = _FALLBACK_BUDGET, selection_method: str | None = None) -> str:
    """Turn a short scope description into a focused research query
    instead of just researching the raw, noisy tender text."""
    query = (
        f"market landscape and competing firms/consultants for a project involving: {scope}."
    )
    if budget and budget != _FALLBACK_BUDGET:
        query += (
            f" The project budget is approximately {budget} — prioritize firms and "
            f"consultancies realistically sized to compete for a contract at this budget "
            f"level, not large enterprise vendors whose typical engagements are far larger."
        )
    if selection_method:
        query += f" Procurement is via {selection_method}."
    query += " Identify likely competitors, their typical positioning, and recent similar awarded projects."
    return query


async def _run_research(query: str) -> str:
    researcher = GPTResearcher(query=query, report_type="research_report")
    await researcher.conduct_research()
    report = await researcher.write_report()
    return report


def research_agent(state: dict) -> dict:
    if not state.get("is_verified"):
        # Partial-return convention — see extraction_agent.py's matching
        # guard and state.py's docstring for why this must be `{}` and
        # not a full state passthrough now that this runs in parallel.
        return {}

    # NOTE: deliberately NOT reading state.get("requirements") here — see
    # the module docstring. `workspace_slug` is safe to read because it's
    # written by the Verifier, which always completes (and joins) BEFORE
    # the parallel Extraction/Research fan-out even starts.
    workspace_slug = state["workspace_slug"]
    scope = _get_scope_from_tender(workspace_slug)
    budget = _get_budget_from_tender(workspace_slug)
    query = _build_query(scope, budget)

    try:
        # research_agent is a plain sync function (LangGraph node), but
        # GPTResearcher's API is async — run it in its own event loop.
        research_summary = asyncio.run(_run_research(query))
    except Exception as e:
        # Surface the ACTUAL exception instead of a generic "failed"
        # string, and flag likely missing env vars, since that's the
        # most common cause of a silent failure here.
        detail = f"{type(e).__name__}: {e}"
        missing = [name for name in _EXPECTED_ENV_VARS if not os.environ.get(name)]
        hint = f" Missing env var(s): {', '.join(missing)}." if missing else ""
        error_msg = f"Research agent failed: {detail}.{hint}"

        return {
            "research_summary": f"(No research available — research step failed: {detail}.{hint})",
            "errors": [error_msg],
        }

    return {"research_summary": research_summary}