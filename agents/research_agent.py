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
import logging
import os

from gpt_researcher import GPTResearcher

from anythingllm_client import AnythingLLMClient
from .prompts import (
    RESEARCH_SCOPE_PROMPT as _SCOPE_PROMPT,
    RESEARCH_BUDGET_PROMPT as _BUDGET_PROMPT,
    RESEARCH_FALLBACK_SCOPE as _FALLBACK_SCOPE,
    RESEARCH_FALLBACK_BUDGET as _FALLBACK_BUDGET,
    RESEARCH_QUERY_BASE as _QUERY_BASE,
    RESEARCH_QUERY_BUDGET_CLAUSE as _QUERY_BUDGET_CLAUSE,
    RESEARCH_QUERY_SELECTION_METHOD_CLAUSE as _QUERY_SELECTION_METHOD_CLAUSE,
    RESEARCH_QUERY_GUARDRAILS as _QUERY_GUARDRAILS,
)
from providers import get_provider
from retrieval import get_relevant_chunks

logger = logging.getLogger(__name__)

# Env vars GPT Researcher needs by default. If you've configured a
# different retriever/LLM provider, adjust this list — it's only used
# to give a more useful error message, not to enforce anything.
_EXPECTED_ENV_VARS = ["TAVILY_API_KEY"]


def _get_scope_from_tender(workspace_slug: str) -> str:
    """Independent, lightweight read of the embedded tender doc — does NOT
    rely on the Extraction agent's output, so this stays safe to run in
    parallel with it. Falls back to a generic scope string (old behavior)
    if the workspace isn't ready yet or the call fails for any reason,
    rather than blowing up the whole Research branch over it."""
    try:
        client = AnythingLLMClient()
        context = get_relevant_chunks(
            client, workspace_slug,
            "project scope, deliverables, sector, domain, technical or "
            "regulatory requirements",
            top_n=6,
        )
        prompt = f"TENDER DOCUMENT EXCERPTS:\n\n{context}\n\n{_SCOPE_PROMPT}"
        response_text = get_provider().complete(prompt)
        scope = response_text.strip().strip('"')
        return scope if scope else _FALLBACK_SCOPE
    except Exception as e:
        # Non-fatal — Research can still run, just with the generic
        # fallback query instead of a tender-specific one.
        logger.warning(
            "Failed to read tender scope for workspace %r, falling back to generic "
            "scope: %s", workspace_slug, e,
        )
        return _FALLBACK_SCOPE


def _get_budget_from_tender(workspace_slug: str) -> str:
    """Independent, lightweight read of the embedded tender doc for the
    stated budget/price ceiling — same pattern as _get_scope_from_tender.
    Used to steer GPT Researcher toward firms actually sized to compete
    for this contract, rather than category-leading enterprise vendors
    a much bigger budget would attract."""
    try:
        client = AnythingLLMClient()
        context = get_relevant_chunks(
            client, workspace_slug, "total budget, price ceiling, contract value", top_n=4
        )
        prompt = f"TENDER DOCUMENT EXCERPTS:\n\n{context}\n\n{_BUDGET_PROMPT}"
        response_text = get_provider().complete(prompt)
        budget = response_text.strip().strip('"')
        return budget if budget else _FALLBACK_BUDGET
    except Exception as e:
        logger.warning(
            "Failed to read tender budget for workspace %r, falling back to %r: %s",
            workspace_slug, _FALLBACK_BUDGET, e,
        )
        return _FALLBACK_BUDGET


def _build_query(scope: str, budget: str = _FALLBACK_BUDGET, selection_method: str | None = None) -> str:
    """Turn a short scope description into a focused research query
    instead of just researching the raw, noisy tender text."""
    query = _QUERY_BASE.format(scope=scope)
    if budget and budget != _FALLBACK_BUDGET:
        query += _QUERY_BUDGET_CLAUSE.format(budget=budget)
    if selection_method:
        query += _QUERY_SELECTION_METHOD_CLAUSE.format(selection_method=selection_method)
    query += _QUERY_GUARDRAILS
    return query


async def _run_research(query: str) -> str:
    researcher = GPTResearcher(query=query, report_type="research_report")

    # Setting these directly on `.cfg` after construction rather than via
    # `config_path=` — passing a config JSON path is a known-unreliable
    # path in gpt-researcher (values are sometimes silently ignored; see
    # assafelovic/gpt-researcher issues #489 and #1041). Setting attributes
    # on the already-constructed Config object is the documented workaround.
    #
    # Lower MAX_SEARCH_RESULTS_PER_QUERY and MAX_SUBTOPICS so the report
    # visits fewer low-value pages (e.g. "Top 40 healthcare consultancies"
    # listicles) instead of citing every firm mentioned on a page it barely
    # used — this is what caused the ~50-entry reference list bloat.
    researcher.cfg.max_search_results_per_query = 4
    researcher.cfg.max_subtopics = 3
    researcher.cfg.total_words = 900
    logger.debug(
        "GPT Researcher config tuned: max_search_results_per_query=4, "
        "max_subtopics=3, total_words=900 (source-list bloat mitigation)"
    )

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
        logger.error(error_msg, exc_info=True)

        return {
            "research_summary": f"(No research available — research step failed: {detail}.{hint})",
            "errors": [error_msg],
        }

    logger.info("Research completed for workspace %r (%d chars)", workspace_slug, len(research_summary))
    return {"research_summary": research_summary}