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
"""

import asyncio
import os

from gpt_researcher import GPTResearcher

# Env vars GPT Researcher needs by default. If you've configured a
# different retriever/LLM provider, adjust this list — it's only used
# to give a more useful error message, not to enforce anything.
_EXPECTED_ENV_VARS = ["TAVILY_API_KEY"]


def _build_query(requirements: dict) -> str:
    """Turn the extracted requirements into a focused research query
    instead of just researching the raw, noisy tender text.

    NOTE: this reads `requirements` from the FULL state passed into this
    node — that's still safe even though Extraction and Research run in
    parallel, because LangGraph gives every node in a step the state as
    of the START of that step (i.e. before either branch has written
    anything). In practice `requirements` won't be populated yet when
    Research runs, so this falls back to the generic query below. If you
    want Research to be able to use Extraction's output, they can no
    longer run in parallel — see the module docstring in graph.py."""
    scope = requirements.get("scope_summary") or "the scope of this tender"
    selection_method = requirements.get("selection_method")

    query = f"market landscape and competing firms/consultants for a project involving: {scope}."
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

    requirements = state.get("requirements", {})
    query = _build_query(requirements)

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