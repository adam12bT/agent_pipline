"""
Research Agent Implementation
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
import re
import unicodedata

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

logger = logging.getLogger(__name__)

# Env vars GPT Researcher needs by default. If you've configured a
# different retriever/LLM provider, adjust this list — it's only used
# to give a more useful error message, not to enforce anything.
_EXPECTED_ENV_VARS = ["TAVILY_API_KEY"]

_RELEVANCE_STOPWORDS = {
    # Generic procurement/project vocabulary does not establish topicality.
    "about", "also", "around", "based", "client", "consultant", "consultants",
    "contract", "development", "including", "market", "project", "proposal",
    "requirements", "services", "solution", "specific", "tender", "work",
    # French equivalents.
    "appel", "besoin", "client", "consultant", "contrat", "developpement",
    "exigences", "incluant", "marche", "offre", "projet", "services", "solution",
    "travaux",
    # Common glue words in both languages.
    "avec", "dans", "des", "does", "for", "from", "have", "les", "pour",
    "that", "the", "this", "une", "will", "with",
}

_REJECTED_RESEARCH_SUMMARY = (
    "(No external research used - relevance validation rejected the report "
    "because it did not match the tender scope.)"
)


def _normalized_keywords(text: str) -> set[str]:
    """Return stable topic terms for a cheap, language-agnostic comparison."""
    decomposed = unicodedata.normalize("NFKD", text or "")
    ascii_text = "".join(char for char in decomposed if not unicodedata.combining(char))
    tokens = re.findall(r"[a-z0-9]+", ascii_text.lower())
    return {
        token
        for token in tokens
        if len(token) >= 3
        and not token.isdigit()
        and token not in _RELEVANCE_STOPWORDS
    }


def _evaluate_research_relevance(scope: str, report: str) -> dict:
    """Measure whether a web report covers the tender's meaningful scope terms.

    This intentionally avoids an LLM judge: the guard must remain available when
    providers are rate-limited and must not consume another paid API request.
    """
    try:
        minimum_coverage = min(
            1.0,
            max(0.0, float(os.environ.get("RESEARCH_MIN_SCOPE_COVERAGE", "0.25"))),
        )
    except ValueError:
        minimum_coverage = 0.25
    try:
        minimum_matches = max(
            1, int(os.environ.get("RESEARCH_MIN_MATCHED_KEYWORDS", "3"))
        )
    except ValueError:
        minimum_matches = 3

    scope_keywords = _normalized_keywords(scope)
    report_keywords = _normalized_keywords(report)
    matched = sorted(scope_keywords & report_keywords)
    coverage = len(matched) / len(scope_keywords) if scope_keywords else 0.0
    scope_is_usable = len(scope_keywords) >= minimum_matches
    relevant = (
        scope_is_usable
        and len(matched) >= minimum_matches
        and coverage >= minimum_coverage
    )

    reason = "accepted"
    if not scope_is_usable:
        reason = "insufficient_tender_scope"
    elif not relevant:
        reason = "low_scope_overlap"

    return {
        "relevant": relevant,
        "reason": reason,
        "coverage": round(coverage, 3),
        "minimum_coverage": minimum_coverage,
        "matched_keyword_count": len(matched),
        "minimum_matched_keywords": minimum_matches,
        "scope_keyword_count": len(scope_keywords),
        "matched_keywords": matched[:25],
    }


def _configure_research_groq_credentials(uses_groq: bool) -> bool:
    """Give GPT Researcher its dedicated key without exposing its value.

    GPT Researcher constructs its own ChatGroq clients and reads the standard
    ``GROQ_API_KEY`` variable internally. Direct pipeline calls do not depend
    on this mutation because GroqProvider prefers ``PIPELINE_GROQ_API_KEY``.
    The legacy single-key setup remains supported when the dedicated research
    key is absent.
    """
    research_key = os.environ.get("RESEARCH_GROQ_API_KEY")
    if not uses_groq or not research_key:
        return False
    os.environ["GROQ_API_KEY"] = research_key
    logger.info("GPT Researcher configured with RESEARCH_GROQ_API_KEY")
    return True


def _get_scope_from_tender(workspace_slug: str, rag=None) -> str:
    """Independent, lightweight read of the embedded tender doc — does NOT
    rely on the Extraction agent's output, so this stays safe to run in
    parallel with it. Falls back to a generic scope string (old behavior)
    if the workspace isn't ready yet or the call fails for any reason,
    rather than blowing up the whole Research branch over it."""
    try:
        query = (
            "project scope, deliverables, sector, domain, technical or "
            "regulatory requirements"
        )
        if rag is None:
            raise RuntimeError("RagQuery dependency was not provided")
        context = rag.query(workspace_slug, query, top_n=6)
        prompt = f"TENDER DOCUMENT EXCERPTS:\n\n{context}\n\n{_SCOPE_PROMPT}"
        response_text = get_provider().complete(
            prompt,
            request_label="research.scope",
            reasoning_effort="low",
            include_reasoning=False,
        )
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


def _get_budget_from_tender(workspace_slug: str, rag=None) -> str:
    """Independent, lightweight read of the embedded tender doc for the
    stated budget/price ceiling — same pattern as _get_scope_from_tender.
    Used to steer GPT Researcher toward firms actually sized to compete
    for this contract, rather than category-leading enterprise vendors
    a much bigger budget would attract."""
    try:
        query = "total budget, price ceiling, contract value"
        if rag is None:
            raise RuntimeError("RagQuery dependency was not provided")
        context = rag.query(workspace_slug, query, top_n=4)
        prompt = f"TENDER DOCUMENT EXCERPTS:\n\n{context}\n\n{_BUDGET_PROMPT}"
        response_text = get_provider().complete(
            prompt,
            request_label="research.budget",
            reasoning_effort="low",
            include_reasoning=False,
        )
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
    # Import lazily so an optional GPT Researcher packaging/import problem
    # cannot prevent the FastAPI application from starting. Any failure here
    # is caught by research_agent(), reported in the run output, and the rest
    # of the proposal pipeline can continue without external research.
    groq_models = (
        os.environ.get("FAST_LLM", ""),
        os.environ.get("SMART_LLM", ""),
        os.environ.get("STRATEGIC_LLM", ""),
    )
    uses_groq = any(model.strip().lower().startswith("groq:") for model in groq_models)
    _configure_research_groq_credentials(uses_groq)

    from gpt_researcher import GPTResearcher

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
    researcher.cfg.fast_token_limit = 1200
    researcher.cfg.smart_token_limit = 1800
    researcher.cfg.strategic_token_limit = 1200
    researcher.cfg.summary_token_limit = 500

    # GPT Researcher creates its own ChatGroq clients, so those calls do not
    # pass through providers/groq_provider.py. Give every internal LLM client
    # one shared LangChain rate limiter and add boundary cooldowns so research
    # cannot collide with the direct calls immediately before/after this step.
    groq_interval = max(
        0.0, float(os.environ.get("GROQ_MIN_INTERVAL_SECONDS", "30"))
    )
    if uses_groq and groq_interval > 0:
        from langchain_core.rate_limiters import InMemoryRateLimiter

        researcher.cfg.llm_kwargs["rate_limiter"] = InMemoryRateLimiter(
            requests_per_second=1.0 / groq_interval,
            check_every_n_seconds=min(1.0, groq_interval),
            max_bucket_size=1,
        )
        await asyncio.sleep(groq_interval)
    logger.debug(
        "GPT Researcher config tuned: max_search_results_per_query=4, "
        "max_subtopics=3, total_words=900 (source-list bloat mitigation)"
    )

    await researcher.conduct_research()
    report = await researcher.write_report()
    if uses_groq and groq_interval > 0:
        await asyncio.sleep(groq_interval)
    return report


def research_agent(state: dict, *, rag=None, web=None) -> dict:
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
    scope = _get_scope_from_tender(workspace_slug, rag=rag)
    budget = _get_budget_from_tender(workspace_slug, rag=rag)

    # A generic fallback cannot safely anchor external research. Skipping here
    # is preferable to feeding an unrelated market report into Generation.
    if scope == _FALLBACK_SCOPE or len(_normalized_keywords(scope)) < 3:
        relevance_report = _evaluate_research_relevance(scope, "")
        error_msg = (
            "Research skipped: tender scope was not specific enough for "
            "relevance validation."
        )
        logger.warning(error_msg)
        return {
            "research_summary": _REJECTED_RESEARCH_SUMMARY,
            "research_relevant": False,
            "relevance_report": relevance_report,
            "errors": [error_msg],
        }

    query = _build_query(scope, budget)

    try:
        # research_agent is a plain sync function (LangGraph node), but
        # GPTResearcher's API is async — run it in its own event loop.
        if web is None:
            raise RuntimeError("WebResearch dependency was not provided")
        research_summary = web.research(query)
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
            "research_relevant": False,
            "relevance_report": {"relevant": False, "reason": "research_failed"},
            "errors": [error_msg],
        }

    relevance_report = _evaluate_research_relevance(scope, research_summary)
    if not relevance_report["relevant"]:
        error_msg = (
            "Research relevance gate rejected the external report: "
            f"coverage={relevance_report['coverage']:.3f} "
            f"(minimum={relevance_report['minimum_coverage']:.3f}), "
            f"matched={relevance_report['matched_keyword_count']} "
            f"(minimum={relevance_report['minimum_matched_keywords']})."
        )
        logger.warning(error_msg)
        return {
            "research_summary": _REJECTED_RESEARCH_SUMMARY,
            "research_relevant": False,
            "relevance_report": relevance_report,
            "errors": [error_msg],
        }

    logger.info(
        "Research completed for workspace %r (%d chars, relevance coverage %.3f)",
        workspace_slug,
        len(research_summary),
        relevance_report["coverage"],
    )
    return {
        "research_summary": research_summary,
        "research_relevant": True,
        "relevance_report": relevance_report,
    }
