"""Tender-scoped external research with an inexpensive relevance gate.

The agent receives verified scope, budget, and selection-method fields through
its input contract. It owns web research only and has no AnythingLLM access.
"""

import logging
import os
import re
import unicodedata

from .prompts import (
    RESEARCH_FALLBACK_SCOPE as _FALLBACK_SCOPE,
    RESEARCH_FALLBACK_BUDGET as _FALLBACK_BUDGET,
    RESEARCH_QUERY_BASE as _QUERY_BASE,
    RESEARCH_QUERY_BUDGET_CLAUSE as _QUERY_BUDGET_CLAUSE,
    RESEARCH_QUERY_SELECTION_METHOD_CLAUSE as _QUERY_SELECTION_METHOD_CLAUSE,
    RESEARCH_QUERY_GUARDRAILS as _QUERY_GUARDRAILS,
)

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

_DOMAIN_PHRASES = {
    "digital_software": {
        "api first",
        "back office",
        "cloud",
        "containerized",
        "data repository",
        "digital platform",
        "information system",
        "mfa",
        "microservice",
        "portal",
        "rbac",
        "software",
        "tls",
        "user interface",
        "web application",
    },
    "physical_pipeline": {
        "flowline",
        "gas pipeline",
        "hdpe",
        "hydraulic test",
        "ndt",
        "oil and gas",
        "oil pipeline",
        "pipe manufacturer",
        "pipeline construction",
        "pipeline inspection",
        "pipeline welding",
        "piping",
        "water pipeline",
    },
    "civil_infrastructure": {
        "bridge construction",
        "civil engineering",
        "concrete",
        "construction site",
        "geotechnical",
        "road construction",
        "structural engineering",
    },
}


def _normalized_text(text: str) -> str:
    decomposed = unicodedata.normalize("NFKD", text or "")
    ascii_text = "".join(char for char in decomposed if not unicodedata.combining(char))
    return re.sub(r"\s+", " ", ascii_text.lower()).strip()


def _domain_scores(text: str) -> dict[str, int]:
    normalized = _normalized_text(text)
    return {
        domain: sum(1 for phrase in phrases if phrase in normalized)
        for domain, phrases in _DOMAIN_PHRASES.items()
    }


def _primary_domain(text: str) -> str:
    scores = _domain_scores(text)
    domain, score = max(scores.items(), key=lambda item: item[1])
    return domain if score >= 2 else "unknown"


def _scope_context(state: dict) -> str:
    parts = [str(state.get("scope_summary") or "").strip()]
    for field in ("deliverables", "technical_constraints", "mandatory_requirements"):
        values = state.get(field) or []
        parts.extend(str(value).strip() for value in values if str(value).strip())
    return ". ".join(part for part in parts if part)


def _report_completeness(report: str) -> dict:
    """Reject clearly cut-off output instead of feeding fragments to Generation."""
    text = (report or "").strip()
    tail = text[-160:].strip()
    last_line = text.splitlines()[-1].strip() if text else ""
    last_word = re.findall(r"[a-zA-Z]+", tail.lower())
    dangling_words = {
        "a", "an", "and", "avec", "de", "des", "du", "et", "for", "of", "or",
        "the", "to", "with",
    }
    incomplete = (
        len(text) < 100
        or tail.endswith(("-", ",", ":", ";", "/", "(", "["))
        or (last_word and last_word[-1] in dangling_words)
        or text.count("[") != text.count("]")
        or text.count("(") != text.count(")")
        or (last_line.startswith("|") and not last_line.endswith("|"))
    )
    return {
        "complete": not incomplete,
        "report_chars": len(text),
        "tail": tail[-80:],
    }


def _report_source_quality(report: str) -> dict:
    try:
        minimum_citations = max(
            0, int(os.environ.get("RESEARCH_MIN_VERIFIABLE_CITATIONS", "1"))
        )
    except ValueError:
        minimum_citations = 1
    urls = {
        url.rstrip(".,;]")
        for url in re.findall(r"https?://[^\s)>]+", report or "")
    }
    return {
        "has_enough_sources": len(urls) >= minimum_citations,
        "citation_count": len(urls),
        "minimum_citations": minimum_citations,
    }


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
    lexical_relevance = (
        scope_is_usable
        and len(matched) >= minimum_matches
        and coverage >= minimum_coverage
    )

    scope_domain = _primary_domain(scope)
    scope_domain_scores = _domain_scores(scope)
    report_domain_scores = _domain_scores(report)
    conflicting_domains: list[str] = []
    if scope_domain == "digital_software":
        conflicting_domains = [
            domain
            for domain in ("physical_pipeline", "civil_infrastructure")
            if report_domain_scores[domain] >= 2
        ]

    completeness = _report_completeness(report)
    source_quality = _report_source_quality(report)
    relevant = all(
        (
            lexical_relevance,
            not conflicting_domains,
            completeness["complete"],
            source_quality["has_enough_sources"],
        )
    )

    reason = "accepted"
    if not scope_is_usable:
        reason = "insufficient_tender_scope"
    elif conflicting_domains:
        reason = "conflicting_domain"
    elif not completeness["complete"]:
        reason = "truncated_or_incomplete_report"
    elif not source_quality["has_enough_sources"]:
        reason = "missing_verifiable_sources"
    elif not lexical_relevance:
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
        "scope_domain": scope_domain,
        "scope_domain_scores": scope_domain_scores,
        "report_domain_scores": report_domain_scores,
        "conflicting_domains": conflicting_domains,
        "report_complete": completeness["complete"],
        "report_chars": completeness["report_chars"],
        "citation_count": source_quality["citation_count"],
        "minimum_citations": source_quality["minimum_citations"],
    }


def _build_query(
    scope: str,
    budget: str = _FALLBACK_BUDGET,
    selection_method: str | None = None,
) -> str:
    """Turn a short scope description into a focused research query
    instead of just researching the raw, noisy tender text."""
    query = _QUERY_BASE.format(scope=scope)
    if budget and budget != _FALLBACK_BUDGET:
        query += _QUERY_BUDGET_CLAUSE.format(budget=budget)
    if selection_method:
        query += _QUERY_SELECTION_METHOD_CLAUSE.format(selection_method=selection_method)
    if _primary_domain(scope) == "digital_software":
        query += (
            " This is a digital/software and information-system procurement. "
            "Interpret workflow, processing, or testing pipelines as software concepts. "
            "Exclude oil/gas pipelines, water pipes, HDPE, NDT, bridge construction, "
            "and civil-engineering suppliers unless the tender explicitly concerns them."
        )
    query += (
        " Treat a company as a likely competitor only when a cited source shows that "
        "it delivers relevant consulting, implementation, or integration services. "
        "Do not present an AI model, API product, software library, or cloud feature as "
        "a bidding firm. Every named competitor must have an inline source URL that "
        "supports its relevant capability; otherwise omit it."
    )
    query += _QUERY_GUARDRAILS
    return query


def research_agent(state: dict, *, web=None) -> dict:
    if not state.get("is_verified"):
        return {}

    scope = str(state.get("scope_summary") or "").strip()
    scope_context = _scope_context(state)
    budget = str(state.get("budget") or _FALLBACK_BUDGET).strip()
    selection_method = state.get("selection_method")

    # A generic fallback cannot safely anchor external research. Skipping here
    # is preferable to feeding an unrelated market report into Generation.
    if scope == _FALLBACK_SCOPE or len(_normalized_keywords(scope_context)) < 3:
        relevance_report = _evaluate_research_relevance(scope_context, "")
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

    query = _build_query(scope_context, budget, selection_method)

    try:
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

    relevance_report = _evaluate_research_relevance(scope_context, research_summary)
    if (
        not relevance_report["relevant"]
        and relevance_report["reason"] == "conflicting_domain"
    ):
        error_msg = (
            "Research relevance gate rejected the external report: "
            f"reason={relevance_report['reason']}, "
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

    # Coverage, completeness, and citation checks are advisory. Web-research
    # providers can return a useful partial report when they hit a response
    # limit, and discarding that report removes all external context from the
    # generation step. Keep only the domain-conflict check blocking so clearly
    # unrelated research (for example oil pipelines for a software tender) is
    # still prevented from contaminating the proposal.
    if not relevance_report["relevant"]:
        advisory_reason = relevance_report["reason"]
        warning = (
            "Research relevance advisory: the external report was retained "
            f"despite {advisory_reason}."
        )
        relevance_report = {
            **relevance_report,
            "relevant": True,
            "meets_relevance_quality_gate": False,
            "advisory_warning": advisory_reason,
            "reason": f"accepted_with_warning:{advisory_reason}",
            "blocking": False,
        }
        logger.warning(warning)
    else:
        relevance_report = {
            **relevance_report,
            "meets_relevance_quality_gate": True,
            "blocking": False,
        }

    logger.info(
        "Research completed (%d chars, relevance coverage %.3f)",
        len(research_summary),
        relevance_report["coverage"],
    )
    return {
        "research_summary": research_summary,
        "research_relevant": True,
        "relevance_report": relevance_report,
    }
