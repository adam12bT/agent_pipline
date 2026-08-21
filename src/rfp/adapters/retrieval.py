"""
Retrieval-only helper on top of AnythingLLMClient.

Deliberately separate from LLM completion: AnythingLLM's vector-search
endpoint does pure similarity search against embedded documents (no LLM
call involved), so it works identically no matter which LLMProvider
(Groq, Ollama, ...) is doing the actual text generation. Agents call
this to pull relevant chunks, then hand them to a provider's
`complete()` as plain-text context.
"""

import logging
import os
import re

from .anythingllm_client import AnythingLLMClient

logger = logging.getLogger(__name__)


def _as_bool(value: str) -> bool:
    return value.strip().lower() not in {"0", "false", "no", "off"}


RERANK_ENABLED = _as_bool(os.environ.get("RAG_RERANK_ENABLED", "true"))
RERANK_CANDIDATE_MULTIPLIER = max(
    1, int(os.environ.get("RAG_RERANK_CANDIDATE_MULTIPLIER", "2"))
)


def _terms(text: str) -> set[str]:
    """Return useful lowercase terms for the lightweight lexical reranker."""
    return {term for term in re.findall(r"(?u)\b\w{3,}\b", text.lower())}


def rerank_results(results: list[dict], query: str, top_n: int) -> list[dict]:
    """Deduplicate chunks and blend vector relevance with query-term coverage.

    Qdrant/AnythingLLM remains the semantic retriever. This small second pass
    promotes chunks that also contain important words from the query and keeps
    duplicate embeddings from consuming the context window. The original
    result dictionaries and metadata are preserved.
    """
    query_terms = _terms(query)
    unique: list[dict] = []
    seen_text: set[str] = set()

    for result in results:
        text = str(result.get("text", "")).strip()
        fingerprint = " ".join(text.lower().split())
        if not text or fingerprint in seen_text:
            continue
        seen_text.add(fingerprint)

        text_terms = _terms(text)
        lexical_score = (
            len(query_terms & text_terms) / len(query_terms) if query_terms else 0.0
        )
        try:
            vector_score = float(result.get("score", 0.0))
        except (TypeError, ValueError):
            vector_score = 0.0

        enriched = dict(result)
        enriched["rerank_score"] = (0.85 * vector_score) + (0.15 * lexical_score)
        unique.append(enriched)

    unique.sort(
        key=lambda item: float(item.get("rerank_score", item.get("score", 0.0))),
        reverse=True,
    )
    return unique[:top_n]


def search_relevant_chunks(
    client: AnythingLLMClient,
    workspace_slug: str,
    query: str,
    top_n: int = 6,
    score_threshold: float = 0.3,
) -> list[dict]:
    """Return ranked raw results, including AnythingLLM source metadata."""
    candidate_count = top_n * RERANK_CANDIDATE_MULTIPLIER if RERANK_ENABLED else top_n
    results = client.vector_search(
        workspace_slug,
        query,
        top_n=candidate_count,
        score_threshold=score_threshold,
    )
    if RERANK_ENABLED:
        return rerank_results(results, query, top_n)
    return rerank_results(results, "", top_n)


def get_relevant_chunks(
    client: AnythingLLMClient,
    workspace_slug: str,
    query: str,
    top_n: int = 6,
    score_threshold: float = 0.3,
) -> str:
    """Vector-search `workspace_slug` for `query` and return the matched
    chunks joined as plain text, or a clear "nothing found" message so
    downstream prompts read naturally either way."""
    try:
        results = search_relevant_chunks(
            client,
            workspace_slug,
            query,
            top_n=top_n,
            score_threshold=score_threshold,
        )
    except Exception as e:
        logger.warning("Retrieval failed for workspace %r query %r: %s", workspace_slug, query, e)
        results = []

    if not results:
        return "(no relevant content found in the document for this query)"

    chunks = [r.get("text", "").strip() for r in results if r.get("text", "").strip()]
    return "\n\n---\n\n".join(chunks) if chunks else "(no relevant content found in the document for this query)"
