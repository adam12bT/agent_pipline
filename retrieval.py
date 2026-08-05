"""
Retrieval-only helper on top of AnythingLLMClient.

Deliberately separate from LLM completion: AnythingLLM's vector-search
endpoint does pure similarity search against embedded documents (no LLM
call involved), so it works identically no matter which LLMProvider
(Groq, Ollama, ...) is doing the actual text generation. Agents call
this to pull relevant chunks, then hand them to a provider's
`complete()` as plain-text context.
"""

from anythingllm_client import AnythingLLMClient


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
        results = client.vector_search(
            workspace_slug, query, top_n=top_n, score_threshold=score_threshold
        )
    except Exception:
        results = []

    if not results:
        return "(no relevant content found in the document for this query)"

    chunks = [r.get("text", "").strip() for r in results if r.get("text", "").strip()]
    return "\n\n---\n\n".join(chunks) if chunks else "(no relevant content found in the document for this query)"
