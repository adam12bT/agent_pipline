"""
Extraction Agent
-----------------
Runs in PARALLEL with the Research agent — both fan out from the Verifier
and join at Generation, since neither depends on the other (this one reads
the embedded tender doc via RAG; Research goes out to the open web).

Uses mode="query" on every call — this means AnythingLLM will ONLY answer
from the embedded document chunks (no general LLM knowledge, no chat
history), which is exactly what you want when extracting facts instead
of having a conversation.
"""

import json
import re

from anythingllm_client import AnythingLLMClient

EXTRACTION_PROMPT = """Based ONLY on the tender document provided, extract the following \
information and respond with ONLY a valid JSON object (no markdown fences, no extra text):

{
  "scope_summary": "2-3 sentence summary of what work is being requested",
  "deliverables": ["list", "of", "expected", "deliverables"],
  "deadlines": {"submission_deadline": "date if stated, else null", "project_duration": "if stated, else null"},
  "budget": "budget or price range if stated, else null",
  "evaluation_criteria": ["list of how proposals will be scored"],
  "selection_method": "e.g. QCBS, QBS, LCS, if stated, else null"
}

If a field cannot be found in the document, use null or an empty list — do not guess."""


def _repair_truncated_json(text: str) -> str:
    """Best-effort repair for JSON that was cut off mid-object/array —
    e.g. the LLM's response got truncated by a token limit before it
    could close its braces (this is what happened in the observed run:
    the response ended right after the last complete key/value pair,
    with no closing '}' for the outer object).

    Walks the text respecting string literals/escapes, tracks how many
    '{'/'[' are still open, trims any dangling incomplete token at the
    very end (an unterminated string, a trailing comma, a half-written
    key), then appends the missing closing brackets in the right order.
    """
    stack = []
    in_string = False
    escape = False
    last_safe_index = 0  # index right after the last structurally complete point

    for i, ch in enumerate(text):
        if in_string:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_string = False
                last_safe_index = i + 1
            continue

        if ch == '"':
            in_string = True
        elif ch in "{[":
            stack.append(ch)
        elif ch in "}]":
            if stack:
                stack.pop()
            last_safe_index = i + 1
        elif ch not in " \t\n\r,":
            last_safe_index = i + 1

    repaired = text[:last_safe_index].rstrip()
    if repaired.endswith(","):
        repaired = repaired[:-1]

    closers = {"{": "}", "[": "]"}
    while stack:
        repaired += closers[stack.pop()]

    return repaired


def _extract_json(text: str) -> dict:
    """LLMs often wrap JSON in prose or markdown fences despite instructions,
    and can also get cut off mid-object by a token limit — handle both."""
    fence_match = re.search(r"```(?:json)?\s*(\{.*\})\s*```", text, re.DOTALL)
    candidate = fence_match.group(1) if fence_match else text

    brace_match = re.search(r"\{.*\}", candidate, re.DOTALL)
    if brace_match:
        candidate = brace_match.group(0)
    else:
        # No closing brace anywhere — very likely truncated. Fall back to
        # everything from the first '{' onward so repair has something to
        # work with, instead of giving up immediately.
        open_brace = candidate.find("{")
        if open_brace != -1:
            candidate = candidate[open_brace:]

    try:
        return json.loads(candidate)
    except json.JSONDecodeError:
        pass

    try:
        repaired = _repair_truncated_json(candidate)
        parsed = json.loads(repaired)
        parsed["_extraction_note"] = (
            "Response appeared truncated (likely hit a token limit); "
            "auto-repaired by closing the open brackets. Spot-check the "
            "last field before trusting it fully."
        )
        return parsed
    except json.JSONDecodeError:
        return {"raw_response": text, "parse_error": True}


def extraction_agent(state: dict) -> dict:
    if not state.get("is_verified"):
        # Should never actually run if the graph is wired correctly, but
        # guard against it anyway rather than silently doing bad work.
        # Partial-return convention: nothing to contribute == empty dict,
        # not a full state passthrough (see state.py docstring — this
        # node runs in parallel with Research, so it must never spread
        # `**state` back).
        return {}

    client = AnythingLLMClient()
    workspace_slug = state["workspace_slug"]

    try:
        response_text = client.chat(workspace_slug, EXTRACTION_PROMPT, mode="query")
        requirements = _extract_json(response_text)
    except Exception as e:
        error_msg = f"Extraction agent failed: {e}"
        return {
            "requirements": {},
            "errors": [error_msg],
        }

    return {"requirements": requirements}