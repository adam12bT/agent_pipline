"""
Shared state object passed between every agent in the pipeline.

This is the "single source of truth" that flows through the LangGraph
StateGraph: each agent reads what it needs from it, does its job, and
writes its results back into it before handing off to the next agent.

IMPORTANT — every agent now returns a PARTIAL dict (only the keys it
actually adds or changes), not `{**state, ...}`. This matters because
Extraction and Research now run in PARALLEL (both fan out from the
Verifier and join at Generation). If two nodes in the same step each
returned the full state, they'd both be "writing" every unchanged key
(workspace_slug, is_verified, tender_file_path, ...) at once, and
LangGraph raises an InvalidUpdateError the moment two nodes in the same
step try to set the same key without a reducer telling it how to combine
them — even if the two values happen to be identical.

`errors` is the one field multiple nodes can genuinely want to append to
independently (Extraction and Research could each hit a real failure in
the same step), so it's the one field that needs a reducer:
`Annotated[list[str], operator.add]` tells LangGraph to concatenate lists
from parallel branches instead of rejecting the concurrent write. Because
of this reducer, every agent must return ONLY the new error(s) it wants
appended — never `state.get("errors", []) + [msg]` — or you'll end up
double-counting the errors already in state.
"""

import operator
from typing import Annotated, TypedDict


class RFPState(TypedDict, total=False):
    # --- input ---
    tender_file_path: str          # path to the RFP/cahier des charges (PDF/DOCX)
    workspace_slug: str            # AnythingLLM workspace this run uses

    # --- Verifier agent output ---
    is_verified: bool
    verification_errors: list[str]

    # --- Extraction agent output (parallel branch) ---
    requirements: dict             # {deliverables, deadlines, budget, evaluation_criteria, scope_summary}

    # --- Research agent output (parallel branch) ---
    research_summary: str          # market/competitor research findings

    # --- Generation agent output ---
    draft_proposal: str            # the generated technical proposal (markdown)
    generation_attempts: int       # retry counter, capped by the graph

    # --- Security agent output (BLOCKING: PII, secrets, malicious/injected content) ---
    security_passed: bool
    security_report: dict          # {findings, notes}

    # --- Quality agent output (GRADED: coherence, template compliance, hallucination risk) ---
    quality_passed: bool
    quality_report: dict           # {word_count, missing_sections, quality_findings, notes}

    # --- pipeline control / bookkeeping ---
    status: str                    # "running" | "blocked" | "security_blocked" | "retry_generation" | "done" | "failed"
    errors: Annotated[list[str], operator.add]  # accumulated via reducer — see module docstring