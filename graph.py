"""
Wires the 6 agents into a single LangGraph StateGraph, matching the
architecture diagram:

  Dossier AO -> verifier -> (blocked?) -> END
                         -> extraction ─┐
                         -> research    ─┴─> generation -> security -> (blocked?) -> END (human alert)
                                                                     -> quality -> (retry?) -> generation
                                                                                -> END (done or failed)

Extraction and Research run in PARALLEL: neither depends on the other
(Extraction reads the embedded tender doc via RAG, Research goes out to
the open web), and both feed into Generation. See state.py for why this
required every agent to switch from `{**state, ...}` to partial returns,
and agents/research_agent.py for the one real trade-off it introduces
(Research can no longer see Extraction's output when building its query,
since they start from the same pre-fork state).

The `_dispatch` node below is plumbing, not a real pipeline stage: a
single conditional edge can only route to ONE set of targets, so gating
"is_verified" has to happen once, then fan out unconditionally from
there. Without it you'd need two separate conditional edges off
`verifier` (one for extraction, one for research), and a plain
`add_edge("verifier", "research")` would fire even when blocked, since
unconditional edges from a node aren't cancelled by that node's
conditional edges. `_dispatch` is filtered out of the UI stepper in
backend/run_store.py so it doesn't show up as a fake stage.
"""

from langgraph.graph import END, StateGraph

from agents import (
    extraction_agent,
    generation_agent,
    quality_agent,
    research_agent,
    security_agent,
    verifier_agent,
)
from state import RFPState


def _dispatch(state: RFPState) -> dict:
    """No-op pass-through — see module docstring for why this exists."""
    return {}


def _route_after_verifier(state: RFPState) -> str:
    return "dispatch" if state.get("is_verified") else END


def _route_after_security(state: RFPState) -> str:
    # Blocking — no retry path. A failed scan goes straight to END; the
    # caller (backend/run_store.py) surfaces `security_report` so the UI
    # can show the human-alert state instead of a normal "failed" run.
    return END if not state.get("security_passed", True) else "quality"


def _route_after_quality(state: RFPState) -> str:
    status = state.get("status")
    if status == "retry_generation":
        return "generation"
    return END  # "done" or "failed" both end the graph — check status/quality_passed after


def build_graph():
    graph = StateGraph(RFPState)

    graph.add_node("verifier", verifier_agent)
    graph.add_node("dispatch", _dispatch)
    graph.add_node("extraction", extraction_agent)
    graph.add_node("research", research_agent)
    graph.add_node("generation", generation_agent)
    graph.add_node("security", security_agent)
    graph.add_node("quality", quality_agent)

    graph.set_entry_point("verifier")

    graph.add_conditional_edges("verifier", _route_after_verifier, {
        "dispatch": "dispatch",
        END: END,
    })

    # Fan-out: both fire unconditionally off `dispatch`, which itself only
    # ever runs on the verified path.
    graph.add_edge("dispatch", "extraction")
    graph.add_edge("dispatch", "research")

    # Join: `generation` only runs once BOTH `extraction` and `research`
    # have finished — LangGraph waits for every incoming edge of a step
    # before running the target node, so two `add_edge(..., "generation")`
    # calls are enough, no extra config needed.
    graph.add_edge("extraction", "generation")
    graph.add_edge("research", "generation")

    graph.add_edge("generation", "security")

    graph.add_conditional_edges("security", _route_after_security, {
        "quality": "quality",
        END: END,
    })

    graph.add_conditional_edges("quality", _route_after_quality, {
        "generation": "generation",
        END: END,
    })

    return graph.compile()