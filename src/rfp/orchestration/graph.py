"""The only module that composes all six independently contracted agents."""

from copy import deepcopy
from functools import partial
import time

from langgraph.graph import END, StateGraph

from rfp.agents._shared import dump_model
from rfp.agents.extraction import run as run_extraction
from rfp.agents.generation import run as run_generation
from rfp.agents.quality import run as run_quality
from rfp.agents.research import run as run_research
from rfp.agents.security import run as run_security
from rfp.agents.verifier import run as run_verifier
from rfp.orchestration.projections import (
    extraction_input,
    generation_input,
    quality_input,
    research_input,
    security_input,
    verifier_input,
)
from rfp.orchestration.dependencies import PipelineDependencies
from rfp.orchestration.routing import (
    after_generation,
    after_quality,
    after_security,
    after_verifier,
    quality_status,
)
from rfp.orchestration.state import PipelineState
from providers.telemetry import collect_llm_usage


_USAGE_COUNTERS = (
    "request_count",
    "successful_calls",
    "failed_calls",
    "prompt_tokens",
    "completion_tokens",
    "total_tokens",
    "duration_seconds",
)


def _add_usage(target: dict, addition: dict) -> dict:
    """Accumulate provider telemetry while preserving individual call records."""

    for key in _USAGE_COUNTERS:
        target[key] = target.get(key, 0) + addition.get(key, 0)
    target.setdefault("calls", []).extend(deepcopy(addition.get("calls") or []))
    providers = target.setdefault("providers", {})
    for provider_name, values in (addition.get("providers") or {}).items():
        provider = providers.setdefault(
            provider_name,
            {
                "request_count": 0,
                "successful_calls": 0,
                "failed_calls": 0,
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_tokens": 0,
                "duration_seconds": 0.0,
                "models": [],
            },
        )
        for key in _USAGE_COUNTERS:
            provider[key] = provider.get(key, 0) + values.get(key, 0)
        for model in values.get("models") or []:
            if model not in provider["models"]:
                provider["models"].append(model)
    return target


def _instrumented(
    state: PipelineState,
    *,
    node_name: str,
    runner,
    dependencies: PipelineDependencies,
) -> dict:
    started_at = time.time()
    started = time.perf_counter()
    with collect_llm_usage() as llm_usage:
        result = runner(state, dependencies=dependencies)
    ended_at = time.time()
    duration = max(0.0, time.perf_counter() - started)

    telemetry = deepcopy(state.get("telemetry") or {})
    agents = telemetry.setdefault("agents", {})
    agent = agents.setdefault(
        node_name,
        {"attempts": [], "duration_seconds": 0.0, "llm_usage": {}},
    )
    attempt = {
        "attempt": len(agent["attempts"]) + 1,
        "started_at_epoch": started_at,
        "ended_at_epoch": ended_at,
        "duration_seconds": duration,
        "llm_usage": deepcopy(llm_usage),
    }
    agent["attempts"].append(attempt)
    agent["duration_seconds"] = agent.get("duration_seconds", 0.0) + duration
    agent["llm_usage"] = _add_usage(agent.get("llm_usage") or {}, llm_usage)
    telemetry["updated_at_epoch"] = ended_at
    telemetry["total_duration_seconds"] = max(
        0.0, ended_at - float(telemetry.get("started_at_epoch", started_at))
    )
    telemetry["llm_usage"] = _add_usage(
        telemetry.get("llm_usage") or {}, llm_usage
    )
    result["telemetry"] = telemetry
    return result


def _verifier(state: PipelineState, *, dependencies: PipelineDependencies) -> dict:
    output = dump_model(
        run_verifier(verifier_input(state), ingestion=dependencies.ingestion)
    )
    return {
        "verifier": output,
        "control": {"status": "running" if output["is_verified"] else "blocked"},
        "errors": output.pop("errors", []),
    }


def _dispatch(_: PipelineState) -> dict:
    return {}


def _extraction(state: PipelineState, *, dependencies: PipelineDependencies) -> dict:
    output = dump_model(run_extraction(extraction_input(state), rag=dependencies.rag))
    return {"extraction": output, "errors": output.pop("errors", [])}


def _research(state: PipelineState, *, dependencies: PipelineDependencies) -> dict:
    output = dump_model(
        run_research(
            research_input(state),
            web=dependencies.web,
        )
    )
    return {"research": output, "errors": output.pop("errors", [])}


def _generation(state: PipelineState, *, dependencies: PipelineDependencies) -> dict:
    output = dump_model(
        run_generation(
            generation_input(state),
            rag=dependencies.rag,
            knowledge=dependencies.knowledge,
        )
    )
    status = "running" if output.get("draft_proposal", "").strip() else "failed"
    return {
        "generation": output,
        "control": {"status": status},
        "errors": output.pop("errors", []),
    }


def _security(state: PipelineState, *, dependencies: PipelineDependencies) -> dict:
    output = dump_model(
        run_security(security_input(state), scanner=dependencies.security_scanner)
    )
    status = "running" if output.get("security_passed", True) else "security_blocked"
    return {"security": output, "control": {"status": status}}


def _quality(state: PipelineState, *, dependencies: PipelineDependencies) -> dict:
    agent_input = quality_input(state)
    output = dump_model(run_quality(agent_input, scanner=dependencies.quality_scanner))
    return {
        "quality": output,
        "control": {"status": quality_status(output, agent_input.generation_attempts)},
    }


def build_graph(dependencies: PipelineDependencies | None = None):
    dependencies = dependencies or PipelineDependencies.defaults()
    graph = StateGraph(PipelineState)
    graph.add_node(
        "verifier",
        partial(
            _instrumented,
            node_name="verifier",
            runner=_verifier,
            dependencies=dependencies,
        ),
    )
    graph.add_node("dispatch", _dispatch)
    for node_name, runner in (
        ("extraction", _extraction),
        ("research", _research),
        ("generation", _generation),
        ("security", _security),
        ("quality", _quality),
    ):
        graph.add_node(
            node_name,
            partial(
                _instrumented,
                node_name=node_name,
                runner=runner,
                dependencies=dependencies,
            ),
        )
    graph.set_entry_point("verifier")
    graph.add_conditional_edges("verifier", after_verifier, {"dispatch": "dispatch", END: END})
    graph.add_edge("dispatch", "extraction")
    graph.add_edge("extraction", "research")
    graph.add_edge("research", "generation")
    graph.add_conditional_edges("generation", after_generation, {"security": "security", END: END})
    graph.add_conditional_edges("security", after_security, {"quality": "quality", END: END})
    graph.add_conditional_edges("quality", after_quality, {"generation": "generation", END: END})
    return graph.compile()
