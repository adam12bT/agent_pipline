"""The only module that composes all six independently contracted agents."""

from functools import partial

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
            rag=dependencies.rag,
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
    graph.add_node("verifier", partial(_verifier, dependencies=dependencies))
    graph.add_node("dispatch", _dispatch)
    graph.add_node("extraction", partial(_extraction, dependencies=dependencies))
    graph.add_node("research", partial(_research, dependencies=dependencies))
    graph.add_node("generation", partial(_generation, dependencies=dependencies))
    graph.add_node("security", partial(_security, dependencies=dependencies))
    graph.add_node("quality", partial(_quality, dependencies=dependencies))
    graph.set_entry_point("verifier")
    graph.add_conditional_edges("verifier", after_verifier, {"dispatch": "dispatch", END: END})
    graph.add_edge("dispatch", "extraction")
    graph.add_edge("dispatch", "research")
    graph.add_edge("extraction", "generation")
    graph.add_edge("research", "generation")
    graph.add_conditional_edges("generation", after_generation, {"security": "security", END: END})
    graph.add_conditional_edges("security", after_security, {"quality": "quality", END: END})
    graph.add_conditional_edges("quality", after_quality, {"generation": "generation", END: END})
    return graph.compile()
