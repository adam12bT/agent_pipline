from .implementation import research_agent
from rfp.agents._shared import dump_model, validate_output
from rfp.agents.research.contract import Input, Output
from rfp.contracts import RagQuery, WebResearch


def run(
    input: Input,
    *,
    rag: RagQuery | None = None,
    web: WebResearch | None = None,
) -> Output:
    return validate_output(
        Output,
        research_agent(dump_model(input), rag=rag, web=web),
    )
