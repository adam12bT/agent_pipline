from .implementation import extraction_agent
from rfp.agents._shared import dump_model, validate_output
from rfp.agents.extraction.contract import Input, Output
from rfp.contracts import RagQuery


def run(input: Input, *, rag: RagQuery | None = None) -> Output:
    return validate_output(Output, extraction_agent(dump_model(input), rag=rag))
