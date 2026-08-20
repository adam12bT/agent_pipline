from .implementation import generation_agent
from rfp.agents._shared import dump_model, validate_output
from rfp.agents.generation.contract import Input, Output
from rfp.contracts import KnowledgeSearch, RagQuery


def run(
    input: Input,
    *,
    rag: RagQuery | None = None,
    knowledge: KnowledgeSearch | None = None,
) -> Output:
    return validate_output(
        Output,
        generation_agent(dump_model(input), rag=rag, knowledge=knowledge),
    )
