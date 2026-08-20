from .implementation import quality_agent
from rfp.agents._shared import dump_model, validate_output
from rfp.agents.quality.contract import Input, Output
from rfp.contracts import OutputScanner


def run(input: Input, *, scanner: OutputScanner | None = None) -> Output:
    return validate_output(Output, quality_agent(dump_model(input), scanner=scanner))
