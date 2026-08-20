from .implementation import verifier_agent
from rfp.agents._shared import dump_model, validate_output
from rfp.agents.verifier.contract import Input, Output
from rfp.contracts import TenderIngestion


def run(input: Input, *, ingestion: TenderIngestion | None = None) -> Output:
    return validate_output(
        Output,
        verifier_agent(dump_model(input), ingestion=ingestion),
    )
