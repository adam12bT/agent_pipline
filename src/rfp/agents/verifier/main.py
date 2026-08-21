from rfp.agents._shared import run_cli
from rfp.agents.verifier.agent import run
from rfp.agents.verifier.contract import Input, Output
from rfp.adapters import AnythingLLMAdapter


def main():
    adapter = AnythingLLMAdapter()
    run_cli(Input, Output, lambda value: run(value, ingestion=adapter))
