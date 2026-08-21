from rfp.agents._shared import run_cli
from rfp.agents.extraction.agent import run
from rfp.agents.extraction.contract import Input, Output
from rfp.adapters import AnythingLLMAdapter


def main():
    adapter = AnythingLLMAdapter()
    run_cli(Input, Output, lambda value: run(value, rag=adapter))
