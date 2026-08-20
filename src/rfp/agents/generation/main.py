from rfp.agents._shared import run_cli
from rfp.agents.generation.agent import run
from rfp.agents.generation.contract import Input
from rfp.adapters import AnythingLLMAdapter


def main():
    adapter = AnythingLLMAdapter()
    run_cli(Input, lambda value: run(value, rag=adapter, knowledge=adapter))
