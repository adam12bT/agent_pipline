from rfp.agents._shared import run_cli
from rfp.agents.research.agent import run
from rfp.agents.research.contract import Input
from rfp.adapters import AnythingLLMAdapter, GPTResearcherAdapter


def main():
    rag = AnythingLLMAdapter()
    web = GPTResearcherAdapter()
    run_cli(Input, lambda value: run(value, rag=rag, web=web))
