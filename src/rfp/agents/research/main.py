from rfp.agents._shared import run_cli
from rfp.agents.research.agent import run
from rfp.agents.research.contract import Input, Output
from rfp.adapters import GPTResearcherAdapter


def main():
    web = GPTResearcherAdapter()
    run_cli(Input, Output, lambda value: run(value, web=web))
