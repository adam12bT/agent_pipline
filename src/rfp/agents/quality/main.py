from rfp.agents._shared import run_cli
from rfp.agents.quality.agent import run
from rfp.agents.quality.contract import Input, Output
from rfp.adapters import ConfiguredQualityScanner


def main():
    scanner = ConfiguredQualityScanner()
    run_cli(Input, Output, lambda value: run(value, scanner=scanner))
