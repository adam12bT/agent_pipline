from rfp.agents._shared import run_cli
from rfp.agents.quality.agent import run
from rfp.agents.quality.contract import Input
from rfp.adapters import ConfiguredQualityScanner


def main():
    scanner = ConfiguredQualityScanner()
    run_cli(Input, lambda value: run(value, scanner=scanner))
