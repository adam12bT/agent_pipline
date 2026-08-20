from rfp.agents._shared import run_cli
from rfp.agents.security.agent import run
from rfp.agents.security.contract import Input
from rfp.adapters import ConfiguredSecurityScanner


def main():
    scanner = ConfiguredSecurityScanner()
    run_cli(Input, lambda value: run(value, scanner=scanner))
