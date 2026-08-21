import json
import os
import subprocess
import sys
import unittest
from pathlib import Path

from rfp.agents.extraction.contract import Input as ExtractionInput, Output as ExtractionOutput
from rfp.agents.generation.contract import Input as GenerationInput, Output as GenerationOutput
from rfp.agents.quality.contract import Input as QualityInput, Output as QualityOutput
from rfp.agents.research.contract import Input as ResearchInput, Output as ResearchOutput
from rfp.agents.security.contract import Input as SecurityInput, Output as SecurityOutput
from rfp.agents.verifier.contract import Input as VerifierInput, Output as VerifierOutput
from rfp.orchestration.state import flatten_pipeline_state


FIXTURES = Path(__file__).parent / "fixtures"
AGENT_CONTRACTS = {
    "verifier": (VerifierInput, VerifierOutput),
    "extraction": (ExtractionInput, ExtractionOutput),
    "research": (ResearchInput, ResearchOutput),
    "generation": (GenerationInput, GenerationOutput),
    "security": (SecurityInput, SecurityOutput),
    "quality": (QualityInput, QualityOutput),
}


class SavedAgentFixtureTests(unittest.TestCase):
    def test_saved_inputs_and_outputs_match_every_public_contract(self):
        for agent, (input_contract, output_contract) in AGENT_CONTRACTS.items():
            with self.subTest(agent=agent):
                folder = FIXTURES / "agents" / agent
                input_contract.model_validate_json((folder / "input.json").read_text(encoding="utf-8"))
                output_contract.model_validate_json((folder / "output.json").read_text(encoding="utf-8"))

    def test_every_agent_cli_accepts_its_saved_real_run_pair(self):
        environment = dict(os.environ)
        environment["PYTHONPATH"] = str(Path(__file__).parents[1] / "src")
        for agent in AGENT_CONTRACTS:
            with self.subTest(agent=agent):
                folder = FIXTURES / "agents" / agent
                completed = subprocess.run(
                    [
                        sys.executable,
                        "-m",
                        f"rfp.agents.{agent}",
                        "--in",
                        str(folder / "input.json"),
                        "--expected",
                        str(folder / "output.json"),
                        "--contract-only",
                    ],
                    cwd=Path(__file__).parents[1],
                    env=environment,
                    capture_output=True,
                    text=True,
                    timeout=30,
                )
                self.assertEqual(completed.returncode, 0, completed.stderr)

    def test_namespaced_state_matches_golden_public_api(self):
        internal = json.loads((FIXTURES / "pipeline" / "internal_state.json").read_text(encoding="utf-8"))
        expected = json.loads((FIXTURES / "pipeline" / "public_state.json").read_text(encoding="utf-8"))
        self.assertEqual(flatten_pipeline_state(internal), expected)


if __name__ == "__main__":
    unittest.main()
