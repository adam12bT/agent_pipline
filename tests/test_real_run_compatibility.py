import json
import unittest
from pathlib import Path
from unittest.mock import patch

from fastapi.testclient import TestClient

from rfp.agents.extraction.contract import Input as ExtractionInput, Output as ExtractionOutput
from rfp.agents.generation.contract import Input as GenerationInput, Output as GenerationOutput
from rfp.agents.quality.contract import Input as QualityInput, Output as QualityOutput
from rfp.agents.research.contract import Input as ResearchInput, Output as ResearchOutput
from rfp.agents.security.contract import Input as SecurityInput, Output as SecurityOutput
from rfp.agents.verifier.contract import Input as VerifierInput, Output as VerifierOutput
from rfp.api.app import app
from rfp import cli
from rfp.compatibility import namespace_legacy_state, replay_public_state


REAL_RUN_PATH = Path(__file__).parent / "fixtures" / "real_run" / "legacy_api_response.json"


def load_real_run() -> dict:
    return json.loads(REAL_RUN_PATH.read_text(encoding="utf-8"))


class RealRunCompatibilityTests(unittest.TestCase):
    def test_real_pre_migration_state_is_byte_for_byte_preserved_at_api_boundary(self):
        legacy = load_real_run()["state"]
        self.assertEqual(replay_public_state(legacy), legacy)

    def test_real_run_can_be_projected_into_every_agent_contract(self):
        legacy = load_real_run()["state"]
        state = namespace_legacy_state(legacy)
        request = state["request"]
        verifier = state["verifier"]
        requirements = state["extraction"]["requirements"]

        pairs = (
            (
                VerifierInput(**request),
                VerifierOutput(**verifier),
            ),
            (
                ExtractionInput(
                    is_verified=verifier["is_verified"],
                    workspace_slug=verifier["workspace_slug"],
                    response_template_workspace_slug=verifier[
                        "response_template_workspace_slug"
                    ],
                    response_template_file_path=request[
                        "response_template_file_path"
                    ],
                ),
                ExtractionOutput(requirements=requirements),
            ),
            (
                ResearchInput(
                    is_verified=verifier["is_verified"],
                    scope_summary=requirements["scope_summary"],
                    budget=str(requirements.get("budget") or "none stated"),
                    selection_method=requirements.get("selection_method"),
                ),
                ResearchOutput(
                    research_summary=state["research"]["research_summary"],
                    research_relevant=False,
                    relevance_report={"reason": "recorded_before_relevance_gate"},
                ),
            ),
            (
                GenerationInput(
                    run_id=request["run_id"],
                    is_verified=verifier["is_verified"],
                    workspace_slug=verifier["workspace_slug"],
                    response_template_workspace_slug=verifier[
                        "response_template_workspace_slug"
                    ],
                    requirements=requirements,
                    research_summary=state["research"]["research_summary"],
                ),
                GenerationOutput(**state["generation"]),
            ),
            (
                SecurityInput(
                    is_verified=verifier["is_verified"],
                    draft_proposal=state["generation"]["draft_proposal"],
                ),
                SecurityOutput(**state["security"]),
            ),
            (
                QualityInput(
                    is_verified=verifier["is_verified"],
                    security_passed=state["security"]["security_passed"],
                    draft_proposal=state["generation"]["draft_proposal"],
                    generation_evidence=state["generation"]["generation_evidence"],
                    generation_attempts=state["generation"]["generation_attempts"],
                    requirements=requirements,
                ),
                QualityOutput(**state["quality"]),
            ),
        )
        self.assertEqual(len(pairs), 6)

    def test_existing_run_and_download_routes_preserve_real_response(self):
        record = load_real_run()
        with patch("rfp.api.app.get_run", return_value=record):
            client = TestClient(app)
            detail = client.get("/api/runs/577336344eda")
            download = client.get("/api/runs/577336344eda/download")

        self.assertEqual(detail.status_code, 200)
        self.assertEqual(detail.json(), record)
        self.assertEqual(download.status_code, 200)
        self.assertEqual(download.text, record["state"]["draft_proposal"])

    def test_cli_preserves_the_complete_real_public_result(self):
        legacy = load_real_run()["state"]
        namespaced = namespace_legacy_state(legacy)

        class FakeAdapter:
            def ensure_ready(self):
                return {}

        class FakePipeline:
            def invoke(self, _state):
                return namespaced

        with patch.object(cli, "AnythingLLMAdapter", return_value=FakeAdapter()), patch.object(
            cli, "build_graph", return_value=FakePipeline()
        ), patch("builtins.print"):
            result = cli.run(
                legacy["tender_file_path"],
                legacy["response_template_file_path"],
            )

        self.assertEqual(result, legacy)


if __name__ == "__main__":
    unittest.main()
