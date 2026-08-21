import os
import unittest
from unittest.mock import patch

from rfp.agents.extraction.contract import Output as ExtractionOutput
from rfp.agents.generation.contract import Output as GenerationOutput
from rfp.agents.quality.contract import Output as QualityOutput
from rfp.agents.research.contract import Output as ResearchOutput
from rfp.agents.security.contract import Output as SecurityOutput
from rfp.agents.verifier.contract import Output as VerifierOutput
from rfp.orchestration.dependencies import PipelineDependencies
from rfp.orchestration.graph import build_graph
from rfp.orchestration.state import flatten_pipeline_state, initial_pipeline_state


class NoopPort:
    def ingest(self, *args, **kwargs):
        return {}

    def query(self, *args, **kwargs):
        return ""

    def ensure_ready(self):
        return None

    def search(self, *args, **kwargs):
        return []

    def research(self, *args, **kwargs):
        return ""

    def scan(self, *args, **kwargs):
        return {}


class PackagedOrchestrationTests(unittest.TestCase):
    def test_namespaced_graph_preserves_flat_api_and_retry_behavior(self):
        port = NoopPort()
        dependencies = PipelineDependencies(port, port, port, port, port, port)
        generation_outputs = [
            GenerationOutput(
                draft_proposal="first draft",
                generation_attempts=1,
                generation_evidence={},
            ),
            GenerationOutput(
                draft_proposal="revised draft",
                generation_attempts=2,
                generation_evidence={},
            ),
        ]
        quality_outputs = [
            QualityOutput(
                quality_passed=False,
                quality_report={"evaluation_available": True},
            ),
            QualityOutput(
                quality_passed=True,
                quality_report={"evaluation_available": True},
            ),
        ]

        with patch.dict(os.environ, {"MAX_GENERATION_ATTEMPTS": "2"}), patch(
            "rfp.orchestration.graph.run_verifier",
            return_value=VerifierOutput(
                is_verified=True,
                workspace_slug="tender-ws",
                response_template_workspace_slug="template-ws",
            ),
        ), patch(
            "rfp.orchestration.graph.run_extraction",
            return_value=ExtractionOutput(requirements={"scope": "test"}),
        ), patch(
            "rfp.orchestration.graph.run_research",
            return_value=ResearchOutput(research_summary="research"),
        ), patch(
            "rfp.orchestration.graph.run_generation",
            side_effect=generation_outputs,
        ), patch(
            "rfp.orchestration.graph.run_security",
            return_value=SecurityOutput(security_passed=True),
        ), patch(
            "rfp.orchestration.graph.run_quality",
            side_effect=quality_outputs,
        ):
            final_internal = build_graph(dependencies).invoke(
                initial_pipeline_state("tender.pdf", "template.docx", run_id="run-1")
            )

        self.assertIn("generation", final_internal)
        self.assertNotIn("draft_proposal", final_internal)
        public = flatten_pipeline_state(final_internal)
        self.assertEqual(public["draft_proposal"], "revised draft")
        self.assertEqual(public["generation_attempts"], 2)
        self.assertEqual(public["status"], "done")


if __name__ == "__main__":
    unittest.main()
