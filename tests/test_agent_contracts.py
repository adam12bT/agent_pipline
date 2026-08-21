import unittest

from rfp.agents.extraction.contract import Input as ExtractionInput
from rfp.agents.generation.contract import Input as GenerationInput
from rfp.agents.quality.contract import Input as QualityInput
from rfp.agents.research.contract import Input as ResearchInput
from rfp.agents.security.contract import Input as SecurityInput
from rfp.agents.verifier.contract import Input as VerifierInput


class PerAgentContractTests(unittest.TestCase):
    def test_verifier_contract(self):
        value = VerifierInput(
            tender_file_path="tender.pdf",
            response_template_file_path="template.docx",
        )
        self.assertEqual(value.tender_file_path, "tender.pdf")

    def test_extraction_contract(self):
        value = ExtractionInput(
            is_verified=True,
            workspace_slug="tender",
            response_template_workspace_slug="template",
            response_template_file_path="template.docx",
        )
        self.assertTrue(value.is_verified)

    def test_research_contract(self):
        value = ResearchInput(
            is_verified=True,
            scope_summary="Secure citizen-services platform",
            budget="480000 TND",
        )
        self.assertIn("platform", value.scope_summary)

    def test_generation_contract(self):
        value = GenerationInput(
            is_verified=True,
            workspace_slug="tender",
            response_template_workspace_slug="template",
        )
        self.assertEqual(value.generation_attempts, 0)

    def test_security_contract(self):
        value = SecurityInput(is_verified=True, draft_proposal="draft")
        self.assertEqual(value.draft_proposal, "draft")

    def test_quality_contract(self):
        value = QualityInput(is_verified=True, draft_proposal="draft")
        self.assertTrue(value.security_passed)


if __name__ == "__main__":
    unittest.main()
