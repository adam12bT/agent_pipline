import unittest

from agents.generation_agent import _proposal_structure
from agents.quality_agent import (
    _check_section_order,
    _check_template_compliance,
    _template_sections,
)


class ResponseTemplateQualityTests(unittest.TestCase):
    def test_client_template_replaces_default_generation_outline(self):
        outline = _proposal_structure(
            {
                "required_sections": ["1. Introduction", "2. Compréhension du besoin"],
                "section_order": ["1. Introduction", "2. Compréhension du besoin"],
            }
        )

        self.assertIn("## 1. Introduction", outline)
        self.assertIn("## 2. Compréhension du besoin", outline)
        self.assertNotIn("Executive Summary", outline)

    def test_client_template_sections_override_defaults(self):
        state = {
            "requirements": {
                "response_template": {
                    "required_sections": ["Contexte", "Solution", "Planning"],
                    "section_order": ["Contexte", "Solution", "Planning"],
                }
            }
        }

        required, ordered = _template_sections(state)

        self.assertEqual(required, ["Contexte", "Solution", "Planning"])
        self.assertEqual(ordered, required)

    def test_missing_and_out_of_order_sections_are_reported(self):
        draft = "# Solution\nDetails\n# Contexte\nDetails"
        sections = ["Contexte", "Solution", "Planning"]

        self.assertEqual(_check_template_compliance(draft, sections), ["Planning"])
        self.assertEqual(_check_section_order(draft, sections), sections)

    def test_numbered_template_title_matches_unnumbered_markdown_heading(self):
        draft = "# Introduction\nTexte\n## **2. Compréhension du besoin**\nTexte"
        sections = ["1. Introduction", "2. Compréhension du besoin"]

        self.assertEqual(_check_template_compliance(draft, sections), [])
        self.assertEqual(_check_section_order(draft, sections), [])


if __name__ == "__main__":
    unittest.main()
