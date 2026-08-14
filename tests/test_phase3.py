import unittest

from agents.quality_agent import (
    _check_section_order,
    _check_template_compliance,
    _template_sections,
)


class ResponseTemplateQualityTests(unittest.TestCase):
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


if __name__ == "__main__":
    unittest.main()
