import os
import unittest
from unittest.mock import patch

from rfp.agents.research.implementation import (
    _configure_research_groq_credentials,
    _evaluate_research_relevance,
    research_agent,
)


class FakeWebResearch:
    def __init__(self, report):
        self.report = report
        self.calls = 0

    def research(self, query):
        self.calls += 1
        return self.report


class ResearchCredentialTests(unittest.TestCase):
    def test_dedicated_research_key_is_mapped_for_gpt_researcher(self):
        with patch.dict(
            os.environ,
            {
                "RESEARCH_GROQ_API_KEY": "research-key",
                "GROQ_API_KEY": "legacy-key",
            },
            clear=True,
        ):
            configured = _configure_research_groq_credentials(True)
            self.assertTrue(configured)
            self.assertEqual(os.environ["GROQ_API_KEY"], "research-key")

    def test_research_key_is_not_mapped_when_research_uses_another_provider(self):
        with patch.dict(
            os.environ,
            {"RESEARCH_GROQ_API_KEY": "research-key"},
            clear=True,
        ):
            configured = _configure_research_groq_credentials(False)
            self.assertFalse(configured)
            self.assertNotIn("GROQ_API_KEY", os.environ)


class ResearchRelevanceTests(unittest.TestCase):
    scope = (
        "Digital platform with a user portal, back-office, reference data "
        "repository, API-first architecture, sovereign cloud hosting and security."
    )

    def test_relevant_report_is_accepted(self):
        report = (
            "The digital platform market includes portal and back-office vendors. "
            "API architecture, sovereign cloud hosting, reference data and security "
            "are important differentiators."
        )
        result = _evaluate_research_relevance(self.scope, report)
        self.assertTrue(result["relevant"])
        self.assertGreaterEqual(result["matched_keyword_count"], 3)

    def test_unrelated_bridge_report_is_rejected(self):
        report = (
            "Road bridge construction requires civil engineering, concrete, site "
            "supervision, traffic management and geotechnical surveys."
        )
        result = _evaluate_research_relevance(self.scope, report)
        self.assertFalse(result["relevant"])
        self.assertEqual(result["reason"], "low_scope_overlap")

    def test_agent_does_not_forward_rejected_research(self):
        web = FakeWebResearch(
            "Road bridge construction requires concrete, structural engineering, "
            "traffic planning and construction supervision."
        )
        with patch(
            "rfp.agents.research.implementation._get_scope_from_tender",
            return_value=self.scope,
        ), patch(
            "rfp.agents.research.implementation._get_budget_from_tender",
            return_value="480,000 TND",
        ):
            result = research_agent(
                {"is_verified": True, "workspace_slug": "tender"},
                rag=object(),
                web=web,
            )

        self.assertEqual(web.calls, 1)
        self.assertFalse(result["research_relevant"])
        self.assertNotIn("bridge construction", result["research_summary"].lower())
        self.assertIn("relevance gate rejected", result["errors"][0].lower())


if __name__ == "__main__":
    unittest.main()
