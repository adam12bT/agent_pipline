import os
import unittest
from unittest.mock import patch

from rfp.agents.research.implementation import (
    _build_query,
    _evaluate_research_relevance,
    research_agent,
)
from rfp.adapters.web_research import configure_research_groq_credentials


class FakeWebResearch:
    def __init__(self, report):
        self.report = report
        self.calls = 0
        self.last_query = None

    def research(self, query):
        self.calls += 1
        self.last_query = query
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
            configured = configure_research_groq_credentials(True)
            self.assertTrue(configured)
            self.assertEqual(os.environ["GROQ_API_KEY"], "research-key")

    def test_research_key_is_not_mapped_when_research_uses_another_provider(self):
        with patch.dict(
            os.environ,
            {"RESEARCH_GROQ_API_KEY": "research-key"},
            clear=True,
        ):
            configured = configure_research_groq_credentials(False)
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
            "are important differentiators. [Source](https://example.com/market)"
        )
        result = _evaluate_research_relevance(self.scope, report)
        self.assertTrue(result["relevant"])
        self.assertGreaterEqual(result["matched_keyword_count"], 3)

    def test_unrelated_report_is_flagged_by_generic_scope_overlap(self):
        report = (
            "Road bridge construction requires civil engineering, concrete, site "
            "supervision, traffic management and geotechnical surveys. "
            "[Source](https://example.com/bridge)"
        )
        result = _evaluate_research_relevance(self.scope, report)
        self.assertFalse(result["relevant"])
        self.assertEqual(result["reason"], "low_scope_overlap")

    def test_agent_retains_low_overlap_research_with_advisory(self):
        web = FakeWebResearch(
            "Road bridge construction requires concrete, structural engineering, "
            "traffic planning and construction supervision."
        )
        result = research_agent(
            {
                "is_verified": True,
                "scope_summary": self.scope,
                "budget": "480000 TND",
            },
            web=web,
        )

        self.assertEqual(web.calls, 1)
        self.assertTrue(result["research_relevant"])
        self.assertIn("bridge construction", result["research_summary"].lower())
        self.assertFalse(result["relevance_report"]["meets_relevance_quality_gate"])

    def test_agent_retains_truncated_research_with_advisory_warning(self):
        report = (
            "Digital platform competitors provide portal, back-office, API-first "
            "cloud hosting, data repositories, MFA, RBAC, and software security. "
            "[Source](https://example.com/market) The strongest competitor offers a large-"
        )
        web = FakeWebResearch(report)
        result = research_agent(
            {
                "is_verified": True,
                "scope_summary": self.scope,
                "budget": "480000 TND",
            },
            web=web,
        )

        self.assertTrue(result["research_relevant"])
        self.assertEqual(result["research_summary"], report)
        self.assertTrue(result["relevance_report"]["relevant"])
        self.assertFalse(
            result["relevance_report"]["meets_relevance_quality_gate"]
        )
        self.assertEqual(
            result["relevance_report"]["advisory_warning"],
            "truncated_or_incomplete_report",
        )

    def test_relevance_has_no_hard_coded_domain_classifier(self):
        report = (
            "The API-first market for pipeline inspection includes oil and gas "
            "pipeline construction, HDPE pipe manufacturers, NDT control, hydraulic "
            "test services, cloud reporting, MFA and security monitoring."
        )
        result = _evaluate_research_relevance(self.scope, report)

        self.assertFalse(result["relevant"])
        self.assertEqual(result["reason"], "missing_verifiable_sources")
        self.assertNotIn("conflicting_domains", result)

    def test_query_uses_tender_scope_without_domain_specific_exclusions(self):
        query = _build_query(self.scope, "480000 TND")

        self.assertIn(self.scope, query)
        self.assertNotIn("Exclude oil/gas pipelines", query)

    def test_truncated_report_is_rejected(self):
        report = (
            "Digital platform competitors provide portal, back-office, API-first "
            "cloud hosting, data repositories, MFA, RBAC, and software security. "
            "The strongest competitor offers a large-"
        )
        result = _evaluate_research_relevance(self.scope, report)

        self.assertFalse(result["relevant"])
        self.assertEqual(result["reason"], "truncated_or_incomplete_report")
        self.assertFalse(result["report_complete"])

    def test_agent_adds_extracted_constraints_to_query(self):
        report = (
            "Digital platform vendors build user portals and back-office software "
            "using API-first architecture, sovereign cloud hosting, MFA, RBAC and "
            "data security controls. The cited market evidence is complete. "
            "[Source](https://example.com/digital-platform-market)"
        )
        web = FakeWebResearch(report)
        result = research_agent(
            {
                "is_verified": True,
                "scope_summary": "Digital tender response platform",
                "deliverables": ["User portal", "Back-office application"],
                "technical_constraints": ["API-first", "Sovereign cloud", "RBAC"],
                "mandatory_requirements": ["MFA administrator access"],
            },
            web=web,
        )

        self.assertTrue(result["research_relevant"])
        self.assertIn("Back-office application", web.last_query)
        self.assertIn("MFA administrator access", web.last_query)

    def test_uncited_report_is_rejected(self):
        report = (
            "Digital platform vendors build portals, back-office software, API-first "
            "architecture, sovereign cloud hosting, data repositories and security."
        )
        result = _evaluate_research_relevance(self.scope, report)

        self.assertFalse(result["relevant"])
        self.assertEqual(result["reason"], "missing_verifiable_sources")


if __name__ == "__main__":
    unittest.main()
