import unittest
from unittest.mock import patch

from agents.security_agent import security_agent


class SecurityScannerConfigurationTests(unittest.TestCase):
    def test_phone_number_is_not_scanned_when_all_scanners_are_disabled(self):
        with (
            patch("agents.security_agent._LLM_GUARD_AVAILABLE", False),
            patch("agents.security_agent.SECURITY_FALLBACK_ENABLED", False),
        ):
            result = security_agent(
                {
                    "is_verified": True,
                    "draft_proposal": "Contact: +216 20 123 456",
                }
            )

        self.assertTrue(result["security_passed"])
        self.assertFalse(result["security_report"]["scan_performed"])
        self.assertEqual(result["security_report"]["scanner"]["mode"], "disabled")
        self.assertEqual(result["security_report"]["findings"], {})

    def test_phone_number_is_blocked_when_fallback_is_enabled(self):
        with (
            patch("agents.security_agent._LLM_GUARD_AVAILABLE", False),
            patch("agents.security_agent.SECURITY_FALLBACK_ENABLED", True),
        ):
            result = security_agent(
                {
                    "is_verified": True,
                    "draft_proposal": "Contact: 20 123 456",
                }
            )

        self.assertFalse(result["security_passed"])
        self.assertEqual(result["status"], "security_blocked")
        self.assertEqual(
            result["security_report"]["scanner"]["mode"], "regex_fallback"
        )


if __name__ == "__main__":
    unittest.main()
