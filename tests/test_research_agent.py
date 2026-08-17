import os
import unittest
from unittest.mock import patch

from agents.research_agent import _configure_research_groq_credentials


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


if __name__ == "__main__":
    unittest.main()
