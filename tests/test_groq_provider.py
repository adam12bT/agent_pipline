import os
import unittest
from unittest.mock import Mock, patch

import requests

from providers.base import LLMProviderError
from providers.groq_provider import GroqProvider


class GroqProviderRetryTests(unittest.TestCase):
    def test_long_retry_after_fails_without_sleeping_or_retrying(self):
        response = Mock()
        response.ok = False
        response.status_code = 429
        response.headers = {"Retry-After": "5489"}
        response.text = "rate limited"
        response.raise_for_status.side_effect = requests.HTTPError("429")

        with (
            patch.dict(
                os.environ,
                {
                    "GROQ_API_KEY": "test-key",
                    "GROQ_MAX_RETRIES": "1",
                    "GROQ_MAX_RETRY_AFTER_SECONDS": "60",
                    "GROQ_RETRY_JITTER_SECONDS": "0",
                },
                clear=False,
            ),
            patch("providers.groq_provider.requests.post", return_value=response) as post,
            patch("providers.groq_provider.time.sleep") as sleep,
        ):
            provider = GroqProvider()
            with self.assertRaises(LLMProviderError):
                provider.complete("hello")

        self.assertEqual(post.call_count, 1)
        sleep.assert_not_called()


if __name__ == "__main__":
    unittest.main()
