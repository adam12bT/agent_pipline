import os
import unittest
from unittest.mock import Mock, patch

import requests

from providers.base import LLMProviderError
from providers.groq_provider import GroqProvider


class GroqProviderRetryTests(unittest.TestCase):
    def test_pipeline_key_is_preferred_over_legacy_key(self):
        response = Mock()
        response.ok = True
        response.status_code = 200
        response.headers = {}
        response.json.return_value = {
            "choices": [{"message": {"content": "ok"}}]
        }

        with (
            patch.dict(
                os.environ,
                {
                    "PIPELINE_GROQ_API_KEY": "pipeline-key",
                    "GROQ_API_KEY": "legacy-or-research-key",
                    "GROQ_MIN_INTERVAL_SECONDS": "0",
                },
                clear=True,
            ),
            patch(
                "providers.groq_provider.requests.post", return_value=response
            ) as post,
        ):
            self.assertEqual(GroqProvider().complete("hello"), "ok")

        self.assertEqual(
            post.call_args.kwargs["headers"]["Authorization"],
            "Bearer pipeline-key",
        )

    def test_429_uses_retry_delay_from_error_body_when_header_is_missing(self):
        limited = Mock()
        limited.ok = False
        limited.status_code = 429
        limited.headers = {}
        limited.text = (
            '{"error":{"message":"Rate limit reached. '
            'Please try again in 1.0725s."}}'
        )
        limited.raise_for_status.side_effect = requests.HTTPError("429")

        success = Mock()
        success.ok = True
        success.status_code = 200
        success.json.return_value = {
            "choices": [{"message": {"content": "ok"}}]
        }

        with (
            patch.dict(
                os.environ,
                {
                    "GROQ_API_KEY": "test-key",
                    "GROQ_MAX_RETRIES": "1",
                    "GROQ_RETRY_JITTER_SECONDS": "0",
                    "GROQ_MIN_INTERVAL_SECONDS": "0",
                },
                clear=False,
            ),
            patch(
                "providers.groq_provider.requests.post",
                side_effect=[limited, success],
            ) as post,
            patch("providers.groq_provider.time.sleep") as sleep,
        ):
            provider = GroqProvider()
            self.assertEqual(provider.complete("hello"), "ok")

        self.assertEqual(post.call_count, 2)
        self.assertTrue(sleep.called)

    def test_success_reserves_minimum_interval_before_next_request(self):
        response = Mock()
        response.ok = True
        response.status_code = 200
        response.json.return_value = {
            "choices": [{"message": {"content": "ok"}}]
        }

        with (
            patch.dict(
                os.environ,
                {
                    "GROQ_API_KEY": "test-key",
                    "GROQ_MIN_INTERVAL_SECONDS": "30",
                },
                clear=False,
            ),
            patch(
                "providers.groq_provider.requests.post", return_value=response
            ) as post,
            patch("providers.groq_provider.time.monotonic", return_value=100.0),
        ):
            provider = GroqProvider()
            self.assertEqual(
                provider.complete(
                    "Return JSON",
                    model="gpt-oss-120b",
                    response_format={"type": "json_object"},
                    reasoning_effort="low",
                    include_reasoning=False,
                ),
                "ok",
            )

        self.assertEqual(provider._next_request_at, 130.0)
        self.assertEqual(
            post.call_args.kwargs["json"]["model"],
            "openai/gpt-oss-120b",
        )
        self.assertEqual(
            post.call_args.kwargs["json"]["response_format"],
            {"type": "json_object"},
        )
        self.assertEqual(post.call_args.kwargs["json"]["reasoning_effort"], "low")
        self.assertFalse(post.call_args.kwargs["json"]["include_reasoning"])

    def test_bare_gpt_oss_name_is_translated_for_groq(self):
        response = Mock()
        response.ok = True
        response.status_code = 200
        response.json.return_value = {
            "choices": [{"message": {"content": "ok"}}]
        }

        with (
            patch.dict(
                os.environ,
                {"GROQ_API_KEY": "test-key", "GROQ_MIN_INTERVAL_SECONDS": "0"},
                clear=False,
            ),
            patch(
                "providers.groq_provider.requests.post", return_value=response
            ) as post,
        ):
            GroqProvider().complete("hello", model="gpt-oss-120b")

        self.assertEqual(
            post.call_args.kwargs["json"]["model"],
            "openai/gpt-oss-120b",
        )

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
