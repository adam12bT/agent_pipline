import os
import unittest
from unittest.mock import Mock, patch

import requests

from providers import get_provider, reset_cache
from providers.base import LLMProviderError
from providers.cerebras_provider import CerebrasProvider


class CerebrasProviderTests(unittest.TestCase):
    def tearDown(self):
        reset_cache()

    def test_factory_registers_cerebras(self):
        with patch.dict(os.environ, {"CEREBRAS_API_KEY": "test-key"}, clear=False):
            self.assertIsInstance(get_provider("cerebras"), CerebrasProvider)

    def test_api_key_is_required(self):
        with patch.dict(os.environ, {}, clear=True):
            with self.assertRaisesRegex(LLMProviderError, "CEREBRAS_API_KEY"):
                CerebrasProvider()

    def test_completion_uses_cerebras_payload_and_json_mode(self):
        response = Mock()
        response.ok = True
        response.status_code = 200
        response.json.return_value = {
            "choices": [{"message": {"content": '{"score": 1}'}}]
        }

        with (
            patch.dict(
                os.environ,
                {
                    "CEREBRAS_API_KEY": "test-key",
                    "CEREBRAS_REASONING_EFFORT": "low",
                    "CEREBRAS_MIN_INTERVAL_SECONDS": "0",
                },
                clear=False,
            ),
            patch(
                "providers.cerebras_provider.requests.post",
                return_value=response,
            ) as post,
        ):
            provider = CerebrasProvider()
            result = provider.complete(
                "Return JSON",
                max_tokens=512,
                response_format={"type": "json_object"},
            )

        self.assertEqual(result, '{"score": 1}')
        payload = post.call_args.kwargs["json"]
        self.assertEqual(payload["model"], "gpt-oss-120b")
        self.assertEqual(payload["max_completion_tokens"], 512)
        self.assertNotIn("max_tokens", payload)
        self.assertEqual(payload["response_format"], {"type": "json_object"})
        self.assertEqual(payload["reasoning_effort"], "low")

    def test_429_retries_using_token_reset_header(self):
        limited = Mock()
        limited.ok = False
        limited.status_code = 429
        limited.headers = {"x-ratelimit-reset-tokens-minute": "1.25s"}
        limited.text = "rate limited"
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
                    "CEREBRAS_API_KEY": "test-key",
                    "CEREBRAS_MAX_RETRIES": "1",
                    "CEREBRAS_RETRY_JITTER_SECONDS": "0",
                    "CEREBRAS_MIN_INTERVAL_SECONDS": "0",
                },
                clear=False,
            ),
            patch(
                "providers.cerebras_provider.requests.post",
                side_effect=[limited, success],
            ) as post,
            patch("providers.cerebras_provider.time.sleep") as sleep,
        ):
            self.assertEqual(CerebrasProvider().complete("hello"), "ok")

        self.assertEqual(post.call_count, 2)
        self.assertTrue(sleep.called)


if __name__ == "__main__":
    unittest.main()
