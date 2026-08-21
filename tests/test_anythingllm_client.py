import unittest
from unittest.mock import Mock, patch

from rfp.adapters import anythingllm_client as client_module
from rfp.adapters.anythingllm_client import AnythingLLMClient


class AnythingLLMRetryTests(unittest.TestCase):
    def setUp(self):
        client_module._NEXT_REQUEST_AT = 0.0

    def test_base_url_is_resolved_when_client_is_created(self):
        with patch.dict(
            "os.environ",
            {"ANYTHINGLLM_BASE_URL": "https://hosted.example/api"},
        ):
            client = AnythingLLMClient()

        self.assertEqual(client.base_url, "https://hosted.example/api")

    def test_workspace_creation_retries_after_429(self):
        rate_limited = Mock(status_code=429, headers={"Retry-After": "0"})
        created = Mock(status_code=200, headers={})
        created.json.return_value = {"workspace": {"slug": "rfp-test"}}
        created.raise_for_status.return_value = None

        client = AnythingLLMClient("https://anythingllm.example/api")
        client.max_retries = 1
        client.retry_jitter_seconds = 0
        client.request_min_interval_seconds = 0

        with patch(
            "rfp.adapters.anythingllm_client.requests.request",
            side_effect=[rate_limited, created],
        ) as request:
            result = client.create_workspace("rfp-test")

        self.assertEqual(result["workspace"]["slug"], "rfp-test")
        self.assertEqual(request.call_count, 2)


if __name__ == "__main__":
    unittest.main()
