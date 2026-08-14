import importlib.util
import os
import tempfile
import unittest
from unittest.mock import patch

_VERIFIER_PATH = os.path.join(
    os.path.dirname(os.path.dirname(__file__)), "agents", "verifier_agent.py"
)
_SPEC = importlib.util.spec_from_file_location("verifier_agent_under_test", _VERIFIER_PATH)
verifier_module = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(verifier_module)


class FakeAnythingLLMClient:
    def create_workspace(self, name):
        return {"workspace": {"slug": name}}


class FakeExtractorClient:
    calls = []

    def process_and_index(self, file_path, workspace_slug):
        self.calls.append((file_path, workspace_slug))
        return {
            "success": True,
            "document": {
                "filename": os.path.basename(file_path),
                "metadata": {
                    "page_count": 2,
                    "native_pages": 1,
                    "ocr_pages": 1,
                    "table_count": 2,
                },
                "pages": [{"page_number": 1}, {"page_number": 2, "used_ocr": True}],
                "blocks": [{"type": "paragraph"}],
                "warnings": [],
            },
            "index_result": {
                "success": True,
                "workspace_slug": workspace_slug,
                "blocks_sent": 26,
                "documents": [{"location": "large-response-entry"}],
                "skipped_existing": 0,
                "rolled_back": 0,
                "error": None,
            },
            "error": None,
        }


class VerifierExtractorIntegrationTests(unittest.TestCase):
    def setUp(self):
        FakeExtractorClient.calls = []

    def test_verifier_delegates_processing_and_keeps_compact_summary(self):
        handle, file_path = tempfile.mkstemp(suffix=".pdf")
        template_handle, template_path = tempfile.mkstemp(suffix=".docx")
        try:
            with os.fdopen(handle, "wb") as file_obj:
                file_obj.write(b"x" * 2048)
            with os.fdopen(template_handle, "wb") as file_obj:
                file_obj.write(b"t" * 2048)

            with (
                patch.object(
                    verifier_module,
                    "AnythingLLMClient",
                    return_value=FakeAnythingLLMClient(),
                ),
                patch.object(
                    verifier_module,
                    "ExtractorClient",
                    return_value=FakeExtractorClient(),
                ),
                patch.object(verifier_module.uuid, "uuid4") as uuid4,
            ):
                uuid4.return_value.hex = "12345678abcdef"
                result = verifier_module.verifier_agent(
                    {
                        "tender_file_path": file_path,
                        "response_template_file_path": template_path,
                    }
                )

            self.assertTrue(result["is_verified"])
            self.assertEqual(result["workspace_slug"], "rfp-12345678")
            self.assertEqual(
                result["response_template_workspace_slug"],
                "rfp-12345678-template",
            )
            self.assertEqual(
                FakeExtractorClient.calls,
                [
                    (file_path, "rfp-12345678"),
                    (template_path, "rfp-12345678-template"),
                ],
            )
            processing = result["document_processing"]
            self.assertEqual(processing["index_result"]["blocks_sent"], 26)
            self.assertNotIn("documents", processing["index_result"])
            self.assertNotIn("blocks", processing["document"])
            self.assertEqual(
                result["response_template_processing"]["index_result"]["blocks_sent"],
                26,
            )
        finally:
            if os.path.exists(file_path):
                os.unlink(file_path)
            if os.path.exists(template_path):
                os.unlink(template_path)

    def test_verifier_blocks_when_response_template_is_missing(self):
        handle, file_path = tempfile.mkstemp(suffix=".pdf")
        try:
            with os.fdopen(handle, "wb") as file_obj:
                file_obj.write(b"x" * 2048)

            result = verifier_module.verifier_agent({"tender_file_path": file_path})

            self.assertFalse(result["is_verified"])
            self.assertIn("Response template file not found", result["verification_errors"][0])
        finally:
            if os.path.exists(file_path):
                os.unlink(file_path)


if __name__ == "__main__":
    unittest.main()
