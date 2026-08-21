import os
import tempfile
import unittest
from unittest.mock import patch

from rfp.agents.verifier import implementation as verifier_module


class FakeIngestion:
    def __init__(self):
        self.calls = []

    def ingest(self, file_path: str, *, workspace_prefix: str = "rfp") -> dict:
        self.calls.append((file_path, workspace_prefix))
        return {
            "workspace_slug": workspace_prefix,
            "processing": {
                "success": True,
                "document": {
                    "filename": os.path.basename(file_path),
                    "metadata": {"page_count": 2, "native_pages": 1, "ocr_pages": 1},
                    "pages": [{"page_number": 1}, {"page_number": 2, "used_ocr": True}],
                    "warnings": [],
                },
                "index_result": {
                    "success": True,
                    "workspace_slug": workspace_prefix,
                    "blocks_sent": 26,
                    "skipped_existing": 0,
                    "rolled_back": 0,
                    "error": None,
                },
                "error": None,
            },
        }


class VerifierExtractorIntegrationTests(unittest.TestCase):
    def test_verifier_delegates_processing_through_injected_port(self):
        handle, file_path = tempfile.mkstemp(suffix=".pdf")
        template_handle, template_path = tempfile.mkstemp(suffix=".docx")
        ingestion = FakeIngestion()
        try:
            with os.fdopen(handle, "wb") as file_obj:
                file_obj.write(b"x" * 2048)
            with os.fdopen(template_handle, "wb") as file_obj:
                file_obj.write(b"t" * 2048)

            with patch.object(verifier_module.uuid, "uuid4") as uuid4:
                uuid4.return_value.hex = "12345678abcdef"
                result = verifier_module.verifier_agent(
                    {
                        "tender_file_path": file_path,
                        "response_template_file_path": template_path,
                    },
                    ingestion=ingestion,
                )

            self.assertTrue(result["is_verified"])
            self.assertEqual(result["workspace_slug"], "rfp-12345678")
            self.assertEqual(
                result["response_template_workspace_slug"],
                "rfp-12345678-template",
            )
            self.assertEqual(
                ingestion.calls,
                [
                    (file_path, "rfp-12345678"),
                    (template_path, "rfp-12345678-template"),
                ],
            )
            self.assertEqual(result["document_processing"]["index_result"]["blocks_sent"], 26)
        finally:
            os.unlink(file_path)
            os.unlink(template_path)

    def test_verifier_uses_default_when_response_template_is_missing(self):
        handle, file_path = tempfile.mkstemp(suffix=".pdf")
        try:
            with os.fdopen(handle, "wb") as file_obj:
                file_obj.write(b"x" * 2048)
            result = verifier_module.verifier_agent(
                {"tender_file_path": file_path}, ingestion=FakeIngestion()
            )
            self.assertTrue(result["is_verified"])
            self.assertEqual(result["template_source"], "default")
            self.assertIsNone(result["response_template_workspace_slug"])
            self.assertTrue(result["response_template_processing"]["skipped"])
        finally:
            os.unlink(file_path)


if __name__ == "__main__":
    unittest.main()
