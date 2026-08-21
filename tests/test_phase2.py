import os
import tempfile
import unittest
from unittest.mock import patch

import ingest_company_corpus as ingestion
from rfp.adapters import retrieval


class FakeSearchClient:
    def __init__(self, results):
        self.results = results
        self.requested_top_n = None

    def vector_search(self, workspace_slug, query, top_n, score_threshold):
        self.requested_top_n = top_n
        return self.results


class FakeIngestionClient:
    def __init__(self):
        self.uploads = []

    def upload_document(self, file_path, workspace_slug):
        self.uploads.append((file_path, workspace_slug))
        return {"documents": [{"location": os.path.basename(file_path)}]}


class FakeIngestionAdapter:
    def __init__(self, client):
        self.client = client

    def ensure_ready(self):
        return {"company-past-proposals": {"created": False}}

    def upload_knowledge(self, category, file_path):
        return self.client.upload_document(file_path, f"company-{category.replace('_', '-')}")


class RetrievalTests(unittest.TestCase):
    def test_reranker_deduplicates_and_promotes_query_terms(self):
        results = [
            {"text": "generic project delivery", "score": 0.8},
            {"text": "budget deadline compliance matrix", "score": 0.78},
            {"text": "  BUDGET deadline compliance matrix ", "score": 0.78},
        ]

        ranked = retrieval.rerank_results(results, "budget deadline", top_n=3)

        self.assertEqual(len(ranked), 2)
        self.assertEqual(ranked[0]["text"], "budget deadline compliance matrix")
        self.assertIn("rerank_score", ranked[0])

    def test_search_requests_extra_candidates(self):
        client = FakeSearchClient([{"text": "known phrase", "score": 0.9}])

        found = retrieval.search_relevant_chunks(client, "workspace", "known phrase", top_n=4)

        expected = 4 * retrieval.RERANK_CANDIDATE_MULTIPLIER if retrieval.RERANK_ENABLED else 4
        self.assertEqual(client.requested_top_n, expected)
        self.assertEqual(len(found), 1)


class IngestionTests(unittest.TestCase):
    def test_second_run_skips_content_already_in_manifest(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            corpus_root = os.path.join(temp_dir, "company_corpus")
            proposals = os.path.join(corpus_root, "past_proposals")
            os.makedirs(proposals)
            document = os.path.join(proposals, "proposal.docx")
            with open(document, "wb") as file_obj:
                file_obj.write(b"fake docx content for hashing")

            fake_client = FakeIngestionClient()
            folder_map = {"past_proposals": "company-past-proposals"}
            manifest_path = os.path.join(corpus_root, ".ingestion_manifest.json")

            with (
                patch.object(ingestion, "CORPUS_ROOT", corpus_root),
                patch.object(ingestion, "MANIFEST_PATH", manifest_path),
                patch.object(ingestion, "FOLDER_TO_WORKSPACE", folder_map),
                patch.object(
                    ingestion,
                    "AnythingLLMAdapter",
                    return_value=FakeIngestionAdapter(fake_client),
                ),
            ):
                first = ingestion.ingest_all()
                second = ingestion.ingest_all()

            self.assertEqual(first["uploaded"], ["past_proposals/proposal.docx"])
            self.assertEqual(second["skipped"], ["past_proposals/proposal.docx"])
            self.assertEqual(len(fake_client.uploads), 1)


if __name__ == "__main__":
    unittest.main()
