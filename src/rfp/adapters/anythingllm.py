"""AnythingLLM adapter implementing the RAG and company-knowledge ports."""

import uuid

from anythingllm_client import AnythingLLMClient
from extractor_client import ExtractorClient, summarize_extractor_response
from retrieval import get_relevant_chunks

COMPANY_WORKSPACES = (
    "company-past-proposals",
    "company-cvs",
    "company-project-references",
)


class AnythingLLMAdapter:
    def __init__(
        self,
        client: AnythingLLMClient | None = None,
        extractor: ExtractorClient | None = None,
    ):
        self.client = client or AnythingLLMClient()
        self.extractor = extractor or ExtractorClient()

    def query(self, workspace_slug: str, query: str, *, top_n: int = 5) -> str:
        return get_relevant_chunks(self.client, workspace_slug, query, top_n=top_n)

    def search(self, workspace_slug: str, query: str, *, top_n: int = 5) -> list[dict]:
        return self.client.vector_search(workspace_slug, query, top_n=top_n)

    def ensure_ready(self) -> None:
        for workspace_slug in COMPANY_WORKSPACES:
            self.client.get_or_create_workspace(workspace_slug)

    def ingest(self, file_path: str, *, workspace_prefix: str = "rfp") -> dict:
        workspace_name = (
            f"rfp-{uuid.uuid4().hex[:8]}"
            if workspace_prefix == "rfp"
            else workspace_prefix
        )
        response = self.client.create_workspace(workspace_name)
        workspace_slug = response["workspace"]["slug"]
        extracted = self.extractor.process_and_index(file_path, workspace_slug)
        return {
            "workspace_slug": workspace_slug,
            "processing": summarize_extractor_response(extracted),
        }
