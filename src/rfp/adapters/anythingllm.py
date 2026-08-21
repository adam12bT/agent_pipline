"""Single AnythingLLM boundary for ingestion, RAG, and company knowledge."""

import uuid

from .anythingllm_client import AnythingLLMClient
from .extractor import ExtractorClient, summarize_extractor_response
from .retrieval import get_relevant_chunks

COMPANY_WORKSPACES = (
    "company-past-proposals",
    "company-cvs",
    "company-project-references",
)

KNOWLEDGE_CATEGORIES = {
    "past_proposals": COMPANY_WORKSPACES[0],
    "cvs": COMPANY_WORKSPACES[1],
    "project_references": COMPANY_WORKSPACES[2],
}


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

    def ensure_ready(self) -> dict[str, dict[str, bool]]:
        result = {}
        for workspace_slug in COMPANY_WORKSPACES:
            outcome = self.client.get_or_create_workspace(workspace_slug)
            result[workspace_slug] = {"created": bool(outcome["created"])}
        return result

    def knowledge_status(self) -> dict[str, dict]:
        """Return UI-ready status for every persistent knowledge workspace."""
        self.ensure_ready()
        result = {}
        for category, slug in KNOWLEDGE_CATEGORIES.items():
            workspace = self.client.get_workspace(slug)
            documents = []
            if workspace:
                for document in workspace.get("documents", []) or []:
                    documents.append(
                        {
                            "title": document.get("title")
                            or document.get("filename")
                            or "unknown",
                            "id": document.get("id"),
                        }
                    )
            result[category] = {
                "slug": slug,
                "exists": workspace is not None,
                "document_count": len(documents),
                "documents": documents,
            }
        return result

    def upload_knowledge(self, category: str, file_path: str) -> dict:
        """Upload a company document through the same AnythingLLM boundary."""
        if category not in KNOWLEDGE_CATEGORIES:
            raise ValueError(f"Unknown knowledge category: {category}")
        slug = KNOWLEDGE_CATEGORIES[category]
        self.client.get_or_create_workspace(slug)
        return self.client.upload_document(file_path, slug)

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
