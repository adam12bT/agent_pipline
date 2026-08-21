"""Capability protocols injected into agents by the orchestrator."""

from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class RagQuery(Protocol):
    def query(self, workspace_slug: str, query: str, *, top_n: int = 5) -> str: ...


@runtime_checkable
class TenderIngestion(Protocol):
    def ingest(self, file_path: str, *, workspace_prefix: str = "rfp") -> dict[str, Any]: ...


@runtime_checkable
class KnowledgeSearch(Protocol):
    def ensure_ready(self) -> None: ...
    def search(self, workspace_slug: str, query: str, *, top_n: int = 5) -> list[dict]: ...


@runtime_checkable
class WebResearch(Protocol):
    def research(self, query: str) -> str: ...


@runtime_checkable
class OutputScanner(Protocol):
    def scan(self, text: str) -> dict[str, Any]: ...
