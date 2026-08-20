"""Composition root for local agent side-effect dependencies."""

from dataclasses import dataclass

from rfp.adapters import (
    AnythingLLMAdapter,
    ConfiguredQualityScanner,
    ConfiguredSecurityScanner,
    GPTResearcherAdapter,
)
from rfp.contracts import KnowledgeSearch, OutputScanner, RagQuery, TenderIngestion, WebResearch


@dataclass(frozen=True)
class PipelineDependencies:
    ingestion: TenderIngestion
    rag: RagQuery
    knowledge: KnowledgeSearch
    web: WebResearch
    security_scanner: OutputScanner
    quality_scanner: OutputScanner

    @classmethod
    def defaults(cls) -> "PipelineDependencies":
        anythingllm = AnythingLLMAdapter()
        return cls(
            ingestion=anythingllm,
            rag=anythingllm,
            knowledge=anythingllm,
            web=GPTResearcherAdapter(),
            security_scanner=ConfiguredSecurityScanner(),
            quality_scanner=ConfiguredQualityScanner(),
        )
