"""Shared Pydantic primitives used by per-agent contracts."""

from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class ContractModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class VerifierInput(ContractModel):
    run_id: str | None = None
    tender_file_path: str
    response_template_file_path: str | None = None


class VerifierOutput(ContractModel):
    is_verified: bool
    verification_errors: list[str] = Field(default_factory=list)
    workspace_slug: str | None = None
    response_template_workspace_slug: str | None = None
    document_processing: dict[str, Any] = Field(default_factory=dict)
    response_template_processing: dict[str, Any] = Field(default_factory=dict)
    template_source: str = "default"
    template_version: str | None = None
    errors: list[str] = Field(default_factory=list)


class ExtractionInput(ContractModel):
    is_verified: bool
    workspace_slug: str
    response_template_workspace_slug: str | None = None
    response_template_file_path: str | None = None


class ExtractionOutput(ContractModel):
    requirements: dict[str, Any] = Field(default_factory=dict)
    errors: list[str] = Field(default_factory=list)


class ResearchInput(ContractModel):
    is_verified: bool
    scope_summary: str
    budget: str = "none stated"
    selection_method: str | None = None
    deliverables: list[str] = Field(default_factory=list)
    technical_constraints: list[str] = Field(default_factory=list)
    mandatory_requirements: list[str] = Field(default_factory=list)


class ResearchOutput(ContractModel):
    research_summary: str = ""
    research_relevant: bool = False
    relevance_report: dict[str, Any] = Field(default_factory=dict)
    errors: list[str] = Field(default_factory=list)


class GenerationInput(ContractModel):
    run_id: str | None = None
    is_verified: bool
    workspace_slug: str
    response_template_workspace_slug: str | None = None
    requirements: dict[str, Any] = Field(default_factory=dict)
    research_summary: str = ""
    previous_draft: str = ""
    previous_generation_evidence: dict[str, Any] = Field(default_factory=dict)
    generation_attempts: int = 0
    quality_report: dict[str, Any] = Field(default_factory=dict)


class GenerationOutput(ContractModel):
    draft_proposal: str = ""
    generation_evidence: dict[str, Any] = Field(default_factory=dict)
    generation_attempts: int = 0
    errors: list[str] = Field(default_factory=list)


class SecurityInput(ContractModel):
    is_verified: bool
    draft_proposal: str = ""


class SecurityOutput(ContractModel):
    security_passed: bool
    security_report: dict[str, Any] = Field(default_factory=dict)


class QualityInput(ContractModel):
    is_verified: bool
    security_passed: bool = True
    draft_proposal: str = ""
    generation_evidence: dict[str, Any] = Field(default_factory=dict)
    generation_attempts: int = 0
    requirements: dict[str, Any] = Field(default_factory=dict)


class QualityOutput(ContractModel):
    quality_passed: bool
    quality_report: dict[str, Any] = Field(default_factory=dict)
