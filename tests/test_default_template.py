from rfp.default_template import (
    DEFAULT_TEMPLATE_VERSION,
    default_response_template,
    resolve_response_template,
)
from rfp.agents.extraction.implementation import extraction_agent
from rfp.agents.generation.implementation import _proposal_sections
from unittest.mock import Mock, patch


def test_default_template_is_used_when_upload_is_absent():
    resolved = resolve_response_template({})

    assert resolved["template_source"] == "default"
    assert resolved["version"] == DEFAULT_TEMPLATE_VERSION
    assert resolved["section_order"] == resolved["required_sections"]
    assert resolved["section_order"]


def test_uploaded_template_wins_without_mutating_input():
    uploaded = {
        "required_sections": ["Client A", "Client B"],
        "section_order": ["Client A", "Client B"],
        "instructions": ["Follow the client order"],
    }
    requirements = {"response_template": uploaded}

    resolved = resolve_response_template(requirements)

    assert resolved["section_order"] == ["Client A", "Client B"]
    assert resolved["template_source"] == "uploaded"
    assert "template_source" not in uploaded


def test_default_template_returns_an_isolated_copy():
    first = default_response_template()
    first["section_order"].append("Mutated")

    assert "Mutated" not in default_response_template()["section_order"]


def test_generation_uses_the_same_canonical_fallback():
    assert _proposal_sections({}) == default_response_template()["section_order"]


def test_extraction_attaches_default_without_template_workspace():
    rag = Mock()
    rag.query.return_value = "The tender requests an independent audit."
    provider = Mock()
    provider.complete.return_value = (
        '{"scope_summary":"Independent audit","deliverables":[],"mandatory_requirements":[]}'
    )

    with patch(
        "rfp.agents.extraction.implementation.get_provider",
        return_value=provider,
    ):
        result = extraction_agent(
            {
                "is_verified": True,
                "workspace_slug": "tender-workspace",
                "response_template_workspace_slug": None,
                "response_template_file_path": None,
            },
            rag=rag,
        )

    assert rag.query.call_count == 1
    assert result["requirements"]["response_template"]["template_source"] == "default"
