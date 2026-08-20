from rfp.agents.extraction.contract import Input


def test_input_contract():
    value = Input(
        is_verified=True,
        workspace_slug="tender",
        response_template_workspace_slug="template",
        response_template_file_path="template.docx",
    )
    assert value.is_verified
