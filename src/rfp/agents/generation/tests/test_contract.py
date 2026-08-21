from rfp.agents.generation.contract import Input


def test_input_contract():
    value = Input(
        is_verified=True,
        workspace_slug="tender",
        response_template_workspace_slug="template",
    )
    assert value.generation_attempts == 0
