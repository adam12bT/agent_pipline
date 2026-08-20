from rfp.agents.research.contract import Input


def test_input_contract():
    value = Input(is_verified=True, workspace_slug="tender")
    assert value.workspace_slug == "tender"
