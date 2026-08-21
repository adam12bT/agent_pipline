from rfp.agents.research.contract import Input


def test_input_contract():
    value = Input(is_verified=True, scope_summary="Secure citizen-services platform")
    assert "platform" in value.scope_summary
