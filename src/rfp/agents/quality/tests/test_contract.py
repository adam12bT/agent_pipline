from rfp.agents.quality.contract import Input


def test_input_contract():
    value = Input(is_verified=True, draft_proposal="draft")
    assert value.security_passed
