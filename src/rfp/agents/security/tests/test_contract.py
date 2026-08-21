from rfp.agents.security.contract import Input


def test_input_contract():
    value = Input(is_verified=True, draft_proposal="draft")
    assert value.draft_proposal == "draft"
