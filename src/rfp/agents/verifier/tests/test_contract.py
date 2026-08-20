from rfp.agents.verifier.contract import Input


def test_input_contract():
    value = Input(tender_file_path="tender.pdf", response_template_file_path="template.docx")
    assert value.tender_file_path == "tender.pdf"
