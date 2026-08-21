# Recorded compatibility run

These fixtures were captured from real pipeline run `577336344eda` using
`complete_pipeline_test_tender.pdf` and `phase3_response_template.docx`.
`legacy_api_response.json` is the complete pre-migration API record. The
per-agent input/output fixtures under `../agents/` are deterministic projections
of that record into the new contracts; no synthetic proposal content is used.

The recorded run failed quality, which is intentional: it exercises the real
failure reports, terminal status, full generated draft, and retry boundary.
