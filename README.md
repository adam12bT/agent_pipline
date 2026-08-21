---
title: RFP Pipeline
emoji: 📄
colorFrom: blue
colorTo: indigo
sdk: docker
app_port: 7860
pinned: false
---

# RFP Pipeline

Multi-agent RFP/tender proposal pipeline (LangGraph + AnythingLLM RAG).
Deployed automatically from the `main` branch via GitHub Actions.

Canonical package: `src/rfp`. FastAPI entry point: `rfp.api.app:app`.
See `/api/health` for a liveness check.

Each run detail response also exposes a top-level `telemetry` object with
exact per-agent durations, repeated-attempt timings, LLM call counts, and
provider-reported prompt/completion token usage. Groq and Ollama direct calls
report usage; external services that do not return token metadata are marked
implicitly by their absence rather than estimated.

Install the complete synchronous application with `pip install -e ".[full]"`.
Install only one agent's dependencies with an extra such as
`pip install -e ".[quality]"` or `pip install -e ".[research]"`.

The Docker image installs `.[full]` directly from `pyproject.toml`;
`requirements.txt` is no longer used. A complete real pre-migration run and
its six projected agent input/output pairs are stored under `tests/fixtures/`.
Verify exact offline CLI/API compatibility with:

```powershell
python -m rfp.compatibility_cli --replay-only
```

For a live comparison, run the following only after confirming that both
files may be uploaded to the configured hosted AnythingLLM and extractor.
It creates new persistent AnythingLLM workspaces:

```powershell
python -m rfp.compatibility_cli tender.pdf response-template.docx `
  --save-current current-live-result.json
```

## Starting a proposal run

The tender is required and the client's response template is optional. Submit
them as multipart fields named `file` and `template` when both exist:

```powershell
curl.exe -X POST http://localhost:8000/api/runs `
  -F "file=@C:\documents\tender.pdf" `
  -F "template=@C:\documents\response-template.docx"
```

The verifier rejects an absent, unsupported, or empty tender and validates any
uploaded template. Uploaded templates are indexed separately so tender facts
and template instructions remain isolated. When `template` is omitted, every
stage uses the canonical versioned structure in `src/rfp/default_template.py`.

The Quality stage preserves the exact evidence supplied to Generation and
uses it to score groundedness and coherence. A failed review is included as
revision feedback on the next generation attempt. After the first complete
draft, retries regenerate only the template sections associated with missing
content, unsupported claims, contradictions, or localized quality failures;
accepted sections are preserved. The Groq provider honors
`Retry-After` and otherwise uses exponential backoff with jitter. Production
Docker builds also fail if LLM Guard cannot be imported; `/api/health` reports
the security and quality scanner capabilities.

## Phase 2: RAG ingestion and validation

The standalone extractor is responsible for PDF/DOCX parsing, bilingual OCR,
table recovery and layout metadata. It sends structured content to AnythingLLM,
which owns workspaces, token chunking, embeddings and Qdrant retrieval. This
repository orchestrates those services and adds a lightweight retrieval reranker.

1. In AnythingLLM, select **Qdrant** as the vector database and configure its
   URL/API key. Set document chunk size to about 512 tokens and overlap to
   about 50 tokens when those collector settings are available. Restart
   AnythingLLM after changing the vector database or embedding model.
2. Put real company files under `company_corpus/past_proposals`,
   `company_corpus/cvs`, and `company_corpus/project_references`.
3. Run `python ingest_company_corpus.py`. Re-running skips identical content
   using `company_corpus/.ingestion_manifest.json`. Use `--force` only for an
   intentional re-index.
4. Validate real documents with exact, distinctive phrases:

```powershell
python phase2_smoke_test.py `
  --native-pdf samples/native.pdf --native-pdf-phrase "EXACT UNIQUE PDF TEXT" `
  --docx samples/sample.docx --docx-phrase "EXACT UNIQUE DOCX TEXT" `
  --scanned-pdf samples/scan.pdf --scanned-pdf-phrase "EXACT TEXT VISIBLE IN SCAN" `
  --table-pdf samples/table.pdf --table-pdf-phrase "DISTINCTIVE TABLE CELL VALUE"
```

A scanned-PDF pass proves that OCR text is searchable for that real sample.
If it fails, first check that OCR is enabled in the deployed AnythingLLM build.
Only add a separate OCR engine such as Surya if representative scans still fail.
The table case similarly checks that critical numbers/cell values survive the
document parser before the agent pipeline depends on them.

The smoke command reports whether source metadata came back with retrieved
chunks. Qdrant itself is configured and verified in AnythingLLM; this client
cannot identify the vector backend from the generic vector-search response.

## Required Space secrets

Set these under this Space's **Settings → Repository secrets**:

- `ANYTHINGLLM_BASE_URL` — URL of your deployed AnythingLLM Space's API
  (e.g. `https://your-username-anythingllm.hf.space/api`), NOT localhost.
- `EXTRACTOR_BASE_URL` — URL of the deployed extraction Space, without an
  `/api` suffix (e.g. `https://your-username-extractor.hf.space`).
- `EXTRACTOR_API_KEY` — optional Bearer token if extractor authentication is enabled.
- `TAVILY_API_KEY` — search retriever key used by GPT Researcher.
- Whichever LLM provider key GPT Researcher is configured to use
  (e.g. `OPENAI_API_KEY`), if not already covered above.
