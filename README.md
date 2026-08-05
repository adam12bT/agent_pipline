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

Backend: FastAPI (`backend/api.py`). See `/api/health` for a liveness check.

## Required Space secrets

Set these under this Space's **Settings → Repository secrets**:

- `ANYTHINGLLM_BASE_URL` — URL of your deployed AnythingLLM Space's API
  (e.g. `https://your-username-anythingllm.hf.space/api`), NOT localhost.
- `TAVILY_API_KEY` — search retriever key used by GPT Researcher.
- Whichever LLM provider key GPT Researcher is configured to use
  (e.g. `OPENAI_API_KEY`), if not already covered above.