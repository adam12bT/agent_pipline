"""
REST API for the RFP pipeline React UI.

Run with (from the project root, so `graph`, `state`, `agents`, etc. are
importable):

    uvicorn backend.api:app --reload --port 8000

Endpoints
---------
Pipeline runs:
  POST   /api/runs                       upload a tender file, start a run
  GET    /api/runs                       list all runs (summary)
  GET    /api/runs/{run_id}              full state of one run (poll this)
  GET    /api/runs/{run_id}/download     download the draft proposal (.md)

Company knowledge base:
  GET    /api/knowledge                  status + doc counts for the 3 workspaces
  POST   /api/knowledge/{category}/upload  upload file(s) into one category

Misc:
  GET    /api/health
"""

import os
import sys
import shutil
import tempfile
import uuid

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import PlainTextResponse
from dotenv import load_dotenv

# Make the project root (parent of backend/) importable, since all the
# pipeline modules (graph, state, agents, anythingllm_client, ...) live
# there as flat top-level modules, not inside the `backend` package.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

load_dotenv()

from backend.run_store import create_run, get_run, list_runs, PIPELINE_STAGES  # noqa: E402
from anythingllm_client import AnythingLLMClient  # noqa: E402
from company_knowledge import (  # noqa: E402
    ALL_COMPANY_WORKSPACES,
    PROPOSALS_WORKSPACE,
    CVS_WORKSPACE,
    REFERENCES_WORKSPACE,
    ensure_company_workspaces,
)

app = FastAPI(title="RFP Pipeline API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # dev-friendly; tighten this before deploying publicly
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

UPLOAD_DIR = os.path.join(tempfile.gettempdir(), "rfp-pipeline-uploads")
os.makedirs(UPLOAD_DIR, exist_ok=True)

SUPPORTED_TENDER_EXTENSIONS = {".pdf", ".docx"}

KNOWLEDGE_CATEGORIES = {
    "past_proposals": PROPOSALS_WORKSPACE,
    "cvs": CVS_WORKSPACE,
    "project_references": REFERENCES_WORKSPACE,
}


def _save_upload(upload: UploadFile, subdir: str) -> str:
    target_dir = os.path.join(UPLOAD_DIR, subdir)
    os.makedirs(target_dir, exist_ok=True)
    ext = os.path.splitext(upload.filename or "")[1].lower()
    safe_name = f"{uuid.uuid4().hex[:8]}_{upload.filename or 'file'}"
    dest_path = os.path.join(target_dir, safe_name)
    with open(dest_path, "wb") as f:
        shutil.copyfileobj(upload.file, f)
    return dest_path, ext


# --------------------------------------------------------------------------
# Health
# --------------------------------------------------------------------------

@app.get("/api/health")
def health():
    return {"ok": True, "stages": PIPELINE_STAGES}


# --------------------------------------------------------------------------
# Pipeline runs
# --------------------------------------------------------------------------

@app.post("/api/runs")
async def start_run(file: UploadFile = File(...)):
    ext = os.path.splitext(file.filename or "")[1].lower()
    if ext not in SUPPORTED_TENDER_EXTENSIONS:
        raise HTTPException(
            400,
            f"Unsupported file type '{ext}'. Supported types: {sorted(SUPPORTED_TENDER_EXTENSIONS)}",
        )

    dest_path, _ = _save_upload(file, "tenders")
    run_id = create_run(tender_file_path=dest_path, original_filename=file.filename)
    return {"run_id": run_id}


@app.get("/api/runs")
def get_runs():
    return list_runs()


@app.get("/api/runs/{run_id}")
def get_run_detail(run_id: str):
    record = get_run(run_id)
    if not record:
        raise HTTPException(404, "Run not found")
    return record


@app.get("/api/runs/{run_id}/download")
def download_draft(run_id: str):
    record = get_run(run_id)
    if not record:
        raise HTTPException(404, "Run not found")
    draft = record.get("state", {}).get("draft_proposal")
    if not draft:
        raise HTTPException(400, "No draft proposal available yet for this run")
    filename = f"proposal-{run_id}.md"
    return PlainTextResponse(
        draft,
        media_type="text/markdown",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


# --------------------------------------------------------------------------
# Company knowledge base
# --------------------------------------------------------------------------

@app.get("/api/knowledge")
def knowledge_status():
    client = AnythingLLMClient()
    ensure_company_workspaces(client)  # idempotent — safe to call on every poll

    result = {}
    for category, slug in KNOWLEDGE_CATEGORIES.items():
        try:
            workspace = client.get_workspace(slug)
        except Exception as e:
            result[category] = {"slug": slug, "error": str(e), "documents": []}
            continue

        documents = []
        if workspace:
            # Real AnythingLLM workspaces include a "documents" array;
            # tolerate its absence gracefully (e.g. against a stripped-down
            # server fork that doesn't return it).
            for doc in workspace.get("documents", []) or []:
                documents.append(
                    {
                        "title": doc.get("title") or doc.get("filename") or "unknown",
                        "id": doc.get("id"),
                    }
                )
        result[category] = {
            "slug": slug,
            "exists": workspace is not None,
            "document_count": len(documents),
            "documents": documents,
        }
    return result


@app.post("/api/knowledge/{category}/upload")
async def upload_knowledge_file(category: str, file: UploadFile = File(...)):
    if category not in KNOWLEDGE_CATEGORIES:
        raise HTTPException(
            400, f"Unknown category '{category}'. Must be one of {list(KNOWLEDGE_CATEGORIES)}"
        )

    ext = os.path.splitext(file.filename or "")[1].lower()
    if ext not in {".pdf", ".docx"}:
        raise HTTPException(400, f"Unsupported file type '{ext}'. Only .pdf and .docx allowed.")

    slug = KNOWLEDGE_CATEGORIES[category]
    client = AnythingLLMClient()
    client.get_or_create_workspace(slug)

    dest_path, _ = _save_upload(file, f"knowledge/{category}")
    try:
        result = client.upload_document(dest_path, slug)
    except Exception as e:
        raise HTTPException(502, f"AnythingLLM upload failed: {e}")

    return {"ok": True, "filename": file.filename, "workspace": slug, "result": result}
