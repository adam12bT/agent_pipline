"""
REST API for the RFP pipeline UI.

Run with (from the project root, so `graph`, `state`, `agents`, etc. are
importable):

    uvicorn rfp.api.app:app --reload --port 8000

Endpoints
---------
Pipeline runs:
  POST   /api/runs                       upload tender and optional response template
  GET    /api/runs                       list all runs (summary)
  GET    /api/runs/{run_id}              full state of one run (poll this)
  GET    /api/runs/{run_id}/download     download the draft proposal (.md)

Company knowledge base:
  GET    /api/knowledge                  status + doc counts for the 3 workspaces
  POST   /api/knowledge/{category}/upload  upload file(s) into one category

Misc:
  GET    /api/health
"""

import logging
import os
import shutil
import tempfile
import uuid

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import PlainTextResponse
from dotenv import load_dotenv

load_dotenv()

from logging_config import configure_logging  # noqa: E402

configure_logging()

from rfp.api.run_store import create_run, get_run, list_runs, PIPELINE_STAGES  # noqa: E402
from rfp.adapters import AnythingLLMAdapter  # noqa: E402
from rfp.adapters.anythingllm import KNOWLEDGE_CATEGORIES  # noqa: E402
from rfp.agents.quality.implementation import (  # noqa: E402
    llm_guard_available as quality_guard_available,
)
from rfp.agents.security.implementation import (  # noqa: E402
    llm_guard_available as security_guard_available,
    security_scanner_status,
)

logger = logging.getLogger(__name__)

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
    return {
        "ok": True,
        "stages": PIPELINE_STAGES,
        "capabilities": {
            "llm_guard_quality": quality_guard_available(),
            "llm_guard_security": security_guard_available(),
            "security_scanner": security_scanner_status(),
        },
    }


# --------------------------------------------------------------------------
# Pipeline runs
# --------------------------------------------------------------------------

@app.post("/api/runs")
async def start_run(
    file: UploadFile = File(...),
    template: UploadFile | None = File(default=None),
):
    ext = os.path.splitext(file.filename or "")[1].lower()
    if ext not in SUPPORTED_TENDER_EXTENSIONS:
        raise HTTPException(
            400,
            f"Unsupported file type '{ext}'. Supported types: {sorted(SUPPORTED_TENDER_EXTENSIONS)}",
        )

    if template is not None:
        template_ext = os.path.splitext(template.filename or "")[1].lower()
        if template_ext not in SUPPORTED_TENDER_EXTENSIONS:
            raise HTTPException(
                400,
                f"Unsupported response template type '{template_ext}'. "
                f"Supported types: {sorted(SUPPORTED_TENDER_EXTENSIONS)}",
            )

    dest_path, _ = _save_upload(file, "tenders")
    template_path = None
    if template is not None:
        template_path, _ = _save_upload(template, "response-templates")
    run_id = create_run(
        tender_file_path=dest_path,
        original_filename=file.filename,
        response_template_file_path=template_path,
        response_template_filename=template.filename if template else None,
    )
    logger.info(
        "Started run %s for tender %r with response template %r",
        run_id,
        file.filename,
        template.filename if template else "built-in default",
    )
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
    try:
        return AnythingLLMAdapter().knowledge_status()
    except Exception as exc:
        logger.error("Failed to read company knowledge status: %s", exc)
        raise HTTPException(502, f"AnythingLLM status request failed: {exc}")


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
    adapter = AnythingLLMAdapter()

    dest_path, _ = _save_upload(file, f"knowledge/{category}")
    try:
        result = adapter.upload_knowledge(category, dest_path)
    except Exception as e:
        logger.error(
            "Knowledge upload failed for category=%r file=%r: %s",
            category, file.filename, e,
        )
        raise HTTPException(502, f"AnythingLLM upload failed: {e}")

    logger.info("Uploaded %r into knowledge category %r (workspace %r)", file.filename, category, slug)
    return {"ok": True, "filename": file.filename, "workspace": slug, "result": result}
