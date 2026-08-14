"""
Verifier Agent
--------------
First node in the pipeline. Checks the tender package is actually usable
before any expensive LLM calls happen: does the file exist, is it a
supported format, is it non-empty? If anything fails, it sets
state["is_verified"] = False and status = "blocked", which the graph
uses to short-circuit straight to the end instead of wasting API calls
on a broken input.

It also does the one-time work of creating a fresh AnythingLLM workspace
for this run and uploading/embedding the tender document into it, since
every later agent needs that workspace to already exist and be searchable.

Returns a PARTIAL state dict (see state.py) — not the full `{**state, ...}`
— since this keeps the pattern consistent across every agent, including
the two (Extraction, Research) that now run in parallel and can't safely
spread the full state back.
"""

import logging
import os
import uuid

from anythingllm_client import AnythingLLMClient
from extractor_client import ExtractorClient, summarize_extractor_response

logger = logging.getLogger(__name__)

SUPPORTED_EXTENSIONS = {".pdf", ".docx"}
MIN_FILE_SIZE_BYTES = 1024  # 1KB — catches empty/corrupt uploads


def verifier_agent(state: dict) -> dict:
    errors = []
    file_path = state.get("tender_file_path", "")

    if not file_path or not os.path.isfile(file_path):
        errors.append(f"File not found: {file_path}")
    else:
        _, ext = os.path.splitext(file_path)
        if ext.lower() not in SUPPORTED_EXTENSIONS:
            errors.append(
                f"Unsupported file type '{ext}'. Supported types: {sorted(SUPPORTED_EXTENSIONS)}"
            )
        if os.path.getsize(file_path) < MIN_FILE_SIZE_BYTES:
            errors.append("File is suspiciously small / possibly empty or corrupt.")

    if errors:
        logger.warning("Verification failed for %r: %s", file_path, errors)
        return {
            "is_verified": False,
            "verification_errors": errors,
            "status": "blocked",
            "errors": errors,
        }

    # Create an isolated AnythingLLM workspace, then delegate parsing, OCR,
    # table recovery, metadata preservation and indexing to the extractor.
    client = AnythingLLMClient()
    extractor = ExtractorClient()
    workspace_name = f"rfp-{uuid.uuid4().hex[:8]}"

    try:
        ws_resp = client.create_workspace(workspace_name)
        workspace_slug = ws_resp["workspace"]["slug"]

        extraction_response = extractor.process_and_index(file_path, workspace_slug)
        document_processing = summarize_extractor_response(extraction_response)
    except Exception as e:
        error_msg = f"Failed to set up workspace / process document: {e}"
        logger.error("Workspace setup failed for %r: %s", file_path, e, exc_info=True)
        return {
            "is_verified": False,
            "verification_errors": [error_msg],
            "status": "blocked",
            "errors": [error_msg],
        }

    logger.info(
        "Verified %r; extractor indexed it into workspace %r",
        file_path,
        workspace_slug,
    )
    return {
        "is_verified": True,
        "verification_errors": [],
        "workspace_slug": workspace_slug,
        "document_processing": document_processing,
        "status": "running",
    }
