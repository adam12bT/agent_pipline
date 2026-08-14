"""
Verifier Agent
--------------
First node in the pipeline. Checks that both the tender and response template
are usable before any expensive LLM calls happen: do they exist, use a
supported format, and contain data? If anything fails, it sets
state["is_verified"] = False and status = "blocked", which the graph
uses to short-circuit straight to the end instead of wasting API calls
on a broken input.

It creates separate AnythingLLM workspaces and indexes both documents through
the extractor. The isolation prevents template boilerplate from contaminating
tender requirement retrieval.

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
    template_path = state.get("response_template_file_path", "")

    for label, path in (("Tender", file_path), ("Response template", template_path)):
        if not path or not os.path.isfile(path):
            errors.append(f"{label} file not found: {path or '(not provided)'}")
            continue

        _, ext = os.path.splitext(path)
        if ext.lower() not in SUPPORTED_EXTENSIONS:
            errors.append(
                f"{label} has unsupported file type '{ext}'. "
                f"Supported types: {sorted(SUPPORTED_EXTENSIONS)}"
            )
        if os.path.getsize(path) < MIN_FILE_SIZE_BYTES:
            errors.append(f"{label} is suspiciously small / possibly empty or corrupt.")

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
    run_token = uuid.uuid4().hex[:8]
    workspace_name = f"rfp-{run_token}"
    template_workspace_name = f"rfp-{run_token}-template"

    try:
        ws_resp = client.create_workspace(workspace_name)
        workspace_slug = ws_resp["workspace"]["slug"]
        template_ws_resp = client.create_workspace(template_workspace_name)
        template_workspace_slug = template_ws_resp["workspace"]["slug"]

        extraction_response = extractor.process_and_index(file_path, workspace_slug)
        document_processing = summarize_extractor_response(extraction_response)
        template_response = extractor.process_and_index(
            template_path, template_workspace_slug
        )
        template_processing = summarize_extractor_response(template_response)
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
        "Verified tender %r and response template %r; indexed into %r and %r",
        file_path,
        template_path,
        workspace_slug,
        template_workspace_slug,
    )
    return {
        "is_verified": True,
        "verification_errors": [],
        "workspace_slug": workspace_slug,
        "response_template_workspace_slug": template_workspace_slug,
        "document_processing": document_processing,
        "response_template_processing": template_processing,
        "status": "running",
    }
