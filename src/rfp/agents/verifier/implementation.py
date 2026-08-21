"""
Verifier Agent Implementation
--------------
First node in the pipeline. Checks that both the tender and response template
are usable before any expensive LLM calls happen: do they exist, use a
supported format, and contain data? It returns only verification facts and
errors; the orchestrator converts a failed verdict into a blocked run.

It creates separate AnythingLLM workspaces and indexes both documents through
the extractor. The isolation prevents template boilerplate from contaminating
tender requirement retrieval.

Returns only the fields declared by its output contract.
"""

import logging
import os
import uuid

logger = logging.getLogger(__name__)

SUPPORTED_EXTENSIONS = {".pdf", ".docx"}
MIN_FILE_SIZE_BYTES = 1024  # 1KB — catches empty/corrupt uploads


def verifier_agent(state: dict, *, ingestion=None) -> dict:
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
            "errors": errors,
        }

    # Create an isolated AnythingLLM workspace, then delegate parsing, OCR,
    # table recovery, metadata preservation and indexing to the extractor.
    run_token = uuid.uuid4().hex[:8]
    workspace_name = f"rfp-{run_token}"
    template_workspace_name = f"rfp-{run_token}-template"

    try:
        if ingestion is None:
            raise RuntimeError("TenderIngestion dependency was not provided")
        tender_result = ingestion.ingest(file_path, workspace_prefix=workspace_name)
        template_result = ingestion.ingest(
            template_path, workspace_prefix=template_workspace_name
        )
        workspace_slug = tender_result["workspace_slug"]
        template_workspace_slug = template_result["workspace_slug"]
        document_processing = tender_result["processing"]
        template_processing = template_result["processing"]
    except Exception as e:
        error_msg = f"Failed to set up workspace / process document: {e}"
        logger.error("Workspace setup failed for %r: %s", file_path, e, exc_info=True)
        return {
            "is_verified": False,
            "verification_errors": [error_msg],
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
    }
