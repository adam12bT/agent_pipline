"""
Verifier Agent Implementation
--------------
First node in the pipeline. Checks that the tender and any optional response
template are usable before any expensive LLM calls happen: do they exist, use a
supported format, and contain data? It returns only verification facts and
errors; the orchestrator converts a failed verdict into a blocked run.

It always indexes the tender. When a response template is supplied it is
indexed in a separate workspace; otherwise the canonical built-in template is
selected without creating a template workspace.

Returns only the fields declared by its output contract.
"""

import logging
import os
import uuid

from rfp.default_template import DEFAULT_TEMPLATE_VERSION

logger = logging.getLogger(__name__)

SUPPORTED_EXTENSIONS = {".pdf", ".docx"}
MIN_FILE_SIZE_BYTES = 1024  # 1KB — catches empty/corrupt uploads


def verifier_agent(state: dict, *, ingestion=None) -> dict:
    errors = []
    file_path = state.get("tender_file_path", "")
    template_path = state.get("response_template_file_path", "")

    documents = [("Tender", file_path)]
    if template_path:
        documents.append(("Response template", template_path))
    for label, path in documents:
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
    template_workspace_name = f"rfp-{run_token}-template" if template_path else None

    try:
        if ingestion is None:
            raise RuntimeError("TenderIngestion dependency was not provided")
        tender_result = ingestion.ingest(file_path, workspace_prefix=workspace_name)
        workspace_slug = tender_result["workspace_slug"]
        document_processing = tender_result["processing"]
        if template_path:
            template_result = ingestion.ingest(
                template_path, workspace_prefix=template_workspace_name
            )
            template_workspace_slug = template_result["workspace_slug"]
            template_processing = template_result["processing"]
            template_source = "uploaded"
            template_version = None
        else:
            template_workspace_slug = None
            template_processing = {
                "success": True,
                "skipped": True,
                "source": "default",
                "version": DEFAULT_TEMPLATE_VERSION,
            }
            template_source = "default"
            template_version = DEFAULT_TEMPLATE_VERSION
    except Exception as e:
        error_msg = f"Failed to set up workspace / process document: {e}"
        logger.error("Workspace setup failed for %r: %s", file_path, e, exc_info=True)
        return {
            "is_verified": False,
            "verification_errors": [error_msg],
            "errors": [error_msg],
        }

    logger.info(
        "Verified tender %r using %s response template; workspaces=%r/%r",
        file_path,
        template_source,
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
        "template_source": template_source,
        "template_version": template_version,
    }
