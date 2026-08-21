"""Bulk-index the company's reusable PDF/DOCX corpus in AnythingLLM.

Expected folder structure::

    company_corpus/
    |-- past_proposals/
    |-- cvs/
    `-- project_references/

Run ``python ingest_company_corpus.py``. A SHA-256 manifest makes normal
re-runs idempotent, preventing duplicate vectors. Use ``--force`` only when a
file must intentionally be re-indexed.
"""

import argparse
import hashlib
import json
import os
from datetime import datetime, timezone

from rfp.adapters import AnythingLLMAdapter
from rfp.adapters.anythingllm import KNOWLEDGE_CATEGORIES

PROPOSALS_WORKSPACE = KNOWLEDGE_CATEGORIES["past_proposals"]
CVS_WORKSPACE = KNOWLEDGE_CATEGORIES["cvs"]
REFERENCES_WORKSPACE = KNOWLEDGE_CATEGORIES["project_references"]

CORPUS_ROOT = os.path.join(os.path.dirname(__file__), "company_corpus")
MANIFEST_PATH = os.path.join(CORPUS_ROOT, ".ingestion_manifest.json")

FOLDER_TO_WORKSPACE = {
    "past_proposals": PROPOSALS_WORKSPACE,
    "cvs": CVS_WORKSPACE,
    "project_references": REFERENCES_WORKSPACE,
}

SUPPORTED_EXTENSIONS = {".pdf", ".docx"}


def _iter_supported_files(folder: str):
    if not os.path.isdir(folder):
        return
    for entry in sorted(os.listdir(folder)):
        path = os.path.join(folder, entry)
        if os.path.isfile(path) and os.path.splitext(entry)[1].lower() in SUPPORTED_EXTENSIONS:
            yield path


def _sha256(file_path: str) -> str:
    digest = hashlib.sha256()
    with open(file_path, "rb") as file_obj:
        for block in iter(lambda: file_obj.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_manifest() -> dict:
    try:
        with open(MANIFEST_PATH, "r", encoding="utf-8") as file_obj:
            data = json.load(file_obj)
        return data if isinstance(data, dict) else {"version": 1, "documents": {}}
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return {"version": 1, "documents": {}}


def _save_manifest(manifest: dict) -> None:
    os.makedirs(CORPUS_ROOT, exist_ok=True)
    temporary_path = f"{MANIFEST_PATH}.tmp"
    with open(temporary_path, "w", encoding="utf-8") as file_obj:
        json.dump(manifest, file_obj, ensure_ascii=False, indent=2, sort_keys=True)
    os.replace(temporary_path, MANIFEST_PATH)


def ingest_all(force: bool = False) -> dict:
    adapter = AnythingLLMAdapter()
    manifest = _load_manifest()
    indexed_documents = manifest.setdefault("documents", {})
    report = {"uploaded": [], "skipped": [], "failed": []}

    print("Ensuring the 3 company knowledge workspaces exist...")
    status = adapter.ensure_ready()
    for slug, info in status.items():
        print(f"  {slug}: {'created' if info['created'] else 'already existed'}")

    for folder_name, workspace_slug in FOLDER_TO_WORKSPACE.items():
        folder_path = os.path.join(CORPUS_ROOT, folder_name)
        files = list(_iter_supported_files(folder_path))

        if not files:
            print(
                f"\n[{folder_name}] No PDF/DOCX files found in {folder_path} - skipping."
            )
            continue

        print(
            f"\n[{folder_name}] Checking {len(files)} file(s) for workspace "
            f"'{workspace_slug}'..."
        )
        for file_path in files:
            relative_path = os.path.relpath(file_path, CORPUS_ROOT).replace("\\", "/")
            content_hash = _sha256(file_path)
            manifest_key = f"{workspace_slug}:{content_hash}"

            if not force and manifest_key in indexed_documents:
                print(f"  SKIPPED (already indexed): {os.path.basename(file_path)}")
                report["skipped"].append(relative_path)
                continue

            try:
                upload_result = adapter.upload_knowledge(folder_name, file_path)
                print(f"  OK: {os.path.basename(file_path)}")
                report["uploaded"].append(relative_path)
                indexed_documents[manifest_key] = {
                    "path": relative_path,
                    "workspace": workspace_slug,
                    "sha256": content_hash,
                    "size_bytes": os.path.getsize(file_path),
                    "indexed_at": datetime.now(timezone.utc).isoformat(),
                    "anythingllm_documents": upload_result.get("documents", []),
                }
                _save_manifest(manifest)
            except Exception as exc:
                print(f"  FAILED: {os.path.basename(file_path)} - {exc}")
                report["failed"].append({"path": relative_path, "error": str(exc)})

    print(
        f"\nDone. {len(report['uploaded'])} uploaded, "
        f"{len(report['skipped'])} skipped, {len(report['failed'])} failed."
    )
    if not any(report.values()):
        print("Create the company_corpus folders and add real PDF/DOCX files, then re-run.")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Index the company corpus in AnythingLLM.")
    parser.add_argument(
        "--force",
        action="store_true",
        help="Upload files even when the same content hash was already indexed.",
    )
    arguments = parser.parse_args()
    result = ingest_all(force=arguments.force)
    raise SystemExit(1 if result["failed"] else 0)
