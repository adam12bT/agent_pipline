"""
Bulk-ingests your company's document corpus into the 3 persistent
knowledge workspaces (see company_knowledge.py).

WHEN YOU HAVE REAL FILES: drop them into this folder structure next to
this script:

    company_corpus/
    ├── past_proposals/     (PDF/DOCX of previously submitted proposals)
    ├── cvs/                 (PDF/DOCX of consultant CVs)
    └── project_references/ (PDF/DOCX describing completed past projects)

Then run:
    python ingest_company_corpus.py

Each file gets uploaded and embedded into its matching workspace. Safe
to re-run — AnythingLLM will just re-embed anything already there
(it does not currently skip duplicates, so avoid re-running on files
already ingested unless you don't mind duplicate chunks in the index).
"""

import os

from anythingllm_client import AnythingLLMClient
from company_knowledge import (
    PROPOSALS_WORKSPACE,
    CVS_WORKSPACE,
    REFERENCES_WORKSPACE,
    ensure_company_workspaces,
)

CORPUS_ROOT = os.path.join(os.path.dirname(__file__), "company_corpus")

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


def ingest_all():
    client = AnythingLLMClient()

    print("Ensuring the 3 company knowledge workspaces exist...")
    status = ensure_company_workspaces(client)
    for slug, info in status.items():
        print(f"  {slug}: {'created' if info['created'] else 'already existed'}")

    total_uploaded = 0
    for folder_name, workspace_slug in FOLDER_TO_WORKSPACE.items():
        folder_path = os.path.join(CORPUS_ROOT, folder_name)
        files = list(_iter_supported_files(folder_path))

        if not files:
            print(f"\n[{folder_name}] No PDF/DOCX files found in {folder_path} — skipping. "
                  f"(Create this folder and add files, then re-run.)")
            continue

        print(f"\n[{folder_name}] Uploading {len(files)} file(s) into workspace '{workspace_slug}'...")
        for file_path in files:
            try:
                client.upload_document(file_path, workspace_slug)
                print(f"  OK: {os.path.basename(file_path)}")
                total_uploaded += 1
            except Exception as e:
                print(f"  FAILED: {os.path.basename(file_path)} — {e}")

    print(f"\nDone. {total_uploaded} file(s) uploaded across all categories.")
    if total_uploaded == 0:
        print(
            "\nNo files were found anywhere. Create the folder structure described "
            "at the top of this script and add your real PDFs/DOCX, then re-run."
        )


if __name__ == "__main__":
    ingest_all()
