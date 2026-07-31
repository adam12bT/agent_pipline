"""
Example entry point.

Usage:
    python main.py /path/to/tender.pdf

Requires:
  - The AnythingLLM server (anything-llm-lightweight/server) running,
    default expected at http://localhost:3001 (override with
    ANYTHINGLLM_BASE_URL in .env)
  - GPT Researcher's own env vars set (an LLM key + a search engine key,
    e.g. TAVILY_API_KEY) — see .env.example
"""

import sys
from dotenv import load_dotenv

load_dotenv()

from graph import build_graph  # noqa: E402  (import after load_dotenv on purpose)
from company_knowledge import ensure_company_workspaces  # noqa: E402


def run(tender_file_path: str):
    print("Ensuring company knowledge base workspaces exist...")
    status = ensure_company_workspaces()
    for slug, info in status.items():
        print(f"  {slug}: {'created (empty — run ingest_company_corpus.py to fill it)' if info['created'] else 'already exists'}")

    pipeline = build_graph()
    initial_state = {
        "tender_file_path": tender_file_path,
        "status": "running",
        "generation_attempts": 0,
        "errors": [],
    }

    final_state = pipeline.invoke(initial_state)

    print("\n" + "=" * 60)
    print(f"STATUS: {final_state.get('status')}")
    print("=" * 60)

    if not final_state.get("is_verified"):
        print("Blocked at verification:")
        for err in final_state.get("verification_errors", []):
            print(f"  - {err}")
        return

    print("\n--- REQUIREMENTS ---")
    print(final_state.get("requirements"))

    print("\n--- QUALITY REPORT ---")
    print(final_state.get("quality_report"))

    print("\n--- DRAFT PROPOSAL ---\n")
    print(final_state.get("draft_proposal"))


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python main.py /path/to/tender.pdf")
        sys.exit(1)

    run(sys.argv[1])
