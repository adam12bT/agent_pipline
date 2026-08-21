"""Command-line entry point for the RFP proposal pipeline."""

import argparse

from dotenv import load_dotenv

from logging_config import configure_logging
from rfp.adapters import AnythingLLMAdapter
from rfp.orchestration.graph import build_graph
from rfp.orchestration.state import flatten_pipeline_state, initial_pipeline_state


def run(tender_file_path: str, response_template_file_path: str | None = None) -> dict:
    """Run the complete pipeline and print its user-facing result."""
    load_dotenv()
    configure_logging()

    print("Ensuring company knowledge base workspaces exist...")
    status = AnythingLLMAdapter().ensure_ready()
    for slug, info in status.items():
        description = "created; run ingest_company_corpus.py to fill it" if info["created"] else "already exists"
        print(f"  {slug}: {description}")

    pipeline = build_graph()
    initial_state = initial_pipeline_state(
        tender_file_path,
        response_template_file_path,
    )
    final_state = flatten_pipeline_state(pipeline.invoke(initial_state))

    print("\n" + "=" * 60)
    print(f"STATUS: {final_state.get('status')}")
    print("=" * 60)

    if not final_state.get("is_verified"):
        print("Blocked at verification:")
        for error in final_state.get("verification_errors", []):
            print(f"  - {error}")
        return final_state

    print("\n--- REQUIREMENTS ---")
    print(final_state.get("requirements"))
    print("\n--- QUALITY REPORT ---")
    print(final_state.get("quality_report"))
    print("\n--- DRAFT PROPOSAL ---\n")
    print(final_state.get("draft_proposal"))
    return final_state


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the RFP proposal pipeline")
    parser.add_argument("tender")
    parser.add_argument(
        "response_template",
        nargs="?",
        help="Optional PDF/DOCX response template; the built-in template is used when omitted.",
    )
    args = parser.parse_args()
    run(args.tender, args.response_template)


if __name__ == "__main__":
    main()
