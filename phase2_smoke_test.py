"""Executable Phase 2 validation for AnythingLLM retrieval and OCR.

Use a distinctive phrase that appears verbatim in each document. For a
scanned PDF, the phrase must exist only in the image so a passing result proves
that AnythingLLM's OCR produced searchable text.
"""

import argparse
import hashlib
import json
import os
from dataclasses import dataclass

from rfp.adapters.anythingllm_client import AnythingLLMClient
from rfp.adapters.retrieval import search_relevant_chunks


@dataclass(frozen=True)
class SmokeCase:
    kind: str
    file_path: str
    expected_text: str
    extension: str


def _file_hash(file_path: str) -> str:
    digest = hashlib.sha256()
    with open(file_path, "rb") as file_obj:
        for block in iter(lambda: file_obj.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _workspace_slug(case: SmokeCase) -> str:
    return f"phase2-smoke-{case.kind}-{_file_hash(case.file_path)[:12]}"


def _validate_case(case: SmokeCase) -> None:
    if not os.path.isfile(case.file_path):
        raise ValueError(f"{case.kind}: file does not exist: {case.file_path}")
    actual_extension = os.path.splitext(case.file_path)[1].lower()
    if actual_extension != case.extension:
        raise ValueError(
            f"{case.kind}: expected a {case.extension} file, got {actual_extension or 'no extension'}"
        )
    if not case.expected_text.strip():
        raise ValueError(f"{case.kind}: the expected phrase cannot be empty")


def run_case(client: AnythingLLMClient, case: SmokeCase) -> dict:
    _validate_case(case)
    slug = _workspace_slug(case)
    workspace = client.get_or_create_workspace(slug)
    if workspace["created"]:
        client.upload_document(case.file_path, slug)

    results = search_relevant_chunks(
        client,
        slug,
        case.expected_text,
        top_n=5,
        score_threshold=0.0,
    )

    # A workspace can survive an interrupted first upload. Repair that state
    # once, while deterministic file-hash slugs avoid normal duplicate uploads.
    if not results and not workspace["created"]:
        client.upload_document(case.file_path, slug)
        results = search_relevant_chunks(
            client,
            slug,
            case.expected_text,
            top_n=5,
            score_threshold=0.0,
        )

    combined_text = "\n".join(str(item.get("text", "")) for item in results)
    phrase_found = case.expected_text.casefold() in combined_text.casefold()
    metadata_found = any(bool(item.get("metadata")) for item in results)
    best_result = results[0] if results else {}

    return {
        "kind": case.kind,
        "file": os.path.abspath(case.file_path),
        "workspace": slug,
        "passed": bool(results) and phrase_found,
        "retrieved_chunks": len(results),
        "expected_phrase_found": phrase_found,
        "metadata_returned": metadata_found,
        "best_score": best_result.get("score"),
        "best_rerank_score": best_result.get("rerank_score"),
        "best_metadata": best_result.get("metadata", {}),
    }


def _optional_case(
    cases: list[SmokeCase], kind: str, file_path: str | None, phrase: str | None, extension: str
) -> None:
    if bool(file_path) != bool(phrase):
        raise ValueError(f"--{kind} and --{kind}-phrase must be supplied together")
    if file_path and phrase:
        cases.append(SmokeCase(kind, file_path, phrase, extension))


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Upload sample documents and prove that their text is retrievable."
    )
    parser.add_argument("--native-pdf", help="Path to a normal text-based PDF")
    parser.add_argument("--native-pdf-phrase", help="Distinctive exact phrase in the PDF")
    parser.add_argument("--docx", help="Path to a DOCX file")
    parser.add_argument("--docx-phrase", help="Distinctive exact phrase in the DOCX")
    parser.add_argument("--scanned-pdf", help="Path to an image-only scanned PDF")
    parser.add_argument(
        "--scanned-pdf-phrase",
        help="Distinctive exact phrase visible in the scan; verifies OCR",
    )
    parser.add_argument("--table-pdf", help="Path to a PDF containing an important table")
    parser.add_argument(
        "--table-pdf-phrase",
        help="Exact row text or distinctive cell value that must remain searchable",
    )
    arguments = parser.parse_args()

    cases: list[SmokeCase] = []
    try:
        _optional_case(cases, "native-pdf", arguments.native_pdf, arguments.native_pdf_phrase, ".pdf")
        _optional_case(cases, "docx", arguments.docx, arguments.docx_phrase, ".docx")
        _optional_case(
            cases,
            "scanned-pdf",
            arguments.scanned_pdf,
            arguments.scanned_pdf_phrase,
            ".pdf",
        )
        _optional_case(
            cases,
            "table-pdf",
            arguments.table_pdf,
            arguments.table_pdf_phrase,
            ".pdf",
        )
        if not cases:
            raise ValueError("provide at least one document and its expected phrase")
        for case in cases:
            _validate_case(case)
    except ValueError as exc:
        parser.error(str(exc))

    client = AnythingLLMClient()
    reports = []
    for case in cases:
        print(f"Testing {case.kind}: {case.file_path}")
        try:
            report = run_case(client, case)
        except Exception as exc:
            report = {
                "kind": case.kind,
                "file": os.path.abspath(case.file_path),
                "passed": False,
                "error": str(exc),
            }
        reports.append(report)
        print(json.dumps(report, ensure_ascii=False, indent=2))

    passed = sum(1 for report in reports if report["passed"])
    print(f"\nPhase 2 smoke result: {passed}/{len(reports)} passed.")
    if any(report["kind"] == "scanned-pdf" and not report["passed"] for report in reports):
        print("OCR is not sufficient for this sample; inspect AnythingLLM OCR before adding another OCR stack.")
    return 0 if passed == len(reports) else 1


if __name__ == "__main__":
    raise SystemExit(main())
