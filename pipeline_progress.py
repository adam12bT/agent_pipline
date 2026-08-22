"""Thread-safe, transient progress snapshots for long-running pipeline nodes.

LangGraph only returns a node's state after that node finishes. Generation,
however, performs several sequential LLM batches. This small side channel lets
the HTTP API expose each completed batch while the generation node is still
running, without coupling an agent to FastAPI or the backend run store.
"""

from copy import deepcopy
import threading
import time
import re


_PROGRESS: dict[str, dict] = {}
_LOCK = threading.Lock()
_MIN_SECTION_BODY_WORDS = 12


def _word_count(value: str) -> int:
    return len(re.findall(r"\b[\w'-]+\b", value, flags=re.UNICODE))


def start_generation(run_id: str | None, batches: list[list[str]]) -> None:
    if not run_id:
        return
    sections = [
        {"title": title, "status": "waiting", "content": "", "batch": batch_number}
        for batch_number, batch in enumerate(batches, start=1)
        for title in batch
    ]
    with _LOCK:
        _PROGRESS[run_id] = {
            "status": "generating",
            "batch_number": 0,
            "batch_count": len(batches),
            "sections": sections,
            "draft": "",
            "updated_at": time.time(),
        }


def mark_batch_started(run_id: str | None, batch_number: int) -> None:
    if not run_id:
        return
    with _LOCK:
        progress = _PROGRESS.get(run_id)
        if not progress:
            return
        progress["batch_number"] = batch_number
        for section in progress["sections"]:
            if section["batch"] == batch_number:
                section["status"] = "generating"
        progress["updated_at"] = time.time()


def mark_batch_completed(
    run_id: str | None,
    batch_number: int,
    section_content: dict[str, str],
) -> None:
    if not run_id:
        return
    with _LOCK:
        progress = _PROGRESS.get(run_id)
        if not progress:
            return
        for section in progress["sections"]:
            if section["batch"] != batch_number:
                continue
            section["content"] = section_content.get(section["title"], "").strip()
            section["word_count"] = _word_count(section["content"])
            section["status"] = (
                "complete"
                if section["word_count"] >= _MIN_SECTION_BODY_WORDS
                else "incomplete"
            )
        progress["draft"] = "\n\n".join(
            section["content"] for section in progress["sections"] if section["content"].strip()
        )
        progress["updated_at"] = time.time()


def finish_generation(run_id: str | None, *, failed: bool = False) -> None:
    if not run_id:
        return
    with _LOCK:
        progress = _PROGRESS.get(run_id)
        if not progress:
            return
        has_incomplete_sections = any(
            section.get("status") in {"waiting", "generating", "incomplete", "failed"}
            for section in progress["sections"]
        )
        progress["status"] = (
            "failed" if failed else "incomplete" if has_incomplete_sections else "complete"
        )
        progress["updated_at"] = time.time()


def get_progress(run_id: str) -> dict | None:
    with _LOCK:
        progress = _PROGRESS.get(run_id)
        return deepcopy(progress) if progress else None
