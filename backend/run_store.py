"""
In-memory store for pipeline runs + the background thread that actually
drives the LangGraph pipeline.

Uses `pipeline.stream(...)` instead of `pipeline.invoke(...)` so the UI can
poll for live, per-agent progress (which node just ran, what it produced)
instead of only seeing a result once the whole graph finishes.

This is intentionally in-memory (a dict) rather than a database — good
enough for a single-process dev/demo deployment. Swap RUNS for a real
store (Redis, Postgres) if you need multi-process or persistence across
restarts.
"""

import threading
import time
import traceback
import uuid
from typing import Optional

from graph import build_graph

# Node execution order, used to render a fixed-position stepper in the UI
# regardless of which nodes have actually run yet. Deliberately excludes
# "dispatch" (graph.py's internal fan-out plumbing node) — see the filter
# in _execute_run() below.
PIPELINE_STAGES = ["verifier", "extraction", "research", "generation", "security", "quality"]

# Stages the UI stepper should never see — currently just the internal
# fan-out node from graph.py.
_HIDDEN_STAGES = {"dispatch"}

RUNS: dict[str, dict] = {}
_LOCK = threading.Lock()


def _new_run_record(run_id: str, tender_file_path: str, original_filename: str) -> dict:
    return {
        "run_id": run_id,
        "tender_filename": original_filename,
        "tender_file_path": tender_file_path,
        "created_at": time.time(),
        "updated_at": time.time(),
        # "queued" | "running" | "blocked" | "security_blocked" | "done" | "failed"
        "run_status": "queued",
        "current_stage": None,
        "completed_stages": [],
        "state": {
            "tender_file_path": tender_file_path,
            "status": "running",
            "generation_attempts": 0,
            "errors": [],
        },
        "error": None,
    }


def create_run(tender_file_path: str, original_filename: str) -> str:
    run_id = uuid.uuid4().hex[:12]
    record = _new_run_record(run_id, tender_file_path, original_filename)
    with _LOCK:
        RUNS[run_id] = record
    thread = threading.Thread(target=_execute_run, args=(run_id,), daemon=True)
    thread.start()
    return run_id


def get_run(run_id: str) -> Optional[dict]:
    with _LOCK:
        record = RUNS.get(run_id)
        return dict(record) if record else None


def list_runs() -> list[dict]:
    with _LOCK:
        records = list(RUNS.values())
    records.sort(key=lambda r: r["created_at"], reverse=True)
    # Lightweight summaries for the run list view.
    return [
        {
            "run_id": r["run_id"],
            "tender_filename": r["tender_filename"],
            "created_at": r["created_at"],
            "updated_at": r["updated_at"],
            "run_status": r["run_status"],
            "current_stage": r["current_stage"],
            "completed_stages": r["completed_stages"],
        }
        for r in records
    ]


def _update_run(run_id: str, **fields):
    with _LOCK:
        if run_id in RUNS:
            RUNS[run_id].update(fields)
            RUNS[run_id]["updated_at"] = time.time()


def _execute_run(run_id: str):
    record = get_run(run_id)
    if not record:
        return

    _update_run(run_id, run_status="running")

    try:
        pipeline = build_graph()
        cumulative_state = dict(record["state"])
        completed_stages = []

        for step in pipeline.stream(cumulative_state):
            # `step` is like {"verifier": {...partial state returned by node...}}
            # or, once extraction/research run in parallel, potentially
            # {"extraction": {...}, "research": {...}} in the same step.
            for node_name, node_output in step.items():
                if node_output:
                    cumulative_state.update(node_output)

                if node_name in _HIDDEN_STAGES:
                    # Internal plumbing (graph.py's fan-out node) — merge
                    # its (empty) output above, but don't surface it as a
                    # fake pipeline stage in the UI.
                    continue

                completed_stages = completed_stages + [node_name]

                run_status = "running"
                node_status = cumulative_state.get("status")
                if node_name == "verifier" and not cumulative_state.get("is_verified", True):
                    run_status = "blocked"
                elif node_name == "security" and not cumulative_state.get("security_passed", True):
                    run_status = "security_blocked"
                elif node_status == "failed":
                    run_status = "failed"
                elif node_status == "done":
                    run_status = "done"

                _update_run(
                    run_id,
                    state=cumulative_state,
                    current_stage=node_name,
                    completed_stages=completed_stages,
                    run_status=run_status,
                )

        # Graph finished streaming without an explicit terminal status
        # (defensive fallback — shouldn't normally happen).
        final = get_run(run_id)
        if final and final["run_status"] == "running":
            _update_run(run_id, run_status=final["state"].get("status", "done"))

    except Exception as e:
        _update_run(
            run_id,
            run_status="failed",
            error=f"{e}\n{traceback.format_exc()}",
        )