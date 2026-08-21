"""
In-memory store for pipeline runs and the background thread that actually
drives the LangGraph pipeline.

Uses `pipeline.stream(...)` instead of `pipeline.invoke(...)` so the UI can
poll for live, per-agent progress (which node just ran, what it produced)
instead of only seeing a result once the whole graph finishes.

This is intentionally in-memory (a dict) rather than a database — good
enough for a single-process dev/demo deployment. Swap RUNS for a real
store (Redis, Postgres) if you need multi-process or persistence across
restarts.
"""

import logging
import os
import threading
import time
import traceback
import uuid
from typing import Optional

from rfp.orchestration.graph import build_graph
from rfp.orchestration.state import flatten_pipeline_state, initial_pipeline_state
from pipeline_progress import get_progress

logger = logging.getLogger(__name__)

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
_MAX_CONCURRENT_RUNS = max(
    1, int(os.environ.get("PIPELINE_MAX_CONCURRENT_RUNS", "1"))
)
_RUN_SLOTS = threading.BoundedSemaphore(_MAX_CONCURRENT_RUNS)


def _new_run_record(
    run_id: str,
    tender_file_path: str,
    original_filename: str,
    response_template_file_path: str | None,
    response_template_filename: str | None,
) -> dict:
    internal_state = initial_pipeline_state(
        tender_file_path,
        response_template_file_path,
        run_id=run_id,
    )
    return {
        "run_id": run_id,
        "tender_filename": original_filename,
        "tender_file_path": tender_file_path,
        "response_template_filename": response_template_filename,
        "response_template_file_path": response_template_file_path,
        "created_at": time.time(),
        "updated_at": time.time(),
        # "queued" | "running" | "blocked" | "security_blocked" | "done" | "failed"
        "run_status": "queued",
        "current_stage": None,
        "completed_stages": [],
        "internal_state": internal_state,
        "state": flatten_pipeline_state(internal_state),
        "telemetry": dict(internal_state.get("telemetry") or {}),
        "error": None,
    }


def create_run(
    tender_file_path: str,
    original_filename: str,
    response_template_file_path: str | None = None,
    response_template_filename: str | None = None,
) -> str:
    run_id = uuid.uuid4().hex[:12]
    record = _new_run_record(
        run_id,
        tender_file_path,
        original_filename,
        response_template_file_path,
        response_template_filename,
    )
    with _LOCK:
        RUNS[run_id] = record
    thread = threading.Thread(target=_queued_execute_run, args=(run_id,), daemon=True)
    thread.start()
    return run_id


def get_run(run_id: str) -> Optional[dict]:
    with _LOCK:
        record = RUNS.get(run_id)
        result = dict(record) if record else None
    if result:
        # The frontend intentionally receives the stable flat projection, not
        # orchestration's namespaced implementation detail.
        result.pop("internal_state", None)
        progress = get_progress(run_id)
        result["generation_progress"] = progress
        # LangGraph emits a node result only after the whole node returns. The
        # side-channel snapshot lets polling clients see that generation is the
        # active stage while its internal batches are still running.
        if progress and progress.get("status") == "generating":
            result["current_stage"] = "generation"
    return result


def _get_internal_run(run_id: str) -> Optional[dict]:
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
            "response_template_filename": r["response_template_filename"],
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


def _queued_execute_run(run_id: str):
    logger.info("Run %s: waiting for an available pipeline slot", run_id)
    with _RUN_SLOTS:
        _execute_run(run_id)


def _execute_run(run_id: str):
    record = _get_internal_run(run_id)
    if not record:
        return

    logger.info("Run %s: starting pipeline execution", run_id)
    _update_run(run_id, run_status="running")

    try:
        pipeline = build_graph()
        cumulative_state = dict(record["internal_state"])
        execution_started = time.time()
        telemetry = dict(cumulative_state.get("telemetry") or {})
        telemetry.update(
            {
                "started_at_epoch": execution_started,
                "updated_at_epoch": execution_started,
                "total_duration_seconds": 0.0,
            }
        )
        cumulative_state["telemetry"] = telemetry
        completed_stages = []

        for step in pipeline.stream(cumulative_state):
            # `step` is like {"verifier": {...partial state returned by node...}}
            # or, once extraction/research run in parallel, potentially
            # {"extraction": {...}, "research": {...}} in the same step.
            for node_name, node_output in step.items():
                if node_output:
                    incoming = dict(node_output)
                    new_errors = list(incoming.pop("errors", []) or [])
                    cumulative_state.update(incoming)
                    if new_errors:
                        cumulative_state["errors"] = (
                            list(cumulative_state.get("errors") or []) + new_errors
                        )

                public_state = flatten_pipeline_state(cumulative_state)

                if node_name in _HIDDEN_STAGES:
                    # Internal plumbing (graph.py's fan-out node) — merge
                    # its (empty) output above, but don't surface it as a
                    # fake pipeline stage in the UI.
                    continue

                completed_stages = completed_stages + [node_name]

                run_status = "running"
                node_status = public_state.get("status")
                if node_name == "verifier" and not public_state.get("is_verified", True):
                    run_status = "blocked"
                    logger.warning(
                        "Run %s: blocked at verification: %s",
                        run_id, public_state.get("verification_errors"),
                    )
                elif node_name == "security" and not public_state.get("security_passed", True):
                    run_status = "security_blocked"
                    logger.warning(
                        "Run %s: security-blocked, escalating to human review: %s",
                        run_id, public_state.get("security_report"),
                    )
                elif node_status == "failed":
                    run_status = "failed"
                    logger.warning("Run %s: pipeline finished with status=failed", run_id)
                elif node_status == "done":
                    run_status = "done"
                    logger.info("Run %s: completed successfully", run_id)

                logger.debug("Run %s: stage %r finished (run_status=%s)", run_id, node_name, run_status)

                _update_run(
                    run_id,
                    internal_state=cumulative_state,
                    state=public_state,
                    telemetry=dict(cumulative_state.get("telemetry") or {}),
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
        # This runs in a daemon background thread — if it isn't logged
        # here, the failure is only ever visible to someone who happens
        # to poll this specific run's state via the API. Log it loudly.
        logger.exception("Run %s: pipeline crashed unexpectedly", run_id)
        _update_run(
            run_id,
            run_status="failed",
            error=f"{e}\n{traceback.format_exc()}",
        )
