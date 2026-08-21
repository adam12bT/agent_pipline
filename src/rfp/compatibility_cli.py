"""Compare a current live pipeline execution with the recorded legacy run."""

import argparse
import json
from pathlib import Path

from dotenv import load_dotenv

from logging_config import configure_logging
from rfp.adapters import AnythingLLMAdapter
from rfp.compatibility import replay_public_state
from rfp.orchestration.graph import build_graph
from rfp.orchestration.state import flatten_pipeline_state, initial_pipeline_state


ROOT = Path(__file__).parents[2]
DEFAULT_BASELINE = (
    ROOT / "tests" / "fixtures" / "real_run" / "legacy_api_response.json"
)
COMPARED_FIELDS = (
    "draft_proposal",
    "security_passed",
    "security_report",
    "quality_passed",
    "quality_report",
    "status",
    "errors",
)


def load_baseline(path: Path) -> dict:
    record = json.loads(path.read_text(encoding="utf-8"))
    return record["state"] if "state" in record else record


def exact_comparison(baseline: dict, current: dict) -> dict:
    fields = {
        field: baseline.get(field) == current.get(field)
        for field in COMPARED_FIELDS
    }
    return {
        "exact_match": all(fields.values()),
        "fields": fields,
        "baseline_status": baseline.get("status"),
        "current_status": current.get("status"),
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run and compare the packaged pipeline with a real legacy result."
    )
    parser.add_argument("tender", nargs="?")
    parser.add_argument("template", nargs="?")
    parser.add_argument("--baseline", type=Path, default=DEFAULT_BASELINE)
    parser.add_argument("--save-current", type=Path)
    parser.add_argument(
        "--replay-only",
        action="store_true",
        help="Verify the migration projection without uploading or calling services.",
    )
    args = parser.parse_args()

    baseline = load_baseline(args.baseline)
    if args.replay_only:
        current = replay_public_state(baseline)
    else:
        if not args.tender or not args.template:
            parser.error("tender and template are required unless --replay-only is used")
        # This is an explicit local compatibility command, so its project
        # configuration must win over stale variables inherited by the shell
        # (for example ANYTHINGLLM_BASE_URL=http://localhost:3001/api).
        load_dotenv(ROOT / ".env", override=True)
        configure_logging()
        AnythingLLMAdapter().ensure_ready()
        current = flatten_pipeline_state(
            build_graph().invoke(initial_pipeline_state(args.tender, args.template))
        )

    if args.save_current:
        args.save_current.write_text(
            json.dumps(current, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    comparison = exact_comparison(baseline, current)
    print(json.dumps(comparison, indent=2))
    return 0 if comparison["exact_match"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
