"""Utilities shared only by per-agent package boundaries."""

import argparse
import json
from pathlib import Path
from typing import Callable, Type

from pydantic import BaseModel


def dump_model(model: BaseModel) -> dict:
    return model.model_dump(exclude_none=True)


def validate_output(model_type: Type[BaseModel], payload: dict) -> BaseModel:
    clean = {key: value for key, value in payload.items() if key != "status"}
    return model_type(**clean)


def run_cli(
    input_type: Type[BaseModel],
    output_type: Type[BaseModel],
    run: Callable[[BaseModel], BaseModel],
) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--in", dest="input_path", required=True)
    parser.add_argument("--out", dest="output_path")
    parser.add_argument("--expected")
    parser.add_argument(
        "--contract-only",
        action="store_true",
        help="Validate a saved real-run input/output pair without side effects.",
    )
    args = parser.parse_args()
    payload = json.loads(Path(args.input_path).read_text(encoding="utf-8"))
    agent_input = input_type(**payload)
    expected = None
    if args.expected:
        expected_payload = json.loads(Path(args.expected).read_text(encoding="utf-8"))
        expected = output_type(**expected_payload)
    if args.contract_only:
        if expected is None:
            parser.error("--contract-only requires --expected")
        result = {
            "valid": True,
            "input_contract": input_type.__name__,
            "output_contract": output_type.__name__,
        }
    else:
        actual = run(agent_input)
        result = dump_model(actual)
        if expected is not None and actual != expected:
            raise SystemExit("Agent output did not match the expected saved output")
    rendered = json.dumps(result, ensure_ascii=False, indent=2)
    if args.output_path:
        Path(args.output_path).write_text(rendered + "\n", encoding="utf-8")
    else:
        print(rendered)
