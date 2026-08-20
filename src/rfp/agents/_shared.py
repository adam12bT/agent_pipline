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


def run_cli(input_type: Type[BaseModel], run: Callable[[BaseModel], BaseModel]) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--in", dest="input_path", required=True)
    parser.add_argument("--out", dest="output_path")
    args = parser.parse_args()
    payload = json.loads(Path(args.input_path).read_text(encoding="utf-8"))
    result = dump_model(run(input_type(**payload)))
    rendered = json.dumps(result, ensure_ascii=False, indent=2)
    if args.output_path:
        Path(args.output_path).write_text(rendered + "\n", encoding="utf-8")
    else:
        print(rendered)
