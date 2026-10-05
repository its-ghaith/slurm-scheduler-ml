from __future__ import annotations

import argparse
import base64
import json
from pathlib import Path

from .loader import load_controller
from .manifest import load_manifest
from .planner import build_plan, write_plan
from .api import ControllerContext
from .baseline_cache import load_baseline_catalog


def parse_json_object(value: str) -> dict:
    data = json.loads(value)
    if not isinstance(data, dict):
        raise argparse.ArgumentTypeError("Controller parameters must be a JSON object.")
    return data


def parse_base64_json_object(value: str) -> dict:
    try:
        decoded = base64.b64decode(value, validate=True).decode("utf-8")
    except Exception as exc:
        raise argparse.ArgumentTypeError(f"Invalid base64 controller parameters: {exc}") from exc
    return parse_json_object(decoded)


def add_parameter_arguments(parser: argparse.ArgumentParser) -> None:
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--parameters", type=parse_json_object)
    group.add_argument("--parameters-base64", type=parse_base64_json_object)


def main() -> None:
    parser = argparse.ArgumentParser(prog="controller-benchmark")
    sub = parser.add_subparsers(dest="command", required=True)
    validate = sub.add_parser("validate")
    validate.add_argument("--manifest", type=Path, required=True)
    validate.add_argument("--controller", required=True)
    validate.add_argument("--controller-id", required=True)
    add_parameter_arguments(validate)
    plan_parser = sub.add_parser("plan")
    plan_parser.add_argument("--manifest", type=Path, required=True)
    plan_parser.add_argument("--controller", required=True)
    plan_parser.add_argument("--controller-id", required=True)
    add_parameter_arguments(plan_parser)
    plan_parser.add_argument("--output-dir", type=Path, required=True)
    plan_parser.add_argument("--baseline-catalog", type=Path)
    args = parser.parse_args()

    parameters = args.parameters if args.parameters is not None else args.parameters_base64
    parameters = parameters or {}
    manifest = load_manifest(args.manifest)
    load_controller(
        args.controller,
        ControllerContext(
            controller_id=args.controller_id,
            benchmark_version=manifest["benchmark_version"],
            task_type="validation",
            quality_metric="quality",
            scenario="validation",
            max_epochs=int(manifest["execution"]["max_epochs"]),
            parameters=parameters,
        ),
    ).close()
    baseline_catalog = None
    if args.command == "plan" and args.baseline_catalog:
        baseline_catalog = load_baseline_catalog(args.baseline_catalog, manifest)
    plan = build_plan(
        manifest,
        controller_id=args.controller_id,
        controller_plugin=args.controller,
        controller_parameters=parameters,
        baseline_catalog=baseline_catalog,
    )
    if args.command == "plan":
        matrix = write_plan(plan, args.output_dir)
        print(matrix)
    print(json.dumps({
        "benchmark_version": plan["benchmark_version"],
        "planned_runs": len(plan["runs"]),
        "planned_pairs": plan["planned_cases"],
        "planned_baseline_runs": plan["planned_baseline_runs"],
        "planned_candidate_runs": plan["planned_candidate_runs"],
        "cached_baselines": len(plan["baseline_cache"]["cached_case_ids"]),
        "blocked_stages": plan["blocked_stages"],
    }, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
