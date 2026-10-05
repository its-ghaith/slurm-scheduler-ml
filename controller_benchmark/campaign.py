from __future__ import annotations

import argparse
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .api import ControllerContext
from .loader import load_controller
from .manifest import load_manifest


def load_controller_specifications(path: str | Path, manifest: dict[str, Any]) -> dict[str, Any]:
    specification = json.loads(Path(path).read_text(encoding="utf-8"))
    controllers = specification.get("controllers", [])
    if not controllers:
        raise ValueError("The campaign must contain at least one controller.")
    ids: set[str] = set()
    baseline_id = manifest["baselines"][0]["id"]
    for controller in controllers:
        controller_id = str(controller.get("id", ""))
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,63}", controller_id):
            raise ValueError(f"Invalid controller id: {controller_id!r}")
        if controller_id == baseline_id or controller_id in ids:
            raise ValueError(f"Duplicate or reserved controller id: {controller_id}")
        ids.add(controller_id)
        instance = load_controller(
            controller["controller_plugin"],
            ControllerContext(
                controller_id=controller_id,
                benchmark_version=manifest["benchmark_version"],
                task_type="validation",
                quality_metric="quality_score",
                scenario="validation",
                max_epochs=int(manifest["execution"]["max_epochs"]),
                parameters=controller.get("controller_parameters", {}),
            ),
        )
        instance.close()
    return specification


def build_campaign_plan(
    manifest: dict[str, Any],
    specification: dict[str, Any],
    *,
    run_id: str | None = None,
) -> dict[str, Any]:
    campaign_id = str(specification.get("campaign_id", "controller-campaign"))
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,63}", campaign_id):
        raise ValueError(f"Invalid campaign id: {campaign_id!r}")
    run_id = run_id or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + f"-{campaign_id}"
    baseline = manifest["baselines"][0]
    controllers = specification["controllers"]
    runs: list[dict[str, Any]] = []
    baseline_references: dict[str, str] = {}
    sequence = 1
    case_index = 0

    def append_run(stage: dict[str, Any], case: dict[str, Any], strategy: dict[str, Any]) -> None:
        nonlocal sequence
        role = "baseline" if strategy["id"] == baseline["id"] else "candidate"
        runs.append(
            {
                "sequence": sequence,
                "role": role,
                "benchmark_run_id": run_id,
                "benchmark_version": manifest["benchmark_version"],
                "benchmark_stage": stage["id"],
                "benchmark_case_id": case["id"],
                "runner": case["runner"],
                "task_type": case["task_type"],
                "quality_metric": case["quality_metric"],
                "scenario": case["scenario"],
                "training_seed": case["training_seed"],
                "strategy": strategy["id"],
                "controller_id": strategy["id"],
                "controller_mode": strategy["controller_mode"],
                "controller_plugin": strategy.get("controller_plugin", ""),
                "controller_parameters": strategy.get("controller_parameters", {}),
                "case": case,
            }
        )
        sequence += 1

    for stage in manifest["stages"]:
        if not stage.get("enabled"):
            continue
        for case in stage["cases"]:
            baseline_references[case["id"]] = case["id"]
            strategies = [baseline, *controllers]
            rotation = case_index % len(strategies)
            strategies = strategies[rotation:] + strategies[:rotation]
            for strategy in strategies:
                append_run(stage, case, strategy)
            case_index += 1

    planned_cases = sum(len(stage.get("cases", [])) for stage in manifest["stages"] if stage.get("enabled"))
    return {
        "schema_version": 2,
        "plan_type": "multi_controller_campaign",
        "campaign_id": campaign_id,
        "benchmark_run_id": run_id,
        "benchmark_version": manifest["benchmark_version"],
        "manifest_sha256": manifest["manifest_sha256"],
        "baseline": baseline,
        "controller": {"id": campaign_id},
        "controllers": controllers,
        "execution": manifest["execution"],
        "acceptance": manifest["acceptance"],
        "blocked_stages": [],
        "baseline_cache": {"catalog_path": None, "cached_case_ids": [], "entries": {}},
        "baseline_references": baseline_references,
        "planned_cases": planned_cases,
        "planned_baseline_runs": planned_cases,
        "planned_candidate_runs": planned_cases * len(controllers),
        "runs": runs,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--controllers", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--run-id")
    args = parser.parse_args()
    manifest = load_manifest(args.manifest)
    specification = load_controller_specifications(args.controllers, manifest)
    plan = build_campaign_plan(manifest, specification, run_id=args.run_id)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    output = args.output_dir / "run-matrix.json"
    output.write_text(json.dumps(plan, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(output)
    print(
        json.dumps(
            {
                "benchmark_version": plan["benchmark_version"],
                "planned_runs": len(plan["runs"]),
                "planned_pairs": plan["planned_cases"] * len(plan["controllers"]),
                "planned_baseline_runs": plan["planned_baseline_runs"],
                "planned_candidate_runs": plan["planned_candidate_runs"],
                "controllers": [controller["id"] for controller in plan["controllers"]],
                "fresh_baselines": True,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
