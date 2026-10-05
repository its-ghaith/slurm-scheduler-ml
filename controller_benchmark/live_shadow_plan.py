from __future__ import annotations

import argparse
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from controller_benchmark.api import ControllerContext
from controller_benchmark.loader import load_controller
from controller_benchmark.manifest import load_manifest


SUITE_PLUGIN = (
    "controller_benchmark.controllers.live_shadow_suite:LiveShadowSuiteController"
)


def load_shadow_configuration(path: str | Path, manifest: dict[str, Any]) -> dict[str, Any]:
    configuration = json.loads(Path(path).read_text(encoding="utf-8"))
    controllers = configuration.get("controllers", [])
    if not isinstance(controllers, list) or not controllers:
        raise ValueError("Live shadow configuration requires controllers.")
    ids: set[str] = set()
    for definition in controllers:
        controller_id = str(definition.get("id", ""))
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,63}", controller_id):
            raise ValueError(f"Invalid shadow controller id: {controller_id!r}")
        if controller_id in ids:
            raise ValueError(f"Duplicate shadow controller id: {controller_id}")
        ids.add(controller_id)
        instance = load_controller(
            str(definition.get("controller_plugin", "")),
            ControllerContext(
                controller_id=controller_id,
                benchmark_version=manifest["benchmark_version"],
                task_type="validation",
                quality_metric="quality_score",
                scenario="validation",
                max_epochs=int(manifest["execution"]["max_epochs"]),
                parameters=dict(definition.get("controller_parameters", {})),
            ),
        )
        instance.close()
    return configuration


def build_live_shadow_plan(
    manifest: dict[str, Any],
    configuration: dict[str, Any],
    *,
    run_id: str | None = None,
    expected_cases: int = 9,
    benchmark_version: str = "controller-live-shadow-nine-dataset-v1",
    dashboard_file: str = "live-shadow-nine-dataset.json",
    experiment_name: str = "controller-live-shadow-development",
    pretraining_job_id: str | None = None,
) -> dict[str, Any]:
    run_id = run_id or (
        datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        + "-live-shadow-dev"
    )
    execution = dict(manifest["execution"])
    execution.update(
        {
            "experiment_name": experiment_name,
            "analysis_module": "controller_benchmark.analyze_live_shadow",
            "dashboard_files": [dashboard_file],
            "energy_comparison_scope": "epoch",
            "measurement_protocol_version": "live-shadow-v1",
        }
    )
    if pretraining_job_id:
        if not re.fullmatch(r"\d+", pretraining_job_id):
            raise ValueError(f"Invalid SLURM pretraining job id: {pretraining_job_id!r}")
        execution.update(
            {
                "pretraining_job_id": pretraining_job_id,
                "pretraining_timeout_seconds": 172800,
            }
        )
    suite_parameters = {
        "controllers": configuration["controllers"],
        "fallback_watts_per_cpu_second": float(
            configuration.get("fallback_watts_per_cpu_second", 5.0)
        ),
    }
    runs: list[dict[str, Any]] = []
    sequence = 1
    for stage in manifest["stages"]:
        if not stage.get("enabled"):
            continue
        for case in stage["cases"]:
            shadow_case = dict(case)
            shadow_case["training_seed"] = 0
            runs.append(
                {
                    "sequence": sequence,
                    "role": "live_shadow_full100",
                    "benchmark_run_id": run_id,
                    "benchmark_version": benchmark_version,
                    "benchmark_stage": stage["id"],
                    "benchmark_case_id": shadow_case["id"],
                    "runner": shadow_case["runner"],
                    "task_type": shadow_case["task_type"],
                    "quality_metric": shadow_case["quality_metric"],
                    "scenario": shadow_case["scenario"],
                    "training_seed": 0,
                    "strategy": "live-shadow-suite",
                    "controller_id": "live-shadow-suite",
                    "controller_mode": "plugin",
                    "controller_plugin": SUITE_PLUGIN,
                    "controller_parameters": suite_parameters,
                    "case": shadow_case,
                }
            )
            sequence += 1
    if len(runs) != expected_cases:
        raise ValueError(
            f"Development shadow benchmark requires exactly {expected_cases} cases, got {len(runs)}."
        )
    return {
        "schema_version": 2,
        "plan_type": "live_shadow_campaign",
        "campaign_id": str(configuration.get("campaign_id", "live-shadow-development")),
        "benchmark_run_id": run_id,
        "benchmark_version": benchmark_version,
        "source_manifest_version": manifest["benchmark_version"],
        "manifest_sha256": manifest["manifest_sha256"],
        "baseline": {"id": "shared-full100", "controller_mode": "none"},
        "controller": {"id": "live-shadow-suite"},
        "controllers": configuration["controllers"],
        "execution": execution,
        "acceptance": {
            "minimum_energy_saving_fraction": 0.0,
            "maximum_quality_regret": 0.10,
            "minimum_success_rate": 0.80,
        },
        "blocked_stages": [],
        "baseline_cache": {"catalog_path": None, "cached_case_ids": [], "entries": {}},
        "baseline_references": {},
        "planned_cases": len(runs),
        "planned_baseline_runs": 0,
        "planned_candidate_runs": len(runs),
        "runs": runs,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--controllers", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--run-id")
    parser.add_argument("--expected-cases", type=int, default=9)
    parser.add_argument(
        "--benchmark-version", default="controller-live-shadow-nine-dataset-v1"
    )
    parser.add_argument("--dashboard-file", default="live-shadow-nine-dataset.json")
    parser.add_argument(
        "--experiment-name", default="controller-live-shadow-development"
    )
    parser.add_argument("--pretraining-job-id")
    args = parser.parse_args()
    manifest = load_manifest(args.manifest)
    configuration = load_shadow_configuration(args.controllers, manifest)
    plan = build_live_shadow_plan(
        manifest,
        configuration,
        run_id=args.run_id,
        expected_cases=args.expected_cases,
        benchmark_version=args.benchmark_version,
        dashboard_file=args.dashboard_file,
        experiment_name=args.experiment_name,
        pretraining_job_id=args.pretraining_job_id,
    )
    args.output_dir.mkdir(parents=True, exist_ok=False)
    output = args.output_dir / "run-matrix.json"
    output.write_text(
        json.dumps(plan, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(output)
    print(
        json.dumps(
            {
                "benchmark_version": plan["benchmark_version"],
                "planned_training_jobs": len(plan["runs"]),
                "shadow_controllers": len(plan["controllers"]),
                "training_seeds": sorted({run["training_seed"] for run in plan["runs"]}),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
