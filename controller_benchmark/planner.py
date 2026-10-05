from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def build_plan(
    manifest: dict[str, Any],
    *,
    controller_id: str,
    controller_plugin: str,
    controller_parameters: dict[str, Any],
    baseline_catalog: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,63}", controller_id):
        raise ValueError("controller_id must contain 1-64 letters, digits, dots, underscores, or hyphens.")
    if not controller_plugin.strip():
        raise ValueError("controller_plugin must not be empty.")
    baseline = manifest["baselines"][0]
    if controller_id == baseline["id"]:
        raise ValueError(f"controller_id must differ from baseline id {baseline['id']!r}.")
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + f"-{controller_id}"
    rows = []
    blocked_stages = []
    sequence = 1
    candidate = {
        "id": controller_id,
        "controller_mode": "plugin",
        "controller_plugin": controller_plugin,
        "controller_parameters": controller_parameters,
    }
    cached_entries = (baseline_catalog or {}).get("entries", {})
    baseline_references: dict[str, str] = {}
    scheduled_baselines: set[str] = set()

    def append_run(stage: dict[str, Any], case: dict[str, Any], strategy: dict[str, Any]) -> None:
        nonlocal sequence
        rows.append({
            "sequence": sequence,
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
        })
        sequence += 1
    for stage_index, stage in enumerate(manifest["stages"]):
        if not stage.get("enabled"):
            blocked_stages.append(
                {
                    "stage": stage["id"],
                    "reason": stage.get("blocked_reason", "disabled"),
                    "required_extension": stage.get("required_extension"),
                }
            )
            continue
        stage_cases = {case["id"]: case for case in stage["cases"]}
        for case in stage["cases"]:
            source_case_id = stage.get("baseline_case_id", case["id"])
            baseline_references[case["id"]] = source_case_id
            if source_case_id not in cached_entries and source_case_id not in scheduled_baselines:
                append_run(stage, stage_cases[source_case_id], baseline)
                scheduled_baselines.add(source_case_id)
            append_run(stage, case, candidate)
    return {
        "benchmark_run_id": run_id,
        "benchmark_version": manifest["benchmark_version"],
        "manifest_sha256": manifest["manifest_sha256"],
        "baseline": baseline,
        "baseline_cache": {
            "catalog_path": (baseline_catalog or {}).get("catalog_path"),
            "cached_case_ids": sorted(set(baseline_references.values()) & set(cached_entries)),
            "entries": cached_entries,
        },
        "baseline_references": baseline_references,
        "controller": candidate,
        "execution": manifest["execution"],
        "acceptance": manifest["acceptance"],
        "blocked_stages": blocked_stages,
        "planned_cases": sum(len(stage.get("cases", [])) for stage in manifest["stages"] if stage.get("enabled")),
        "planned_baseline_runs": sum(1 for row in rows if row["controller_id"] == baseline["id"]),
        "planned_candidate_runs": sum(1 for row in rows if row["controller_id"] == controller_id),
        "runs": rows,
    }


def write_plan(plan: dict[str, Any], output_dir: str | Path) -> Path:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=False)
    path = output / "run-matrix.json"
    path.write_text(json.dumps(plan, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return path
