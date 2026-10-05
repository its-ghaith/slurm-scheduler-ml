from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


def load_manifest(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    data = json.loads(path.read_text(encoding="utf-8"))
    validate_manifest(data)
    data["manifest_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    return data


def validate_manifest(data: dict[str, Any]) -> None:
    required = {"benchmark_version", "execution", "acceptance", "baselines", "stages"}
    missing = sorted(required - data.keys())
    if missing:
        raise ValueError(f"Manifest is missing keys: {', '.join(missing)}")
    if not data["baselines"]:
        raise ValueError("At least one baseline must be configured.")
    baseline = data["baselines"][0]
    if not baseline.get("id") or "controller_mode" not in baseline:
        raise ValueError("The primary baseline requires id and controller_mode.")
    acceptance = data["acceptance"]
    for key in ("minimum_energy_saving_fraction", "maximum_quality_regret", "minimum_success_rate"):
        if key not in acceptance:
            raise ValueError(f"Acceptance criteria are missing {key}.")
    if not 0.0 <= float(acceptance["minimum_success_rate"]) <= 1.0:
        raise ValueError("minimum_success_rate must be between 0 and 1.")
    stage_ids: set[str] = set()
    case_ids: set[str] = set()
    for stage in data["stages"]:
        stage_id = str(stage.get("id", "")).strip()
        if not stage_id or stage_id in stage_ids:
            raise ValueError(f"Invalid or duplicate stage id: {stage_id!r}")
        stage_ids.add(stage_id)
        if stage.get("enabled") and not stage.get("cases"):
            raise ValueError(f"Enabled stage {stage_id} has no cases.")
        stage_case_ids = {str(case.get("id", "")).strip() for case in stage.get("cases", [])}
        if stage.get("baseline_case_id") and stage["baseline_case_id"] not in stage_case_ids:
            raise ValueError(f"Stage {stage_id} references an unknown baseline_case_id.")
        for case in stage.get("cases", []):
            case_id = str(case.get("id", "")).strip()
            if not case_id or case_id in case_ids:
                raise ValueError(f"Invalid or duplicate case id: {case_id!r}")
            case_ids.add(case_id)
            for key in ("runner", "task_type", "quality_metric", "scenario", "training_seed"):
                if key not in case:
                    raise ValueError(f"Case {case_id} is missing {key}.")
    if not any(stage.get("enabled") for stage in data["stages"]):
        raise ValueError("At least one benchmark stage must be enabled.")
