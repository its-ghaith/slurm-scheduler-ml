from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
from typing import Any

from controller_benchmark.generalized_manifest import build_manifest
from controller_benchmark.manifest import validate_manifest


SELECTED_CASE_IDS = (
    "carpk-aerial-vehicle",
    "cifar100-classification",
    "uavid-segmentation",
    "wikitext103-language-modeling-scratch",
    "traffic-hourly-time-series-forecasting-scratch",
    "minatar-breakout-reinforcement-learning-scratch",
)


def build_small_manifest(
    vision_source: dict[str, Any],
    classification_source: dict[str, Any],
) -> dict[str, Any]:
    source = build_manifest(
        vision_source,
        classification_source,
        include_vision=True,
        suite="hardest",
    )
    selected = set(SELECTED_CASE_IDS)
    stages: list[dict[str, Any]] = []
    for source_stage in source["stages"]:
        cases = [
            copy.deepcopy(case)
            for case in source_stage["cases"]
            if case["id"] in selected
        ]
        if not cases:
            continue
        stage = copy.deepcopy(source_stage)
        stage["cases"] = cases
        stage["enabled"] = True
        stage.pop("baseline_case_id", None)
        stages.append(stage)
    found = {case["id"] for stage in stages for case in stage["cases"]}
    missing = selected - found
    if missing:
        raise ValueError(f"Small RAPEC-G v2 manifest is missing: {sorted(missing)}")
    source["stages"] = stages
    source["benchmark_version"] = "rapec-g-v2-small-validation-v1"
    source["description"] = (
        "Six-case scratch-only live-shadow validation of RAPEC-G v2 across "
        "computer vision, language modeling, forecasting, and reinforcement learning."
    )
    source["execution"].update(
        {
            "experiment_name": "rapec-g-v2-small-validation-v1",
            "analysis_module": "controller_benchmark.analyze_live_shadow",
            "dashboard_files": ["rapec-g-v2-small-validation.json"],
        }
    )
    source["acceptance"] = {
        "minimum_energy_saving_fraction": 0.0,
        "maximum_quality_regret": 0.10,
        "minimum_success_rate": 0.80,
    }
    validate_manifest(source)
    return source


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--vision-source", type=Path, required=True)
    parser.add_argument("--classification-source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    manifest = build_small_manifest(
        json.loads(args.vision_source.read_text(encoding="utf-8")),
        json.loads(args.classification_source.read_text(encoding="utf-8")),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(args.output)
    print(
        json.dumps(
            {
                "cases": sum(len(stage["cases"]) for stage in manifest["stages"]),
                "case_ids": [
                    case["id"]
                    for stage in manifest["stages"]
                    for case in stage["cases"]
                ],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
