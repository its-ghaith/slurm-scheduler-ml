from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path


def build_validation_manifest(source: dict, seeds: list[int]) -> dict:
    if not seeds:
        raise ValueError("At least one validation seed is required")
    result = copy.deepcopy(source)
    seed_tag = "-".join(str(seed) for seed in seeds)
    result["benchmark_version"] = (
        f"{source['benchmark_version']}-rapec-v3-ti-fresh-seeds-{seed_tag}"
    )
    result["description"] = (
        "Fresh paired Full100 and task-independent RAPEC-v3 validation. "
        "Generated from the nine-dataset benchmark without modifying it."
    )
    result["execution"]["experiment_name"] = result["benchmark_version"]
    result["execution"]["dashboard_files"] = ["controller-benchmark.json"]
    for stage in result["stages"]:
        expanded = []
        for case in stage.get("cases", []):
            for seed in seeds:
                replica = copy.deepcopy(case)
                replica["id"] = f"{case['id']}-seed-{seed}"
                replica["training_seed"] = seed
                replica["source_case_id"] = case["id"]
                expanded.append(replica)
        stage["cases"] = expanded
        stage.pop("baseline_case_id", None)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seeds", type=int, nargs="+", required=True)
    args = parser.parse_args()
    source = json.loads(args.source.read_text(encoding="utf-8-sig"))
    result = build_validation_manifest(source, args.seeds)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(args.output)


if __name__ == "__main__":
    main()
