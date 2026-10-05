from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path
from typing import Any

import torch

from controller_benchmark.runners.cross_domain_tasks import build_task


def _cases(manifest: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {
        case["id"]: case
        for stage in manifest["stages"]
        if stage.get("enabled", True)
        for case in stage["cases"]
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--case-id", action="append", default=[])
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    args = parser.parse_args()

    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    cases = _cases(manifest)
    selected = args.case_id or [
        case_id for case_id, case in cases.items() if not case.get("pretrained")
    ]
    missing = sorted(set(selected) - set(cases))
    if missing:
        parser.error(f"unknown case ids: {', '.join(missing)}")

    device = torch.device(args.device)
    for case_id in selected:
        case = cases[case_id]
        if case.get("pretrained"):
            raise ValueError(f"Smoke loader check expects a scratch case: {case_id}")
        spec = {
            "case": case,
            "execution": {**manifest["execution"], "max_epochs": 100},
            "training_seed": int(case.get("training_seed", 0)),
        }
        task = build_task(spec, device)
        print(
            json.dumps(
                {
                    "case_id": case_id,
                    "dataset_key": case["dataset_key"],
                    "task_type": case["task_type"],
                    "model": type(task.model).__name__,
                    "parameters": sum(parameter.numel() for parameter in task.model.parameters()),
                    "status": "loader_initialized",
                },
                sort_keys=True,
            ),
            flush=True,
        )
        del task
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
