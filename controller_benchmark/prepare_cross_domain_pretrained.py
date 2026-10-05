from __future__ import annotations

import argparse
import copy
import json
import os
import time
from pathlib import Path
from typing import Any

import torch

from controller_benchmark.runners.common import GpuSampler
from controller_benchmark.runners.cross_domain_tasks import assert_cuda_model, build_task
from slurm.gpu_energy_utils import load_gpu_rows, summarize_rows


def scratch_cases(manifest: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        case
        for stage in manifest["stages"]
        if stage.get("enabled")
        for case in stage["cases"]
        if case.get("runner") == "cross_domain_task"
        and not bool(case.get("pretrained", False))
    ]


def prepare(
    manifest_path: Path,
    output_root: Path,
    pretraining_checkpoints: int,
    force: bool,
    case_ids: set[str] | None = None,
    status_path: Path | None = None,
) -> None:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    output_root.mkdir(parents=True, exist_ok=True)
    status_handle = None
    if status_path is not None:
        status_path.parent.mkdir(parents=True, exist_ok=True)
        status_handle = status_path.open("w", encoding="utf-8")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rows = []
    try:
        for case in scratch_cases(manifest):
            if case_ids and case["id"] not in case_ids:
                continue
            output = output_root / f"{case['id']}.pt"
            if output.exists() and not force:
                print(f"SKIP {case['id']}: {output}", flush=True)
                continue
            spec = {
                "case": case,
                "execution": {
                    **manifest["execution"],
                    "max_epochs": pretraining_checkpoints,
                },
                "training_seed": int(case.get("training_seed", 0)),
            }
            if bool(case.get("require_cuda", False)) and device.type != "cuda":
                raise RuntimeError(
                    f"Source case {case['id']} requires CUDA, but no GPU is available."
                )
            task = build_task(spec, device)
            if bool(case.get("require_cuda", False)):
                assert_cuda_model(task.model, device, str(case["id"]))
            energy_path = output_root / "energy" / f"{case['id']}.csv"
            sampler = GpuSampler(energy_path)
            sampler.start()
            started = time.time()
            final_metrics: dict[str, float] = {}
            best_metrics: dict[str, float] = {}
            best_quality = float("-inf")
            best_checkpoint = 0
            best_state: dict[str, torch.Tensor] | None = None
            try:
                for checkpoint in range(1, pretraining_checkpoints + 1):
                    cuda_start = None
                    cuda_end = None
                    if device.type == "cuda":
                        torch.cuda.synchronize(device)
                        torch.cuda.reset_peak_memory_stats(device)
                        cuda_start = torch.cuda.Event(enable_timing=True)
                        cuda_end = torch.cuda.Event(enable_timing=True)
                        cuda_start.record()
                    quality, final_metrics = task.epoch(checkpoint)
                    if device.type == "cuda":
                        cuda_end.record()
                        torch.cuda.synchronize(device)
                        final_metrics.update(
                            {
                                "cuda_training_elapsed_ms": float(
                                    cuda_start.elapsed_time(cuda_end)
                                ),
                                "cuda_peak_memory_allocated_mb": float(
                                    torch.cuda.max_memory_allocated(device) / (1024.0**2)
                                ),
                                "cuda_execution_verified": 1.0,
                            }
                        )
                    if quality > best_quality:
                        best_quality = quality
                        best_checkpoint = checkpoint
                        best_metrics = dict(final_metrics)
                        best_state = copy.deepcopy(task.model.state_dict())
                    line = json.dumps(
                        {
                            "source_case": case["id"],
                            "checkpoint": checkpoint,
                            "pretraining_checkpoints": pretraining_checkpoints,
                            **final_metrics,
                        }
                    )
                    print(line, flush=True)
                    if status_handle is not None:
                        status_handle.write(line + "\n")
                        status_handle.flush()
            finally:
                sampler.stop()
            if best_state is None:
                raise RuntimeError(f"No valid source checkpoint was produced for {case['id']}.")
            gpu = summarize_rows(load_gpu_rows(energy_path))
            document = {
                "schema_version": 1,
                "model_state": {
                    key: value.detach().cpu()
                    for key, value in best_state.items()
                },
                "source_case_id": case["id"],
                "task_type": case["task_type"],
                "task_family": case["task_family"],
                "dataset_key": case["dataset_key"],
                "model_version": case["model_version"],
                "training_seed": spec["training_seed"],
                "pretraining_checkpoints": pretraining_checkpoints,
                "best_pretraining_checkpoint": best_checkpoint,
                "best_pretraining_quality_score": best_quality,
                "pretraining_scope": "source_domain_training_split_only",
                "duration_seconds": time.time() - started,
                "pretraining_gpu_energy_wh": float(gpu["gpu_energy_kwh"]) * 1000.0,
                "pretraining_gpu_power_avg_w": float(gpu["gpu_power_avg_w"]),
                "pretraining_gpu_util_avg_pct": float(gpu["gpu_util_avg_pct"]),
                "training_device": str(device),
                "cuda_execution_required": bool(case.get("require_cuda", False)),
                "cuda_device_name": (
                    torch.cuda.get_device_name(device) if device.type == "cuda" else None
                ),
                "best_training_metrics": best_metrics,
                "final_training_metrics": final_metrics,
            }
            temporary = output.with_suffix(".pt.tmp")
            torch.save(document, temporary)
            os.replace(temporary, output)
            rows.append({key: value for key, value in document.items() if key != "model_state"})
            del task
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    finally:
        if status_handle is not None:
            status_handle.close()
    catalog = output_root / "catalog.json"
    existing = []
    if catalog.exists():
        existing = json.loads(catalog.read_text(encoding="utf-8")).get("checkpoints", [])
    by_case = {str(row["source_case_id"]): row for row in [*existing, *rows]}
    catalog.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "pretraining_checkpoints": pretraining_checkpoints,
                "energy_scope": "pretraining_reported_separately_not_charged_to_finetuning",
                "checkpoints": [by_case[key] for key in sorted(by_case)],
            },
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    print(catalog)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("/workspace-cache/controller-pretrained-cross-domain"),
    )
    parser.add_argument("--pretraining-checkpoints", type=int, default=20)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--case-id", action="append", default=[])
    parser.add_argument(
        "--status-path",
        type=Path,
        default=Path(os.environ.get("PRETRAINING_STATUS_PATH", "")) if os.environ.get("PRETRAINING_STATUS_PATH") else None,
    )
    args = parser.parse_args()
    if args.pretraining_checkpoints < 1:
        parser.error("--pretraining-checkpoints must be positive")
    prepare(
        args.manifest,
        args.output_root,
        args.pretraining_checkpoints,
        args.force,
        set(args.case_id) or None,
        args.status_path,
    )


if __name__ == "__main__":
    main()
