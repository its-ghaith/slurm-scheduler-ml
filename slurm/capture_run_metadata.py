from __future__ import annotations

import argparse
import json
import os
import platform
import statistics
import subprocess
import time
from pathlib import Path


def _run(command: list[str]) -> str:
    try:
        return subprocess.check_output(command, text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:
        return ""


def _gpu_query(fields: str) -> list[str]:
    output = _run(["nvidia-smi", f"--query-gpu={fields}", "--format=csv,noheader,nounits"])
    return [part.strip() for part in output.splitlines()[0].split(",")] if output else []


def _package_version(name: str) -> str | None:
    try:
        from importlib.metadata import version

        return version(name)
    except Exception:
        return None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument("--idle-sample-seconds", type=float, default=10.0)
    parser.add_argument("--idle-sample-interval", type=float, default=0.5)
    parser.add_argument("--print-idle-power", action="store_true")
    args = parser.parse_args()

    idle_samples: list[float] = []
    deadline = time.time() + max(0.0, args.idle_sample_seconds)
    while time.time() < deadline:
        values = _gpu_query("power.draw")
        if values:
            try:
                idle_samples.append(float(values[0]))
            except ValueError:
                pass
        time.sleep(max(0.1, args.idle_sample_interval))

    fields = "name,driver_version,power.limit,clocks.current.graphics,clocks.current.memory,temperature.gpu"
    gpu = _gpu_query(fields)
    keys = ["name", "driver_version", "power_limit_w", "graphics_clock_mhz", "memory_clock_mhz", "temperature_c"]
    gpu_metadata = dict(zip(keys, gpu))
    utilization = _gpu_query("utilization.gpu,memory.used")
    if utilization:
        gpu_metadata["initial_utilization_pct"] = utilization[0]
    if len(utilization) > 1:
        gpu_metadata["initial_memory_used_mb"] = utilization[1]
    idle_power = statistics.fmean(idle_samples) if idle_samples else 0.0

    metadata = {
        "captured_at": time.time(),
        "hostname": platform.node(),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "slurm_node": os.environ.get("SLURMD_NODENAME") or os.environ.get("SLURM_NODELIST"),
        "scenario": os.environ.get("EXPERIMENT_SCENARIO", "unspecified"),
        "comparison_strategy": os.environ.get("COMPARISON_STRATEGY", "unspecified"),
        "training_seed": int(os.environ.get("TRAINING_SEED", "0")),
        "split_seed": int(os.environ.get("SPLIT_SEED", "0")),
        "cache_policy": os.environ.get("CACHE_POLICY", "ram"),
        "runtime_image_id": os.environ.get("RUNTIME_IMAGE_ID", "unknown"),
        "idle_power_w": idle_power,
        "idle_samples": len(idle_samples),
        "gpu": gpu_metadata,
        "other_compute_processes": _run(
            [
                "nvidia-smi",
                "--query-compute-apps=pid,process_name,used_memory",
                "--format=csv,noheader,nounits",
            ]
        ).splitlines(),
        "python_version": platform.python_version(),
        "packages": {
            name: _package_version(name)
            for name in ("torch", "ultralytics", "mlflow", "codecarbon", "numpy")
        },
    }
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    if args.print_idle_power:
        print(f"{idle_power:.6f}")


if __name__ == "__main__":
    main()
