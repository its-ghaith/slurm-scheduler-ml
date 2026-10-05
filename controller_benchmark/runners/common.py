from __future__ import annotations

import csv
import json
import math
import os
import subprocess
import threading
import time
from pathlib import Path
from typing import Any

from controller_benchmark.runtime import ControllerPluginRuntime
from slurm.gpu_energy_utils import load_gpu_rows, summarize_rows, summarize_window


def decode_run_spec(encoded: str) -> dict[str, Any]:
    import base64

    return json.loads(base64.b64decode(encoded, validate=True).decode("utf-8"))


def number(value: Any, default: float | None = None) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return default
    return result if math.isfinite(result) else default


class GpuSampler:
    HEADER = ["ts", "gpu_index", "power_w", "util_gpu_pct", "util_mem_pct", "mem_used_mb", "temp_c"]

    def __init__(self, path: Path, interval_seconds: float = 0.5):
        self.path = path
        self.interval_seconds = interval_seconds
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("w", encoding="utf-8", newline="") as handle:
            csv.writer(handle).writerow(self.HEADER)
        self._thread = threading.Thread(target=self._run, name="gpu-sampler", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=10)

    def _run(self) -> None:
        query = "index,power.draw,utilization.gpu,utilization.memory,memory.used,temperature.gpu"
        while not self._stop.is_set():
            try:
                output = subprocess.check_output(
                    ["nvidia-smi", f"--query-gpu={query}", "--format=csv,noheader,nounits"],
                    text=True,
                    stderr=subprocess.DEVNULL,
                    timeout=5,
                )
                now = time.time()
                with self.path.open("a", encoding="utf-8", newline="") as handle:
                    writer = csv.writer(handle)
                    for line in output.splitlines():
                        values = [part.strip() for part in line.split(",")]
                        if len(values) == 6:
                            writer.writerow([now, *values])
            except Exception:
                pass
            self._stop.wait(self.interval_seconds)


def gpu_runtime_metadata(spec: dict[str, Any]) -> dict[str, Any]:
    query = "name,driver_version,power.limit,clocks.current.graphics,clocks.current.memory,temperature.gpu"
    values = ["unknown"] * 6
    try:
        output = subprocess.check_output(
            ["nvidia-smi", f"--query-gpu={query}", "--format=csv,noheader,nounits"], text=True, timeout=5
        ).splitlines()[0]
        values = [part.strip() for part in output.split(",")]
    except Exception:
        pass
    return {
        "captured_at": time.time(),
        "runtime_image_id": os.environ.get("RUNTIME_IMAGE_ID", spec["execution"].get("runtime_image_id", "unknown")),
        "scenario": spec["scenario"],
        "training_seed": spec["training_seed"],
        "split_seed": spec["execution"]["split_seed"],
        "cache_policy": spec["execution"]["cache_policy"],
        "gpu": {
            "name": values[0],
            "driver_version": values[1],
            "power_limit_w": values[2],
            "graphics_clock_mhz": values[3],
            "memory_clock_mhz": values[4],
            "temperature_c": values[5],
        },
    }


class EpochTelemetry:
    def __init__(self, spec: dict[str, Any], output_dir: Path):
        self.spec = spec
        self.output_dir = output_dir
        self.job_id = str(os.environ.get("SLURM_JOB_ID", "local"))
        self.resume_from_job_id = str(spec.get("case", {}).get("resume_from_job_id", "") or "")
        self.csv_path = output_dir / f"gpu_metrics_job_{self.job_id}.csv"
        self.timeline_path = output_dir / f"epoch_timeline_job_{self.job_id}.jsonl"
        self.epoch_summary_path = output_dir / f"epoch_summary_job_{self.job_id}.json"
        self.gpu_summary_path = output_dir / f"gpu_summary_job_{self.job_id}.json"
        self.phase_timeline_path = output_dir / f"phase_timeline_job_{self.job_id}.jsonl"
        self.phase_codecarbon_dir = output_dir / f"codecarbon_job_{self.job_id}"
        self.sampler = GpuSampler(self.csv_path)
        self.codecarbon_tracker = None
        self.codecarbon_available = False
        self.phase_tracker = None
        self.phase_name: str | None = None
        self.phase_started_at: float | None = None
        self.phase_records: dict[str, dict[str, float]] = {}
        self.started = False
        self.closed = False
        self.history: list[dict[str, Any]] = []
        self.epoch_start = 0.0
        self.best_quality: float | None = None
        self.plugin = None
        self.resume_energy_kwh = 0.0
        self.resume_duration_seconds = 0.0
        self._load_resume_history()
        if spec["controller_mode"] == "plugin":
            self.plugin = ControllerPluginRuntime(
                plugin_path=spec["controller_plugin"],
                parameters_json=json.dumps(spec.get("controller_parameters", {})),
                controller_id=spec["controller_id"],
                benchmark_version=spec["benchmark_version"],
                task_type=spec["task_type"],
                quality_metric=spec["quality_metric"],
                scenario=spec["scenario"],
                max_epochs=int(spec["execution"]["max_epochs"]),
                metadata={"benchmark_case_id": spec["benchmark_case_id"], "benchmark_stage": spec["benchmark_stage"]},
            )

    def _load_resume_history(self) -> None:
        if not self.resume_from_job_id:
            return
        previous_timeline = self.output_dir / f"epoch_timeline_job_{self.resume_from_job_id}.jsonl"
        if not previous_timeline.exists():
            raise FileNotFoundError(f"Resume timeline not found: {previous_timeline}")
        metric = self.spec["quality_metric"]
        with previous_timeline.open("r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                event = json.loads(line)
                self.history.append(event)
                self.resume_energy_kwh += float(event.get("gpu_energy_kwh", event.get("total_energy_kwh", 0.0)) or 0.0)
                self.resume_duration_seconds += float(event.get("duration_seconds", 0.0) or 0.0)
                value = number(event.get(metric, event.get("quality_score")))
                if value is not None:
                    self.best_quality = value if self.best_quality is None else max(self.best_quality, value)

    def start(self) -> None:
        if self.started:
            return
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.phase_codecarbon_dir.mkdir(parents=True, exist_ok=True)
        self.sampler.start()
        if self.history:
            with self.timeline_path.open("w", encoding="utf-8") as handle:
                for event in self.history:
                    handle.write(json.dumps(event, ensure_ascii=False) + "\n")
        try:
            from codecarbon import OfflineEmissionsTracker

            self.codecarbon_tracker = OfflineEmissionsTracker(
                log_level="error",
                output_dir=str(self.output_dir),
                output_file=f"codecarbon_job_total_job_{self.job_id}.csv",
                country_iso_code="DEU",
            )
            self.codecarbon_tracker.start()
            self.codecarbon_available = True
        except Exception:
            self.codecarbon_tracker = None
            self.codecarbon_available = False
        self.started = True
        self.begin_phase("preprocessing_initialization")

    def _write_phase_event(self, event: dict[str, Any]) -> None:
        with self.phase_timeline_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(event, ensure_ascii=False) + "\n")

    def _close_phase(self) -> None:
        if self.phase_name is None or self.phase_started_at is None:
            return
        ended_at = time.time()
        codecarbon_total = 0.0
        codecarbon_gpu = 0.0
        codecarbon_cpu = 0.0
        if self.phase_tracker is not None:
            try:
                self.phase_tracker.stop()
                codecarbon_total = float(
                    getattr(getattr(self.phase_tracker, "_total_energy", None), "kWh", 0.0)
                )
                codecarbon_gpu = float(
                    getattr(getattr(self.phase_tracker, "_total_gpu_energy", None), "kWh", 0.0)
                )
                codecarbon_cpu = float(
                    getattr(getattr(self.phase_tracker, "_total_cpu_energy", None), "kWh", 0.0)
                )
            except Exception:
                self.codecarbon_available = False
        window = summarize_window(self.csv_path, self.phase_started_at, ended_at)
        duration = max(0.0, ended_at - self.phase_started_at)
        gpu_kwh = max(0.0, float(window.get("gpu_energy_kwh", 0.0)))
        idle_power = max(0.0, float(os.environ.get("GPU_IDLE_POWER_W", "0") or 0.0))
        idle_kwh = idle_power * duration / 3_600_000.0
        pue = float(os.environ.get("PUE_FACTOR", "1.0"))
        price = float(os.environ.get("ELECTRICITY_PRICE_EUR_PER_KWH", "0.30"))
        co2_factor = float(os.environ.get("GRID_CO2_KG_PER_KWH", "0.40"))
        record = {
            "duration_seconds": duration,
            "gpu_energy_kwh": gpu_kwh,
            "total_energy_kwh": gpu_kwh * pue,
            "idle_energy_kwh": idle_kwh,
            "net_gpu_energy_kwh": max(0.0, gpu_kwh - idle_kwh),
            "net_total_energy_kwh": max(0.0, gpu_kwh - idle_kwh) * pue,
            "codecarbon_duration_seconds": duration,
            "codecarbon_energy_kwh": max(0.0, codecarbon_total),
            "codecarbon_gpu_energy_kwh": max(0.0, codecarbon_gpu),
            "codecarbon_cpu_energy_kwh": max(0.0, codecarbon_cpu),
            "estimated_electricity_cost_eur": gpu_kwh * pue * price,
            "estimated_co2_kg": gpu_kwh * pue * co2_factor,
            "codecarbon_estimated_electricity_cost_eur": max(0.0, codecarbon_total) * price,
            "codecarbon_estimated_co2_kg": max(0.0, codecarbon_total) * co2_factor,
        }
        self.phase_records[self.phase_name] = record
        self._write_phase_event(
            {
                "phase": self.phase_name,
                "event": "end",
                "ts": ended_at,
                **record,
            }
        )
        self.phase_name = None
        self.phase_started_at = None
        self.phase_tracker = None

    def begin_phase(self, name: str) -> None:
        if not self.started or self.closed:
            return
        if self.phase_name == name:
            return
        self._close_phase()
        self.phase_name = name
        self.phase_started_at = time.time()
        self._write_phase_event({"phase": name, "event": "start", "ts": self.phase_started_at})
        try:
            from codecarbon import OfflineEmissionsTracker

            self.phase_tracker = OfflineEmissionsTracker(
                log_level="error",
                output_dir=str(self.phase_codecarbon_dir),
                output_file=f"codecarbon_{name}.csv",
                country_iso_code="DEU",
            )
            self.phase_tracker.start()
        except Exception:
            self.phase_tracker = None
            self.codecarbon_available = False

    def start_epoch(self) -> None:
        if self.phase_name != "training":
            self.begin_phase("training")
        self.epoch_start = time.time()

    def finish_epoch(self, epoch: int, quality: float, extra_metrics: dict[str, Any] | None = None) -> bool:
        # Ultralytics can emit on_fit_epoch_end once more during final validation.
        if self.history and epoch <= int(self.history[-1]["epoch"]):
            return bool(self.history[-1].get("should_stop"))
        end = time.time()
        gpu = summarize_window(self.csv_path, self.epoch_start, end)
        quality = float(quality)
        if not 0.0 <= quality <= 1.0:
            raise ValueError(f"quality_score must be in [0, 1], got {quality}.")
        self.best_quality = quality if self.best_quality is None else max(self.best_quality, quality)
        cumulative_energy = sum(float(row["total_energy_kwh"]) for row in self.history) + gpu["gpu_energy_kwh"]
        event = {
            "epoch": epoch,
            "epoch_index": epoch,
            "decision_checkpoint": epoch,
            "progress_unit": str(self.spec.get("case", {}).get("progress_unit", "epoch")),
            self.spec["quality_metric"]: quality,
            f"best_{self.spec['quality_metric']}": self.best_quality,
            "duration_seconds": end - self.epoch_start,
            "total_energy_kwh": gpu["gpu_energy_kwh"],
            "gpu_energy_kwh": gpu["gpu_energy_kwh"],
            "cumulative_total_energy_kwh": cumulative_energy,
            "interval_energy_wh": gpu["gpu_energy_kwh"] * 1000.0,
            "cumulative_energy_wh": cumulative_energy * 1000.0,
            "gpu_power_avg_w": gpu["gpu_power_avg_w"],
            "gpu_power_max_w": gpu["gpu_power_max_w"],
            "gpu_util_avg_pct": gpu["gpu_util_avg_pct"],
            "gpu_mem_used_avg_mb": gpu["gpu_mem_used_avg_mb"],
            "gpu_memory_used_mb": gpu["gpu_mem_used_avg_mb"],
            "gpu_temp_avg_c": gpu["gpu_temp_avg_c"],
            "controller_mode": self.spec["controller_mode"],
            "comparison_strategy": self.spec["strategy"],
            "controller_id": self.spec["controller_id"],
            "benchmark_version": self.spec["benchmark_version"],
            "benchmark_run_id": self.spec["benchmark_run_id"],
            "benchmark_case_id": self.spec["benchmark_case_id"],
            "benchmark_stage": self.spec["benchmark_stage"],
            "task_type": self.spec["task_type"],
            "quality_metric": self.spec["quality_metric"],
            "adaptive_monitor_metric": "quality_score",
        }
        event.update(extra_metrics or {})
        stop, reason, diagnostics = (False, None, {})
        if self.plugin is not None:
            stop, reason, diagnostics = self.plugin.evaluate(event, self.history)
        event.update(diagnostics)
        event["should_stop"] = int(stop)
        event["stop_reason"] = reason
        self.history.append(event)
        with self.timeline_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(event, ensure_ascii=False) + "\n")
        try:
            import mlflow

            mlflow.log_metric(f"epoch/{self.spec['quality_metric']}", quality, step=epoch)
            mlflow.log_metric("epoch/slurm_gpu_energy_wh", gpu["gpu_energy_kwh"] * 1000.0, step=epoch)
            for key, value in event.items():
                if key.startswith("controller_plugin_") and number(value) is not None:
                    mlflow.log_metric(f"epoch/{key}", float(value), step=epoch)
        except Exception:
            pass
        return stop

    def finalize(self) -> tuple[dict[str, Any], dict[str, Any]]:
        # Training has ended. Keep job and phase telemetry alive while checkpoint
        # evaluation, artifact logging, and cleanup execute.
        self.begin_phase("finalization_evaluation")
        return self._write_summaries(final=False)

    def _write_summaries(self, *, final: bool) -> tuple[dict[str, Any], dict[str, Any]]:
        gpu = summarize_rows(load_gpu_rows(self.csv_path))
        gpu["gpu_energy_kwh"] = float(gpu.get("gpu_energy_kwh", 0.0)) + self.resume_energy_kwh
        gpu["duration_seconds"] = float(gpu.get("duration_seconds", 0.0)) + self.resume_duration_seconds
        runtime = gpu_runtime_metadata(self.spec)
        pue = float(os.environ.get("PUE_FACTOR", "1.0"))
        price = float(os.environ.get("ELECTRICITY_PRICE_EUR_PER_KWH", "0.30"))
        co2_factor = float(os.environ.get("GRID_CO2_KG_PER_KWH", "0.40"))
        codecarbon_total_kwh = float(
            getattr(getattr(self.codecarbon_tracker, "_total_energy", None), "kWh", 0.0)
        )
        codecarbon_gpu_kwh = float(
            getattr(getattr(self.codecarbon_tracker, "_total_gpu_energy", None), "kWh", 0.0)
        )
        codecarbon_cpu_kwh = float(
            getattr(getattr(self.codecarbon_tracker, "_total_cpu_energy", None), "kWh", 0.0)
        )
        gpu_summary = {
            "job_id": self.job_id,
            **gpu,
            "scenario": self.spec["scenario"],
            "comparison_strategy": self.spec["strategy"],
            "training_seed": self.spec["training_seed"],
            "split_seed": self.spec["execution"]["split_seed"],
            "cache_policy": self.spec["execution"]["cache_policy"],
            "controller_id": self.spec["controller_id"],
            "benchmark_version": self.spec["benchmark_version"],
            "benchmark_run_id": self.spec["benchmark_run_id"],
            "benchmark_case_id": self.spec["benchmark_case_id"],
            "benchmark_stage": self.spec["benchmark_stage"],
            "task_type": self.spec["task_type"],
            "quality_metric": self.spec["quality_metric"],
            "adaptive_monitor_metric": "quality_score",
            "training_energy_kwh": float(
                self.phase_records.get("training", {}).get("gpu_energy_kwh", 0.0)
            ) or sum(float(row["gpu_energy_kwh"]) for row in self.history),
            "estimated_electricity_cost_eur": gpu["gpu_energy_kwh"] * pue * price,
            "estimated_co2_kg": gpu["gpu_energy_kwh"] * pue * co2_factor,
            "codecarbon_job_total_energy_kwh": codecarbon_total_kwh,
            "codecarbon_job_total_gpu_energy_kwh": codecarbon_gpu_kwh,
            "codecarbon_job_total_cpu_energy_kwh": codecarbon_cpu_kwh,
            "codecarbon_measurement_available": int(self.codecarbon_available and final),
            "codecarbon_estimated_electricity_cost_eur": codecarbon_total_kwh * pue * price,
            "codecarbon_estimated_co2_kg": codecarbon_total_kwh * pue * co2_factor,
            "phase_metrics": self.phase_records,
            "lifecycle_energy_complete": int(
                final
                and self.codecarbon_available
                and {"preprocessing_initialization", "training", "finalization_evaluation"}
                <= set(self.phase_records)
            ),
            "price_eur_kwh": price,
            "co2_kg_kwh": co2_factor,
            "pue_factor": pue,
            "run_metadata": runtime,
        }
        last = self.history[-1] if self.history else {}
        summary = {
            "job_id": self.job_id,
            "scenario": self.spec["scenario"],
            "comparison_strategy": self.spec["strategy"],
            "training_seed": self.spec["training_seed"],
            "split_seed": self.spec["execution"]["split_seed"],
            "cache_policy": self.spec["execution"]["cache_policy"],
            "controller_id": self.spec["controller_id"],
            "controller_mode": self.spec["controller_mode"],
            "benchmark_version": self.spec["benchmark_version"],
            "benchmark_run_id": self.spec["benchmark_run_id"],
            "benchmark_case_id": self.spec["benchmark_case_id"],
            "benchmark_stage": self.spec["benchmark_stage"],
            "task_type": self.spec["task_type"],
            "quality_metric": self.spec["quality_metric"],
            "epochs_completed": len(self.history),
            "stop_epoch": last.get("epoch") if last.get("should_stop") else None,
            "stop_reason": last.get("stop_reason"),
            "epochs": self.history,
            "total_gpu_energy_kwh": sum(float(row["gpu_energy_kwh"]) for row in self.history),
            "total_duration_seconds": sum(float(row["duration_seconds"]) for row in self.history),
        }
        if self.history:
            metric = self.spec["quality_metric"]
            summary[f"best_{metric}"] = max(float(row[metric]) for row in self.history)
            summary[f"final_{metric}"] = float(self.history[-1][metric])
        for key, value in last.items():
            if key.startswith("controller_plugin_"):
                summary[key] = value
        self.gpu_summary_path.write_text(json.dumps(gpu_summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        self.epoch_summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        return summary, gpu_summary

    def close(self) -> None:
        if self.closed:
            return
        if self.started:
            self._close_phase()
        if self.codecarbon_tracker is not None:
            try:
                self.codecarbon_tracker.stop()
            except Exception:
                self.codecarbon_available = False
        self.sampler.stop()
        if self.plugin is not None:
            self.plugin.close()
        self.closed = True
        if self.history:
            existing: dict[str, Any] = {}
            if self.epoch_summary_path.exists():
                try:
                    existing = json.loads(self.epoch_summary_path.read_text(encoding="utf-8"))
                except (OSError, json.JSONDecodeError):
                    existing = {}
            summary, gpu = self._write_summaries(final=True)
            summary.update(
                {
                    key: value
                    for key, value in existing.items()
                    if key not in {"epochs", "total_gpu_energy_kwh", "total_duration_seconds"}
                }
            )
            summary.update(
                {
                    "lifecycle_energy_complete": bool(gpu["lifecycle_energy_complete"]),
                    "phase_metrics": gpu["phase_metrics"],
                    "job_gpu_energy_kwh": gpu["gpu_energy_kwh"],
                    "job_duration_seconds": gpu["duration_seconds"],
                }
            )
            self.epoch_summary_path.write_text(
                json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
            )
            shadow_path = self.output_dir / f"shadow_summary_job_{self.job_id}.json"
            if shadow_path.exists():
                try:
                    shadow = json.loads(shadow_path.read_text(encoding="utf-8"))
                    shadow.update(
                        {
                            "lifecycle_energy_complete": bool(gpu["lifecycle_energy_complete"]),
                            "preprocessing_energy_wh": self.phase_records.get(
                                "preprocessing_initialization", {}
                            ).get("gpu_energy_kwh", 0.0)
                            * 1000.0,
                            "finalization_energy_wh": self.phase_records.get(
                                "finalization_evaluation", {}
                            ).get("gpu_energy_kwh", 0.0)
                            * 1000.0,
                            "phase_metrics": self.phase_records,
                        }
                    )
                    shadow_path.write_text(
                        json.dumps(shadow, indent=2, ensure_ascii=False) + "\n",
                        encoding="utf-8",
                    )
                except (OSError, json.JSONDecodeError):
                    pass
