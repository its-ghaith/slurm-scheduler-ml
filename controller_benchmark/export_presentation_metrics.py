from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

from controller_benchmark.case_labels import CASE_LABELS, case_metadata


DEFAULT_RUN_ID = "thesis-presentation-48"
DEFAULT_VERSION = "thesis-presentation-48-v1"
SOURCE_MODE = "recorded_full100_replay"
PHASES = (
    "preprocessing_initialization",
    "training",
    "finalization_evaluation",
    "other",
)
COMMON_RAW_METRICS = (
    "accuracy",
    "macro_f1",
    "map50",
    "map50_95",
    "precision",
    "recall",
    "miou",
    "dice",
    "pixel_accuracy",
    "mae",
    "rmse",
)


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def _number(value: object) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _escape(value: object) -> str:
    return str(value).replace("\\", "\\\\").replace("\n", "\\n").replace('"', '\\"')


def _labels(values: dict[str, object]) -> str:
    return ",".join(f'{key}="{_escape(value)}"' for key, value in values.items())


def _sample(lines: list[str], metric: str, labels: dict[str, object], value: object) -> None:
    number = _number(value)
    if number is not None:
        lines.append(f"{metric}{{{_labels(labels)}}} {number:.12g}")


def _quality_at_epoch(summary: dict[str, Any], epoch: int) -> float:
    rows = summary.get("epochs") or []
    observed = [
        _number(row.get("quality_score"))
        for row in rows
        if int(row.get("epoch", row.get("epoch_index", -1))) <= epoch
    ]
    return max((value for value in observed if value is not None), default=0.0)


def _index_source(source_root: Path) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, Any]]]:
    summaries: dict[str, dict[str, Any]] = {}
    shadows: dict[str, dict[str, Any]] = {}
    for path in sorted(source_root.rglob("epoch_summary_job_*.json")):
        data = _load(path)
        case_id = str(data["benchmark_case_id"])
        if case_id in summaries:
            raise ValueError(f"Duplicate epoch summary for {case_id}")
        summaries[case_id] = data
    for path in sorted(source_root.rglob("shadow_summary_job_*.json")):
        data = _load(path)
        case_id = str(data["benchmark_case_id"])
        if case_id in shadows:
            raise ValueError(f"Duplicate shadow summary for {case_id}")
        shadows[case_id] = data
    expected = set(CASE_LABELS)
    if set(summaries) != expected or set(shadows) != expected:
        raise ValueError(
            "Presentation export requires exactly the registered 48 cases; "
            f"epoch missing={sorted(expected - set(summaries))}, "
            f"shadow missing={sorted(expected - set(shadows))}"
        )
    return summaries, shadows


def _report_rows(report: dict[str, Any]) -> dict[str, dict[str, Any]]:
    rows = report.get("internal_rows") or []
    indexed = {str(row["_v4"]["case_id"]): row for row in rows}
    if set(indexed) != set(CASE_LABELS):
        raise ValueError("Replay report does not contain the exact 48-case registry")
    return indexed


def build(
    source_root: Path,
    replay_report: Path,
    *,
    run_id: str = DEFAULT_RUN_ID,
    benchmark_version: str = DEFAULT_VERSION,
) -> str:
    summaries, shadows = _index_source(source_root)
    report = _load(replay_report)
    rows = _report_rows(report)
    lines = [
        "# Presentation dataset built only from recorded Full100 trajectories and their deterministic controller replay.",
        "# Provenance remains machine-readable in source_mode and measurement_status labels.",
    ]

    run_labels = {
        "benchmark_run_id": run_id,
        "benchmark_version": benchmark_version,
        "source_mode": SOURCE_MODE,
        "measurement_status": "recorded_and_replayed",
    }
    for metric, value in (
        ("thesis_live_shadow_real_jobs", 48),
        ("thesis_live_shadow_virtual_es_jobs", 48),
        ("thesis_live_shadow_virtual_rapec_g_v4_jobs", 48),
        ("thesis_live_shadow_represented_jobs_total", 144),
        ("thesis_live_shadow_lifecycle_complete", 0),
    ):
        _sample(lines, metric, run_labels, value)

    seen_raw: set[tuple[str, str, str]] = set()
    for case_id, (_, _, _) in sorted(CASE_LABELS.items(), key=lambda item: item[1][2]):
        summary = summaries[case_id]
        shadow = shadows[case_id]
        report_row = rows[case_id]
        case_code, case_name, case_order = case_metadata(case_id)
        source_run_id = str(summary.get("benchmark_run_id", "unknown"))
        job_id = str(summary.get("job_id", shadow.get("job_id", "unknown")))
        stage = str(summary.get("benchmark_stage", shadow.get("benchmark_stage", "unknown")))
        task_type = str(summary.get("task_type", shadow.get("task_type", "unknown")))
        physical = {
            "benchmark_run_id": run_id,
            "benchmark_version": benchmark_version,
            "benchmark_case_id": case_id,
            "case_code": case_code,
            "case_name": case_name,
            "case_order": case_order,
            "benchmark_stage": stage,
            "task_type": task_type,
            "job_id": job_id,
            "source_mode": SOURCE_MODE,
            "source_run_id": source_run_id,
        }

        gpu_energy_kwh = _number(summary.get("total_gpu_energy_kwh")) or 0.0
        total_energy_kwh = _number(summary.get("total_energy_kwh")) or gpu_energy_kwh
        duration_seconds = _number(summary.get("total_duration_seconds")) or 0.0
        cost_eur = _number(summary.get("estimated_electricity_cost_eur")) or 0.0
        co2_kg = _number(summary.get("estimated_co2_kg")) or 0.0
        measured = {**physical, "measurement_status": "measured"}
        proxy = {**physical, "measurement_status": "derived_proxy"}
        _sample(lines, "slurm_job_gpu_energy_kwh", measured, gpu_energy_kwh)
        _sample(lines, "slurm_job_estimated_electricity_cost_eur", measured, cost_eur)
        _sample(lines, "slurm_job_estimated_co2_kg", measured, co2_kg)
        _sample(lines, "slurm_job_codecarbon_job_total_gpu_energy_kwh", proxy, gpu_energy_kwh)
        _sample(lines, "slurm_job_codecarbon_job_total_energy_kwh", proxy, total_energy_kwh)
        _sample(lines, "slurm_job_codecarbon_estimated_electricity_cost_eur", proxy, cost_eur)
        _sample(lines, "slurm_job_codecarbon_estimated_co2_kg", proxy, co2_kg)

        for phase in PHASES:
            phase_is_training = phase == "training"
            status = "measured" if phase_is_training else "not_measured"
            phase_labels = {**physical, "phase": phase, "measurement_status": status}
            phase_duration = duration_seconds if phase_is_training else 0.0
            phase_energy = gpu_energy_kwh if phase_is_training else 0.0
            _sample(lines, "slurm_job_phase_duration_seconds", phase_labels, phase_duration)
            _sample(lines, "slurm_job_phase_gpu_energy_kwh", phase_labels, phase_energy)
            cc_status = "derived_proxy" if phase_is_training else "not_measured"
            _sample(
                lines,
                "slurm_job_phase_codecarbon_gpu_energy_kwh",
                {**physical, "phase": phase, "measurement_status": cc_status},
                phase_energy,
            )

        epoch_base = {
            **physical,
            "measurement_status": "measured",
        }
        thesis_base = {
            "benchmark_run_id": run_id,
            "benchmark_version": benchmark_version,
            "case_id": case_id,
            "case_code": case_code,
            "case_name": case_name,
            "case_order": case_order,
            "task_type": task_type,
            "stage": stage,
            "job_id": job_id,
            "source_mode": SOURCE_MODE,
            "source_run_id": source_run_id,
            "measurement_status": "measured",
        }
        for epoch_row in summary.get("epochs") or []:
            epoch = str(epoch_row.get("epoch", epoch_row.get("epoch_index", "unknown")))
            physical_epoch = {**epoch_base, "epoch": epoch}
            thesis_epoch = {**thesis_base, "epoch": epoch}
            energy_kwh = _number(epoch_row.get("gpu_energy_kwh"))
            quality = _number(epoch_row.get("quality_score"))
            utilization = _number(epoch_row.get("gpu_util_avg_pct"))
            _sample(lines, "slurm_job_epoch_gpu_energy_kwh", physical_epoch, energy_kwh)
            _sample(lines, "slurm_job_epoch_gpu_util_avg_pct", physical_epoch, utilization)
            _sample(lines, "slurm_job_epoch_quality_score", physical_epoch, quality)
            if energy_kwh is not None:
                _sample(lines, "thesis_live_shadow_epoch_gpu_energy_wh", thesis_epoch, energy_kwh * 1000.0)
            _sample(lines, "thesis_live_shadow_epoch_quality_score", thesis_epoch, quality)

            raw_metric = epoch_row.get("raw_quality_metric")
            raw_value = _number(epoch_row.get("raw_quality_value"))
            if raw_metric and raw_value is not None:
                key = (case_id, epoch, str(raw_metric))
                seen_raw.add(key)
                _sample(
                    lines,
                    "thesis_live_shadow_epoch_raw_quality",
                    {**thesis_epoch, "raw_quality_metric": raw_metric},
                    raw_value,
                )
            for raw_name in COMMON_RAW_METRICS:
                key = (case_id, epoch, raw_name)
                value = _number(epoch_row.get(raw_name))
                if value is None or key in seen_raw:
                    continue
                seen_raw.add(key)
                _sample(
                    lines,
                    "thesis_live_shadow_epoch_raw_quality",
                    {**thesis_epoch, "raw_quality_metric": raw_name},
                    value,
                )

        shadow_controllers = {
            str(controller["controller_id"]): controller
            for controller in shadow.get("controllers") or []
        }
        for report_key, controller_id in (
            ("_standard", "standard-es-10"),
            ("_v4", "rapec-g-v4"),
        ):
            result = report_row[report_key]
            controller = shadow_controllers[controller_id]
            stop_epoch = int(result["stop_epoch"])
            if int(controller["stop_epoch"]) != stop_epoch:
                raise ValueError(f"Stop-epoch mismatch for {case_id}/{controller_id}")
            overhead = _number(controller.get("controller_overhead_energy_wh")) or 0.0
            stopped_energy = float(result["stopped_energy_wh"])
            common = {
                "benchmark_run_id": run_id,
                "benchmark_version": benchmark_version,
                "controller_id": controller_id,
                "stage": stage,
                "case_id": case_id,
                "case_code": case_code,
                "case_name": case_name,
                "case_order": case_order,
                "task_type": task_type,
                "energy_measurement_method": controller.get("energy_measurement_method", "unknown"),
                "source_mode": SOURCE_MODE,
                "source_run_id": source_run_id,
                "measurement_status": "recorded_and_replayed",
            }
            values = {
                "live_shadow_controller_stop_epoch": stop_epoch,
                "live_shadow_full100_best_quality": result["baseline_best_quality"],
                "live_shadow_best_quality_at_stop": result["stopped_best_quality"],
                "live_shadow_quality_regret": result["quality_regret"],
                "live_shadow_full100_training_energy_wh": result["baseline_energy_wh"],
                "live_shadow_training_energy_to_stop_wh": stopped_energy,
                "live_shadow_controller_overhead_energy_wh": overhead,
                "live_shadow_counterfactual_total_energy_wh": stopped_energy + overhead,
                "live_shadow_energy_saving_fraction": result["energy_saving_fraction"],
                "live_shadow_controller_compute_seconds": controller.get("controller_compute_seconds", 0.0),
                "live_shadow_training_time_to_stop_seconds": controller.get("training_time_to_stop_s", 0.0),
                "live_shadow_stopped_early": int(bool(result.get("stopped_early"))),
            }
            for metric, value in values.items():
                _sample(lines, metric, common, value)

    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--replay-report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--run-id", default=DEFAULT_RUN_ID)
    parser.add_argument("--benchmark-version", default=DEFAULT_VERSION)
    args = parser.parse_args()
    content = build(
        args.source_root,
        args.replay_report,
        run_id=args.run_id,
        benchmark_version=args.benchmark_version,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(content, encoding="utf-8", newline="\n")
    print(f"Wrote {args.output} ({content.count(chr(10))} lines)")


if __name__ == "__main__":
    main()
