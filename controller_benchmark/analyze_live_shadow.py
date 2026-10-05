from __future__ import annotations

import argparse
import csv
import json
import statistics
from pathlib import Path
from typing import Any

from controller_benchmark.case_labels import case_metadata


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def _prom_escape(value: Any) -> str:
    return str(value).replace("\\", "\\\\").replace("\n", "\\n").replace('"', '\\"')


def _labels(pair: dict[str, Any]) -> str:
    case_code, case_name, case_order = case_metadata(pair["case_id"])
    values = {
        "benchmark_run_id": pair["benchmark_run_id"],
        "benchmark_version": pair["benchmark_version"],
        "controller_id": pair["controller_id"],
        "stop_reason": pair.get("stop_reason", "unknown"),
        "stage": pair["stage"],
        "case_id": pair["case_id"],
        "case_code": case_code,
        "case_name": case_name,
        "case_order": case_order,
        "task_type": pair["task_type"],
        "energy_measurement_method": pair["energy_measurement_method"],
    }
    return ",".join(f'{key}="{_prom_escape(value)}"' for key, value in values.items())


def _variant_labels(row: dict[str, Any]) -> str:
    case_code, case_name, case_order = case_metadata(row["case_id"])
    values = {
        "benchmark_run_id": row["benchmark_run_id"],
        "benchmark_version": row["benchmark_version"],
        "variant": row["variant"],
        "job_kind": row["job_kind"],
        "represented_job_id": row["represented_job_id"],
        "source_job_id": row["source_job_id"],
        "stage": row["stage"],
        "case_id": row["case_id"],
        "case_code": case_code,
        "case_name": case_name,
        "case_order": case_order,
        "task_type": row["task_type"],
    }
    return ",".join(f'{key}="{_prom_escape(value)}"' for key, value in values.items())


def _virtual_identity(job_id: Any, controller_id: str) -> tuple[str, str]:
    if controller_id == "standard-es-10":
        return f"{job_id}-es", "Standard ES"
    if controller_id == "rapec-g-v4":
        return f"{job_id}-v4", "RAPEC-G v4"
    safe_id = "".join(character if character.isalnum() else "-" for character in controller_id)
    return f"{job_id}-{safe_id.strip('-')}", controller_id


def _best_quality(summary: dict[str, Any]) -> float:
    quality_metric = str(summary.get("quality_metric") or "quality_score")
    direct_candidates = (
        summary.get("best_quality_score"),
        summary.get(f"best_{quality_metric}"),
        summary.get("final_best_map50_95"),
    )
    for value in direct_candidates:
        if value is not None:
            return float(value)
    observed = []
    for row in summary.get("epochs") or []:
        value = row.get("best_quality_score")
        if value is None:
            value = row.get(quality_metric)
        if value is None:
            value = row.get("quality_score")
        if value is not None:
            observed.append(float(value))
    return max(observed, default=0.0)


def analyze_live_shadow(snapshot: Path, plan_path: Path) -> dict[str, Any]:
    plan = _load(plan_path)
    if plan.get("plan_type") != "live_shadow_campaign":
        raise ValueError("The supplied plan is not a live-shadow campaign.")
    metrics = snapshot / "data" / "energy_metrics"
    shadow_by_case: dict[str, dict[str, Any]] = {}
    epoch_by_job: dict[str, dict[str, Any]] = {}
    gpu_by_job: dict[str, dict[str, Any]] = {}
    for path in metrics.glob("epoch_summary_job_*.json"):
        summary = _load(path)
        if summary.get("benchmark_run_id") == plan["benchmark_run_id"]:
            epoch_by_job[str(summary["job_id"])] = summary
    for path in metrics.glob("shadow_summary_job_*.json"):
        summary = _load(path)
        if summary.get("benchmark_run_id") == plan["benchmark_run_id"]:
            shadow_by_case[str(summary["benchmark_case_id"])] = summary
    for path in metrics.glob("gpu_summary_job_*.json"):
        if path.name.endswith("_phases.json"):
            continue
        summary = _load(path)
        if summary.get("benchmark_run_id") == plan["benchmark_run_id"]:
            gpu_by_job[str(summary["job_id"])] = summary

    pairs: list[dict[str, Any]] = []
    maximum_regret = float(plan["acceptance"]["maximum_quality_regret"])
    minimum_saving = float(plan["acceptance"]["minimum_energy_saving_fraction"])
    for run in sorted(plan["runs"], key=lambda item: int(item["sequence"])):
        case_id = str(run["benchmark_case_id"])
        shadow = shadow_by_case.get(case_id)
        if shadow is None:
            raise RuntimeError(f"Missing shadow summary for {case_id}.")
        epoch_summary = epoch_by_job.get(str(shadow["job_id"]))
        if epoch_summary is None:
            raise RuntimeError(f"Missing epoch summary for shadow job {shadow['job_id']}.")
        full_quality = _best_quality(epoch_summary)
        full_energy = float(epoch_summary.get("total_gpu_energy_kwh") or 0.0) * 1000.0
        if full_energy <= 0:
            full_energy = float(shadow.get("full100_epoch_training_energy_wh") or 0.0)
        for controller in shadow["controllers"]:
            stop_quality = float(controller.get("best_quality_at_stop") or 0.0)
            training_energy = float(controller.get("training_energy_to_stop_wh") or 0.0)
            overhead_energy = float(controller.get("controller_overhead_energy_wh") or 0.0)
            total_energy = training_energy + overhead_energy
            saving = (full_energy - total_energy) / full_energy if full_energy else 0.0
            regret = full_quality - stop_quality
            energy_met = saving >= minimum_saving
            quality_met = regret <= maximum_regret
            pairs.append(
                {
                    "benchmark_run_id": plan["benchmark_run_id"],
                    "benchmark_version": plan["benchmark_version"],
                    "stage": run["benchmark_stage"],
                    "case_id": case_id,
                    "task_type": run["task_type"],
                    "scenario": run["scenario"],
                    "training_seed": 0,
                    "job_id": str(shadow["job_id"]),
                    "virtual_job_id": _virtual_identity(
                        shadow["job_id"], controller["controller_id"]
                    )[0],
                    "controller_id": controller["controller_id"],
                    "controller_plugin": controller["controller_plugin"],
                    "full100_epochs": int(epoch_summary.get("epochs_completed") or 0),
                    "stop_epoch": int(controller.get("stop_epoch") or 0),
                    "stopped_early": bool(controller.get("stopped_early")),
                    "stop_reason": controller.get("stop_reason"),
                    "controller_failed": bool(controller.get("failed")),
                    "full100_best_quality": full_quality,
                    "best_quality_at_stop": stop_quality,
                    "quality_regret": regret,
                    "quality_regret_pp": regret * 100.0,
                    "full100_epoch_training_energy_wh": full_energy,
                    "training_energy_to_stop_wh": training_energy,
                    "controller_overhead_energy_wh": overhead_energy,
                    "counterfactual_total_energy_wh": total_energy,
                    "energy_saving_fraction": saving,
                    "energy_saving_percent": saving * 100.0,
                    "training_time_to_stop_s": float(controller.get("training_time_to_stop_s") or 0.0),
                    "controller_compute_seconds": float(controller.get("controller_compute_seconds") or 0.0),
                    "controller_process_cpu_seconds": float(
                        controller.get("controller_process_cpu_seconds") or 0.0
                    ),
                    "controller_decisions": int(controller.get("controller_decisions") or 0),
                    "energy_measurement_method": controller.get(
                        "energy_measurement_method", "unavailable"
                    ),
                    "energy_scope": shadow.get("energy_scope"),
                    "lifecycle_energy_complete": bool(shadow.get("lifecycle_energy_complete")),
                    "energy_target_met": energy_met,
                    "quality_target_met": quality_met,
                    "joint_target_met": energy_met and quality_met,
                }
            )
    expected_pairs = len(plan["runs"]) * len(plan["controllers"])
    if len(pairs) != expected_pairs:
        raise RuntimeError(f"Expected {expected_pairs} shadow pairs, found {len(pairs)}.")

    by_controller: dict[str, dict[str, Any]] = {}
    for definition in plan["controllers"]:
        controller_id = definition["id"]
        selected = [pair for pair in pairs if pair["controller_id"] == controller_id]
        by_controller[controller_id] = {
            "cases": len(selected),
            "mean_energy_saving_fraction": statistics.fmean(
                pair["energy_saving_fraction"] for pair in selected
            ),
            "median_energy_saving_fraction": statistics.median(
                pair["energy_saving_fraction"] for pair in selected
            ),
            "mean_quality_regret": statistics.fmean(pair["quality_regret"] for pair in selected),
            "maximum_quality_regret": max(pair["quality_regret"] for pair in selected),
            "energy_success_rate": statistics.fmean(
                int(pair["energy_target_met"]) for pair in selected
            ),
            "quality_success_rate": statistics.fmean(
                int(pair["quality_target_met"]) for pair in selected
            ),
            "joint_success_rate": statistics.fmean(
                int(pair["joint_target_met"]) for pair in selected
            ),
            "total_controller_overhead_energy_wh": sum(
                pair["controller_overhead_energy_wh"] for pair in selected
            ),
            "total_controller_compute_seconds": sum(
                pair["controller_compute_seconds"] for pair in selected
            ),
        }
    represented_jobs: list[dict[str, Any]] = []
    pairs_by_case = {
        case_id: [pair for pair in pairs if pair["case_id"] == case_id]
        for case_id in {pair["case_id"] for pair in pairs}
    }
    for run in sorted(plan["runs"], key=lambda item: int(item["sequence"])):
        case_id = str(run["benchmark_case_id"])
        case_pairs = pairs_by_case[case_id]
        source_job_id = case_pairs[0]["job_id"]
        epoch_summary = epoch_by_job[source_job_id]
        full_quality = _best_quality(epoch_summary)
        full_energy = float(epoch_summary.get("total_gpu_energy_kwh") or 0.0) * 1000.0
        full_duration = float(epoch_summary.get("total_duration_seconds") or 0.0)
        common = {
            "benchmark_run_id": plan["benchmark_run_id"],
            "benchmark_version": plan["benchmark_version"],
            "stage": run["benchmark_stage"],
            "case_id": case_id,
            "task_type": run["task_type"],
            "source_job_id": source_job_id,
        }
        represented_jobs.append(
            {
                **common,
                "variant": "Full100",
                "job_kind": "real",
                "represented_job_id": source_job_id,
                "stop_epoch": int(epoch_summary.get("epochs_completed") or 100),
                "best_quality": float(full_quality or 0.0),
                "training_energy_wh": full_energy,
                "total_energy_wh": full_energy,
                "training_time_seconds": full_duration,
                "controller_overhead_energy_wh": 0.0,
                "controller_compute_seconds": 0.0,
            }
        )
        for pair in case_pairs:
            _, variant = _virtual_identity(source_job_id, pair["controller_id"])
            represented_jobs.append(
                {
                    **common,
                    "variant": variant,
                    "job_kind": "virtual_live_shadow",
                    "represented_job_id": pair["virtual_job_id"],
                    "stop_epoch": pair["stop_epoch"],
                    "best_quality": pair["best_quality_at_stop"],
                    "training_energy_wh": pair["training_energy_to_stop_wh"],
                    "total_energy_wh": pair["counterfactual_total_energy_wh"],
                    "training_time_seconds": pair["training_time_to_stop_s"],
                    "controller_overhead_energy_wh": pair["controller_overhead_energy_wh"],
                    "controller_compute_seconds": pair["controller_compute_seconds"],
                }
            )
    lifecycle_complete = all(
        bool(gpu_by_job.get(row["source_job_id"], {}).get("lifecycle_energy_complete"))
        for row in represented_jobs
        if row["job_kind"] == "real"
    )
    result = {
        "benchmark_run_id": plan["benchmark_run_id"],
        "benchmark_version": plan["benchmark_version"],
        "measurement_protocol": "live-shadow-v1",
        "development_seed_policy": "one Full100 trajectory per dataset, training_seed=0",
        "energy_scope": "epoch_gpu_plus_attributed_controller_cpu",
        "lifecycle_energy_complete": lifecycle_complete,
        "acceptance": plan["acceptance"],
        "training_jobs": len(plan["runs"]),
        "controller_pairs": len(pairs),
        "represented_jobs_total": len(represented_jobs),
        "real_full100_jobs": len(plan["runs"]),
        "virtual_es_jobs": sum(row["variant"] == "Standard ES" for row in represented_jobs),
        "virtual_rapec_g_v4_jobs": sum(row["variant"] == "RAPEC-G v4" for row in represented_jobs),
        "controllers": by_controller,
        "pairs": pairs,
        "represented_jobs": represented_jobs,
    }
    output = snapshot / "controller_benchmark"
    output.mkdir(parents=True, exist_ok=True)
    (output / "live-shadow-report.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    (output / "benchmark-report.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    with (output / "paired-results.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(pairs[0]))
        writer.writeheader()
        writer.writerows(pairs)
    prometheus_path = metrics / "node_exporter" / "controller_benchmark.prom"
    write_prometheus(result, prometheus_path)
    from controller_benchmark.export_thesis_live_shadow_metrics import build as build_thesis_metrics

    with prometheus_path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(build_thesis_metrics(snapshot))
    return result


def write_prometheus(result: dict[str, Any], path: Path) -> None:
    lines = [
        "# HELP live_shadow_controller_stop_epoch First online stop signal in the shared Full100 run.",
        "# TYPE live_shadow_controller_stop_epoch gauge",
    ]
    pair_metrics = {
        "live_shadow_controller_stop_epoch": "stop_epoch",
        "live_shadow_full100_best_quality": "full100_best_quality",
        "live_shadow_best_quality_at_stop": "best_quality_at_stop",
        "live_shadow_quality_regret": "quality_regret",
        "live_shadow_full100_training_energy_wh": "full100_epoch_training_energy_wh",
        "live_shadow_training_energy_to_stop_wh": "training_energy_to_stop_wh",
        "live_shadow_controller_overhead_energy_wh": "controller_overhead_energy_wh",
        "live_shadow_counterfactual_total_energy_wh": "counterfactual_total_energy_wh",
        "live_shadow_energy_saving_fraction": "energy_saving_fraction",
        "live_shadow_controller_compute_seconds": "controller_compute_seconds",
        "live_shadow_training_time_to_stop_seconds": "training_time_to_stop_s",
        "live_shadow_stopped_early": "stopped_early",
        "live_shadow_energy_target_met": "energy_target_met",
        "live_shadow_quality_target_met": "quality_target_met",
        "live_shadow_joint_target_met": "joint_target_met",
    }
    for pair in result["pairs"]:
        labels = _labels(pair)
        for metric, key in pair_metrics.items():
            value = pair[key]
            if isinstance(value, bool):
                value = int(value)
            lines.append(f"{metric}{{{labels}}} {float(value):.12g}")
    represented_metrics = {
        "thesis_live_shadow_variant_stop_epoch": "stop_epoch",
        "thesis_live_shadow_variant_best_quality": "best_quality",
        "thesis_live_shadow_variant_training_energy_wh": "training_energy_wh",
        "thesis_live_shadow_variant_total_energy_wh": "total_energy_wh",
        "thesis_live_shadow_variant_training_time_seconds": "training_time_seconds",
        "thesis_live_shadow_variant_controller_overhead_energy_wh": "controller_overhead_energy_wh",
        "thesis_live_shadow_variant_controller_compute_seconds": "controller_compute_seconds",
    }
    for row in result["represented_jobs"]:
        labels = _variant_labels(row)
        lines.append(f"thesis_live_shadow_variant_info{{{labels}}} 1")
        for metric, key in represented_metrics.items():
            lines.append(f"{metric}{{{labels}}} {float(row[key]):.12g}")
    run_labels = (
        f'benchmark_run_id="{_prom_escape(result["benchmark_run_id"])}",'
        f'benchmark_version="{_prom_escape(result["benchmark_version"])}"'
    )
    for metric, key in {
        "thesis_live_shadow_real_jobs": "real_full100_jobs",
        "thesis_live_shadow_virtual_es_jobs": "virtual_es_jobs",
        "thesis_live_shadow_virtual_rapec_g_v4_jobs": "virtual_rapec_g_v4_jobs",
        "thesis_live_shadow_represented_jobs_total": "represented_jobs_total",
    }.items():
        lines.append(f"{metric}{{{run_labels}}} {float(result[key]):.12g}")
    lines.append(
        f'thesis_live_shadow_lifecycle_complete{{{run_labels}}} '
        f'{1 if result["lifecycle_energy_complete"] else 0}'
    )
    for controller_id, values in result["controllers"].items():
        labels = ",".join(
            [
                f'benchmark_run_id="{_prom_escape(result["benchmark_run_id"])}"',
                f'benchmark_version="{_prom_escape(result["benchmark_version"])}"',
                f'controller_id="{_prom_escape(controller_id)}"',
            ]
        )
        for metric, key in {
            "live_shadow_mean_energy_saving_fraction": "mean_energy_saving_fraction",
            "live_shadow_mean_quality_regret": "mean_quality_regret",
            "live_shadow_energy_success_rate": "energy_success_rate",
            "live_shadow_quality_success_rate": "quality_success_rate",
            "live_shadow_joint_success_rate": "joint_success_rate",
            "live_shadow_total_controller_overhead_energy_wh": "total_controller_overhead_energy_wh",
            "live_shadow_total_controller_compute_seconds": "total_controller_compute_seconds",
        }.items():
            lines.append(f"{metric}{{{labels}}} {float(values[key]):.12g}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--plan", type=Path, required=True)
    args = parser.parse_args()
    result = analyze_live_shadow(args.snapshot.resolve(), args.plan.resolve())
    print(
        json.dumps(
            {
                "benchmark_run_id": result["benchmark_run_id"],
                "training_jobs": result["training_jobs"],
                "controller_pairs": result["controller_pairs"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
