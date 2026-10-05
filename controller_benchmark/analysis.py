from __future__ import annotations

import csv
import hashlib
import json
import math
import statistics
from pathlib import Path
from typing import Any


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load_cached_baselines(plan: dict[str, Any]) -> dict[tuple[str, str], dict[str, Any]]:
    cache = plan.get("baseline_cache", {})
    catalog_path = cache.get("catalog_path")
    if not catalog_path:
        return {}
    root = Path(catalog_path).resolve().parent
    baseline_id = plan["baseline"]["id"]
    result = {}
    for case_id, entry in cache.get("entries", {}).items():
        summary_path = root / entry["epoch_summary"]
        gpu_path = root / entry["gpu_summary"]
        if _file_hash(summary_path) != entry["epoch_summary_sha256"] or _file_hash(gpu_path) != entry["gpu_summary_sha256"]:
            raise RuntimeError(f"Cached baseline integrity check failed for {case_id}.")
        result[(case_id, baseline_id)] = {
            "summary": _load_json(summary_path),
            "gpu": _load_json(gpu_path),
            "job_id": str(entry["source_job_id"]),
            "source": "cache",
        }
    return result


def _validate_runtime(gpu: dict[str, Any], execution: dict[str, Any], label: str) -> None:
    metadata = gpu.get("run_metadata", {})
    actual_gpu = metadata.get("gpu", {})
    checks = {
        "runtime_image_id": (metadata.get("runtime_image_id"), execution.get("runtime_image_id")),
        "gpu_name": (actual_gpu.get("name"), execution.get("gpu_name")),
        "gpu_power_limit_w": (str(actual_gpu.get("power_limit_w")), str(execution.get("gpu_power_limit_w"))),
    }
    mismatches = [f"{key}: {actual!r} != {expected!r}" for key, (actual, expected) in checks.items() if actual != expected]
    if mismatches:
        raise RuntimeError(f"Runtime mismatch for {label}: " + "; ".join(mismatches))


def _best_quality(summary: dict[str, Any]) -> float:
    metric = str(summary.get("quality_metric", "map50_95"))
    values = [row.get(metric) for row in summary.get("epochs", [])]
    if metric == "quality_score" and not any(value is not None for value in values):
        # Cached YOLO baselines created before the generic quality contract.
        values = [row.get("map50_95") for row in summary.get("epochs", [])]
    values = [float(value) for value in values if value is not None]
    return max(values, default=0.0)


def _quality_values(summary: dict[str, Any]) -> list[float]:
    metric = str(summary.get("quality_metric", "map50_95"))
    values = [row.get(metric) for row in summary.get("epochs", [])]
    if metric == "quality_score" and not any(value is not None for value in values):
        values = [row.get("map50_95") for row in summary.get("epochs", [])]
    return [float(value) for value in values if value is not None]


def _mad(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    center = statistics.median(values)
    return 1.4826 * statistics.median(abs(value - center) for value in values)


def _dynamic_quality_tolerance(summary: dict[str, Any]) -> float:
    values = _quality_values(summary)
    if len(values) < 3:
        return 0.0
    deltas = [right - left for left, right in zip(values, values[1:])]
    innovation_noise = _mad(deltas) / math.sqrt(2.0)
    late_count = max(5, int(math.ceil(math.sqrt(len(values)))))
    late_uncertainty = _mad(values[-late_count:])
    return 1.96 * math.sqrt(innovation_noise**2 + late_uncertainty**2)


def _comparable_training_gpu_energy_wh(gpu: dict[str, Any]) -> tuple[float, float]:
    training = gpu.get("phase_metrics", {}).get("training", {})
    slurm_kwh = training.get("gpu_energy_kwh", gpu.get("gpu_energy_kwh", 0.0))
    codecarbon_kwh = training.get(
        "codecarbon_gpu_energy_kwh",
        gpu.get("codecarbon_job_total_gpu_energy_kwh", 0.0),
    )
    return float(slurm_kwh or 0.0) * 1000.0, float(codecarbon_kwh or 0.0) * 1000.0


def _phase_values(gpu: dict[str, Any]) -> dict[str, dict[str, float]]:
    price = float(gpu.get("price_eur_kwh", 0.30) or 0.30)
    co2 = float(gpu.get("co2_kg_kwh", 0.40) or 0.40)
    pue = float(gpu.get("pue_factor", 1.0) or 1.0)
    phases: dict[str, dict[str, float]] = {}
    for phase, values in gpu.get("phase_metrics", {}).items():
        if not isinstance(values, dict):
            continue
        gpu_energy_kwh = float(values.get("gpu_energy_kwh", values.get("total_energy_kwh", 0.0)) or 0.0)
        total_energy_kwh = float(values.get("total_energy_kwh", gpu_energy_kwh) or gpu_energy_kwh)
        phases[str(phase)] = {
            "gpu_energy_wh": gpu_energy_kwh * 1000.0,
            "total_energy_wh": total_energy_kwh * 1000.0,
            "duration_seconds": float(values.get("duration_seconds", 0.0) or 0.0),
            "cost_eur": float(values.get("estimated_electricity_cost_eur", total_energy_kwh * pue * price) or 0.0),
            "co2_kg": float(values.get("estimated_co2_kg", total_energy_kwh * pue * co2) or 0.0),
            "codecarbon_gpu_energy_wh": float(values.get("codecarbon_gpu_energy_kwh", 0.0) or 0.0) * 1000.0,
            "codecarbon_total_energy_wh": float(values.get("codecarbon_energy_kwh", 0.0) or 0.0) * 1000.0,
            "codecarbon_cost_eur": float(values.get("codecarbon_estimated_electricity_cost_eur", 0.0) or 0.0),
            "codecarbon_co2_kg": float(values.get("codecarbon_estimated_co2_kg", 0.0) or 0.0),
        }
    if not phases:
        gpu_energy_kwh = float(gpu.get("training_energy_kwh", gpu.get("gpu_energy_kwh", 0.0)) or 0.0)
        phases["training"] = {
            "gpu_energy_wh": gpu_energy_kwh * 1000.0,
            "total_energy_wh": gpu_energy_kwh * 1000.0,
            "duration_seconds": float(gpu.get("duration_seconds", 0.0) or 0.0),
            "cost_eur": gpu_energy_kwh * pue * price,
            "co2_kg": gpu_energy_kwh * pue * co2,
            "codecarbon_gpu_energy_wh": float(gpu.get("codecarbon_job_total_gpu_energy_kwh", 0.0) or 0.0) * 1000.0,
            "codecarbon_total_energy_wh": float(gpu.get("codecarbon_job_total_energy_kwh", 0.0) or 0.0) * 1000.0,
            "codecarbon_cost_eur": float(gpu.get("codecarbon_estimated_electricity_cost_eur", 0.0) or 0.0),
            "codecarbon_co2_kg": float(gpu.get("codecarbon_estimated_co2_kg", 0.0) or 0.0),
        }
    return phases


def analyze_snapshot(snapshot: str | Path, plan_path: str | Path) -> dict[str, Any]:
    snapshot = Path(snapshot)
    plan = _load_json(Path(plan_path))
    metrics = snapshot / "data" / "energy_metrics"
    runs: dict[tuple[str, str], dict[str, Any]] = _load_cached_baselines(plan)
    for summary_path in metrics.glob("epoch_summary_job_*.json"):
        summary = _load_json(summary_path)
        if summary.get("benchmark_run_id") != plan["benchmark_run_id"]:
            continue
        job_id = str(summary["job_id"])
        gpu = _load_json(metrics / f"gpu_summary_job_{job_id}.json")
        runs[(summary["benchmark_case_id"], summary["controller_id"])] = {
            "summary": summary,
            "gpu": gpu,
            "job_id": job_id,
            "source": "fresh",
        }
    candidate_id = plan["controller"]["id"]
    baseline_id = plan.get("baseline", {"id": "full100"})["id"]
    pairs = []
    for case_id in sorted({row["benchmark_case_id"] for row in plan["runs"]}):
        baseline_case_id = plan.get("baseline_references", {}).get(case_id, case_id)
        baseline = runs.get((baseline_case_id, baseline_id))
        candidate = runs.get((case_id, candidate_id))
        if baseline is None or candidate is None:
            raise RuntimeError(f"Incomplete benchmark pair for {case_id}")
        b_summary, c_summary = baseline["summary"], candidate["summary"]
        _validate_runtime(baseline["gpu"], plan["execution"], f"{case_id}/baseline")
        _validate_runtime(candidate["gpu"], plan["execution"], f"{case_id}/candidate")
        b_job_energy = float(baseline["gpu"].get("gpu_energy_kwh", 0.0)) * 1000.0
        c_job_energy = float(candidate["gpu"].get("gpu_energy_kwh", 0.0)) * 1000.0
        energy_scope = str(
            plan.get("execution", {}).get("energy_comparison_scope", "job")
        ).lower()
        if energy_scope == "epoch":
            b_energy = float(b_summary.get("total_gpu_energy_kwh", 0.0)) * 1000.0
            c_energy = float(c_summary.get("total_gpu_energy_kwh", 0.0)) * 1000.0
        elif energy_scope == "job":
            b_energy = b_job_energy
            c_energy = c_job_energy
        else:
            raise ValueError(f"Unsupported energy_comparison_scope: {energy_scope!r}")
        b_codecarbon_gpu = float(baseline["gpu"].get("codecarbon_job_total_gpu_energy_kwh", 0.0)) * 1000.0
        c_codecarbon_gpu = float(candidate["gpu"].get("codecarbon_job_total_gpu_energy_kwh", 0.0)) * 1000.0
        b_codecarbon_total = float(baseline["gpu"].get("codecarbon_job_total_energy_kwh", 0.0)) * 1000.0
        c_codecarbon_total = float(candidate["gpu"].get("codecarbon_job_total_energy_kwh", 0.0)) * 1000.0
        b_slurm_comparable, b_codecarbon_comparable = _comparable_training_gpu_energy_wh(baseline["gpu"])
        c_slurm_comparable, c_codecarbon_comparable = _comparable_training_gpu_energy_wh(candidate["gpu"])
        b_quality, c_quality = _best_quality(b_summary), _best_quality(c_summary)
        saving = (b_energy - c_energy) / b_energy if b_energy else 0.0
        regret = b_quality - c_quality
        acceptance = plan["acceptance"]
        tolerance_mode = str(
            acceptance.get("quality_tolerance_mode", "fixed")
        ).lower()
        if tolerance_mode == "full100_dynamic":
            quality_tolerance = _dynamic_quality_tolerance(b_summary)
        elif tolerance_mode == "fixed":
            quality_tolerance = float(acceptance["maximum_quality_regret"])
        else:
            raise ValueError(f"Unsupported quality_tolerance_mode: {tolerance_mode!r}")
        pairs.append({
            "benchmark_run_id": plan["benchmark_run_id"],
            "benchmark_version": plan["benchmark_version"],
            "stage": c_summary["benchmark_stage"],
            "case_id": case_id,
            "scenario": c_summary.get("scenario", "unspecified"),
            "task_type": c_summary.get("task_type", "unknown"),
            "quality_metric": c_summary.get("quality_metric", "quality"),
            "training_seed": c_summary.get("training_seed", 0),
            "baseline_job_id": baseline["job_id"],
            "baseline_source": baseline["source"],
            "baseline_case_id": baseline_case_id,
            "candidate_job_id": candidate["job_id"],
            "energy_comparison_scope": energy_scope,
            "baseline_energy_wh": b_energy,
            "candidate_energy_wh": c_energy,
            "baseline_job_energy_wh": b_job_energy,
            "candidate_job_energy_wh": c_job_energy,
            "baseline_codecarbon_gpu_energy_wh": b_codecarbon_gpu,
            "candidate_codecarbon_gpu_energy_wh": c_codecarbon_gpu,
            "baseline_codecarbon_total_energy_wh": b_codecarbon_total,
            "candidate_codecarbon_total_energy_wh": c_codecarbon_total,
            "baseline_cost_eur": float(baseline["gpu"].get("estimated_electricity_cost_eur", 0.0) or 0.0),
            "candidate_cost_eur": float(candidate["gpu"].get("estimated_electricity_cost_eur", 0.0) or 0.0),
            "baseline_codecarbon_cost_eur": float(
                baseline["gpu"].get("codecarbon_estimated_electricity_cost_eur", 0.0) or 0.0
            ),
            "candidate_codecarbon_cost_eur": float(
                candidate["gpu"].get("codecarbon_estimated_electricity_cost_eur", 0.0) or 0.0
            ),
            "baseline_co2_kg": float(baseline["gpu"].get("estimated_co2_kg", 0.0) or 0.0),
            "candidate_co2_kg": float(candidate["gpu"].get("estimated_co2_kg", 0.0) or 0.0),
            "baseline_codecarbon_co2_kg": float(baseline["gpu"].get("codecarbon_estimated_co2_kg", 0.0) or 0.0),
            "candidate_codecarbon_co2_kg": float(candidate["gpu"].get("codecarbon_estimated_co2_kg", 0.0) or 0.0),
            "baseline_comparable_slurm_gpu_energy_wh": b_slurm_comparable,
            "candidate_comparable_slurm_gpu_energy_wh": c_slurm_comparable,
            "baseline_comparable_codecarbon_gpu_energy_wh": b_codecarbon_comparable,
            "candidate_comparable_codecarbon_gpu_energy_wh": c_codecarbon_comparable,
            "baseline_codecarbon_available": bool(
                baseline["gpu"].get("codecarbon_measurement_available", b_codecarbon_gpu > 0.0)
            ),
            "candidate_codecarbon_available": bool(
                candidate["gpu"].get("codecarbon_measurement_available", c_codecarbon_gpu > 0.0)
            ),
            "energy_saving_fraction": saving,
            "baseline_best_quality": b_quality,
            "candidate_best_quality": c_quality,
            "quality_regret": regret,
            "quality_tolerance_mode": tolerance_mode,
            "quality_regret_tolerance": quality_tolerance,
            "baseline_epochs": int(b_summary.get("epochs_completed", 0)),
            "candidate_epochs": int(c_summary.get("epochs_completed", 0)),
            "baseline_duration_seconds": float(baseline["gpu"].get("duration_seconds", 0.0)),
            "candidate_duration_seconds": float(candidate["gpu"].get("duration_seconds", 0.0)),
            "baseline_phases": _phase_values(baseline["gpu"]),
            "candidate_phases": _phase_values(candidate["gpu"]),
            "energy_target_met": saving >= float(acceptance["minimum_energy_saving_fraction"]),
            "quality_target_met": regret <= quality_tolerance,
        })
    stage_results = {}
    for stage in sorted({pair["stage"] for pair in pairs}):
        values = [pair for pair in pairs if pair["stage"] == stage]
        stage_results[stage] = {
            "cases": len(values),
            "energy_success_rate": sum(pair["energy_target_met"] for pair in values) / len(values),
            "quality_success_rate": sum(pair["quality_target_met"] for pair in values) / len(values),
            "joint_success_rate": sum(pair["energy_target_met"] and pair["quality_target_met"] for pair in values) / len(values),
            "mean_energy_saving_fraction": statistics.fmean(pair["energy_saving_fraction"] for pair in values),
            "minimum_energy_saving_fraction": min(pair["energy_saving_fraction"] for pair in values),
            "mean_quality_regret": statistics.fmean(pair["quality_regret"] for pair in values),
            "maximum_quality_regret": max(pair["quality_regret"] for pair in values),
        }
    required_rate = float(plan["acceptance"]["minimum_success_rate"])
    return {
        "benchmark_run_id": plan["benchmark_run_id"],
        "benchmark_version": plan["benchmark_version"],
        "manifest_sha256": plan["manifest_sha256"],
        "controller": plan["controller"],
        "acceptance": plan["acceptance"],
        "blocked_stages": plan["blocked_stages"],
        "pairs": pairs,
        "stages": stage_results,
        "overall": {
            "cases": len(pairs),
            "energy_success_rate": sum(pair["energy_target_met"] for pair in pairs) / len(pairs),
            "quality_success_rate": sum(pair["quality_target_met"] for pair in pairs) / len(pairs),
            "joint_success_rate": sum(pair["energy_target_met"] and pair["quality_target_met"] for pair in pairs) / len(pairs),
            "mean_energy_saving_fraction": statistics.fmean(pair["energy_saving_fraction"] for pair in pairs),
            "mean_quality_regret": statistics.fmean(pair["quality_regret"] for pair in pairs),
            "benchmark_passed": all(result["joint_success_rate"] >= required_rate for result in stage_results.values()),
            "full_generalisation_claim_available": not plan["blocked_stages"],
        },
    }


def write_analysis(result: dict[str, Any], output_dir: str | Path) -> None:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    (output / "benchmark-report.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    csv_pairs = [
        {key: value for key, value in pair.items() if key not in {"baseline_phases", "candidate_phases"}}
        for pair in result["pairs"]
    ]
    with (output / "paired-results.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(csv_pairs[0]))
        writer.writeheader()
        writer.writerows(csv_pairs)


def prom_escape(value: Any) -> str:
    return str(value).replace("\\", "\\\\").replace('"', '\\"')


def write_prometheus(result: dict[str, Any], path: str | Path) -> None:
    lines = []
    for pair in result["pairs"]:
        labels = ",".join(
            f'{key}="{prom_escape(pair[key])}"'
            for key in (
                "benchmark_run_id", "benchmark_version", "stage", "case_id", "scenario",
                "task_type", "quality_metric", "baseline_job_id", "candidate_job_id",
                "energy_comparison_scope",
            )
        )
        values = {
            "controller_benchmark_energy_saving_fraction": pair["energy_saving_fraction"],
            "controller_benchmark_duration_saving_fraction": (
                (pair["baseline_duration_seconds"] - pair["candidate_duration_seconds"]) / pair["baseline_duration_seconds"]
                if pair["baseline_duration_seconds"] else 0.0
            ),
            "controller_benchmark_epochs_saved": pair["baseline_epochs"] - pair["candidate_epochs"],
            "controller_benchmark_quality_regret": pair["quality_regret"],
            "controller_benchmark_quality_regret_tolerance": pair["quality_regret_tolerance"],
            "controller_benchmark_baseline_energy_wh": pair["baseline_energy_wh"],
            "controller_benchmark_candidate_energy_wh": pair["candidate_energy_wh"],
            "controller_benchmark_baseline_best_quality": pair["baseline_best_quality"],
            "controller_benchmark_candidate_best_quality": pair["candidate_best_quality"],
            "controller_benchmark_baseline_epochs": pair["baseline_epochs"],
            "controller_benchmark_candidate_epochs": pair["candidate_epochs"],
            "controller_benchmark_energy_target_met": int(pair["energy_target_met"]),
            "controller_benchmark_quality_target_met": int(pair["quality_target_met"]),
            "controller_benchmark_codecarbon_coverage": int(
                pair["baseline_codecarbon_available"] and pair["candidate_codecarbon_available"]
            ),
        }
        lines.append(f"controller_benchmark_case_info{{{labels}}} 1")
        lines.extend(f"{metric}{{{labels}}} {value:.12g}" for metric, value in values.items())
        for (
            variant, job_key, energy_key, codecarbon_key, codecarbon_total_key, cost_key,
            codecarbon_cost_key, co2_key, codecarbon_co2_key, comparable_slurm_key,
            comparable_codecarbon_key, quality_key, epoch_key,
        ) in (
            (
                "Full100", "baseline_job_id", "baseline_energy_wh", "baseline_codecarbon_gpu_energy_wh",
                "baseline_codecarbon_total_energy_wh", "baseline_cost_eur", "baseline_codecarbon_cost_eur",
                "baseline_co2_kg", "baseline_codecarbon_co2_kg", "baseline_comparable_slurm_gpu_energy_wh",
                "baseline_comparable_codecarbon_gpu_energy_wh", "baseline_best_quality", "baseline_epochs",
            ),
            (
                "Candidate", "candidate_job_id", "candidate_energy_wh", "candidate_codecarbon_gpu_energy_wh",
                "candidate_codecarbon_total_energy_wh", "candidate_cost_eur", "candidate_codecarbon_cost_eur",
                "candidate_co2_kg", "candidate_codecarbon_co2_kg", "candidate_comparable_slurm_gpu_energy_wh",
                "candidate_comparable_codecarbon_gpu_energy_wh", "candidate_best_quality", "candidate_epochs",
            ),
        ):
            variant_labels = f'{labels},variant="{variant}",job_id="{prom_escape(pair[job_key])}"'
            lines.append(f"controller_benchmark_energy_wh{{{variant_labels}}} {pair[energy_key]:.12g}")
            job_energy_key = (
                "baseline_job_energy_wh"
                if variant == "Full100"
                else "candidate_job_energy_wh"
            )
            lines.append(
                f"controller_benchmark_job_energy_wh{{{variant_labels}}} "
                f"{pair[job_energy_key]:.12g}"
            )
            lines.append(
                f"controller_benchmark_codecarbon_gpu_energy_wh{{{variant_labels}}} "
                f"{pair[codecarbon_key]:.12g}"
            )
            lines.append(
                f"controller_benchmark_codecarbon_total_energy_wh{{{variant_labels}}} "
                f"{pair[codecarbon_total_key]:.12g}"
            )
            lines.append(f"controller_benchmark_cost_eur{{{variant_labels}}} {pair[cost_key]:.12g}")
            lines.append(
                f"controller_benchmark_codecarbon_cost_eur{{{variant_labels}}} "
                f"{pair[codecarbon_cost_key]:.12g}"
            )
            lines.append(f"controller_benchmark_co2_kg{{{variant_labels}}} {pair[co2_key]:.12g}")
            lines.append(
                f"controller_benchmark_codecarbon_co2_kg{{{variant_labels}}} "
                f"{pair[codecarbon_co2_key]:.12g}"
            )
            lines.append(
                f"controller_benchmark_comparable_slurm_gpu_energy_wh{{{variant_labels}}} "
                f"{pair[comparable_slurm_key]:.12g}"
            )
            lines.append(
                f"controller_benchmark_comparable_codecarbon_gpu_energy_wh{{{variant_labels}}} "
                f"{pair[comparable_codecarbon_key]:.12g}"
            )
            lines.append(f"controller_benchmark_best_quality{{{variant_labels}}} {pair[quality_key]:.12g}")
            lines.append(f"controller_benchmark_epochs{{{variant_labels}}} {pair[epoch_key]:.12g}")
            duration_key = "baseline_duration_seconds" if variant == "Full100" else "candidate_duration_seconds"
            lines.append(f"controller_benchmark_duration_seconds{{{variant_labels}}} {pair[duration_key]:.12g}")
        for variant, job_key, phase_key, quality_key, epoch_key in (
            ("Full100", "baseline_job_id", "baseline_phases", "baseline_best_quality", "baseline_epochs"),
            ("Candidate", "candidate_job_id", "candidate_phases", "candidate_best_quality", "candidate_epochs"),
        ):
            for phase, phase_values in pair.get(phase_key, {}).items():
                phase_labels = (
                    f'{labels},variant="{variant}",job_id="{prom_escape(pair[job_key])}",'
                    f'phase="{prom_escape(phase)}"'
                )
                phase_metrics = {
                    "controller_benchmark_phase_gpu_energy_wh": phase_values.get("gpu_energy_wh", 0.0),
                    "controller_benchmark_phase_total_energy_wh": phase_values.get("total_energy_wh", 0.0),
                    "controller_benchmark_phase_duration_seconds": phase_values.get("duration_seconds", 0.0),
                    "controller_benchmark_phase_cost_eur": phase_values.get("cost_eur", 0.0),
                    "controller_benchmark_phase_co2_kg": phase_values.get("co2_kg", 0.0),
                    "controller_benchmark_phase_codecarbon_gpu_energy_wh": phase_values.get("codecarbon_gpu_energy_wh", 0.0),
                    "controller_benchmark_phase_codecarbon_total_energy_wh": phase_values.get("codecarbon_total_energy_wh", 0.0),
                    "controller_benchmark_phase_codecarbon_cost_eur": phase_values.get("codecarbon_cost_eur", 0.0),
                    "controller_benchmark_phase_codecarbon_co2_kg": phase_values.get("codecarbon_co2_kg", 0.0),
                    "controller_benchmark_phase_best_quality": pair[quality_key],
                    "controller_benchmark_phase_epochs": pair[epoch_key],
                }
                lines.extend(
                    f"{metric}{{{phase_labels}}} {float(value or 0.0):.12g}"
                    for metric, value in phase_metrics.items()
                )
    overall_labels = (
        f'benchmark_run_id="{prom_escape(result["benchmark_run_id"])}",'
        f'benchmark_version="{prom_escape(result["benchmark_version"])}",'
        f'controller_id="{prom_escape(result["controller"]["id"])}"'
    )
    overall_values = {
        "controller_benchmark_pairs_total": result["overall"]["cases"],
        "controller_benchmark_energy_success_rate": result["overall"]["energy_success_rate"],
        "controller_benchmark_quality_success_rate": result["overall"]["quality_success_rate"],
        "controller_benchmark_joint_success_rate": result["overall"]["joint_success_rate"],
        "controller_benchmark_mean_energy_saving_fraction": result["overall"]["mean_energy_saving_fraction"],
        "controller_benchmark_mean_quality_regret": result["overall"]["mean_quality_regret"],
        "controller_benchmark_passed": int(result["overall"]["benchmark_passed"]),
        "controller_benchmark_full_generalisation_claim_available": int(
            result["overall"]["full_generalisation_claim_available"]
        ),
    }
    lines.extend(f"{metric}{{{overall_labels}}} {value:.12g}" for metric, value in overall_values.items())
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
