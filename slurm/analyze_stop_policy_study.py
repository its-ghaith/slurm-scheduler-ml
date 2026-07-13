#!/usr/bin/env python3
"""Aggregate CARPK stop-policy runs into reproducible research tables."""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any


TARGETS = (0.50, 0.60, 0.68)
PUE_VALUES = (1.0, 1.2, 1.4)


def as_float(value: Any, default: float | None = None) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    return number if math.isfinite(number) else default


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def energy_to_target(epochs: list[dict[str, Any]], target: float) -> float | None:
    cumulative_wh = 0.0
    for epoch in sorted(epochs, key=lambda item: int(item.get("epoch", 0))):
        cumulative_wh += 1000.0 * (as_float(epoch.get("gpu_energy_kwh"), 0.0) or 0.0)
        if (as_float(epoch.get("map50_95"), -math.inf) or -math.inf) >= target:
            return cumulative_wh
    return None


def load_runs(input_dir: Path) -> list[dict[str, Any]]:
    summaries = sorted(input_dir.rglob("epoch_summary_job_*.json"))
    runs: list[dict[str, Any]] = []
    for epoch_path in summaries:
        epoch = read_json(epoch_path)
        job_id = str(epoch.get("job_id") or epoch_path.stem.rsplit("_", 1)[-1])
        phase_candidates = list(input_dir.rglob(f"gpu_summary_job_{job_id}_phases.json"))
        phase = read_json(phase_candidates[0]) if phase_candidates else {}
        metadata_candidates = list(input_dir.rglob(f"run_metadata_job_{job_id}.json"))
        metadata = read_json(metadata_candidates[0]) if metadata_candidates else {}
        epochs = epoch.get("epochs") or []

        quality = as_float(epoch.get("test_best_map50_95"), as_float(epoch.get("final_best_map50_95")))
        map50 = as_float(epoch.get("test_best_map50"), as_float(epoch.get("final_best_map50")))
        gross_job_kwh = as_float(phase.get("gpu_energy_kwh"), as_float(epoch.get("total_gpu_energy_kwh"), 0.0)) or 0.0
        net_job_kwh = as_float(phase.get("net_gpu_energy_kwh"), as_float(epoch.get("total_net_gpu_energy_kwh"), 0.0)) or 0.0
        duration = as_float(phase.get("duration_seconds"), as_float(epoch.get("total_duration_seconds"), 0.0)) or 0.0

        row: dict[str, Any] = {
            "job_id": job_id,
            "scenario": epoch.get("scenario") or phase.get("scenario") or metadata.get("scenario") or "unspecified",
            "strategy": epoch.get("comparison_strategy") or phase.get("comparison_strategy") or "unspecified",
            "training_seed": int(epoch.get("training_seed", metadata.get("training_seed", 0))),
            "split_seed": int(epoch.get("split_seed", metadata.get("split_seed", 0))),
            "cache_policy": epoch.get("cache_policy") or metadata.get("cache_policy") or "unknown",
            "controller_mode": epoch.get("controller_mode", "none"),
            "epochs_completed": int(epoch.get("epochs_completed", len(epochs))),
            "stop_epoch": epoch.get("stop_epoch"),
            "stop_reason": epoch.get("stop_reason"),
            "test_best_map50": map50,
            "test_best_map50_95": quality,
            "test_best_precision": as_float(epoch.get("test_best_precision")),
            "test_best_recall": as_float(epoch.get("test_best_recall")),
            "final_map50_95": as_float(epoch.get("final_map50_95")),
            "job_gpu_energy_wh": gross_job_kwh * 1000.0,
            "job_net_gpu_energy_wh": net_job_kwh * 1000.0,
            "epoch_gpu_energy_wh": (as_float(epoch.get("total_gpu_energy_kwh"), 0.0) or 0.0) * 1000.0,
            "duration_seconds": duration,
            "idle_power_w": as_float(phase.get("idle_power_w"), as_float(metadata.get("idle_power_w"))),
            "runtime_image_id": metadata.get("runtime_image_id", phase.get("runtime_image_id", "unknown")),
            "gpu_name": metadata.get("gpu", {}).get("name", phase.get("gpu_name", "unknown")),
        }
        for target in TARGETS:
            row[f"energy_to_map_{target:.2f}_wh"] = energy_to_target(epochs, target)
        runs.append(row)
    return runs


def enrich_regret_and_pareto(runs: list[dict[str, Any]]) -> None:
    references: dict[tuple[str, int], float] = {}
    for row in runs:
        if row["strategy"] == "full100" and row["test_best_map50_95"] is not None:
            references[(row["scenario"], row["training_seed"])] = row["test_best_map50_95"]

    for row in runs:
        reference = references.get((row["scenario"], row["training_seed"]))
        quality = row["test_best_map50_95"]
        row["accuracy_regret_map50_95_pp"] = (
            (reference - quality) * 100.0 if reference is not None and quality is not None else None
        )
        row["accuracy_per_wh"] = quality / row["job_gpu_energy_wh"] if quality is not None and row["job_gpu_energy_wh"] > 0 else None

    for row in runs:
        quality = row["test_best_map50_95"]
        energy = row["job_gpu_energy_wh"]
        row["pareto_dominated"] = False
        if quality is None:
            row["pareto_dominated"] = True
            continue
        for other in runs:
            if other is row or other["scenario"] != row["scenario"]:
                continue
            other_quality = other["test_best_map50_95"]
            other_energy = other["job_gpu_energy_wh"]
            if other_quality is None:
                continue
            if other_energy <= energy and other_quality >= quality and (other_energy < energy or other_quality > quality):
                row["pareto_dominated"] = True
                break


def confidence_interval(values: list[float]) -> tuple[float, float, float, float, int]:
    count = len(values)
    mean = statistics.fmean(values)
    std = statistics.stdev(values) if count > 1 else 0.0
    # Two-sided 95% Student-t critical values; n=5 uses 2.776 instead of 1.96.
    t_critical = {
        2: 12.706,
        3: 4.303,
        4: 3.182,
        5: 2.776,
        6: 2.571,
        7: 2.447,
        8: 2.365,
        9: 2.306,
        10: 2.262,
    }.get(count, 1.96)
    half_width = t_critical * std / math.sqrt(count) if count > 1 else 0.0
    return mean, std, mean - half_width, mean + half_width, count


def aggregate(runs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    metrics = (
        "test_best_map50_95",
        "test_best_map50",
        "job_gpu_energy_wh",
        "job_net_gpu_energy_wh",
        "duration_seconds",
        "epochs_completed",
        "accuracy_regret_map50_95_pp",
        "accuracy_per_wh",
        *(f"energy_to_map_{target:.2f}_wh" for target in TARGETS),
    )
    groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in runs:
        groups[(row["scenario"], row["strategy"])].append(row)

    output: list[dict[str, Any]] = []
    for (scenario, strategy), rows in sorted(groups.items()):
        result: dict[str, Any] = {"scenario": scenario, "strategy": strategy, "runs": len(rows)}
        for metric in metrics:
            values = [as_float(row.get(metric)) for row in rows]
            valid = [value for value in values if value is not None]
            if not valid:
                continue
            mean, std, ci_low, ci_high, count = confidence_interval(valid)
            result.update(
                {
                    f"{metric}_mean": mean,
                    f"{metric}_std": std,
                    f"{metric}_ci95_low": ci_low,
                    f"{metric}_ci95_high": ci_high,
                    f"{metric}_n": count,
                }
            )
        output.append(result)

    for row in output:
        row["aggregate_pareto_dominated"] = False
        quality = row.get("test_best_map50_95_mean")
        energy = row.get("job_gpu_energy_wh_mean")
        if quality is None or energy is None:
            row["aggregate_pareto_dominated"] = True
            continue
        for other in output:
            if other is row or other["scenario"] != row["scenario"]:
                continue
            other_quality = other.get("test_best_map50_95_mean")
            other_energy = other.get("job_gpu_energy_wh_mean")
            if other_quality is None or other_energy is None:
                continue
            if other_energy <= energy and other_quality >= quality and (other_energy < energy or other_quality > quality):
                row["aggregate_pareto_dominated"] = True
                break
    return output


def pue_sensitivity(runs: list[dict[str, Any]], price: float, co2: float) -> list[dict[str, Any]]:
    output = []
    for row in runs:
        gpu_kwh = row["job_gpu_energy_wh"] / 1000.0
        for pue in PUE_VALUES:
            total_kwh = gpu_kwh * pue
            output.append(
                {
                    "job_id": row["job_id"],
                    "scenario": row["scenario"],
                    "strategy": row["strategy"],
                    "training_seed": row["training_seed"],
                    "pue": pue,
                    "estimated_total_energy_kwh": total_kwh,
                    "estimated_cost_eur": total_kwh * price,
                    "estimated_co2_kg": total_kwh * co2,
                }
            )
    return output


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def write_pareto_plot(path: Path, runs: list[dict[str, Any]]) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    figure, axes = plt.subplots(1, max(1, len({row["scenario"] for row in runs})), figsize=(12, 5), squeeze=False)
    for axis, scenario in zip(axes[0], sorted({row["scenario"] for row in runs})):
        for row in (item for item in runs if item["scenario"] == scenario and item["test_best_map50_95"] is not None):
            axis.scatter(row["job_gpu_energy_wh"], row["test_best_map50_95"] * 100.0, s=35)
            axis.annotate(f"{row['strategy']} s{row['training_seed']}", (row["job_gpu_energy_wh"], row["test_best_map50_95"] * 100.0), fontsize=6)
        axis.set(title=scenario, xlabel="GPU energy (Wh)", ylabel="best checkpoint test mAP50-95 (%)")
        axis.grid(alpha=0.25)
    figure.tight_layout()
    figure.savefig(path, dpi=180)
    plt.close(figure)


def validate_provenance(runs: list[dict[str, Any]]) -> dict[str, Any]:
    split_seeds = sorted({row["split_seed"] for row in runs})
    image_ids = sorted({row["runtime_image_id"] for row in runs if row["runtime_image_id"] != "unknown"})
    gpu_names = sorted({row["gpu_name"] for row in runs if row["gpu_name"] != "unknown"})
    cache_policies = sorted({row["cache_policy"] for row in runs})
    warnings = []
    if len(split_seeds) != 1:
        warnings.append("Runs use different split seeds.")
    if len(image_ids) > 1:
        warnings.append("Runs use different runtime image digests.")
    if len(gpu_names) > 1:
        warnings.append("Runs use different GPU models.")
    if len(cache_policies) != 1:
        warnings.append("Runs use different cache policies.")
    return {
        "split_seeds": split_seeds,
        "runtime_image_ids": image_ids,
        "gpu_names": gpu_names,
        "cache_policies": cache_policies,
        "warnings": warnings,
        "valid": not warnings,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--price-eur-kwh", type=float, default=0.30)
    parser.add_argument("--co2-kg-kwh", type=float, default=0.40)
    args = parser.parse_args()

    runs = load_runs(args.input_dir)
    if not runs:
        raise SystemExit(f"No epoch_summary_job_*.json files found below {args.input_dir}")
    enrich_regret_and_pareto(runs)
    aggregates = aggregate(runs)
    sensitivity = pue_sensitivity(runs, args.price_eur_kwh, args.co2_kg_kwh)
    provenance = validate_provenance(runs)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "runs.csv", runs)
    write_csv(args.output_dir / "aggregates.csv", aggregates)
    write_csv(args.output_dir / "pue_sensitivity.csv", sensitivity)
    (args.output_dir / "study_summary.json").write_text(
        json.dumps({"provenance_validation": provenance, "runs": runs, "aggregates": aggregates}, indent=2),
        encoding="utf-8",
    )
    write_pareto_plot(args.output_dir / "pareto_front.png", runs)
    print(json.dumps({"runs": len(runs), "groups": len(aggregates), "provenance": provenance, "output_dir": str(args.output_dir)}))


if __name__ == "__main__":
    main()
