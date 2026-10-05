from __future__ import annotations

import csv
import json
import math
import statistics
from pathlib import Path
from typing import Any, Mapping, Sequence

from controller_benchmark.optimization.replay import BaselineTrace, ideal_stop_epoch

from .metrics import calibrated_tolerances, trace_group


def _energy_wh(row: Mapping[str, Any]) -> float:
    if row.get("epoch_energy_wh") is not None:
        return max(0.0, float(row["epoch_energy_wh"]))
    value = row.get("total_energy_kwh", row.get("gpu_energy_kwh", 0.0))
    return max(0.0, float(value or 0.0) * 1000.0)


def _quantile(values: Sequence[float], probability: float) -> float:
    ordered = sorted(float(value) for value in values if math.isfinite(float(value)))
    if not ordered:
        return 0.0
    if len(ordered) == 1:
        return ordered[0]
    position = min(1.0, max(0.0, probability)) * (len(ordered) - 1)
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def build_oracle_report(traces: Sequence[BaselineTrace]) -> dict[str, Any]:
    """Compute the non-causal upper bound allowed by the quality tolerance."""
    tolerances = calibrated_tolerances(traces)
    rows: list[dict[str, Any]] = []
    grouped: dict[str, list[dict[str, Any]]] = {}
    for trace in traces:
        group = trace_group(trace)
        tolerance = tolerances[group]
        stop_epoch = ideal_stop_epoch(trace, tolerance)
        prefix = [
            row
            for row in trace.epochs
            if int(row.get("epoch_index", row["epoch"])) <= stop_epoch
        ]
        baseline_quality = max(float(row[trace.quality_metric]) for row in trace.epochs)
        stopped_quality = max(float(row[trace.quality_metric]) for row in prefix)
        baseline_energy = sum(_energy_wh(row) for row in trace.epochs)
        stopped_energy = sum(_energy_wh(row) for row in prefix)
        saving = (
            (baseline_energy - stopped_energy) / baseline_energy
            if baseline_energy > 0.0
            else 0.0
        )
        regret = max(0.0, baseline_quality - stopped_quality)
        result = {
            "case_id": trace.case_id,
            "dataset_group": group,
            "task_type": trace.task_type,
            "training_seed": int(trace.metadata.get("training_seed", 0)),
            "source_job_id": trace.source_job_id,
            "full_epochs": trace.max_epochs,
            "oracle_stop_epoch": stop_epoch,
            "baseline_best_quality": baseline_quality,
            "oracle_stopped_best_quality": stopped_quality,
            "quality_regret": regret,
            "calibrated_quality_tolerance": tolerance,
            "quality_feasible": regret <= tolerance,
            "baseline_training_energy_wh": baseline_energy,
            "oracle_training_energy_wh": stopped_energy,
            "oracle_energy_saving_fraction": saving,
        }
        rows.append(result)
        grouped.setdefault(group, []).append(result)

    groups = []
    for group, replicas in sorted(grouped.items()):
        groups.append(
            {
                "dataset_group": group,
                "task_type": replicas[0]["task_type"],
                "replicates": len(replicas),
                "mean_oracle_stop_epoch": statistics.fmean(
                    row["oracle_stop_epoch"] for row in replicas
                ),
                "mean_quality_regret": statistics.fmean(
                    row["quality_regret"] for row in replicas
                ),
                "mean_oracle_energy_saving_fraction": statistics.fmean(
                    row["oracle_energy_saving_fraction"] for row in replicas
                ),
                "all_quality_feasible": all(row["quality_feasible"] for row in replicas),
            }
        )

    group_savings = [row["mean_oracle_energy_saving_fraction"] for row in groups]
    replicate_counts = {
        group: len(replicas) for group, replicas in sorted(grouped.items())
    }
    minimum_replicates = min(replicate_counts.values(), default=0)
    return {
        "schema_version": 1,
        "interpretation": (
            "Non-causal upper bound: earliest Full100 epoch whose best-so-far quality "
            "is within the data-derived tolerance of the Full100 best quality."
        ),
        "energy_scope": "training/epoch GPU energy",
        "trace_count": len(rows),
        "dataset_group_count": len(groups),
        "replication_audit": {
            "replicates_per_dataset_group": replicate_counts,
            "minimum_replicates": minimum_replicates,
            "probabilistic_noninferiority_estimable": minimum_replicates >= 3,
            "single_replica_probability_is_binary": minimum_replicates == 1,
        },
        "oracle_energy_saving_q25": _quantile(group_savings, 0.25),
        "mean_oracle_energy_saving_fraction": (
            statistics.fmean(group_savings) if group_savings else 0.0
        ),
        "groups": groups,
        "cases": rows,
    }


def write_oracle_report(output_dir: str | Path, report: Mapping[str, Any]) -> None:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    (output / "oracle-upper-bound.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    rows = list(report.get("cases", []))
    if rows:
        with (output / "oracle-upper-bound.csv").open(
            "w", encoding="utf-8", newline=""
        ) as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
