#!/usr/bin/env python3
"""Build paired research metrics for the Full100 vs energy-guard study."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from pathlib import Path


def prom_escape(value: object) -> str:
    return str(value).replace("\\", "\\\\").replace('"', '\\"')


def job_number(path: Path) -> int:
    return int(path.stem.rsplit("_", 1)[1])


def load_runs(snapshot: Path) -> list[dict]:
    metrics = snapshot / "data" / "energy_metrics"
    runs = []
    for path in sorted(metrics.glob("epoch_summary_job_*.json"), key=job_number):
        summary = json.loads(path.read_text(encoding="utf-8"))
        job_id = str(summary["job_id"])
        gpu = json.loads((metrics / f"gpu_summary_job_{job_id}.json").read_text(encoding="utf-8"))
        scenario = str(summary.get("scenario", "unspecified"))
        stage = "A" if scenario == "train128-scratch" else "B"
        seed = int(summary.get("training_seed", 0))
        case_label = f"A seed {seed}" if stage == "A" else f"B {scenario}"
        epochs = summary.get("epochs", [])
        runs.append(
            {
                "stage": stage,
                "case_label": case_label,
                "scenario": scenario,
                "training_seed": seed,
                "strategy": str(summary.get("comparison_strategy", "unspecified")),
                "job_id": job_id,
                "epochs": int(summary.get("epochs_completed", len(epochs))),
                "best_map50_95": max((float(row.get("map50_95") or 0.0) for row in epochs), default=0.0),
                "energy_wh": float(gpu.get("gpu_energy_kwh", 0.0)) * 1000.0,
                "duration_seconds": float(gpu.get("duration_seconds", 0.0)),
                "adaptive_stopped": bool(summary.get("adaptive_stopped")),
                "quality_conflict": bool(summary.get("energy_guard_quality_conflict") or (epochs[-1].get("energy_guard_quality_conflict") if epochs else 0)),
                "stop_reason": str(summary.get("stop_reason") or "target_epoch_reached"),
            }
        )
    return runs


def pair_runs(runs: list[dict], target: float) -> list[dict]:
    groups: dict[tuple[str, str, int], list[dict]] = {}
    for run in runs:
        groups.setdefault((run["stage"], run["scenario"], run["training_seed"]), []).append(run)
    pairs = []
    for key, group in sorted(groups.items()):
        full = next((run for run in group if run["strategy"] == "full100"), None)
        guard = next((run for run in group if run["strategy"] == "generalized_energy_guard_20pct"), None)
        if full is None or guard is None:
            raise RuntimeError(f"Incomplete pair {key}: {[run['strategy'] for run in group]}")
        saving = (full["energy_wh"] - guard["energy_wh"]) / full["energy_wh"] if full["energy_wh"] else 0.0
        pairs.append(
            {
                "stage": full["stage"],
                "case_label": full["case_label"],
                "scenario": full["scenario"],
                "training_seed": full["training_seed"],
                "full_job_id": full["job_id"],
                "controller_job_id": guard["job_id"],
                "full_energy_wh": full["energy_wh"],
                "controller_energy_wh": guard["energy_wh"],
                "energy_saving_pct": saving * 100.0,
                "target_met": saving >= target,
                "full_best_map50_95_pct": full["best_map50_95"] * 100.0,
                "controller_best_map50_95_pct": guard["best_map50_95"] * 100.0,
                "quality_regret_pp": (full["best_map50_95"] - guard["best_map50_95"]) * 100.0,
                "full_epochs": full["epochs"],
                "controller_epochs": guard["epochs"],
                "epochs_saved": full["epochs"] - guard["epochs"],
                "full_duration_seconds": full["duration_seconds"],
                "controller_duration_seconds": guard["duration_seconds"],
                "duration_saving_pct": ((full["duration_seconds"] - guard["duration_seconds"]) / full["duration_seconds"] * 100.0) if full["duration_seconds"] else 0.0,
                "quality_conflict": guard["quality_conflict"],
                "stop_reason": guard["stop_reason"],
            }
        )
    return pairs


def write_prom(path: Path, pairs: list[dict], target: float) -> None:
    lines = []
    for pair in pairs:
        labels = ",".join(
            f'{key}="{prom_escape(pair[key])}"'
            for key in ("stage", "case_label", "scenario", "training_seed", "full_job_id", "controller_job_id")
        )
        lines.append(f"slurm_energy_guard_case_info{{{labels}}} 1")
        paired_metrics = {
            "slurm_energy_guard_energy_wh": (("Full100", "full_energy_wh"), ("Energy Guard", "controller_energy_wh")),
            "slurm_energy_guard_best_map50_95_pct": (("Full100", "full_best_map50_95_pct"), ("Energy Guard", "controller_best_map50_95_pct")),
            "slurm_energy_guard_epochs": (("Full100", "full_epochs"), ("Energy Guard", "controller_epochs")),
            "slurm_energy_guard_duration_seconds": (("Full100", "full_duration_seconds"), ("Energy Guard", "controller_duration_seconds")),
        }
        for metric, variants in paired_metrics.items():
            for series, key in variants:
                lines.append(f'{metric}{{{labels},series="{series}"}} {pair[key]:.12g}')
        single_metrics = {
            "slurm_energy_guard_energy_saving_pct": ("Energy saving", "energy_saving_pct"),
            "slurm_energy_guard_target_met": ("20% target met", "target_met"),
            "slurm_energy_guard_quality_regret_pp": ("mAP50-95 regret", "quality_regret_pp"),
            "slurm_energy_guard_epochs_saved": ("Epochs saved", "epochs_saved"),
            "slurm_energy_guard_duration_saving_pct": ("Duration saving", "duration_saving_pct"),
            "slurm_energy_guard_quality_conflict": ("Quality conflict", "quality_conflict"),
        }
        for metric, (series, key) in single_metrics.items():
            value = int(pair[key]) if isinstance(pair[key], bool) else pair[key]
            lines.append(f'{metric}{{{labels},series="{series}"}} {value:.12g}')
    savings = [pair["energy_saving_pct"] for pair in pairs]
    regrets = [pair["quality_regret_pp"] for pair in pairs]
    aggregate = {
        "slurm_energy_guard_total_pairs": len(pairs),
        "slurm_energy_guard_target_saving_pct": target * 100.0,
        "slurm_energy_guard_target_success_rate_pct": 100.0 * sum(pair["target_met"] for pair in pairs) / len(pairs),
        "slurm_energy_guard_min_energy_saving_pct": min(savings),
        "slurm_energy_guard_mean_energy_saving_pct": statistics.fmean(savings),
        "slurm_energy_guard_mean_quality_regret_pp": statistics.fmean(regrets),
        "slurm_energy_guard_max_quality_regret_pp": max(regrets),
        "slurm_energy_guard_quality_conflict_rate_pct": 100.0 * sum(pair["quality_conflict"] for pair in pairs) / len(pairs),
    }
    lines.extend(f"{name} {value:.12g}" for name, value in aggregate.items())
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--target-saving", type=float, default=0.20)
    args = parser.parse_args()
    runs = load_runs(args.snapshot)
    if len(runs) != 24:
        raise RuntimeError(f"Expected 24 runs, found {len(runs)}")
    pairs = pair_runs(runs, args.target_saving)
    reports = args.snapshot / "reports"
    reports.mkdir(parents=True, exist_ok=True)
    csv_path = reports / "generalized-energy-guard-paired-results.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(pairs[0]))
        writer.writeheader()
        writer.writerows(pairs)
    report = {
        "target_energy_saving_fraction": args.target_saving,
        "pairs": pairs,
        "aggregate": {
            "pairs": len(pairs),
            "target_met": sum(pair["target_met"] for pair in pairs),
            "minimum_energy_saving_pct": min(pair["energy_saving_pct"] for pair in pairs),
            "mean_energy_saving_pct": statistics.fmean(pair["energy_saving_pct"] for pair in pairs),
            "mean_quality_regret_pp": statistics.fmean(pair["quality_regret_pp"] for pair in pairs),
            "maximum_quality_regret_pp": max(pair["quality_regret_pp"] for pair in pairs),
        },
    }
    (reports / "generalized-energy-guard-report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    write_prom(
        args.snapshot / "data" / "energy_metrics" / "node_exporter" / "generalized_energy_guard_study.prom",
        pairs,
        args.target_saving,
    )
    print(json.dumps(report["aggregate"], indent=2))


if __name__ == "__main__":
    main()
