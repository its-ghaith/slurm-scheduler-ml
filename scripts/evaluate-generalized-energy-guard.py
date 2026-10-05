#!/usr/bin/env python3
"""Replay the generalized energy guard on completed Full100 reference curves."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from slurm.epoch_energy_controller import EnergyAdaptiveConfig, EpochEnergyAdaptiveController
from slurm.gpu_energy_utils import summarize_window


DEFAULT_STAGE_A = Path(
    r"D:\Masterarbeit_Vergleichssicherung\CARPK_StageA_HybridGeneralisation_Min30Eval1_10Jobs_2026-07-20_234033"
)
DEFAULT_STAGE_B = Path(
    r"D:\Masterarbeit_Vergleichssicherung\CARPK_StageB_ScenarioGeneralisation_Min30Eval1_14Jobs_2026-07-21_204649"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage-a", type=Path, default=DEFAULT_STAGE_A)
    parser.add_argument("--stage-b", type=Path, default=DEFAULT_STAGE_B)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results" / "generalized-energy-guard-replay")
    parser.add_argument("--target-saving", type=float, default=0.20)
    parser.add_argument("--fallback-fraction", type=float, default=0.77)
    parser.add_argument("--post-training-reserve-wh", type=float, default=0.90)
    return parser.parse_args()


def job_number(path: Path) -> int:
    return int(path.stem.rsplit("_", 1)[1])


def replay_snapshot(stage: str, snapshot: Path, args: argparse.Namespace) -> list[dict]:
    metrics_dir = snapshot / "data" / "energy_metrics"
    summaries = []
    for path in sorted(metrics_dir.glob("epoch_summary_job_*.json"), key=job_number):
        data = json.loads(path.read_text(encoding="utf-8"))
        if data.get("comparison_strategy") == "full100" and int(data.get("epochs_completed", 0)) >= 100:
            summaries.append((path, data))
    if not summaries:
        raise RuntimeError(f"No Full100 references found in {snapshot}")

    rows = []
    for summary_path, summary in summaries:
        job_id = str(summary["job_id"])
        gpu_csv = metrics_dir / f"gpu_metrics_job_{job_id}.csv"
        gpu_summary_path = metrics_dir / f"gpu_summary_job_{job_id}.json"
        gpu_summary = json.loads(gpu_summary_path.read_text(encoding="utf-8"))
        full_job_wh = float(gpu_summary["gpu_energy_kwh"]) * 1000.0
        events = list(summary["epochs"])
        final_epoch_energy = summarize_window(gpu_csv, end_ts=float(events[-1]["end_ts"]))
        post_training_wh = max(0.0, full_job_wh - float(final_epoch_energy["gpu_energy_kwh"]) * 1000.0)

        controller = EpochEnergyAdaptiveController(
            gpu_csv_path=gpu_csv,
            epoch_timeline_path=args.output_dir / "replay-timelines" / f"{stage}-job-{job_id}.jsonl",
            config=EnergyAdaptiveConfig(
                enabled=True,
                controller_mode="generalized_energy_guard",
                monitor_metric="map50_95",
                min_epochs=30,
                patience=3,
                uncertainty_target_epoch=100,
                uncertainty_alpha=0.05,
                uncertainty_bootstrap_samples=24,
                uncertainty_min_fit_points=12,
                uncertainty_min_quality=0.55,
                conformal_regret_tolerance=0.015,
                controller_evaluation_interval=1,
                energy_guard_target_saving_fraction=args.target_saving,
                energy_guard_safety_margin_fraction=0.01,
                energy_guard_post_training_reserve_wh=args.post_training_reserve_wh,
                energy_guard_window=5,
                energy_guard_fallback_epoch_fraction=args.fallback_fraction,
                energy_guard_strict=True,
            ),
        )
        stop_event = None
        stop_reason = None
        for source_event in events:
            event = dict(source_event)
            stop, reason, diagnostics = controller._evaluate_generalized_energy_guard_stop(event)
            event.update(diagnostics)
            event["should_stop"] = int(stop)
            controller.history.append(event)
            if stop:
                stop_event = event
                stop_reason = reason
                break
        if stop_event is None:
            stop_event = dict(events[-1])
            stop_reason = "target_epoch_reached"

        stop_observed_wh = float(
            summarize_window(gpu_csv, end_ts=float(stop_event["end_ts"]))["gpu_energy_kwh"]
        ) * 1000.0
        hypothetical_job_wh = stop_observed_wh + post_training_wh
        actual_saving = (full_job_wh - hypothetical_job_wh) / full_job_wh if full_job_wh else 0.0
        full_best = max(float(item.get("map50_95") or 0.0) for item in events)
        stop_best = max(
            float(item.get("map50_95") or 0.0)
            for item in events
            if int(item.get("epoch_index", item.get("epoch", 0))) <= int(stop_event["epoch_index"])
        )
        rows.append(
            {
                "stage": stage,
                "scenario": summary.get("scenario"),
                "training_seed": summary.get("training_seed"),
                "reference_job_id": job_id,
                "stop_epoch": int(stop_event["epoch_index"]),
                "full_job_energy_wh": round(full_job_wh, 9),
                "replayed_job_energy_wh": round(hypothetical_job_wh, 9),
                "energy_saving_pct": round(actual_saving * 100.0, 6),
                "full_best_map50_95_pct": round(full_best * 100.0, 6),
                "stop_best_map50_95_pct": round(stop_best * 100.0, 6),
                "quality_regret_pp": round((full_best - stop_best) * 100.0, 6),
                "target_met": actual_saving + 1e-9 >= args.target_saving,
                "quality_conflict": bool(stop_event.get("energy_guard_quality_conflict")),
                "stop_reason": stop_reason,
            }
        )
    return rows


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows = replay_snapshot("A", args.stage_a, args) + replay_snapshot("B", args.stage_b, args)
    csv_path = args.output_dir / "replay-results.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    savings = [float(row["energy_saving_pct"]) for row in rows]
    regrets = [float(row["quality_regret_pp"]) for row in rows]
    report = {
        "method": "historical replay on independent completed Full100 traces",
        "target_energy_saving_fraction": args.target_saving,
        "fallback_epoch_fraction": args.fallback_fraction,
        "runs": rows,
        "aggregate": {
            "references": len(rows),
            "target_met_count": sum(bool(row["target_met"]) for row in rows),
            "minimum_energy_saving_pct": min(savings),
            "median_energy_saving_pct": sorted(savings)[len(savings) // 2],
            "maximum_quality_regret_pp": max(regrets),
            "mean_quality_regret_pp": sum(regrets) / len(regrets),
        },
        "limitation": (
            "Replay validates the energy target on the recorded CARPK traces; it is not a mathematical "
            "guarantee for unseen hardware, tasks, epoch budgets, or energy profiles. Confirm with fresh runs."
        ),
    }
    json_path = args.output_dir / "replay-report.json"
    json_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(report["aggregate"], indent=2))
    if not all(row["target_met"] for row in rows):
        raise SystemExit("The 20% target was not met for every replayed reference.")


if __name__ == "__main__":
    main()
