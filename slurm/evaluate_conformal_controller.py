from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

try:
    from slurm.epoch_energy_controller import EnergyAdaptiveConfig, EpochEnergyAdaptiveController
except ImportError:  # pragma: no cover
    from epoch_energy_controller import EnergyAdaptiveConfig, EpochEnergyAdaptiveController


SCENARIO_DEFAULTS = {
    "train128-scratch": {"min_epochs": 40, "min_quality": 0.60},
    "train256-pretrained": {"min_epochs": 35, "min_quality": 0.78},
}


def replay(path: Path, calibration: Path, regret_tolerance: float) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8-sig"))
    scenario = str(payload.get("scenario", "unspecified"))
    defaults = SCENARIO_DEFAULTS.get(scenario, {"min_epochs": 40, "min_quality": 0.55})
    rows = payload.get("epochs", [])[:100]
    final_best = max(float(row.get("map50_95", 0.0)) for row in rows)
    oracle_epoch = next(
        (
            index
            for index, row in enumerate(rows, start=1)
            if final_best - max(float(item.get("map50_95", 0.0)) for item in rows[:index]) <= regret_tolerance
        ),
        len(rows),
    )
    controller = EpochEnergyAdaptiveController(
        gpu_csv_path=path.parent / "unused.csv",
        epoch_timeline_path=path.parent / "unused.jsonl",
        config=EnergyAdaptiveConfig(
            enabled=True,
            controller_mode="conformal_energy_aware",
            comparison_strategy="conformal_energy_aware_controller",
            monitor_metric="map50_95",
            min_epochs=defaults["min_epochs"],
            patience=3,
            uncertainty_target_epoch=100,
            uncertainty_alpha=0.05,
            uncertainty_bootstrap_samples=16,
            uncertainty_min_fit_points=12,
            uncertainty_min_quality=defaults["min_quality"],
            scenario=scenario,
            conformal_calibration_path=str(calibration),
            conformal_regret_tolerance=regret_tolerance,
            conformal_future_efficiency_threshold=0.05,
            conformal_max_interval_width=0.08,
            conformal_energy_window=5,
        ),
    )
    stop_epoch = len(rows)
    stop_reason = "target_epoch_reached"
    stop_best = final_best
    for index, source in enumerate(rows, start=1):
        event = dict(source)
        event["epoch"] = index
        event["epoch_index"] = index
        event["best_map50_95"] = max(float(item.get("map50_95", 0.0)) for item in rows[:index])
        stop, reason, diagnostics = controller._evaluate_conformal_energy_stop(event)
        event.update(diagnostics)
        event["low_gain_streak"] = controller.low_gain_streak
        controller.history.append(event)
        if stop:
            stop_epoch = index
            stop_reason = reason or "controller_stop"
            stop_best = event["best_map50_95"]
            break
    return {
        "job_id": payload.get("job_id"),
        "scenario": scenario,
        "training_seed": payload.get("training_seed"),
        "oracle_stop_epoch": oracle_epoch,
        "controller_stop_epoch": stop_epoch,
        "stop_epoch_error": stop_epoch - oracle_epoch,
        "final_best_map50_95": final_best,
        "stop_best_map50_95": stop_best,
        "regret_at_stop": final_best - stop_best,
        "stop_reason": stop_reason,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--calibration", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--regret-tolerance", type=float, default=0.015)
    args = parser.parse_args()
    rows = []
    for path in sorted(args.input_dir.rglob("epoch_summary_job_*.json")):
        payload = json.loads(path.read_text(encoding="utf-8-sig"))
        if payload.get("comparison_strategy") == "full100" and len(payload.get("epochs", [])) >= 100:
            rows.append(replay(path, args.calibration, args.regret_tolerance))
    if not rows:
        raise SystemExit("No complete full100 curves found")
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.output_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {args.output_csv} with {len(rows)} replay rows")


if __name__ == "__main__":
    main()
