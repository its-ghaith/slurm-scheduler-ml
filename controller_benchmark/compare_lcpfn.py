from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path
from typing import Any

from controller_benchmark.runtime import ControllerPluginRuntime


LCPFN_PLUGIN = "controller_benchmark.controllers.lcpfn_quality_baseline:LcpfnQualityBaselineController"


def _case_summaries(snapshot: Path) -> dict[str, dict[str, Any]]:
    result = {}
    for path in (snapshot / "data" / "energy_metrics").glob("epoch_summary_job_*.json"):
        summary = json.loads(path.read_text(encoding="utf-8"))
        case_id = str(summary.get("benchmark_case_id", ""))
        if case_id and summary.get("epochs"):
            result[case_id] = summary
    return result


def _energy_wh(event: dict[str, Any]) -> float:
    if event.get("interval_energy_wh") is not None:
        return float(event["interval_energy_wh"])
    return float(event.get("total_energy_kwh", event.get("gpu_energy_kwh", 0.0)) or 0.0) * 1000.0


def _quality_value(event: dict[str, Any], summary: dict[str, Any]) -> float:
    candidates = [
        "quality_score",
        str(summary.get("quality_metric", "")),
        str(summary.get("primary_metric", "")),
        "map50_95",
        "macro_f1",
        "accuracy",
        "miou",
        "dice",
    ]
    for key in candidates:
        if key and event.get(key) is not None:
            return float(event[key])
    raise ValueError(
        f"No quality value found for case {summary.get('benchmark_case_id', 'unknown')}."
    )


def replay_lcpfn(summary: dict[str, Any], parameters: dict[str, Any]) -> dict[str, Any]:
    epochs = list(summary["epochs"])
    maximum = len(epochs)
    runtime = ControllerPluginRuntime(
        plugin_path=LCPFN_PLUGIN,
        parameters_json=json.dumps(parameters),
        controller_id="lcpfn-quality-baseline",
        benchmark_version="lcpfn-cross-domain-replay-v1",
        task_type=str(summary.get("task_type", "unknown")),
        quality_metric="quality_score",
        scenario="cross-domain-replay",
        max_epochs=maximum,
        metadata={"benchmark_case_id": summary.get("benchmark_case_id")},
    )
    history: list[dict[str, Any]] = []
    stop_epoch = maximum
    stop_reason = "max_checkpoints_reached"
    compute_seconds = 0.0
    try:
        for index, source in enumerate(epochs, 1):
            event = dict(source)
            event["quality_score"] = _quality_value(source, summary)
            event["epoch"] = index
            event["epoch_index"] = index
            event["decision_checkpoint"] = index
            started = time.perf_counter()
            stop, reason, _ = runtime.evaluate(event, history)
            compute_seconds += time.perf_counter() - started
            history.append(event)
            if stop:
                stop_epoch = index
                stop_reason = str(reason)
                break
    finally:
        runtime.close()
    full_quality = max(_quality_value(row, summary) for row in epochs)
    stop_quality = max(_quality_value(row, summary) for row in epochs[:stop_epoch])
    full_energy = sum(_energy_wh(row) for row in epochs)
    stop_energy = sum(_energy_wh(row) for row in epochs[:stop_epoch])
    return {
        "benchmark_case_id": summary.get("benchmark_case_id"),
        "task_type": summary.get("task_type"),
        "controller_id": "lcpfn-quality-baseline",
        "stop_checkpoint": stop_epoch,
        "max_checkpoints": maximum,
        "full100_best_quality": full_quality,
        "best_quality_at_stop": stop_quality,
        "quality_regret": max(0.0, full_quality - stop_quality),
        "full100_epoch_energy_wh": full_energy,
        "training_energy_to_stop_wh": stop_energy,
        "energy_saving_fraction": (full_energy - stop_energy) / full_energy if full_energy > 0 else None,
        "controller_compute_seconds": compute_seconds,
        "controller_energy_measured": False,
        "energy_scope": "counterfactual_epoch_gpu_only",
        "stop_reason": stop_reason,
        "comparison_note": "LC-PFN quality-only single-run adaptation; not the paper's multi-configuration model-selection protocol.",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--horizon-checkpoints", type=int, default=5)
    parser.add_argument("--minimum-checkpoints", type=int, default=10)
    parser.add_argument("--useful-gain", type=float, default=0.002)
    parser.add_argument("--patience", type=int, default=3)
    args = parser.parse_args()
    parameters = {
        "horizon_epochs": args.horizon_checkpoints,
        "min_epochs": args.minimum_checkpoints,
        "min_fit_points": max(3, args.minimum_checkpoints),
        "patience": args.patience,
        "useful_gain": args.useful_gain,
        "min_expected_gain": args.useful_gain,
        "max_probability_gain_gt_threshold": 0.20,
    }
    rows = [replay_lcpfn(summary, parameters) for summary in _case_summaries(args.snapshot).values()]
    if not rows:
        raise RuntimeError("No completed epoch summaries were found in the snapshot.")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    json_path = args.output_dir / "lcpfn-comparison.json"
    csv_path = args.output_dir / "lcpfn-comparison.csv"
    json_path.write_text(json.dumps({"schema_version": 1, "parameters": parameters, "rows": rows}, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(json_path)
    print(csv_path)


if __name__ == "__main__":
    main()
