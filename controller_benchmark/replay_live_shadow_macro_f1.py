from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
import time
from pathlib import Path
from typing import Any, Mapping

from controller_benchmark.api import ControllerContext, EpochObservation
from controller_benchmark.loader import load_controller


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def _number(value: Any, default: float | None = None) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return default
    return result if math.isfinite(result) else default


def _epoch_energy_wh(row: Mapping[str, Any]) -> float:
    return max(0.0, float(_number(row.get("gpu_energy_kwh"), 0.0) or 0.0) * 1000.0)


def _original_pairs(snapshot: Path) -> dict[tuple[str, str], dict[str, Any]]:
    path = snapshot / "controller_benchmark" / "live-shadow-report.json"
    if not path.exists():
        return {}
    return {
        (str(row["case_id"]), str(row["controller_id"])): row
        for row in _load(path).get("pairs", [])
    }


def replay_macro_f1(
    snapshot: Path,
    plan_path: Path,
    *,
    output_dir: Path | None = None,
) -> dict[str, Any]:
    plan = _load(plan_path)
    metrics = snapshot / "data" / "energy_metrics"
    output = output_dir or snapshot / "controller_benchmark"
    output.mkdir(parents=True, exist_ok=True)
    original = _original_pairs(snapshot)
    acceptance = plan.get("acceptance", {})
    maximum_regret = float(acceptance.get("maximum_quality_regret", 0.10))
    minimum_saving = float(acceptance.get("minimum_energy_saving_fraction", 0.0))
    fallback_watts = float(
        plan["runs"][0]
        .get("controller_parameters", {})
        .get("fallback_watts_per_cpu_second", 5.0)
    )

    summaries_by_case: dict[str, dict[str, Any]] = {}
    for path in metrics.glob("epoch_summary_job_*.json"):
        summary = _load(path)
        if (
            summary.get("benchmark_run_id") == plan["benchmark_run_id"]
            and summary.get("task_type") == "image_classification"
        ):
            summaries_by_case[str(summary["benchmark_case_id"])] = summary

    pairs: list[dict[str, Any]] = []
    classification_runs = [
        run for run in plan["runs"] if run["task_type"] == "image_classification"
    ]
    for run in sorted(classification_runs, key=lambda item: int(item["sequence"])):
        case_id = str(run["benchmark_case_id"])
        summary = summaries_by_case.get(case_id)
        if summary is None:
            raise RuntimeError(f"Missing classification epoch summary for {case_id}.")
        epochs = sorted(
            summary.get("epochs", []),
            key=lambda row: int(row.get("epoch_index", row.get("epoch", 0))),
        )
        if not epochs or any(_number(row.get("macro_f1")) is None for row in epochs):
            raise RuntimeError(f"Incomplete per-epoch macro_f1 trace for {case_id}.")
        full_quality = max(float(row["macro_f1"]) for row in epochs)
        full_energy = sum(_epoch_energy_wh(row) for row in epochs)

        for definition in plan["controllers"]:
            controller_id = str(definition["id"])
            controller = load_controller(
                str(definition["controller_plugin"]),
                ControllerContext(
                    controller_id=controller_id,
                    benchmark_version=f"{plan['benchmark_version']}-macro-f1-replay",
                    task_type="image_classification",
                    quality_metric="quality_score",
                    scenario=str(run["scenario"]),
                    max_epochs=len(epochs),
                    parameters=dict(definition.get("controller_parameters", {})),
                    metadata={
                        **dict(run.get("case", {})),
                        "benchmark_case_id": case_id,
                        "benchmark_stage": run["benchmark_stage"],
                        "replay_quality_metric": "macro_f1",
                    },
                ),
            )
            history: list[dict[str, Any]] = []
            cumulative_energy = 0.0
            cumulative_duration = 0.0
            overhead_energy = 0.0
            compute_seconds = 0.0
            process_cpu_seconds = 0.0
            best_quality: float | None = None
            previous_quality: float | None = None
            stop_epoch = len(epochs)
            stop_reason = "max_epochs_reached"
            try:
                for raw in epochs:
                    # Remove decisions from the original Accuracy run so the replay
                    # receives only telemetry that existed before that decision.
                    event = {
                        key: value
                        for key, value in raw.items()
                        if not key.startswith("controller_plugin_")
                        and key
                        not in {
                            "should_stop",
                            "stop_reason",
                            "controller_decision",
                            "controller_decision_state",
                        }
                    }
                    epoch = int(event.get("epoch_index", event.get("epoch", len(history) + 1)))
                    quality = float(event["macro_f1"])
                    best_quality = quality if best_quality is None else max(best_quality, quality)
                    energy = _epoch_energy_wh(event)
                    duration = float(_number(event.get("duration_seconds"), 0.0) or 0.0)
                    cumulative_energy += energy
                    cumulative_duration += duration
                    event["source_accuracy"] = event.get("accuracy")
                    event["quality_score"] = quality
                    event["best_quality_score"] = best_quality
                    event["quality_metric"] = "macro_f1"
                    observation = EpochObservation(
                        epoch=epoch,
                        max_epochs=len(epochs),
                        task_type="image_classification",
                        quality_metric="quality_score",
                        quality=quality,
                        best_quality=best_quality,
                        delta_quality=(
                            quality - previous_quality if previous_quality is not None else None
                        ),
                        epoch_energy_wh=energy,
                        cumulative_energy_wh=cumulative_energy,
                        epoch_duration_seconds=duration,
                        cumulative_duration_seconds=cumulative_duration,
                        gpu_utilization_pct=_number(event.get("gpu_util_avg_pct")),
                        history=tuple(history),
                        raw_metrics=event,
                    )
                    wall_started = time.perf_counter()
                    cpu_started = time.process_time()
                    decision = controller.evaluate(observation)
                    cpu_delta = max(0.0, time.process_time() - cpu_started)
                    process_cpu_seconds += cpu_delta
                    compute_seconds += max(0.0, time.perf_counter() - wall_started)
                    overhead_energy += cpu_delta * fallback_watts / 3600.0
                    history.append(event)
                    previous_quality = quality
                    if decision.stop:
                        stop_epoch = epoch
                        stop_reason = decision.reason
                        break
            finally:
                controller.close()

            stopped_quality = max(float(row["quality_score"]) for row in history)
            total_energy = cumulative_energy + overhead_energy
            saving = (full_energy - total_energy) / full_energy if full_energy else 0.0
            regret = max(0.0, full_quality - stopped_quality)
            original_pair = original.get((case_id, controller_id), {})
            energy_met = saving >= minimum_saving
            quality_met = regret <= maximum_regret
            pairs.append(
                {
                    "benchmark_run_id": plan["benchmark_run_id"],
                    "replay_id": f"{plan['benchmark_run_id']}-macro-f1-replay",
                    "case_id": case_id,
                    "task_type": "image_classification",
                    "controller_id": controller_id,
                    "quality_metric": "macro_f1",
                    "source_trace_quality_metric": "accuracy",
                    "full100_best_quality": full_quality,
                    "best_quality_at_stop": stopped_quality,
                    "quality_regret": regret,
                    "quality_regret_pp": regret * 100.0,
                    "full100_epoch_training_energy_wh": full_energy,
                    "training_energy_to_stop_wh": cumulative_energy,
                    "controller_overhead_energy_wh": overhead_energy,
                    "counterfactual_total_energy_wh": total_energy,
                    "energy_saving_fraction": saving,
                    "energy_saving_percent": saving * 100.0,
                    "training_time_to_stop_s": cumulative_duration,
                    "stop_epoch": stop_epoch,
                    "stopped_early": stop_epoch < len(epochs),
                    "stop_reason": stop_reason,
                    "controller_decisions": len(history),
                    "controller_compute_seconds": compute_seconds,
                    "controller_process_cpu_seconds": process_cpu_seconds,
                    "controller_energy_method": "replay_process_cpu_time_power_model",
                    "controller_power_model_w": fallback_watts,
                    "original_accuracy_stop_epoch": original_pair.get("stop_epoch"),
                    "original_accuracy_quality_regret_pp": original_pair.get(
                        "quality_regret_pp"
                    ),
                    "energy_target_met": energy_met,
                    "quality_target_met": quality_met,
                    "joint_target_met": energy_met and quality_met,
                }
            )

    expected_pairs = len(classification_runs) * len(plan["controllers"])
    if len(pairs) != expected_pairs:
        raise RuntimeError(f"Expected {expected_pairs} Macro-F1 pairs, found {len(pairs)}.")

    aggregates: dict[str, dict[str, Any]] = {}
    for definition in plan["controllers"]:
        controller_id = str(definition["id"])
        selected = [row for row in pairs if row["controller_id"] == controller_id]
        aggregates[controller_id] = {
            "cases": len(selected),
            "mean_energy_saving_fraction": statistics.fmean(
                row["energy_saving_fraction"] for row in selected
            ),
            "mean_quality_regret": statistics.fmean(
                row["quality_regret"] for row in selected
            ),
            "maximum_quality_regret": max(row["quality_regret"] for row in selected),
            "energy_success_rate": statistics.fmean(
                int(row["energy_target_met"]) for row in selected
            ),
            "quality_success_rate": statistics.fmean(
                int(row["quality_target_met"]) for row in selected
            ),
            "joint_success_rate": statistics.fmean(
                int(row["joint_target_met"]) for row in selected
            ),
        }
    result = {
        "schema_version": 1,
        "replay_id": f"{plan['benchmark_run_id']}-macro-f1-replay",
        "source_benchmark_run_id": plan["benchmark_run_id"],
        "method": "causal_prefix_replay_without_retraining",
        "quality_metric": "macro_f1",
        "replayed_task_type": "image_classification",
        "training_trajectory_reused": True,
        "physical_stop_run": False,
        "controller_energy_method": "process_cpu_time_times_configured_power",
        "controller_power_model_w": fallback_watts,
        "acceptance": acceptance,
        "cases": len(classification_runs),
        "controller_pairs": len(pairs),
        "controllers": aggregates,
        "pairs": pairs,
    }
    json_path = output / "live-shadow-macro-f1-replay-report.json"
    json_path.write_text(
        json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    csv_path = output / "paired-results-macro-f1-replay.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(pairs[0]))
        writer.writeheader()
        writer.writerows(pairs)
    source_report_path = snapshot / "controller_benchmark" / "live-shadow-report.json"
    if source_report_path.exists():
        source_report = _load(source_report_path)
        replay_by_key = {
            (row["case_id"], row["controller_id"]): row for row in pairs
        }
        combined_pairs: list[dict[str, Any]] = []
        for source_pair in source_report["pairs"]:
            combined = dict(source_pair)
            replay = replay_by_key.get(
                (source_pair["case_id"], source_pair["controller_id"])
            )
            if replay is not None:
                for key in (
                    "full100_best_quality",
                    "best_quality_at_stop",
                    "quality_regret",
                    "quality_regret_pp",
                    "full100_epoch_training_energy_wh",
                    "training_energy_to_stop_wh",
                    "controller_overhead_energy_wh",
                    "counterfactual_total_energy_wh",
                    "energy_saving_fraction",
                    "energy_saving_percent",
                    "training_time_to_stop_s",
                    "controller_compute_seconds",
                    "controller_process_cpu_seconds",
                    "controller_decisions",
                    "stop_epoch",
                    "stopped_early",
                    "stop_reason",
                    "energy_target_met",
                    "quality_target_met",
                    "joint_target_met",
                ):
                    combined[key] = replay[key]
                combined["energy_measurement_method"] = replay[
                    "controller_energy_method"
                ]
                combined["quality_metric"] = "macro_f1"
                combined["quality_source"] = "causal_prefix_replay"
            else:
                combined["quality_metric"] = {
                    "aerial_vehicle_counting": "map50_95",
                    "semantic_segmentation": "miou",
                }.get(source_pair["stage"], "quality_score")
                combined["quality_source"] = "original_live_shadow"
            combined_pairs.append(combined)

        combined_controllers: dict[str, dict[str, Any]] = {}
        for definition in plan["controllers"]:
            controller_id = str(definition["id"])
            selected = [
                row for row in combined_pairs if row["controller_id"] == controller_id
            ]
            combined_controllers[controller_id] = {
                "cases": len(selected),
                "mean_energy_saving_fraction": statistics.fmean(
                    row["energy_saving_fraction"] for row in selected
                ),
                "median_energy_saving_fraction": statistics.median(
                    row["energy_saving_fraction"] for row in selected
                ),
                "mean_quality_regret": statistics.fmean(
                    row["quality_regret"] for row in selected
                ),
                "maximum_quality_regret": max(
                    row["quality_regret"] for row in selected
                ),
                "energy_success_rate": statistics.fmean(
                    int(row["energy_target_met"]) for row in selected
                ),
                "quality_success_rate": statistics.fmean(
                    int(row["quality_target_met"]) for row in selected
                ),
                "joint_success_rate": statistics.fmean(
                    int(row["joint_target_met"]) for row in selected
                ),
                "total_controller_overhead_energy_wh": sum(
                    row["controller_overhead_energy_wh"] for row in selected
                ),
                "total_controller_compute_seconds": sum(
                    row["controller_compute_seconds"] for row in selected
                ),
            }
        combined_report = {
            "schema_version": 1,
            "report_id": f"{plan['benchmark_run_id']}-mixed-quality-macro-f1-replay",
            "source_benchmark_run_id": plan["benchmark_run_id"],
            "measurement_protocol": "live-shadow-with-causal-macro-f1-replay-v1",
            "quality_metrics": {
                "aerial_vehicle_counting": "map50_95",
                "image_classification": "macro_f1",
                "semantic_segmentation": "miou",
            },
            "classification_training_reused": True,
            "classification_physical_stop_run": False,
            "acceptance": acceptance,
            "training_jobs": source_report["training_jobs"],
            "controller_pairs": len(combined_pairs),
            "controllers": combined_controllers,
            "pairs": combined_pairs,
        }
        (output / "live-shadow-mixed-quality-macro-f1-report.json").write_text(
            json.dumps(combined_report, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        with (output / "paired-results-mixed-quality-macro-f1.csv").open(
            "w", encoding="utf-8", newline=""
        ) as handle:
            fieldnames = sorted({key for row in combined_pairs for key in row})
            writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(combined_pairs)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    result = replay_macro_f1(
        args.snapshot.resolve(),
        args.plan.resolve(),
        output_dir=args.output_dir.resolve() if args.output_dir else None,
    )
    print(
        json.dumps(
            {
                "replay_id": result["replay_id"],
                "cases": result["cases"],
                "controller_pairs": result["controller_pairs"],
                "output": str(
                    (args.output_dir or args.snapshot / "controller_benchmark")
                    / "live-shadow-macro-f1-replay-report.json"
                ),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
