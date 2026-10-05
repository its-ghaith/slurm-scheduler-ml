from __future__ import annotations

import argparse
import json
import statistics
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from controller_benchmark.optimization.replay import BaselineTrace, ReplayResult
from controller_benchmark.rapec_g_v3_offline_replay import (
    _controller,
    _read_json,
    _replay,
    load_live_shadow_traces,
)
from controller_benchmark.optimization.replay import load_baseline_traces


def _controller_summary(results: list[ReplayResult]) -> dict[str, Any]:
    return {
        "cases": len(results),
        "mean_energy_saving_fraction_vs_full100": statistics.mean(
            result.energy_saving_fraction for result in results
        ),
        "median_energy_saving_fraction_vs_full100": statistics.median(
            result.energy_saving_fraction for result in results
        ),
        "mean_quality_regret": statistics.mean(
            result.quality_regret for result in results
        ),
        "maximum_quality_regret": max(
            result.quality_regret for result in results
        ),
        "quality_success_fraction": sum(
            result.quality_regret <= 0.10 for result in results
        )
        / len(results),
    }


def _group_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    v4 = [row["_v4"] for row in rows]
    direct = [row["energy_saving_vs_standard_es_fraction"] for row in rows]
    total_standard = sum(row["_standard"].stopped_energy_wh for row in rows)
    total_v4 = sum(row["_v4"].stopped_energy_wh for row in rows)
    return {
        "rapec_g_v4": _controller_summary(v4),
        "rapec_g_v4_vs_standard_es": {
            "mean_direct_energy_saving_fraction": statistics.mean(direct),
            "median_direct_energy_saving_fraction": statistics.median(direct),
            "aggregate_direct_energy_saving_fraction": (
                (total_standard - total_v4) / total_standard
                if total_standard > 0.0
                else 0.0
            ),
            "cases_with_lower_energy": sum(value > 0.0 for value in direct),
            "cases_with_equal_energy": sum(
                abs(value) <= 1e-12 for value in direct
            ),
            "cases_with_higher_energy": sum(value < 0.0 for value in direct),
            "joint_success_fraction": sum(
                row["_v4"].quality_regret <= 0.10
                and row["_v4"].stopped_energy_wh
                <= row["_standard"].stopped_energy_wh
                for row in rows
            )
            / len(rows),
        },
    }


def build_report(
    vision_traces: list[BaselineTrace],
    nonvision_traces: list[BaselineTrace],
    configuration: dict[str, Any],
) -> dict[str, Any]:
    standard = _controller(configuration, "standard-es-10")
    v3 = _controller(configuration, "rapec-g-v3")
    v4 = _controller(configuration, "rapec-g-v4")
    internal_rows: list[dict[str, Any]] = []
    for domain, traces in (
        ("computer_vision", vision_traces),
        ("non_vision", nonvision_traces),
    ):
        for trace in traces:
            standard_result = _replay(trace, standard)
            v3_result = _replay(trace, v3)
            v4_result = _replay(trace, v4)
            direct_saving = (
                (standard_result.stopped_energy_wh - v4_result.stopped_energy_wh)
                / standard_result.stopped_energy_wh
                if standard_result.stopped_energy_wh > 0.0
                else 0.0
            )
            internal_rows.append(
                {
                    "domain": domain,
                    "case_id": trace.case_id,
                    "task_type": trace.task_type,
                    "quality_metric": trace.quality_metric,
                    "full100_best_quality": v4_result.baseline_best_quality,
                    "full100_energy_wh": v4_result.baseline_energy_wh,
                    "standard_es_10": standard_result.to_dict(),
                    "rapec_g_v3": v3_result.to_dict(),
                    "rapec_g_v4": v4_result.to_dict(),
                    "energy_saving_vs_standard_es_fraction": direct_saving,
                    "quality_regret_difference_vs_standard_es": (
                        v4_result.quality_regret
                        - standard_result.quality_regret
                    ),
                    "success_vs_standard_es": (
                        v4_result.quality_regret <= 0.10
                        and v4_result.stopped_energy_wh
                        <= standard_result.stopped_energy_wh
                    ),
                    "_standard": standard_result,
                    "_v3": v3_result,
                    "_v4": v4_result,
                }
            )

    cv_rows = [
        row for row in internal_rows if row["domain"] == "computer_vision"
    ]
    nonvision_rows = [
        row for row in internal_rows if row["domain"] == "non_vision"
    ]
    cv_regression_passed = all(
        row["_v3"].stop_epoch == row["_v4"].stop_epoch
        and abs(row["_v3"].quality_regret - row["_v4"].quality_regret)
        <= 1e-12
        and abs(
            row["_v3"].energy_saving_fraction
            - row["_v4"].energy_saving_fraction
        )
        <= 1e-12
        for row in cv_rows
    )
    by_case = {row["case_id"]: row for row in internal_rows}
    target_acceptance = {
        "multi_eurlex_not_later_than_standard_es": (
            by_case["multi-eurlex-text-classification-scratch"]["_v4"].stop_epoch
            <= by_case["multi-eurlex-text-classification-scratch"]["_standard"].stop_epoch
        ),
        "electricity_not_later_than_standard_es": (
            by_case["electricity-hourly-time-series-forecasting-scratch"]["_v4"].stop_epoch
            <= by_case["electricity-hourly-time-series-forecasting-scratch"]["_standard"].stop_epoch
        ),
        "breakout_protected_beyond_standard_es": (
            by_case["minatar-breakout-reinforcement-learning-scratch"]["_v4"].stop_epoch
            > by_case["minatar-breakout-reinforcement-learning-scratch"]["_standard"].stop_epoch
        ),
        "breakout_quality_regret_at_most_10pp": (
            by_case["minatar-breakout-reinforcement-learning-scratch"]["_v4"].quality_regret
            <= 0.10
        ),
    }
    public_rows = [
        {
            key: value
            for key, value in row.items()
            if not key.startswith("_")
        }
        for row in internal_rows
    ]
    return {
        "schema_version": "rapec-g-v4-offline-replay-v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "training_jobs_submitted": 0,
        "evaluation_mode": "offline_replay_of_recorded_full100_epoch_traces",
        "acceptance": {
            "maximum_quality_regret": 0.10,
            "cv_regression_passed": cv_regression_passed,
            **target_acceptance,
            "all_passed": cv_regression_passed
            and all(target_acceptance.values()),
        },
        "summary": {
            "computer_vision": _group_summary(cv_rows),
            "non_vision": _group_summary(nonvision_rows),
            "all": _group_summary(internal_rows),
        },
        "rows": public_rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--vision-source", type=Path, required=True)
    parser.add_argument("--nonvision-run-dir", type=Path, required=True)
    parser.add_argument("--nonvision-metrics-dir", type=Path, required=True)
    parser.add_argument("--controllers", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = build_report(
        load_baseline_traces(args.vision_source),
        load_live_shadow_traces(
            args.nonvision_run_dir,
            args.nonvision_metrics_dir,
        ),
        _read_json(args.controllers),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(args.output.resolve())
    print(json.dumps(report["acceptance"], indent=2))
    print(json.dumps(report["summary"], indent=2))
    if not report["acceptance"]["all_passed"]:
        raise SystemExit("RAPEC-G v4 offline replay acceptance failed.")


if __name__ == "__main__":
    main()
