from __future__ import annotations

import argparse
import json
import statistics
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from controller_benchmark.optimization.replay import (
    BaselineTrace,
    ReplayResult,
    load_baseline_traces,
    replay_controller,
)


def _read_json(path: Path) -> Any:
    raw = path.read_bytes()
    encoding = "utf-16" if raw.startswith((b"\xff\xfe", b"\xfe\xff")) else "utf-8-sig"
    return json.loads(raw.decode(encoding))


def load_live_shadow_traces(
    run_dir: Path,
    metrics_dir: Path,
) -> list[BaselineTrace]:
    matrix = _read_json(run_dir / "run-matrix.json")
    status = _read_json(metrics_dir / "status.json")
    jobs = {
        item["case_id"]: str(item["job_id"])
        for item in status["runs"]
        if item.get("job_id")
    }
    traces: list[BaselineTrace] = []
    for run in matrix["runs"]:
        case_id = run["benchmark_case_id"]
        if case_id not in jobs:
            raise RuntimeError(f"No completed job found for {case_id}.")
        summary = _read_json(
            metrics_dir / f"epoch_summary_job_{jobs[case_id]}.json"
        )
        epochs = tuple(
            sorted(
                summary["epochs"],
                key=lambda row: int(row.get("epoch_index", row["epoch"])),
            )
        )
        traces.append(
            BaselineTrace(
                case_id=case_id,
                stage=run["benchmark_stage"],
                task_type=run["task_type"],
                scenario=run["scenario"],
                quality_metric=run["quality_metric"],
                max_epochs=max(
                    int(row.get("epoch_index", row["epoch"]))
                    for row in epochs
                ),
                metadata=run.get("case", {}),
                epochs=epochs,
                source_job_id=jobs[case_id],
            )
        )
    return traces


def _controller(configuration: dict[str, Any], controller_id: str) -> dict[str, Any]:
    return next(
        item
        for item in configuration["controllers"]
        if item["id"] == controller_id
    )


def _replay(
    trace: BaselineTrace,
    controller: dict[str, Any],
) -> ReplayResult:
    return replay_controller(
        trace,
        controller_plugin=controller["controller_plugin"],
        controller_id=controller["id"],
        parameters=controller["controller_parameters"],
    )


def _summary(
    triples: list[tuple[ReplayResult, ReplayResult, ReplayResult]],
) -> dict[str, Any]:
    def metrics(index: int) -> dict[str, float]:
        results = [triple[index] for triple in triples]
        return {
            "mean_energy_saving_fraction": statistics.mean(
                item.energy_saving_fraction for item in results
            ),
            "mean_quality_regret": statistics.mean(
                item.quality_regret for item in results
            ),
            "maximum_quality_regret": max(
                item.quality_regret for item in results
            ),
            "joint_success_fraction": sum(
                item.energy_saving_fraction > 0.0
                and item.quality_regret <= 0.10
                for item in results
            )
            / len(results),
        }

    standard_results = [triple[0] for triple in triples]
    v3_results = [triple[2] for triple in triples]
    direct_savings = [
        (standard.stopped_energy_wh - v3.stopped_energy_wh)
        / standard.stopped_energy_wh
        for standard, v3 in zip(standard_results, v3_results)
        if standard.stopped_energy_wh > 0.0
    ]
    total_standard_energy = sum(
        item.stopped_energy_wh for item in standard_results
    )
    total_v3_energy = sum(item.stopped_energy_wh for item in v3_results)
    return {
        "cases": len(triples),
        "standard_es_10": metrics(0),
        "rapec_g_v2": metrics(1),
        "rapec_g_v3": metrics(2),
        "rapec_g_v3_vs_standard_es": {
            "mean_direct_energy_saving_fraction": statistics.mean(
                direct_savings
            ),
            "median_direct_energy_saving_fraction": statistics.median(
                direct_savings
            ),
            "aggregate_direct_energy_saving_fraction": (
                (total_standard_energy - total_v3_energy)
                / total_standard_energy
                if total_standard_energy > 0.0
                else 0.0
            ),
            "cases_with_lower_energy": sum(
                saving > 0.0 for saving in direct_savings
            ),
            "cases_with_equal_energy": sum(
                abs(saving) <= 1e-12 for saving in direct_savings
            ),
            "cases_with_higher_energy": sum(
                saving < 0.0 for saving in direct_savings
            ),
            "mean_quality_regret_difference": statistics.mean(
                v3.quality_regret - standard.quality_regret
                for standard, v3 in zip(standard_results, v3_results)
            ),
        },
        "changed_stop_epochs": sum(
            old.stop_epoch != new.stop_epoch for _, old, new in triples
        ),
        "mean_additional_energy_saving_fraction": statistics.mean(
            right.energy_saving_fraction - left.energy_saving_fraction
            for _, left, right in triples
        ),
        "mean_added_quality_regret": statistics.mean(
            right.quality_regret - left.quality_regret
            for _, left, right in triples
        ),
    }


def build_report(
    vision_traces: list[BaselineTrace],
    nonvision_traces: list[BaselineTrace],
    configuration: dict[str, Any],
) -> dict[str, Any]:
    v2 = _controller(configuration, "rapec-g-v2")
    v3 = _controller(configuration, "rapec-g-v3")
    standard = _controller(configuration, "standard-es-10")
    groups = {"computer_vision": vision_traces, "non_vision": nonvision_traces}
    report_rows: list[dict[str, Any]] = []
    group_triples: dict[
        str,
        list[tuple[ReplayResult, ReplayResult, ReplayResult]],
    ] = {}
    for group, traces in groups.items():
        triples = []
        for trace in traces:
            baseline = _replay(trace, standard)
            old = _replay(trace, v2)
            new = _replay(trace, v3)
            triples.append((baseline, old, new))
            direct_saving = (
                (baseline.stopped_energy_wh - new.stopped_energy_wh)
                / baseline.stopped_energy_wh
                if baseline.stopped_energy_wh > 0.0
                else 0.0
            )
            report_rows.append(
                {
                    "domain": group,
                    "case_id": trace.case_id,
                    "task_type": trace.task_type,
                    "standard_es_10": baseline.to_dict(),
                    "rapec_g_v2": old.to_dict(),
                    "rapec_g_v3": new.to_dict(),
                    "rapec_g_v3_direct_energy_saving_vs_standard_es_fraction": (
                        direct_saving
                    ),
                    "rapec_g_v3_quality_regret_difference_vs_standard_es": (
                        new.quality_regret - baseline.quality_regret
                    ),
                    "additional_energy_saving_fraction": (
                        new.energy_saving_fraction - old.energy_saving_fraction
                    ),
                    "added_quality_regret": (
                        new.quality_regret - old.quality_regret
                    ),
                }
            )
        group_triples[group] = triples
    all_triples = [
        triple for triples in group_triples.values() for triple in triples
    ]
    cv_triples = group_triples["computer_vision"]
    return {
        "schema_version": "rapec-g-v3-offline-replay-v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "acceptance": {
            "maximum_quality_regret": 0.10,
            "requires_positive_energy_saving": True,
            "cv_stop_epoch_regression_allowed": False,
            "direct_standard_es_comparison_formula": (
                "(E_standard_es - E_rapec_g_v3) / E_standard_es"
            ),
        },
        "cv_regression_passed": all(
            old.stop_epoch == new.stop_epoch
            and abs(old.quality_regret - new.quality_regret) <= 1e-12
            and abs(old.energy_saving_fraction - new.energy_saving_fraction)
            <= 1e-12
            for _, old, new in cv_triples
        ),
        "summary": {
            "computer_vision": _summary(cv_triples),
            "non_vision": _summary(group_triples["non_vision"]),
            "all": _summary(all_triples),
        },
        "rows": report_rows,
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
    print(json.dumps(report["summary"], indent=2))
    if not report["cv_regression_passed"]:
        raise SystemExit("RAPEC-G v3 changed at least one CV replay result.")


if __name__ == "__main__":
    main()
