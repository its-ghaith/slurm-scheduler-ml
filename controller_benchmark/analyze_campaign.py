from __future__ import annotations

import argparse
import csv
import json
import tempfile
from pathlib import Path
from typing import Any

from .analysis import analyze_snapshot, write_analysis, write_prometheus


TEST_QUALITY_KEYS = {
    "object_detection": "test_best_map50_95",
    "image_classification": "test_accuracy",
    "semantic_segmentation": "test_miou",
}


def _subplan(plan: dict[str, Any], controller: dict[str, Any]) -> dict[str, Any]:
    baseline_id = plan["baseline"]["id"]
    selected_ids = {baseline_id, controller["id"]}
    result = dict(plan)
    result["controller"] = controller
    result["runs"] = [row for row in plan["runs"] if row["controller_id"] in selected_ids]
    result["planned_candidate_runs"] = sum(
        row["controller_id"] == controller["id"] for row in result["runs"]
    )
    return result


def _add_controller_label(text: str, controller_id: str) -> str:
    output: list[str] = []
    for line in text.splitlines():
        if not line or line.startswith("#") or "{" not in line:
            output.append(line)
            continue
        metric, remainder = line.split("{", 1)
        labels, value = remainder.rsplit("}", 1)
        if 'controller_id="' not in labels:
            labels = f'{labels},controller_id="{controller_id}"'
        output.append(f"{metric}{{{labels}}}{value}")
    return "\n".join(output).rstrip() + "\n"


def _summary_by_job(snapshot: Path) -> dict[str, dict[str, Any]]:
    metrics = snapshot / "data" / "energy_metrics"
    summaries = {}
    for path in metrics.glob("epoch_summary_job_*.json"):
        summary = json.loads(path.read_text(encoding="utf-8-sig"))
        summaries[str(summary.get("job_id"))] = summary
    return summaries


def _add_test_quality(report: dict[str, Any], summaries: dict[str, dict[str, Any]]) -> None:
    for pair in report["pairs"]:
        metric = TEST_QUALITY_KEYS.get(pair["task_type"])
        if metric is None:
            continue
        baseline = summaries.get(str(pair["baseline_job_id"]), {})
        candidate = summaries.get(str(pair["candidate_job_id"]), {})
        baseline_value = baseline.get(metric)
        candidate_value = candidate.get(metric)
        if baseline_value is None or candidate_value is None:
            continue
        pair["test_quality_metric"] = metric
        pair["baseline_test_quality"] = float(baseline_value)
        pair["candidate_test_quality"] = float(candidate_value)
        pair["test_quality_regret"] = float(baseline_value) - float(candidate_value)


def _test_quality_prometheus(report: dict[str, Any], controller_id: str) -> str:
    lines = []
    for pair in report["pairs"]:
        if "baseline_test_quality" not in pair:
            continue
        labels = ",".join(
            [
                f'benchmark_run_id="{pair["benchmark_run_id"]}"',
                f'benchmark_version="{pair["benchmark_version"]}"',
                f'controller_id="{controller_id}"',
                f'stage="{pair["stage"]}"',
                f'case_id="{pair["case_id"]}"',
                f'task_type="{pair["task_type"]}"',
                f'test_quality_metric="{pair["test_quality_metric"]}"',
            ]
        )
        lines.extend(
            [
                f"controller_campaign_baseline_test_quality{{{labels}}} {pair['baseline_test_quality']:.12g}",
                f"controller_campaign_candidate_test_quality{{{labels}}} {pair['candidate_test_quality']:.12g}",
                f"controller_campaign_test_quality_regret{{{labels}}} {pair['test_quality_regret']:.12g}",
            ]
        )
    return "\n".join(lines) + ("\n" if lines else "")


def analyze_campaign(snapshot: Path, plan_path: Path) -> dict[str, Any]:
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    if plan.get("plan_type") != "multi_controller_campaign":
        raise ValueError("The supplied plan is not a multi-controller campaign.")
    output = snapshot / "controller_benchmark"
    output.mkdir(parents=True, exist_ok=True)
    reports: dict[str, Any] = {}
    combined_pairs: list[dict[str, Any]] = []
    prometheus_parts: list[str] = []
    summaries = _summary_by_job(snapshot)

    for controller in plan["controllers"]:
        controller_id = controller["id"]
        with tempfile.TemporaryDirectory() as temporary:
            subplan_path = Path(temporary) / "run-matrix.json"
            subplan_path.write_text(
                json.dumps(_subplan(plan, controller), ensure_ascii=False),
                encoding="utf-8",
            )
            report = analyze_snapshot(snapshot, subplan_path)
            _add_test_quality(report, summaries)
            controller_output = output / controller_id
            write_analysis(report, controller_output)
            prom_path = controller_output / "controller_benchmark.prom"
            write_prometheus(report, prom_path)
            prometheus_parts.append(
                _add_controller_label(prom_path.read_text(encoding="utf-8"), controller_id)
            )
            prometheus_parts.append(_test_quality_prometheus(report, controller_id))
        reports[controller_id] = report
        combined_pairs.extend(
            {"controller_id": controller_id, **pair}
            for pair in report["pairs"]
        )

    combined = {
        "benchmark_run_id": plan["benchmark_run_id"],
        "benchmark_version": plan["benchmark_version"],
        "campaign_id": plan["campaign_id"],
        "controllers": reports,
        "pairs": combined_pairs,
    }
    (output / "benchmark-report.json").write_text(
        json.dumps(combined, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    with (output / "paired-results.csv").open("w", encoding="utf-8", newline="") as handle:
        flattened = [
            {key: value for key, value in pair.items() if key not in {"baseline_phases", "candidate_phases"}}
            for pair in combined_pairs
        ]
        writer = csv.DictWriter(handle, fieldnames=list(flattened[0]))
        writer.writeheader()
        writer.writerows(flattened)
    prom_path = snapshot / "data" / "energy_metrics" / "node_exporter" / "controller_benchmark.prom"
    prom_path.parent.mkdir(parents=True, exist_ok=True)
    prom_path.write_text("".join(prometheus_parts), encoding="utf-8", newline="\n")
    return combined


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--plan", type=Path, required=True)
    args = parser.parse_args()
    result = analyze_campaign(args.snapshot.resolve(), args.plan.resolve())
    print(json.dumps({"benchmark_run_id": result["benchmark_run_id"], "controllers": list(result["controllers"])}, indent=2))


if __name__ == "__main__":
    main()
