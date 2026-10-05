from __future__ import annotations

import argparse
import json
from pathlib import Path

from controller_benchmark.case_labels import case_metadata


COMMON_RAW_METRICS = (
    "accuracy",
    "macro_f1",
    "map50",
    "map50_95",
    "precision",
    "recall",
    "miou",
    "dice",
    "pixel_accuracy",
    "mae",
    "rmse",
)


def escape(value: object) -> str:
    return str(value).replace("\\", "\\\\").replace('"', '\\"')


def labels(values: dict[str, object]) -> str:
    return ",".join(f'{key}="{escape(value)}"' for key, value in values.items())


def finite_number(value: object) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if number == number and abs(number) != float("inf") else None


def build(snapshot: Path) -> str:
    metrics_dir = snapshot / "data" / "energy_metrics"
    summaries = sorted(metrics_dir.glob("epoch_summary_job_*.json"))
    if not summaries:
        raise FileNotFoundError(f"No epoch summaries found below {metrics_dir}")

    lines = [
        "# Compact thesis export derived from immutable Full100 epoch summaries.",
        "# HELP thesis_live_shadow_epoch_gpu_energy_wh GPU energy consumed during one Full100 epoch.",
        "# TYPE thesis_live_shadow_epoch_gpu_energy_wh gauge",
        "# HELP thesis_live_shadow_epoch_quality_score Task-independent quality score in [0,1].",
        "# TYPE thesis_live_shadow_epoch_quality_score gauge",
        "# HELP thesis_live_shadow_epoch_raw_quality Raw task-specific quality value.",
        "# TYPE thesis_live_shadow_epoch_raw_quality gauge",
    ]
    seen: set[tuple[str, str, str]] = set()
    for summary_path in summaries:
        data = json.loads(summary_path.read_text(encoding="utf-8"))
        case_id = data.get("benchmark_case_id", "none")
        case_code, case_name, case_order = case_metadata(case_id)
        base = {
            "benchmark_run_id": data.get("benchmark_run_id", "none"),
            "benchmark_version": data.get("benchmark_version", "unversioned"),
            "case_id": case_id,
            "case_code": case_code,
            "case_name": case_name,
            "case_order": case_order,
            "task_type": data.get("task_type", "unknown"),
            "stage": data.get("benchmark_stage", "unknown"),
            "job_id": data.get("job_id", "unknown"),
        }
        for row in data.get("epochs", []):
            epoch = str(row.get("epoch", row.get("epoch_index", "unknown")))
            common = {**base, "epoch": epoch}
            energy_kwh = finite_number(row.get("gpu_energy_kwh"))
            quality = finite_number(row.get("quality_score"))
            if energy_kwh is not None:
                lines.append(
                    f"thesis_live_shadow_epoch_gpu_energy_wh{{{labels(common)}}} {energy_kwh * 1000:.12g}"
                )
            if quality is not None:
                lines.append(
                    f"thesis_live_shadow_epoch_quality_score{{{labels(common)}}} {quality:.12g}"
                )

            raw_metric = row.get("raw_quality_metric")
            raw_value = finite_number(row.get("raw_quality_value"))
            if raw_metric and raw_value is not None:
                key = (str(base["case_id"]), epoch, str(raw_metric))
                seen.add(key)
                lines.append(
                    "thesis_live_shadow_epoch_raw_quality"
                    f'{{{labels({**common, "raw_quality_metric": raw_metric})}}} {raw_value:.12g}'
                )

            for metric in COMMON_RAW_METRICS:
                value = finite_number(row.get(metric))
                key = (str(base["case_id"]), epoch, metric)
                if value is None or key in seen:
                    continue
                lines.append(
                    "thesis_live_shadow_epoch_raw_quality"
                    f'{{{labels({**common, "raw_quality_metric": metric})}}} {value:.12g}'
                )
                seen.add(key)
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    content = build(args.snapshot)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(content, encoding="utf-8", newline="\n")
    print(f"Wrote {args.output} ({content.count(chr(10))} lines)")


if __name__ == "__main__":
    main()
