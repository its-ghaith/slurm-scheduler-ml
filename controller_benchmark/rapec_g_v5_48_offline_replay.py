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
    replay_controller,
)


def _read_json(path: Path) -> Any:
    raw = path.read_bytes()
    encoding = "utf-16" if raw.startswith((b"\xff\xfe", b"\xfe\xff")) else "utf-8-sig"
    return json.loads(raw.decode(encoding))


def _energy_wh(row: dict[str, Any]) -> float:
    for key in ("epoch_energy_wh", "gpu_epoch_energy_wh"):
        if row.get(key) is not None:
            return max(0.0, float(row.get(key) or 0.0))
    for key in ("total_energy_kwh", "gpu_energy_kwh", "net_total_energy_kwh", "net_gpu_energy_kwh"):
        if row.get(key) is not None:
            return max(0.0, float(row.get(key) or 0.0) * 1000.0)
    return 0.0


def _normalize_epochs(epochs: list[dict[str, Any]]) -> tuple[dict[str, Any], ...]:
    normalized: list[dict[str, Any]] = []
    previous_total: float | None = None
    previous_gpu: float | None = None
    for raw in sorted(epochs, key=lambda row: int(row.get("epoch_index", row.get("epoch")))):
        row = dict(raw)
        energy = _energy_wh(row)
        if energy <= 0.0 and row.get("cumulative_total_energy_kwh") is not None:
            cumulative = float(row.get("cumulative_total_energy_kwh") or 0.0) * 1000.0
            energy = max(0.0, cumulative - (previous_total or 0.0))
        if energy <= 0.0 and row.get("cumulative_gpu_energy_kwh") is not None:
            cumulative = float(row.get("cumulative_gpu_energy_kwh") or 0.0) * 1000.0
            energy = max(0.0, cumulative - (previous_gpu or 0.0))
        row["epoch_energy_wh"] = energy
        row.setdefault("duration_seconds", 0.0)
        if row.get("cumulative_total_energy_kwh") is not None:
            previous_total = float(row.get("cumulative_total_energy_kwh") or 0.0) * 1000.0
        if row.get("cumulative_gpu_energy_kwh") is not None:
            previous_gpu = float(row.get("cumulative_gpu_energy_kwh") or 0.0) * 1000.0
        normalized.append(row)
    return tuple(normalized)


def _quality_metric(summary: dict[str, Any], epochs: tuple[dict[str, Any], ...]) -> str:
    metric = str(summary.get("quality_metric") or "quality_score")
    if epochs and metric in epochs[0]:
        return metric
    for candidate in ("quality_score", "map50_95", "macro_f1", "accuracy", "miou", "dice"):
        if epochs and candidate in epochs[0]:
            return candidate
    return metric


def _trace(run_dir: Path, record: dict[str, Any]) -> BaselineTrace | None:
    job_id = str(record.get("job_id") or "")
    case_id = str(record.get("case_id") or "")
    if not job_id or not case_id or record.get("state") != "COMPLETED":
        return None
    summary_path = run_dir / "metrics" / f"epoch_summary_job_{job_id}.json"
    if not summary_path.exists():
        return None
    summary = _read_json(summary_path)
    epochs = _normalize_epochs(summary.get("epochs") or [])
    if not epochs:
        return None
    metric = _quality_metric(summary, epochs)
    return BaselineTrace(
        case_id=case_id,
        stage=str(summary.get("benchmark_stage") or "unknown"),
        task_type=str(summary.get("task_type") or "unknown"),
        scenario=str(summary.get("scenario") or case_id),
        quality_metric=metric,
        max_epochs=max(int(row.get("epoch_index", row.get("epoch"))) for row in epochs),
        metadata={
            "benchmark_case_id": case_id,
            "benchmark_stage": summary.get("benchmark_stage") or "unknown",
            "source_job_id": job_id,
            "pretrained": "pretrained" in case_id,
        },
        epochs=epochs,
        source_job_id=job_id,
    )


def load_traces(root: Path, run_ids: list[str]) -> list[BaselineTrace]:
    traces: list[BaselineTrace] = []
    for run_id in run_ids:
        run_dir = root / run_id
        status = _read_json(run_dir / "status.json")
        for record in status.get("runs") or []:
            trace = _trace(run_dir, record)
            if trace is not None:
                traces.append(trace)
    return traces


def _controller(configuration: dict[str, Any], controller_id: str) -> dict[str, Any]:
    return next(item for item in configuration["controllers"] if item["id"] == controller_id)


def _replay(trace: BaselineTrace, controller: dict[str, Any]) -> ReplayResult:
    return replay_controller(
        trace,
        controller_plugin=controller["controller_plugin"],
        controller_id=controller["id"],
        parameters=controller["controller_parameters"],
    )


def _fmt_percent(value: float | None) -> str:
    return "n/a" if value is None else f"{value * 100.0:.2f} %".replace(".", ",")


def _fmt_pp(value: float | None) -> str:
    return "n/a" if value is None else f"{value * 100.0:.2f} pp".replace(".", ",")


def _fmt_wh(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.2f} Wh".replace(".", ",")


def _saving(base: float, value: float) -> float:
    return (base - value) / base if base > 0.0 else 0.0


def _public_row(trace: BaselineTrace, standard: ReplayResult, v4: ReplayResult, v5: ReplayResult) -> dict[str, Any]:
    full_quality = v5.baseline_best_quality
    full_energy = v5.baseline_energy_wh
    return {
        "Dataset": trace.case_id,
        "Domain": "pretrained" if "pretrained" in trace.case_id else "scratch",
        "Task": trace.task_type,
        "Qualität Full": _fmt_percent(full_quality),
        "Qualität ES": _fmt_percent(standard.stopped_best_quality),
        "Qualität V4": _fmt_percent(v4.stopped_best_quality),
        "Qualität V5": _fmt_percent(v5.stopped_best_quality),
        "V4 Verlust vs Full": _fmt_pp(max(0.0, full_quality - v4.stopped_best_quality)),
        "V4 Verlust vs ES": _fmt_pp(max(0.0, standard.stopped_best_quality - v4.stopped_best_quality)),
        "V5 Verlust vs Full": _fmt_pp(max(0.0, full_quality - v5.stopped_best_quality)),
        "V5 Verlust vs ES": _fmt_pp(max(0.0, standard.stopped_best_quality - v5.stopped_best_quality)),
        "Energie Full100": _fmt_wh(full_energy),
        "Energie ES": _fmt_wh(standard.stopped_energy_wh),
        "Energie V4": _fmt_wh(v4.stopped_energy_wh),
        "Energie V5": _fmt_wh(v5.stopped_energy_wh),
        "Stop ES": str(standard.stop_epoch),
        "Stop V4": str(v4.stop_epoch),
        "Stop V5": str(v5.stop_epoch),
        "V4 Ersparnis vs Full": _fmt_percent(_saving(full_energy, v4.stopped_energy_wh)),
        "V4 Ersparnis vs ES": _fmt_percent(_saving(standard.stopped_energy_wh, v4.stopped_energy_wh)),
        "V5 Ersparnis vs Full": _fmt_percent(_saving(full_energy, v5.stopped_energy_wh)),
        "V5 Ersparnis vs ES": _fmt_percent(_saving(standard.stopped_energy_wh, v5.stopped_energy_wh)),
        "_standard": standard.to_dict(),
        "_v4": v4.to_dict(),
        "_v5": v5.to_dict(),
    }


def _summary(rows: list[dict[str, Any]], controller_key: str) -> dict[str, Any]:
    if not rows:
        return {}
    results = [row[f"_{controller_key}"] for row in rows]
    standard = [row["_standard"] for row in rows]
    return {
        "cases": len(rows),
        "mean_energy_saving_vs_full100": statistics.mean(
            item["energy_saving_fraction"] for item in results
        ),
        "mean_energy_saving_vs_standard_es": statistics.mean(
            _saving(es["stopped_energy_wh"], item["stopped_energy_wh"])
            for es, item in zip(standard, results)
        ),
        "mean_quality_regret": statistics.mean(
            item["quality_regret"] for item in results
        ),
        "max_quality_regret": max(item["quality_regret"] for item in results),
        "cases_lower_energy_than_es": sum(
            item["stopped_energy_wh"] < es["stopped_energy_wh"]
            for es, item in zip(standard, results)
        ),
    }


def render_text(rows: list[dict[str, Any]]) -> str:
    columns = [
        "Dataset",
        "Domain",
        "Task",
        "Qualität Full",
        "Qualität ES",
        "Qualität V4",
        "Qualität V5",
        "V4 Verlust vs Full",
        "V4 Verlust vs ES",
        "V5 Verlust vs Full",
        "V5 Verlust vs ES",
        "Energie Full100",
        "Energie ES",
        "Energie V4",
        "Energie V5",
        "Stop ES",
        "Stop V4",
        "Stop V5",
        "V4 Ersparnis vs Full",
        "V4 Ersparnis vs ES",
        "V5 Ersparnis vs Full",
        "V5 Ersparnis vs ES",
    ]
    public = [{key: str(row[key]) for key in columns} for row in rows]
    widths = {
        key: max(len(key), *(len(row[key]) for row in public))
        for key in columns
    }
    lines = [
        "Offline-Replay Ergebnis-Tabelle: Standard ES vs RAPEC-G v4 vs RAPEC-G v5",
        f"Rows: {len(rows)}",
        "",
        " ".join(f"{key:<{widths[key]}}" for key in columns),
        " ".join("-" * widths[key] for key in columns),
    ]
    for row in public:
        lines.append(" ".join(f"{row[key]:<{widths[key]}}" for key in columns))
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--run-id", action="append", required=True)
    parser.add_argument("--controllers", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    configuration = _read_json(args.controllers)
    standard_controller = _controller(configuration, "standard-es-10")
    v4_controller = _controller(configuration, "rapec-g-v4")
    v5_controller = _controller(configuration, "rapec-g-v5")

    rows: list[dict[str, Any]] = []
    for trace in load_traces(args.source_root, args.run_id):
        standard = _replay(trace, standard_controller)
        v4 = _replay(trace, v4_controller)
        v5 = _replay(trace, v5_controller)
        rows.append(_public_row(trace, standard, v4, v5))

    rows.sort(key=lambda row: (row["Domain"], row["Task"], row["Dataset"]))
    public_rows = [
        {key: value for key, value in row.items() if not key.startswith("_")}
        for row in rows
    ]
    report = {
        "schema_version": "rapec-g-v5-48-offline-replay-v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "training_jobs_submitted": 0,
        "evaluation_mode": "offline_replay_of_recorded_full100_epoch_traces",
        "run_ids": args.run_id,
        "controllers": ["standard-es-10", "rapec-g-v4", "rapec-g-v5"],
        "summary": {
            "rapec_g_v4": _summary(rows, "v4"),
            "rapec_g_v5": _summary(rows, "v5"),
        },
        "rows": public_rows,
        "internal_rows": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    text_path = args.output.with_suffix(".txt")
    text_path.write_text(render_text(rows) + "\n", encoding="utf-8")
    print(args.output.resolve())
    print(text_path.resolve())
    print(json.dumps(report["summary"], indent=2))


if __name__ == "__main__":
    main()
