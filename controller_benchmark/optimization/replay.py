from __future__ import annotations

import json
import math
import statistics
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping

from controller_benchmark.api import ControllerContext, ControllerDecision, EpochObservation
from controller_benchmark.loader import load_controller

from .source_bundle import verify_source_lock


def _number(value: Any, default: float | None = None) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return default
    return result if math.isfinite(result) else default


def _median(values: Iterable[float]) -> float:
    clean = [value for value in values if math.isfinite(value)]
    return statistics.median(clean) if clean else 0.0


def _mad(values: Iterable[float]) -> float:
    clean = [value for value in values if math.isfinite(value)]
    if len(clean) < 2:
        return 0.0
    center = statistics.median(clean)
    return 1.4826 * statistics.median(abs(value - center) for value in clean)


def _energy_wh(row: Mapping[str, Any]) -> float:
    if row.get("epoch_energy_wh") is not None:
        return max(0.0, _number(row.get("epoch_energy_wh"), 0.0) or 0.0)
    value = _number(row.get("total_energy_kwh"), None)
    if value is None:
        value = _number(row.get("gpu_energy_kwh"), 0.0)
    return max(0.0, float(value or 0.0) * 1000.0)


@dataclass(frozen=True)
class BaselineTrace:
    case_id: str
    stage: str
    task_type: str
    scenario: str
    quality_metric: str
    max_epochs: int
    metadata: Mapping[str, Any]
    epochs: tuple[Mapping[str, Any], ...]
    source_job_id: str


@dataclass(frozen=True)
class ReplayResult:
    case_id: str
    stage: str
    task_type: str
    stop_epoch: int
    ideal_stop_epoch: int
    baseline_best_quality: float
    stopped_best_quality: float
    quality_regret: float
    dynamic_quality_tolerance: float
    normalized_quality_regret: float
    baseline_energy_wh: float
    stopped_energy_wh: float
    energy_saving_fraction: float
    normalized_stop_error: float
    stopped_early: bool
    stop_reason: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def load_baseline_traces(source_dir: str | Path) -> list[BaselineTrace]:
    root = Path(source_dir).resolve()
    verify_source_lock(root)
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8-sig"))
    catalog_root = root / "baseline-cache"
    catalog = json.loads((catalog_root / "catalog.json").read_text(encoding="utf-8-sig"))
    case_specs: dict[str, tuple[str, dict[str, Any]]] = {}
    for stage in manifest["stages"]:
        if not stage.get("enabled"):
            continue
        for case in stage["cases"]:
            case_specs[case["id"]] = (stage["id"], case)

    traces: list[BaselineTrace] = []
    for case_id in sorted(case_specs):
        stage_id, case = case_specs[case_id]
        entry = catalog["entries"][case_id]
        summary = json.loads(
            (catalog_root / entry["epoch_summary"]).read_text(encoding="utf-8-sig")
        )
        epochs = sorted(summary["epochs"], key=lambda row: int(row.get("epoch_index", row["epoch"])))
        traces.append(
            BaselineTrace(
                case_id=case_id,
                stage=stage_id,
                task_type=case["task_type"],
                scenario=case["scenario"],
                quality_metric=case["quality_metric"],
                max_epochs=int(manifest["execution"]["max_epochs"]),
                metadata={
                    **case,
                    "benchmark_case_id": case_id,
                    "benchmark_stage": stage_id,
                },
                epochs=tuple(epochs),
                source_job_id=str(entry["source_job_id"]),
            )
        )
    return traces


def dynamic_quality_tolerance(trace: BaselineTrace) -> dict[str, float]:
    values = [float(row[trace.quality_metric]) for row in trace.epochs]
    deltas = [right - left for left, right in zip(values, values[1:])]
    innovation_noise = _mad(deltas) / math.sqrt(2.0) if deltas else 0.0
    late_count = max(5, int(math.ceil(math.sqrt(len(values)))))
    late_values = values[-late_count:]
    late_uncertainty = _mad(late_values)
    tolerance = 1.96 * math.sqrt(innovation_noise**2 + late_uncertainty**2)
    return {
        "innovation_noise": innovation_noise,
        "late_stage_uncertainty": late_uncertainty,
        "dynamic_quality_tolerance": tolerance,
    }


def ideal_stop_epoch(trace: BaselineTrace, tolerance: float) -> int:
    values = [float(row[trace.quality_metric]) for row in trace.epochs]
    target = max(values) - tolerance
    best = -math.inf
    for row, value in zip(trace.epochs, values):
        best = max(best, value)
        if best >= target:
            return int(row.get("epoch_index", row["epoch"]))
    return trace.max_epochs


def replay_controller(
    trace: BaselineTrace,
    *,
    controller_plugin: str,
    controller_id: str,
    parameters: Mapping[str, Any],
    decision_observer: Callable[
        [EpochObservation, ControllerDecision, float], None
    ]
    | None = None,
) -> ReplayResult:
    context = ControllerContext(
        controller_id=controller_id,
        benchmark_version="offline-pareto-optimization-v1",
        task_type=trace.task_type,
        quality_metric=trace.quality_metric,
        scenario=trace.scenario,
        max_epochs=trace.max_epochs,
        parameters=dict(parameters),
        metadata=trace.metadata,
    )
    controller = load_controller(controller_plugin, context)
    history: list[Mapping[str, Any]] = []
    cumulative_energy = 0.0
    cumulative_duration = 0.0
    best_quality: float | None = None
    previous_quality: float | None = None
    stop_reason = "maximum_epochs_reached"
    stop_epoch = trace.max_epochs
    try:
        for raw in trace.epochs:
            event = dict(raw)
            epoch = int(event.get("epoch_index", event["epoch"]))
            quality = _number(event.get(trace.quality_metric), _number(event.get("quality_score")))
            if quality is None or not 0.0 <= quality <= 1.0:
                raise ValueError(f"Invalid quality in {trace.case_id}/epoch {epoch}: {quality}")
            best_quality = quality if best_quality is None else max(best_quality, quality)
            epoch_energy = _energy_wh(event)
            duration = _number(event.get("duration_seconds"), 0.0) or 0.0
            cumulative_energy += epoch_energy
            cumulative_duration += duration
            observation = EpochObservation(
                epoch=epoch,
                max_epochs=trace.max_epochs,
                task_type=trace.task_type,
                quality_metric=trace.quality_metric,
                quality=quality,
                best_quality=best_quality,
                delta_quality=(quality - previous_quality) if previous_quality is not None else None,
                epoch_energy_wh=epoch_energy,
                cumulative_energy_wh=cumulative_energy,
                epoch_duration_seconds=duration,
                cumulative_duration_seconds=cumulative_duration,
                gpu_utilization_pct=_number(event.get("gpu_util_avg_pct")),
                history=tuple(history),
                raw_metrics=event,
            )
            decision_started = time.perf_counter()
            decision = controller.evaluate(observation)
            decision_seconds = time.perf_counter() - decision_started
            if decision_observer is not None:
                decision_observer(observation, decision, decision_seconds)
            history.append(event)
            previous_quality = quality
            if decision.stop:
                stop_epoch = epoch
                stop_reason = decision.reason
                break
    finally:
        controller.close()

    baseline_quality = max(float(row[trace.quality_metric]) for row in trace.epochs)
    stopped_quality = max(float(row[trace.quality_metric]) for row in history)
    baseline_energy = sum(_energy_wh(row) for row in trace.epochs)
    stopped_energy = sum(_energy_wh(row) for row in history)
    regret = max(0.0, baseline_quality - stopped_quality)
    tolerance_info = dynamic_quality_tolerance(trace)
    tolerance = tolerance_info["dynamic_quality_tolerance"]
    denominator = max(tolerance, 1e-12)
    ideal_epoch = ideal_stop_epoch(trace, tolerance)
    return ReplayResult(
        case_id=trace.case_id,
        stage=trace.stage,
        task_type=trace.task_type,
        stop_epoch=stop_epoch,
        ideal_stop_epoch=ideal_epoch,
        baseline_best_quality=baseline_quality,
        stopped_best_quality=stopped_quality,
        quality_regret=regret,
        dynamic_quality_tolerance=tolerance,
        normalized_quality_regret=min(regret / denominator, 1_000_000.0),
        baseline_energy_wh=baseline_energy,
        stopped_energy_wh=stopped_energy,
        energy_saving_fraction=(baseline_energy - stopped_energy) / baseline_energy,
        normalized_stop_error=abs(stop_epoch - ideal_epoch) / trace.max_epochs,
        stopped_early=stop_epoch < trace.max_epochs,
        stop_reason=stop_reason,
    )
