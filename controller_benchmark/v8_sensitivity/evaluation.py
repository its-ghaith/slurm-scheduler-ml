from __future__ import annotations

import math
import statistics
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from controller_benchmark.api import ControllerDecision, EpochObservation
from controller_benchmark.optimization.replay import (
    BaselineTrace,
    ReplayResult,
    replay_controller,
)
from controller_benchmark.scientific_optimization.metrics import (
    calibrated_tolerances,
    trace_group,
)


METRIC_NAMES = (
    "mean_energy_saving_fraction",
    "mean_quality_regret_pp",
    "mean_stop_epoch",
    "mean_stop_epoch_fraction",
    "noninferiority_success_fraction",
    "joint_success_fraction",
    "bayesian_veto_rate",
    "controller_runtime_ms_per_epoch",
    "energy_saving_q25",
    "quality_cvar90_normalized",
)


def _quantile(values: Sequence[float], probability: float) -> float:
    clean = sorted(float(value) for value in values if math.isfinite(float(value)))
    if not clean:
        return 0.0
    if len(clean) == 1:
        return clean[0]
    position = max(0.0, min(1.0, probability)) * (len(clean) - 1)
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return clean[lower]
    weight = position - lower
    return clean[lower] * (1.0 - weight) + clean[upper] * weight


def _cvar90(values: Sequence[float]) -> float:
    clean = sorted(
        (float(value) for value in values if math.isfinite(float(value))),
        reverse=True,
    )
    if not clean:
        return 0.0
    count = max(1, int(math.ceil(0.10 * len(clean))))
    return statistics.fmean(clean[:count])


@dataclass
class ReplayDiagnostics:
    decision_count: int = 0
    decision_seconds: float = 0.0
    bayesian_veto_count: int = 0

    def observe(
        self,
        observation: EpochObservation,
        decision: ControllerDecision,
        decision_seconds: float,
    ) -> None:
        del observation
        self.decision_count += 1
        self.decision_seconds += max(0.0, float(decision_seconds))
        self.bayesian_veto_count += int(
            bool(decision.diagnostics.get("rapec8_bayesian_veto", 0))
        )


@dataclass(frozen=True)
class EvaluatedTrace:
    trace: BaselineTrace
    result: ReplayResult
    diagnostics: ReplayDiagnostics


def _scope_metrics(
    evaluated: Sequence[EvaluatedTrace],
    tolerances: Mapping[str, float],
) -> dict[str, float]:
    if not evaluated:
        return {name: 0.0 for name in METRIC_NAMES}

    grouped: dict[str, list[EvaluatedTrace]] = {}
    for item in evaluated:
        grouped.setdefault(trace_group(item.trace), []).append(item)

    group_energy: list[float] = []
    group_regret: list[float] = []
    group_stop_epoch: list[float] = []
    group_stop_fraction: list[float] = []
    group_noninferiority: list[float] = []
    group_joint: list[float] = []
    group_normalized_regret: list[float] = []
    group_veto_rate: list[float] = []
    group_runtime_ms: list[float] = []
    for group, replicas in grouped.items():
        tolerance = max(float(tolerances[group]), 1e-12)
        energy = statistics.fmean(
            item.result.energy_saving_fraction for item in replicas
        )
        regret = statistics.fmean(item.result.quality_regret for item in replicas)
        group_energy.append(energy)
        group_regret.append(regret)
        group_stop_epoch.append(
            statistics.fmean(item.result.stop_epoch for item in replicas)
        )
        group_stop_fraction.append(
            statistics.fmean(
                item.result.stop_epoch / max(1, item.trace.max_epochs)
                for item in replicas
            )
        )
        group_noninferiority.append(float(regret <= tolerance))
        group_joint.append(float(regret <= tolerance and energy > 0.0))
        group_normalized_regret.append(regret / tolerance)
        decisions = sum(item.diagnostics.decision_count for item in replicas)
        decision_seconds = sum(item.diagnostics.decision_seconds for item in replicas)
        vetoes = sum(item.diagnostics.bayesian_veto_count for item in replicas)
        group_veto_rate.append(vetoes / max(1, decisions))
        group_runtime_ms.append(1000.0 * decision_seconds / max(1, decisions))
    return {
        "mean_energy_saving_fraction": statistics.fmean(group_energy),
        "mean_quality_regret_pp": 100.0 * statistics.fmean(group_regret),
        "mean_stop_epoch": statistics.fmean(group_stop_epoch),
        "mean_stop_epoch_fraction": statistics.fmean(group_stop_fraction),
        "noninferiority_success_fraction": statistics.fmean(group_noninferiority),
        "joint_success_fraction": statistics.fmean(group_joint),
        "bayesian_veto_rate": statistics.fmean(group_veto_rate),
        "controller_runtime_ms_per_epoch": statistics.fmean(group_runtime_ms),
        "energy_saving_q25": _quantile(group_energy, 0.25),
        "quality_cvar90_normalized": _cvar90(group_normalized_regret),
    }


class V8ReplayEvaluator:
    """Evaluate one V8 parameter vector on immutable Full100 traces."""

    def __init__(
        self,
        traces: Sequence[BaselineTrace],
        *,
        controller_plugin: str,
        controller_id: str,
    ) -> None:
        if not traces:
            raise ValueError("At least one Full100 trace is required")
        self.traces = tuple(traces)
        self.controller_plugin = controller_plugin
        self.controller_id = controller_id
        self.tolerances = calibrated_tolerances(self.traces)
        self.task_types = tuple(sorted({trace.task_type for trace in self.traces}))

    @property
    def output_names(self) -> tuple[str, ...]:
        scopes = ("overall", *self.task_types)
        return tuple(
            f"{scope}__{metric}" for scope in scopes for metric in METRIC_NAMES
        )

    def evaluate(self, parameters: Mapping[str, Any]) -> dict[str, float]:
        evaluated: list[EvaluatedTrace] = []
        for trace in self.traces:
            diagnostics = ReplayDiagnostics()
            result = replay_controller(
                trace,
                controller_plugin=self.controller_plugin,
                controller_id=self.controller_id,
                parameters=parameters,
                decision_observer=diagnostics.observe,
            )
            evaluated.append(EvaluatedTrace(trace, result, diagnostics))

        outputs: dict[str, float] = {}
        scopes: dict[str, Sequence[EvaluatedTrace]] = {"overall": evaluated}
        scopes.update(
            {
                task_type: [
                    item for item in evaluated if item.trace.task_type == task_type
                ]
                for task_type in self.task_types
            }
        )
        for scope, values in scopes.items():
            for metric, value in _scope_metrics(values, self.tolerances).items():
                outputs[f"{scope}__{metric}"] = float(value)
        return outputs
