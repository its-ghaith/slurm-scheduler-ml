from __future__ import annotations

import json
import math
import time
from typing import Any

from .api import ControllerContext, ControllerDecision, EpochObservation
from .loader import load_controller


def _number(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


class ControllerPluginRuntime:
    """Adapter from generic epoch dictionaries to the stable controller API."""

    def __init__(
        self,
        *,
        plugin_path: str,
        parameters_json: str,
        controller_id: str,
        benchmark_version: str,
        task_type: str,
        quality_metric: str,
        scenario: str,
        max_epochs: int,
        metadata: dict[str, Any] | None = None,
    ):
        parameters = json.loads(parameters_json or "{}")
        if not isinstance(parameters, dict):
            raise ValueError("Controller parameters must be a JSON object.")
        context = ControllerContext(
            controller_id=controller_id,
            benchmark_version=benchmark_version,
            task_type=task_type,
            quality_metric=quality_metric,
            scenario=scenario,
            max_epochs=max_epochs,
            parameters=parameters,
            metadata=metadata or {},
        )
        self.plugin_path = plugin_path
        self.controller = load_controller(plugin_path, context)

    def evaluate(self, event: dict[str, Any], history: list[dict[str, Any]]) -> tuple[bool, str | None, dict]:
        quality_metric = self.controller.context.quality_metric
        quality = _number(event.get(quality_metric))
        if quality is None or not 0.0 <= quality <= 1.0:
            raise ValueError(f"{quality_metric} must be a finite value in [0, 1], got {quality!r}.")
        best_quality = _number(event.get(f"best_{quality_metric}"))
        if best_quality is None:
            values = [_number(row.get(quality_metric)) for row in [*history, event]]
            best_quality = max((value for value in values if value is not None), default=None)
        previous_quality = _number(history[-1].get(quality_metric)) if history else None
        delta_quality = quality - previous_quality if quality is not None and previous_quality is not None else None
        epoch_energy_wh = _number(event.get("interval_energy_wh"))
        if epoch_energy_wh is None:
            epoch_energy_wh = (_number(event.get("total_energy_kwh")) or 0.0) * 1000.0
        cumulative_energy_wh = _number(event.get("cumulative_energy_wh"))
        if cumulative_energy_wh is None:
            cumulative_energy_wh = (_number(event.get("cumulative_total_energy_kwh")) or 0.0) * 1000.0
        started = time.perf_counter()
        decision = self.controller.evaluate(
            EpochObservation(
                epoch=int(
                    event.get(
                        "decision_checkpoint",
                        event.get("epoch_index", event.get("epoch", len(history) + 1)),
                    )
                ),
                max_epochs=self.controller.context.max_epochs,
                task_type=self.controller.context.task_type,
                quality_metric=quality_metric,
                quality=quality,
                best_quality=best_quality,
                delta_quality=delta_quality,
                epoch_energy_wh=epoch_energy_wh,
                cumulative_energy_wh=cumulative_energy_wh,
                epoch_duration_seconds=_number(event.get("duration_seconds")) or 0.0,
                cumulative_duration_seconds=sum((_number(row.get("duration_seconds")) or 0.0) for row in history)
                + (_number(event.get("duration_seconds")) or 0.0),
                gpu_utilization_pct=_number(event.get("gpu_util_avg_pct")),
                history=tuple(history),
                raw_metrics=event,
            )
        )
        if not isinstance(decision, ControllerDecision):
            raise TypeError(f"{self.plugin_path}.evaluate() must return ControllerDecision")
        compute_seconds = time.perf_counter() - started
        diagnostics = {
            "controller_plugin_confidence": decision.confidence,
            "controller_plugin_predicted_energy_saving_fraction": decision.predicted_energy_saving_fraction,
            "controller_plugin_predicted_quality_regret": decision.predicted_quality_regret,
            "controller_plugin_compute_seconds": compute_seconds,
            "controller_decision": "stop" if decision.stop else "continue",
            "controller_decision_state": 2 if decision.stop else 0,
        }
        for key, value in decision.diagnostics.items():
            if value is not None and not isinstance(value, (bool, int, float)):
                raise TypeError(f"Controller diagnostic '{key}' must be numeric, boolean, or null.")
            diagnostics[f"controller_plugin_{key}"] = value
        reason = decision.reason if decision.stop else None
        return decision.stop, reason, diagnostics

    def close(self) -> None:
        self.controller.close()
