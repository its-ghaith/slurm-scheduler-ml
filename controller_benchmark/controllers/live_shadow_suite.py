from __future__ import annotations

import json
import math
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from controller_benchmark.api import (
    Controller,
    ControllerContext,
    ControllerDecision,
    EpochObservation,
)
from controller_benchmark.loader import load_controller


def _finite(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _slug(value: str) -> str:
    return "".join(char if char.isalnum() else "_" for char in value.lower()).strip("_")


class ControllerEnergyMeter:
    """Attribute controller CPU energy using RAPL with a documented fallback."""

    def __init__(self, fallback_watts_per_cpu_second: float = 5.0):
        self.fallback_watts_per_cpu_second = max(0.0, float(fallback_watts_per_cpu_second))
        self.rapl_domains = self._discover_rapl_domains()

    @staticmethod
    def _discover_rapl_domains() -> list[tuple[Path, float]]:
        root = Path("/sys/class/powercap")
        domains: list[tuple[Path, float]] = []
        if not root.exists():
            return domains
        # Only package-level domains are summed; nested DRAM/core domains would double count.
        for path in sorted(root.glob("intel-rapl:[0-9]/energy_uj")):
            try:
                maximum = float((path.parent / "max_energy_range_uj").read_text().strip())
                float(path.read_text().strip())
            except (OSError, ValueError):
                continue
            domains.append((path, maximum))
        return domains

    def snapshot(self) -> dict[str, float]:
        values: dict[str, float] = {}
        for path, _ in self.rapl_domains:
            try:
                values[str(path)] = float(path.read_text().strip())
            except (OSError, ValueError):
                return {}
        return values

    def energy_wh(
        self,
        before: dict[str, float],
        after: dict[str, float],
        process_cpu_seconds: float,
    ) -> tuple[float, str]:
        if before and after and self.rapl_domains:
            total_uj = 0.0
            for path, maximum in self.rapl_domains:
                key = str(path)
                if key not in before or key not in after:
                    break
                delta = after[key] - before[key]
                if delta < 0:
                    delta += maximum
                total_uj += max(0.0, delta)
            else:
                return total_uj / 3_600_000_000.0, "intel_rapl"
        estimated = (
            max(0.0, process_cpu_seconds)
            * self.fallback_watts_per_cpu_second
            / 3600.0
        )
        return estimated, "cpu_time_power_model"


@dataclass
class ShadowControllerState:
    controller_id: str
    plugin_path: str
    controller: Controller
    stopped: bool = False
    failed: bool = False
    stop_epoch: int | None = None
    stop_reason: str | None = None
    best_quality_at_stop: float | None = None
    training_energy_to_stop_wh: float | None = None
    training_time_to_stop_s: float | None = None
    controller_overhead_energy_wh: float = 0.0
    controller_compute_seconds: float = 0.0
    controller_process_cpu_seconds: float = 0.0
    decisions: int = 0
    energy_measurement_method: str = "unavailable"
    predicted_energy_saving_fraction: float | None = None
    predicted_quality_regret: float | None = None
    confidence: float | None = None
    error: str | None = None
    last_diagnostics: dict[str, Any] = field(default_factory=dict)

    @property
    def counterfactual_total_energy_wh(self) -> float | None:
        if self.training_energy_to_stop_wh is None:
            return None
        return self.training_energy_to_stop_wh + self.controller_overhead_energy_wh


class LiveShadowSuiteController(Controller):
    """Run multiple stop policies live while deliberately keeping Full100 alive."""

    def __init__(self, context: ControllerContext):
        super().__init__(context)
        definitions = context.parameters.get("controllers", [])
        if not isinstance(definitions, list) or not definitions:
            raise ValueError("Live shadow suite requires a non-empty controllers list.")
        fallback_power = float(
            context.parameters.get(
                "fallback_watts_per_cpu_second",
                os.environ.get("SHADOW_CONTROLLER_WATTS_PER_CPU_SECOND", 5.0),
            )
        )
        self.energy_meter = ControllerEnergyMeter(fallback_power)
        self.states: list[ShadowControllerState] = []
        seen: set[str] = set()
        for definition in definitions:
            controller_id = str(definition.get("id", "")).strip()
            plugin_path = str(definition.get("controller_plugin", "")).strip()
            if not controller_id or controller_id in seen:
                raise ValueError(f"Invalid or duplicate shadow controller id: {controller_id!r}")
            if not plugin_path:
                raise ValueError(f"Shadow controller {controller_id!r} has no plugin path.")
            seen.add(controller_id)
            child_context = ControllerContext(
                controller_id=controller_id,
                benchmark_version=context.benchmark_version,
                task_type=context.task_type,
                quality_metric=context.quality_metric,
                scenario=context.scenario,
                max_epochs=context.max_epochs,
                parameters=dict(definition.get("controller_parameters", {})),
                metadata={**dict(context.metadata), "shadow_suite_id": context.controller_id},
            )
            self.states.append(
                ShadowControllerState(
                    controller_id=controller_id,
                    plugin_path=plugin_path,
                    controller=load_controller(plugin_path, child_context),
                )
            )
        self.output_path = self._output_path()
        self.last_epoch = 0
        self.last_best_quality: float | None = None
        self.last_training_energy_wh = 0.0
        self.last_training_time_s = 0.0

    def _output_path(self) -> Path:
        output_dir = Path(
            os.environ.get("BENCHMARK_METRICS_DIR")
            or os.environ.get("GPU_METRICS_DIR")
            or "/workspace/energy_metrics"
        )
        return output_dir / f"shadow_summary_job_{os.environ.get('SLURM_JOB_ID', 'local')}.json"

    def _freeze(
        self,
        state: ShadowControllerState,
        observation: EpochObservation,
        reason: str,
    ) -> None:
        state.stopped = True
        state.stop_epoch = observation.epoch
        state.stop_reason = reason
        state.best_quality_at_stop = _finite(observation.best_quality)
        state.training_energy_to_stop_wh = max(0.0, observation.cumulative_energy_wh)
        state.training_time_to_stop_s = max(0.0, observation.cumulative_duration_seconds)

    def _evaluate_child(
        self,
        state: ShadowControllerState,
        observation: EpochObservation,
    ) -> None:
        rapl_before = self.energy_meter.snapshot()
        wall_started = time.perf_counter()
        cpu_started = time.process_time()
        try:
            decision = state.controller.evaluate(observation)
            if not isinstance(decision, ControllerDecision):
                raise TypeError("Shadow controller did not return ControllerDecision.")
        except Exception as exc:  # A failed shadow must not terminate the shared Full100 run.
            decision = None
            state.failed = True
            state.error = f"{type(exc).__name__}: {exc}"
        process_cpu_seconds = max(0.0, time.process_time() - cpu_started)
        compute_seconds = max(0.0, time.perf_counter() - wall_started)
        rapl_after = self.energy_meter.snapshot()
        energy_wh, method = self.energy_meter.energy_wh(
            rapl_before,
            rapl_after,
            process_cpu_seconds,
        )
        state.decisions += 1
        state.controller_compute_seconds += compute_seconds
        state.controller_process_cpu_seconds += process_cpu_seconds
        state.controller_overhead_energy_wh += max(0.0, energy_wh)
        state.energy_measurement_method = method
        if decision is None:
            self._freeze(state, observation, "shadow_controller_failed")
            return
        state.confidence = _finite(decision.confidence)
        state.predicted_energy_saving_fraction = _finite(
            decision.predicted_energy_saving_fraction
        )
        state.predicted_quality_regret = _finite(decision.predicted_quality_regret)
        state.last_diagnostics = dict(decision.diagnostics)
        if decision.stop:
            self._freeze(state, observation, decision.reason)

    def evaluate(self, observation: EpochObservation) -> ControllerDecision:
        self.last_epoch = observation.epoch
        self.last_best_quality = _finite(observation.best_quality)
        self.last_training_energy_wh = max(0.0, observation.cumulative_energy_wh)
        self.last_training_time_s = max(0.0, observation.cumulative_duration_seconds)
        for state in self.states:
            if not state.stopped:
                self._evaluate_child(state, observation)
        if observation.epoch >= observation.max_epochs:
            for state in self.states:
                if not state.stopped:
                    self._freeze(state, observation, "max_epochs_reached")
        self._persist()
        return ControllerDecision(
            stop=False,
            reason="live_shadow_full100_continues",
            diagnostics=self._diagnostics(),
        )

    def _diagnostics(self) -> dict[str, float | int | bool | None]:
        diagnostics: dict[str, float | int | bool | None] = {
            "shadow_suite_active_controllers": sum(not state.stopped for state in self.states),
            "shadow_suite_stopped_controllers": sum(state.stopped for state in self.states),
            "shadow_suite_failed_controllers": sum(state.failed for state in self.states),
            "shadow_suite_full100_continues": 1,
        }
        for state in self.states:
            prefix = f"shadow_{_slug(state.controller_id)}"
            diagnostics.update(
                {
                    f"{prefix}_stopped": int(state.stopped),
                    f"{prefix}_failed": int(state.failed),
                    f"{prefix}_stop_epoch": state.stop_epoch,
                    f"{prefix}_best_quality_at_stop": state.best_quality_at_stop,
                    f"{prefix}_training_energy_to_stop_wh": state.training_energy_to_stop_wh,
                    f"{prefix}_controller_overhead_energy_wh": state.controller_overhead_energy_wh,
                    f"{prefix}_counterfactual_total_energy_wh": state.counterfactual_total_energy_wh,
                    f"{prefix}_controller_compute_seconds": state.controller_compute_seconds,
                    f"{prefix}_controller_process_cpu_seconds": state.controller_process_cpu_seconds,
                    f"{prefix}_decisions": state.decisions,
                    f"{prefix}_energy_measurement_method_code": (
                        1 if state.energy_measurement_method == "intel_rapl" else 2
                    ),
                }
            )
        return diagnostics

    def _document(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "measurement_protocol": "live-shadow-v1",
            "benchmark_run_id": os.environ.get("BENCHMARK_RUN_ID"),
            "benchmark_version": self.context.benchmark_version,
            "benchmark_case_id": self.context.metadata.get("benchmark_case_id"),
            "benchmark_stage": self.context.metadata.get("benchmark_stage"),
            "task_type": self.context.task_type,
            "quality_metric": self.context.quality_metric,
            "scenario": self.context.scenario,
            "job_id": os.environ.get("SLURM_JOB_ID", "local"),
            "full100_epoch": self.last_epoch,
            "full100_best_quality": self.last_best_quality,
            "full100_epoch_training_energy_wh": self.last_training_energy_wh,
            "full100_epoch_training_time_s": self.last_training_time_s,
            "energy_scope": "epoch_gpu_plus_attributed_controller_cpu",
            "lifecycle_energy_complete": False,
            "preprocessing_energy_wh": None,
            "finalization_energy_wh": None,
            "controllers": [
                {
                    "controller_id": state.controller_id,
                    "controller_plugin": state.plugin_path,
                    "stopped": state.stopped,
                    "stopped_early": bool(
                        state.stop_epoch is not None and state.stop_epoch < self.context.max_epochs
                    ),
                    "failed": state.failed,
                    "stop_epoch": state.stop_epoch,
                    "stop_reason": state.stop_reason,
                    "best_quality_at_stop": state.best_quality_at_stop,
                    "training_energy_to_stop_wh": state.training_energy_to_stop_wh,
                    "training_time_to_stop_s": state.training_time_to_stop_s,
                    "controller_overhead_energy_wh": state.controller_overhead_energy_wh,
                    "counterfactual_total_energy_wh": state.counterfactual_total_energy_wh,
                    "controller_compute_seconds": state.controller_compute_seconds,
                    "controller_process_cpu_seconds": state.controller_process_cpu_seconds,
                    "controller_decisions": state.decisions,
                    "energy_measurement_method": state.energy_measurement_method,
                    "confidence": state.confidence,
                    "predicted_energy_saving_fraction": state.predicted_energy_saving_fraction,
                    "predicted_quality_regret": state.predicted_quality_regret,
                    "error": state.error,
                    "last_diagnostics": state.last_diagnostics,
                }
                for state in self.states
            ],
        }

    def _persist(self) -> None:
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.output_path.with_suffix(self.output_path.suffix + ".tmp")
        temporary.write_text(
            json.dumps(self._document(), indent=2, ensure_ascii=False, default=str) + "\n",
            encoding="utf-8",
        )
        temporary.replace(self.output_path)

    def close(self) -> None:
        self._persist()
        for state in self.states:
            state.controller.close()
