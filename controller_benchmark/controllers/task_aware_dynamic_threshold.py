from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Mapping

from controller_benchmark.api import ControllerDecision, EpochObservation
from controller_benchmark.controllers.risk_aware_predictive_energy import (
    DynamicThresholds,
    RiskAwarePredictiveEnergyController,
    _best_so_far,
    _mad,
    _median,
)


@dataclass(frozen=True)
class TaskThresholdAdapter:
    task_family: str
    horizon_epochs: int
    trend_window: int
    patience: int
    probability_threshold: float
    uncertainty_to_gain_ratio: float
    utility_quantile: float
    min_epoch_fraction: float
    noise_multiplier: float
    recent_gain_fraction: float
    remaining_room_fraction: float
    utility_floor_fraction: float
    minimum_quality_floor: float
    stop_states: frozenset[str]


class TaskAwareDynamicThresholdController(RiskAwarePredictiveEnergyController):
    """RAPEC-v4: task-aware dynamic threshold adapter on top of RAPEC-v3.

    The stop rule remains task-independent and works on quality_score Q_t.
    This controller dynamically adapts the thresholds that feed the stop rule
    from task type, scenario, validation noise and learning stage.
    """

    def __init__(self, context):
        super().__init__(context)
        params = context.parameters
        self.adapter_strength = float(params.get("adapter_strength", 1.0))
        self._active_adapter: TaskThresholdAdapter | None = None

    def evaluate(self, observation: EpochObservation) -> ControllerDecision:
        rows = [*observation.history, observation.raw_metrics]
        adapter = self._build_adapter(rows, observation)
        self._active_adapter = adapter
        original = self._capture_mutable_config()
        self._apply_adapter(adapter)
        try:
            decision = super().evaluate(observation)
        finally:
            self._restore_mutable_config(original)

        diagnostics = dict(decision.diagnostics)
        diagnostics.update(
            {
                "rapec4_adapter_horizon_epochs": adapter.horizon_epochs,
                "rapec4_adapter_trend_window": adapter.trend_window,
                "rapec4_adapter_patience": adapter.patience,
                "rapec4_adapter_probability_threshold": adapter.probability_threshold,
                "rapec4_adapter_uncertainty_to_gain_ratio": adapter.uncertainty_to_gain_ratio,
                "rapec4_adapter_utility_quantile": adapter.utility_quantile,
                "rapec4_adapter_min_epoch_fraction": adapter.min_epoch_fraction,
                "rapec4_adapter_noise_multiplier": adapter.noise_multiplier,
                "rapec4_adapter_recent_gain_fraction": adapter.recent_gain_fraction,
                "rapec4_adapter_remaining_room_fraction": adapter.remaining_room_fraction,
                "rapec4_adapter_utility_floor_fraction": adapter.utility_floor_fraction,
                "rapec4_adapter_minimum_quality_floor": adapter.minimum_quality_floor,
                "rapec4_adapter_task_family_code": self._task_family_code(adapter.task_family),
            }
        )
        reason = decision.reason
        if decision.stop and reason.startswith("rapec_stop:"):
            reason = reason.replace("rapec_stop:", "rapec4_stop:", 1)
        elif not decision.stop and reason.startswith("rapec_"):
            reason = reason.replace("rapec_", "rapec4_", 1)
        return ControllerDecision(
            stop=decision.stop,
            reason=reason,
            confidence=decision.confidence,
            predicted_energy_saving_fraction=decision.predicted_energy_saving_fraction,
            predicted_quality_regret=decision.predicted_quality_regret,
            diagnostics=diagnostics,
        )

    def _dynamic_thresholds(self, rows: list[Mapping[str, Any]], observation: EpochObservation) -> DynamicThresholds:
        thresholds = super()._dynamic_thresholds(rows, observation)
        adapter = self._active_adapter
        if adapter is None:
            return thresholds
        recent_energy = self._recent_horizon_energy(rows)
        utility_floor = 0.0
        if recent_energy and recent_energy > 0:
            utility_floor = thresholds.meaningful_gain / recent_energy * adapter.utility_floor_fraction
        return DynamicThresholds(
            meaningful_gain=thresholds.meaningful_gain,
            utility_threshold=max(thresholds.utility_threshold, utility_floor),
            adaptive_min_epochs=thresholds.adaptive_min_epochs,
            validation_noise=thresholds.validation_noise,
            recent_gain_horizon=thresholds.recent_gain_horizon,
            remaining_quality_room=thresholds.remaining_quality_room,
        )

    def _adaptive_min_epochs(self, rows: list[Mapping[str, Any]], observation: EpochObservation, noise: float) -> int:
        adapter = self._active_adapter
        if adapter is None:
            return super()._adaptive_min_epochs(rows, observation, noise)
        noise_penalty = 0.05 if noise > 0.01 else 0.0
        fraction = min(0.90, adapter.min_epoch_fraction + noise_penalty)
        return min(
            max(1, observation.max_epochs - adapter.horizon_epochs),
            max(self.min_epochs_floor, int(math.ceil(fraction * observation.max_epochs))),
        )

    def _minimum_quality(self, thresholds: DynamicThresholds, observation: EpochObservation) -> float:
        adapter = self._active_adapter
        if adapter is None:
            return super()._minimum_quality(thresholds, observation)
        epoch_fraction = max(0.0, min(1.0, observation.epoch / max(1, observation.max_epochs)))
        ramp = adapter.minimum_quality_floor * min(1.0, epoch_fraction / max(0.10, adapter.min_epoch_fraction))
        return max(0.0, min(0.95, ramp - thresholds.validation_noise))

    def _build_adapter(self, rows: list[Mapping[str, Any]], observation: EpochObservation) -> TaskThresholdAdapter:
        profile = self._base_profile(observation)
        qualities = self._quality_values(rows, observation)
        recent = qualities[-profile.trend_window :] if qualities else []
        best = _best_so_far(qualities)
        best_recent = best[-profile.trend_window :] if best else []
        validation_noise = _mad(recent)
        recent_slope = _median([right - left for left, right in zip(best_recent, best_recent[1:])]) or 0.0
        epoch_fraction = max(0.0, min(1.0, observation.epoch / max(1, observation.max_epochs)))
        horizon = profile.horizon_epochs
        trend_window = profile.trend_window
        patience = profile.patience
        probability = profile.probability_threshold
        uncertainty_ratio = profile.uncertainty_to_gain_ratio
        utility_quantile = profile.utility_quantile
        min_epoch_fraction = profile.min_epoch_fraction

        if profile.task_family != "classification" and validation_noise > max(0.01, abs(recent_slope) * 3.0):
            horizon += 2
            patience += 1
            probability = max(0.05, probability - 0.05)
            uncertainty_ratio = max(1.5, uncertainty_ratio - 0.5)
            min_epoch_fraction = min(0.90, min_epoch_fraction + 0.05)

        if profile.task_family == "scratch_detection" and epoch_fraction < 0.70:
            horizon += 1
            patience += 1
            probability = max(0.05, probability - 0.02)

        if profile.task_family in {"pretrained_detection", "classification", "segmentation"} and epoch_fraction > 0.65:
            patience = max(2, patience - 1)
            probability = min(0.45, probability + 0.04)

        if profile.task_family == "segmentation" and epoch_fraction > 0.45:
            utility_quantile = min(0.45, utility_quantile + 0.05)
            probability = min(0.45, probability + 0.05)

        horizon = max(1, min(observation.max_epochs, horizon))
        trend_window = max(profile.trend_window, horizon * 2)
        return TaskThresholdAdapter(
            task_family=profile.task_family,
            horizon_epochs=horizon,
            trend_window=trend_window,
            patience=max(1, patience),
            probability_threshold=max(0.01, min(0.50, probability)),
            uncertainty_to_gain_ratio=max(1.0, uncertainty_ratio),
            utility_quantile=max(0.01, min(0.50, utility_quantile)),
            min_epoch_fraction=max(0.01, min(0.90, min_epoch_fraction)),
            noise_multiplier=profile.noise_multiplier,
            recent_gain_fraction=profile.recent_gain_fraction,
            remaining_room_fraction=profile.remaining_room_fraction,
            utility_floor_fraction=profile.utility_floor_fraction,
            minimum_quality_floor=profile.minimum_quality_floor,
            stop_states=profile.stop_states,
        )

    def _base_profile(self, observation: EpochObservation) -> TaskThresholdAdapter:
        task_type = self.context.task_type.lower()
        scenario = self.context.scenario.lower()
        if task_type == "image_classification":
            return TaskThresholdAdapter(
                "classification", 5, 10, 2, 0.35, 6.0, 0.30, 0.18, 1.10, 0.35, 0.008, 0.35, 0.45,
                frozenset({"slow_learning", "plateau", "overfit_risk"}),
            )
        if task_type == "semantic_segmentation":
            return TaskThresholdAdapter(
                "segmentation", 10, 22, 4, 0.35, 6.0, 0.32, 0.25, 1.20, 0.30, 0.010, 0.45, 0.55,
                frozenset({"slow_learning", "plateau", "overfit_risk"}),
            )
        if "scratch" in scenario:
            return TaskThresholdAdapter(
                "scratch_detection", 15, 32, 6, 0.10, 2.5, 0.08, 0.55, 1.00, 0.20, 0.004, 0.20, 0.48,
                frozenset({"plateau", "overfit_risk"}),
            )
        if "pretrained" in scenario:
            min_fraction = 0.35 if "128" in scenario else 0.30
            return TaskThresholdAdapter(
                "pretrained_detection", 8, 18, 3, 0.28, 3.5, 0.24, min_fraction, 1.15, 0.30, 0.010, 0.30, 0.65,
                frozenset({"slow_learning", "plateau", "overfit_risk"}),
            )
        return TaskThresholdAdapter(
            "generic_detection", 10, 22, 4, 0.22, 4.0, 0.20, 0.30, 1.20, 0.30, 0.010, 0.30, 0.50,
            frozenset({"slow_learning", "plateau", "overfit_risk"}),
        )

    def _apply_adapter(self, adapter: TaskThresholdAdapter) -> None:
        self.horizon_epochs = adapter.horizon_epochs
        self.trend_window = adapter.trend_window
        self.patience = adapter.patience
        self.max_probability_gain_gt_threshold = adapter.probability_threshold
        self.max_uncertainty_to_gain_ratio = adapter.uncertainty_to_gain_ratio
        self.utility_quantile = adapter.utility_quantile
        self.noise_multiplier = adapter.noise_multiplier
        self.recent_gain_fraction = adapter.recent_gain_fraction
        self.remaining_room_fraction = adapter.remaining_room_fraction
        self.minimum_quality_floor = adapter.minimum_quality_floor
        self.STOP_STATES = set(adapter.stop_states)

    def _capture_mutable_config(self) -> dict[str, Any]:
        return {
            "horizon_epochs": self.horizon_epochs,
            "trend_window": self.trend_window,
            "patience": self.patience,
            "max_probability_gain_gt_threshold": self.max_probability_gain_gt_threshold,
            "max_uncertainty_to_gain_ratio": self.max_uncertainty_to_gain_ratio,
            "utility_quantile": self.utility_quantile,
            "noise_multiplier": self.noise_multiplier,
            "recent_gain_fraction": self.recent_gain_fraction,
            "remaining_room_fraction": self.remaining_room_fraction,
            "minimum_quality_floor": self.minimum_quality_floor,
            "STOP_STATES": set(self.STOP_STATES),
        }

    def _restore_mutable_config(self, original: dict[str, Any]) -> None:
        for key, value in original.items():
            setattr(self, key, value)

    def _task_family_code(self, task_family: str) -> int:
        return {
            "classification": 1,
            "segmentation": 2,
            "scratch_detection": 3,
            "pretrained_detection": 4,
            "generic_detection": 5,
        }.get(task_family, 0)
