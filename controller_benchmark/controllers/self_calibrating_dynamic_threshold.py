from __future__ import annotations

import math
import statistics
from dataclasses import dataclass
from typing import Any, Mapping

from controller_benchmark.api import ControllerDecision, EpochObservation
from controller_benchmark.controllers.risk_aware_predictive_energy import (
    DynamicThresholds,
    RiskAwarePredictiveEnergyController,
    _best_so_far,
    _mad,
    _median,
    _quantile,
)


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


@dataclass(frozen=True)
class SelfCalibration:
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
    validation_noise: float
    curve_stability: float
    delayed_learning_risk: float
    saturation_score: float
    energy_variability: float
    recent_velocity: float
    historical_velocity: float
    stall_recovery_rate: float


class SelfCalibratingDynamicThresholdController(RiskAwarePredictiveEnergyController):
    """RAPEC-v5: task-agnostic self-calibrating dynamic thresholds.

    RAPEC-v5 does not use task names, dataset names, scratch/pretrained flags or
    model identifiers to select thresholds. It infers the threshold adapter from
    the observed quality curve and energy profile of the current run.
    """

    def __init__(self, context):
        super().__init__(context)
        params = context.parameters
        self.min_horizon = max(1, int(params.get("min_horizon_epochs", 5)))
        self.max_horizon = max(self.min_horizon, int(params.get("max_horizon_epochs", 18)))
        self.min_patience_dynamic = max(1, int(params.get("min_patience", 2)))
        self.max_patience_dynamic = max(self.min_patience_dynamic, int(params.get("max_patience", 8)))
        self.min_epoch_fraction_floor = float(params.get("min_epoch_fraction_floor", 0.12))
        self.max_epoch_fraction_cap = float(params.get("max_epoch_fraction_cap", 0.75))
        self._active_calibration: SelfCalibration | None = None

    def evaluate(self, observation: EpochObservation) -> ControllerDecision:
        rows = [*observation.history, observation.raw_metrics]
        calibration = self._self_calibrate(rows, observation)
        self._active_calibration = calibration
        original = self._capture_mutable_config()
        self._apply_calibration(calibration)
        try:
            decision = super().evaluate(observation)
        finally:
            self._restore_mutable_config(original)

        diagnostics = dict(decision.diagnostics)
        diagnostics.update(
            {
                "rapec5_horizon_epochs": calibration.horizon_epochs,
                "rapec5_trend_window": calibration.trend_window,
                "rapec5_patience": calibration.patience,
                "rapec5_probability_threshold": calibration.probability_threshold,
                "rapec5_uncertainty_to_gain_ratio": calibration.uncertainty_to_gain_ratio,
                "rapec5_utility_quantile": calibration.utility_quantile,
                "rapec5_min_epoch_fraction": calibration.min_epoch_fraction,
                "rapec5_noise_multiplier": calibration.noise_multiplier,
                "rapec5_recent_gain_fraction": calibration.recent_gain_fraction,
                "rapec5_remaining_room_fraction": calibration.remaining_room_fraction,
                "rapec5_utility_floor_fraction": calibration.utility_floor_fraction,
                "rapec5_curve_stability": calibration.curve_stability,
                "rapec5_delayed_learning_risk": calibration.delayed_learning_risk,
                "rapec5_saturation_score": calibration.saturation_score,
                "rapec5_energy_variability": calibration.energy_variability,
                "rapec5_recent_velocity": calibration.recent_velocity,
                "rapec5_historical_velocity": calibration.historical_velocity,
                "rapec5_stall_recovery_rate": calibration.stall_recovery_rate,
            }
        )
        reason = decision.reason
        if decision.stop and reason.startswith("rapec_stop:"):
            reason = reason.replace("rapec_stop:", "rapec5_stop:", 1)
        elif not decision.stop and reason.startswith("rapec_"):
            reason = reason.replace("rapec_", "rapec5_", 1)
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
        calibration = self._active_calibration
        if calibration is None:
            return thresholds
        recent_energy = self._recent_horizon_energy(rows)
        utility_floor = 0.0
        if recent_energy and recent_energy > 0:
            utility_floor = thresholds.meaningful_gain / recent_energy * calibration.utility_floor_fraction
        return DynamicThresholds(
            meaningful_gain=thresholds.meaningful_gain,
            utility_threshold=max(thresholds.utility_threshold, utility_floor),
            adaptive_min_epochs=thresholds.adaptive_min_epochs,
            validation_noise=thresholds.validation_noise,
            recent_gain_horizon=thresholds.recent_gain_horizon,
            remaining_quality_room=thresholds.remaining_quality_room,
        )

    def _adaptive_min_epochs(self, rows: list[Mapping[str, Any]], observation: EpochObservation, noise: float) -> int:
        calibration = self._active_calibration
        if calibration is None:
            return super()._adaptive_min_epochs(rows, observation, noise)
        noise_penalty = 0.05 if noise > 0.01 else 0.0
        fraction = min(self.max_epoch_fraction_cap, calibration.min_epoch_fraction + noise_penalty)
        return min(
            max(1, observation.max_epochs - calibration.horizon_epochs),
            max(self.min_epochs_floor, int(math.ceil(fraction * observation.max_epochs))),
        )

    def _minimum_quality(self, thresholds: DynamicThresholds, observation: EpochObservation) -> float:
        calibration = self._active_calibration
        if calibration is None:
            return super()._minimum_quality(thresholds, observation)
        # Task-agnostic guard: require more evidence when the curve itself shows
        # delayed-learning risk, but avoid a task-specific absolute target.
        epoch_fraction = max(0.0, min(1.0, observation.epoch / max(1, observation.max_epochs)))
        evidence_floor = 0.08 * epoch_fraction * (1.0 - calibration.delayed_learning_risk)
        return max(self.minimum_quality_floor, min(0.60, evidence_floor - thresholds.validation_noise))

    def _self_calibrate(self, rows: list[Mapping[str, Any]], observation: EpochObservation) -> SelfCalibration:
        qualities = self._quality_values(rows, observation)
        best = _best_so_far(qualities)
        n = len(qualities)
        default_window = max(6, min(24, int(math.sqrt(max(1, n)) * 4)))
        recent_raw = qualities[-default_window:]
        recent_best = best[-default_window:]
        raw_deltas = [right - left for left, right in zip(recent_raw, recent_raw[1:])]
        best_deltas = [max(0.0, right - left) for left, right in zip(recent_best, recent_best[1:])]
        positive_deltas = [delta for left, right in zip(best, best[1:]) if (delta := max(0.0, right - left)) > 0]
        validation_noise = _mad(recent_raw)
        recent_velocity = _median(best_deltas) or 0.0
        historical_velocity = _quantile(positive_deltas, 0.60) or recent_velocity or 0.0
        velocity_reference = max(historical_velocity, 1e-6)
        velocity_ratio = _clamp(recent_velocity / velocity_reference, 0.0, 1.5)
        noise_ratio = _clamp(validation_noise / (abs(recent_velocity) * max(1, self.min_horizon) + 1e-6), 0.0, 3.0)
        curve_stability = _clamp(1.0 / (1.0 + noise_ratio), 0.0, 1.0)
        saturation_score = _clamp(1.0 - min(1.0, velocity_ratio), 0.0, 1.0)
        remaining_room = max(0.0, 1.0 - (best[-1] if best else 0.0))
        epoch_fraction = _clamp(observation.epoch / max(1, observation.max_epochs), 0.0, 1.0)
        stall_recovery_rate = self._stall_recovery_rate(rows, observation)
        energy_variability = self._energy_variability(rows)
        delayed_learning_risk = _clamp(
            0.35 * saturation_score * (1.0 - epoch_fraction)
            + 0.30 * stall_recovery_rate
            + 0.20 * (1.0 - curve_stability)
            + 0.15 * min(1.0, remaining_room),
            0.0,
            1.0,
        )

        horizon_float = (
            self.min_horizon
            + 8.0 * delayed_learning_risk
            + 3.0 * (1.0 - curve_stability)
            + 2.0 * energy_variability
        )
        horizon = int(round(_clamp(horizon_float, self.min_horizon, self.max_horizon)))
        patience_float = (
            self.min_patience_dynamic
            + 4.0 * delayed_learning_risk
            + 2.0 * (1.0 - curve_stability)
            + 1.0 * energy_variability
            - (1.0 if epoch_fraction > 0.70 and curve_stability > 0.70 else 0.0)
        )
        patience = int(round(_clamp(patience_float, self.min_patience_dynamic, self.max_patience_dynamic)))
        probability_threshold = _clamp(
            0.12 + 0.25 * curve_stability + 0.06 * epoch_fraction - 0.20 * delayed_learning_risk,
            0.05,
            0.45,
        )
        uncertainty_ratio = _clamp(2.0 + 4.0 * curve_stability - 1.5 * delayed_learning_risk, 1.5, 6.0)
        utility_quantile = _clamp(
            0.10 + 0.25 * curve_stability + 0.10 * epoch_fraction - 0.22 * delayed_learning_risk,
            0.05,
            0.45,
        )
        min_epoch_fraction = _clamp(
            0.15 + 0.35 * delayed_learning_risk + 0.12 * remaining_room + 0.08 * (1.0 - curve_stability),
            self.min_epoch_fraction_floor,
            self.max_epoch_fraction_cap,
        )
        trend_window = max(horizon * 2, int(round(8 + 22 * delayed_learning_risk + 10 * (1.0 - curve_stability))))
        noise_multiplier = _clamp(1.0 + 0.75 * (1.0 - curve_stability) + 0.35 * delayed_learning_risk, 0.85, 2.25)
        recent_gain_fraction = _clamp(0.15 + 0.35 * curve_stability - 0.12 * delayed_learning_risk, 0.08, 0.45)
        remaining_room_fraction = _clamp(0.006 + 0.014 * curve_stability - 0.004 * delayed_learning_risk, 0.003, 0.025)
        utility_floor_fraction = _clamp(0.20 + 0.45 * curve_stability - 0.20 * delayed_learning_risk, 0.05, 0.70)
        return SelfCalibration(
            horizon_epochs=horizon,
            trend_window=max(6, min(48, trend_window)),
            patience=patience,
            probability_threshold=probability_threshold,
            uncertainty_to_gain_ratio=uncertainty_ratio,
            utility_quantile=utility_quantile,
            min_epoch_fraction=min_epoch_fraction,
            noise_multiplier=noise_multiplier,
            recent_gain_fraction=recent_gain_fraction,
            remaining_room_fraction=remaining_room_fraction,
            utility_floor_fraction=utility_floor_fraction,
            validation_noise=validation_noise,
            curve_stability=curve_stability,
            delayed_learning_risk=delayed_learning_risk,
            saturation_score=saturation_score,
            energy_variability=energy_variability,
            recent_velocity=recent_velocity,
            historical_velocity=historical_velocity,
            stall_recovery_rate=stall_recovery_rate,
        )

    def _stall_recovery_rate(self, rows: list[Mapping[str, Any]], observation: EpochObservation) -> float:
        qualities = self._quality_values(rows, observation)
        if len(qualities) < self.min_fit_points + self.min_horizon * 2:
            return 0.0
        best = _best_so_far(qualities)
        positive_deltas = [max(0.0, right - left) for left, right in zip(best, best[1:])]
        reference = _quantile([value for value in positive_deltas if value > 0], 0.50) or 0.0
        if reference <= 0:
            return 0.0
        window = self.min_horizon
        stalls = 0
        recoveries = 0
        for end in range(window, len(best) - window):
            past_gain = max(0.0, best[end] - best[end - window])
            future_gain = max(0.0, best[end + window] - best[end])
            if past_gain <= reference * window * 0.35:
                stalls += 1
                if future_gain > reference * window * 0.70:
                    recoveries += 1
        return recoveries / stalls if stalls else 0.0

    def _energy_variability(self, rows: list[Mapping[str, Any]]) -> float:
        values = [self._energy_wh(row) for row in rows[-24:] if self._energy_wh(row) > 0]
        if len(values) < 4:
            return 0.0
        median = statistics.median(values)
        if median <= 0:
            return 0.0
        return _clamp(_mad(values) / median, 0.0, 1.0)

    def _apply_calibration(self, calibration: SelfCalibration) -> None:
        self.horizon_epochs = calibration.horizon_epochs
        self.trend_window = calibration.trend_window
        self.patience = calibration.patience
        self.max_probability_gain_gt_threshold = calibration.probability_threshold
        self.max_uncertainty_to_gain_ratio = calibration.uncertainty_to_gain_ratio
        self.utility_quantile = calibration.utility_quantile
        self.noise_multiplier = calibration.noise_multiplier
        self.recent_gain_fraction = calibration.recent_gain_fraction
        self.remaining_room_fraction = calibration.remaining_room_fraction
        self.STOP_STATES = {"slow_learning", "plateau", "overfit_risk"}

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
            "STOP_STATES": set(self.STOP_STATES),
        }

    def _restore_mutable_config(self, original: dict[str, Any]) -> None:
        for key, value in original.items():
            setattr(self, key, value)
