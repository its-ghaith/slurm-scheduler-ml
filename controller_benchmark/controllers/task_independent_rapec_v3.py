from __future__ import annotations

import math
from dataclasses import replace
from typing import Any, Mapping

from controller_benchmark.api import ControllerDecision, EpochObservation

from .risk_aware_predictive_energy import (
    DynamicThresholds,
    RiskAwarePredictiveEnergyController,
    _best_so_far,
    _mad,
)


class TaskIndependentRapecV3Controller(RiskAwarePredictiveEnergyController):
    """Versioned RAPEC-v3 variant without task or scenario profiles.

    The original RAPEC-v3 remains unchanged for reproducibility. This variant
    releases the warm-up guard only when the online quality signal is no longer
    clearly larger than the observed validation noise. It never inspects task,
    dataset, model, scratch, or pretrained names.
    """

    def __init__(self, context):
        super().__init__(context)
        params = context.parameters
        self.warmup_signal_to_noise_ratio = max(
            0.0, float(params.get("warmup_signal_to_noise_ratio", 2.0))
        )
        self.warmup_noise_stability_ratio = max(
            1.0, float(params.get("warmup_noise_stability_ratio", 2.5))
        )
        self.warmup_required_stable_windows = max(
            1, int(params.get("warmup_required_stable_windows", 2))
        )
        self._stable_warmup_windows = 0
        self._last_warmup_diagnostics: dict[str, float | int] = {}

    def evaluate(self, observation: EpochObservation) -> ControllerDecision:
        decision = super().evaluate(observation)
        diagnostics = {
            **decision.diagnostics,
            **self._last_warmup_diagnostics,
            "rapec_ti_task_independent": 1,
            "rapec_ti_uses_task_or_scenario_profile": 0,
            "rapec_ti_uses_only_current_run_history": 1,
        }
        return replace(decision, diagnostics=diagnostics)

    def _adaptive_min_epochs(
        self,
        rows: list[Mapping[str, Any]],
        observation: EpochObservation,
        noise: float,
    ) -> int:
        history_floor = max(
            self.min_epochs_floor,
            self.min_fit_points + self.horizon_epochs,
            self.trend_window,
        )
        latest_allowed = max(1, observation.max_epochs - self.horizon_epochs)
        history_floor = min(latest_allowed, history_floor)
        if observation.epoch < history_floor:
            self._stable_warmup_windows = 0
            self._last_warmup_diagnostics = {
                "rapec_ti_online_warmup_floor": history_floor,
                "rapec_ti_recent_gain": 0.0,
                "rapec_ti_signal_to_noise_ratio": math.inf,
                "rapec_ti_noise_stability_ratio": math.inf,
                "rapec_ti_stable_warmup_windows": 0,
                "rapec_ti_warmup_released": 0,
            }
            return history_floor

        qualities = self._quality_values(rows, observation)
        best = _best_so_far(qualities)
        horizon = min(self.horizon_epochs, max(1, len(best) - 1))
        recent_gain = max(0.0, best[-1] - best[-1 - horizon]) if len(best) > horizon else 0.0
        noise_floor = max(noise, self.min_meaningful_gain)
        signal_to_noise = recent_gain / noise_floor

        window = min(self.trend_window, len(qualities))
        recent_noise = _mad(qualities[-window:]) if window >= 2 else noise_floor
        previous = qualities[-2 * window : -window] if len(qualities) >= 2 * window else []
        previous_noise = _mad(previous) if len(previous) >= 2 else recent_noise
        low_noise = max(min(recent_noise, previous_noise), self.min_meaningful_gain)
        noise_stability = max(recent_noise, previous_noise) / low_noise

        learning_is_active = signal_to_noise > self.warmup_signal_to_noise_ratio
        noise_is_stable = noise_stability <= self.warmup_noise_stability_ratio
        if not learning_is_active and noise_is_stable:
            self._stable_warmup_windows += 1
        else:
            self._stable_warmup_windows = 0
        released = self._stable_warmup_windows >= self.warmup_required_stable_windows

        self._last_warmup_diagnostics = {
            "rapec_ti_online_warmup_floor": history_floor,
            "rapec_ti_recent_gain": recent_gain,
            "rapec_ti_signal_to_noise_ratio": signal_to_noise,
            "rapec_ti_noise_stability_ratio": noise_stability,
            "rapec_ti_stable_warmup_windows": self._stable_warmup_windows,
            "rapec_ti_warmup_released": int(released),
        }
        if released:
            return history_floor
        return min(latest_allowed, observation.epoch + 1)

    def _minimum_quality(
        self,
        thresholds: DynamicThresholds,
        observation: EpochObservation,
    ) -> float:
        # Absolute task quality is not comparable across mAP, accuracy, and mIoU.
        return max(0.0, min(1.0, self.minimum_quality_floor))
