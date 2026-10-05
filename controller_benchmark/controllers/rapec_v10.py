from __future__ import annotations

import math
from dataclasses import replace
from typing import Any, Mapping

from controller_benchmark.api import ControllerDecision, EpochObservation

from .risk_aware_predictive_energy import (
    RiskAwarePredictiveEnergyController,
    _best_so_far,
    _mad,
)


class RAPECV10Controller(RiskAwarePredictiveEnergyController):
    """RAPEC-v3 with a task-independent, online-calibrated warm-up guard.

    All prediction, utility, uncertainty, quality guard, and stopping logic is
    inherited unchanged from RAPEC-v3. Only its task/scenario-based minimum
    epoch profile is replaced with evidence from the current training run.
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
            "rapec10_task_independent": 1,
            "rapec10_uses_task_or_scenario_profile": 0,
            "rapec10_uses_only_current_run_history": 1,
        }
        return replace(decision, diagnostics=diagnostics)

    def _adaptive_min_epochs(
        self,
        rows: list[Mapping[str, Any]],
        observation: EpochObservation,
        noise: float,
    ) -> int:
        # Prediction and trend estimates need a minimum amount of current-run data.
        history_floor = max(
            self.min_epochs_floor,
            self.min_fit_points + self.horizon_epochs,
            self.trend_window,
        )
        latest_allowed = max(1, observation.max_epochs - self.horizon_epochs)
        history_floor = min(latest_allowed, history_floor)
        if observation.epoch < history_floor:
            self._stable_warmup_windows = 0
            return self._record_warmup(
                effective_min_epochs=history_floor,
                observation=observation,
                history_floor=history_floor,
                recent_gain=0.0,
                detrended_noise=max(noise, self.min_meaningful_gain),
                signal_to_noise=math.inf,
                noise_stability=math.inf,
                learning_is_active=True,
                noise_is_stable=False,
                released=False,
            )

        qualities = self._quality_values(rows, observation)
        best = _best_so_far(qualities)
        horizon = min(self.horizon_epochs, max(1, len(best) - 1))
        recent_gain = (
            max(0.0, best[-1] - best[-1 - horizon])
            if len(best) > horizon
            else 0.0
        )
        online_noise = self._detrended_noise(
            qualities[-self.trend_window :], fallback=noise
        )
        noise_floor = max(online_noise, self.min_meaningful_gain)
        signal_to_noise = recent_gain / noise_floor

        window = min(self.trend_window, len(qualities))
        recent_noise = self._detrended_noise(
            qualities[-window:], fallback=noise_floor
        )
        previous_values = (
            qualities[-2 * window : -window]
            if len(qualities) >= 2 * window
            else []
        )
        previous_noise = (
            self._detrended_noise(previous_values, fallback=recent_noise)
            if len(previous_values) >= 3
            else recent_noise
        )
        lower_noise = max(
            min(recent_noise, previous_noise), self.min_meaningful_gain
        )
        noise_stability = max(recent_noise, previous_noise) / lower_noise

        learning_is_active = (
            signal_to_noise > self.warmup_signal_to_noise_ratio
        )
        noise_is_stable = (
            noise_stability <= self.warmup_noise_stability_ratio
        )
        if not learning_is_active and noise_is_stable:
            self._stable_warmup_windows += 1
        else:
            self._stable_warmup_windows = 0
        released = (
            self._stable_warmup_windows >= self.warmup_required_stable_windows
        )

        effective_min_epochs = (
            history_floor
            if released
            else min(latest_allowed, observation.epoch + 1)
        )
        return self._record_warmup(
            effective_min_epochs=effective_min_epochs,
            observation=observation,
            history_floor=history_floor,
            recent_gain=recent_gain,
            detrended_noise=noise_floor,
            signal_to_noise=signal_to_noise,
            noise_stability=noise_stability,
            learning_is_active=learning_is_active,
            noise_is_stable=noise_is_stable,
            released=released,
        )

    def _record_warmup(
        self,
        *,
        effective_min_epochs: int,
        observation: EpochObservation,
        history_floor: int,
        recent_gain: float,
        detrended_noise: float,
        signal_to_noise: float,
        noise_stability: float,
        learning_is_active: bool,
        noise_is_stable: bool,
        released: bool,
    ) -> int:
        self._last_warmup_diagnostics = {
            "rapec10_online_history_floor": history_floor,
            "rapec10_effective_min_epochs": effective_min_epochs,
            "rapec10_dynamic_min_epoch_fraction": (
                effective_min_epochs / max(1, observation.max_epochs)
            ),
            "rapec10_recent_gain": recent_gain,
            "rapec10_detrended_validation_noise": detrended_noise,
            "rapec10_signal_to_noise_ratio": signal_to_noise,
            "rapec10_noise_stability_ratio": noise_stability,
            "rapec10_learning_is_active": int(learning_is_active),
            "rapec10_noise_is_stable": int(noise_is_stable),
            "rapec10_stable_warmup_windows": self._stable_warmup_windows,
            "rapec10_warmup_released": int(released),
        }
        return effective_min_epochs

    @staticmethod
    def _detrended_noise(values: list[float], *, fallback: float) -> float:
        if len(values) < 3:
            return max(0.0, fallback)
        deltas = [right - left for left, right in zip(values, values[1:])]
        return max(0.0, _mad(deltas))
