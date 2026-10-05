from __future__ import annotations

import math
from dataclasses import replace
from typing import Any, Mapping

from controller_benchmark.api import ControllerDecision, EpochObservation

from .risk_aware_predictive_energy import (
    RiskAwarePredictiveEnergyController,
    _best_so_far,
    _mad,
    _quantile,
)


class ProfileFreeRapecV3Controller(RiskAwarePredictiveEnergyController):
    """RAPEC-v3 with a profile-free, current-run warm-up gate.

    Prediction, utility, uncertainty, quality guard, and stopping logic are
    inherited unchanged from RAPEC-v3. Only the task/scenario-dependent minimum
    epoch rule is replaced. Readiness is inferred from model data requirements
    and dimensionless progress-to-noise evidence observed in the current run.
    """

    def __init__(self, context):
        super().__init__(context)
        params = context.parameters
        self.warmup_significance_z = max(
            0.0, float(params.get("warmup_significance_z", 1.96))
        )
        self.warmup_low_progress_quantile = min(
            0.5,
            max(0.0, float(params.get("warmup_low_progress_quantile", 0.25))),
        )
        self.warmup_required_low_progress_windows = max(
            1, int(params.get("warmup_required_low_progress_windows", 2))
        )
        self.warmup_min_reference_points = max(
            2, int(params.get("warmup_min_reference_points", 4))
        )
        self.warmup_no_learning_multiplier = max(
            1.0, float(params.get("warmup_no_learning_multiplier", 2.0))
        )
        raw_multipliers = params.get(
            "warmup_confirmation_horizon_multipliers", (1, 2, 4)
        )
        if isinstance(raw_multipliers, (int, float, str)):
            raw_multipliers = (raw_multipliers,)
        self.warmup_confirmation_horizon_multipliers = tuple(
            sorted({max(1, int(value)) for value in raw_multipliers})
        )
        self._seen_significant_learning = False
        self._low_progress_windows = 0
        self._warmup_released = False
        self._normalized_progress_history: dict[int, list[float]] = {}
        self._last_warmup_diagnostics: dict[str, float | int] = {}

    def evaluate(self, observation: EpochObservation) -> ControllerDecision:
        decision = super().evaluate(observation)
        diagnostics = {
            **decision.diagnostics,
            **self._last_warmup_diagnostics,
            "rapec_pf_task_independent": 1,
            "rapec_pf_uses_task_or_scenario_profile": 0,
            "rapec_pf_uses_only_current_run_history": 1,
        }
        return replace(decision, diagnostics=diagnostics)

    def _adaptive_min_epochs(
        self,
        rows: list[Mapping[str, Any]],
        observation: EpochObservation,
        noise: float,
    ) -> int:
        structural_floor = self._structural_history_floor(observation)
        latest_allowed = max(1, observation.max_epochs - self.horizon_epochs)
        structural_floor = min(latest_allowed, structural_floor)

        qualities = self._quality_values(rows, observation)
        best = _best_so_far(qualities)
        deltas = [right - left for left, right in zip(qualities, qualities[1:])]
        recent_deltas = deltas[-self.trend_window :]
        per_epoch_noise = max(
            _mad(recent_deltas),
            noise / math.sqrt(max(1, self.horizon_epochs)),
            self.min_meaningful_gain,
        )
        progress = self._multi_horizon_progress(best, per_epoch_noise)
        base_horizon = min(self.horizon_epochs, max(1, len(best) - 1))
        base = progress.get(base_horizon, {})
        recent_gain = float(base.get("gain", 0.0))
        horizon_noise = float(base.get("noise", per_epoch_noise))
        progress_z = float(base.get("z", 0.0))
        progress_quantile = base.get("quantile")
        if any(item["z"] >= self.warmup_significance_z for item in progress.values()):
            self._seen_significant_learning = True

        required_horizons = self._confirmation_horizons()
        low_relative_progress = bool(required_horizons) and all(
            horizon in progress and bool(progress[horizon]["low"])
            for horizon in required_horizons
        )
        if (
            observation.epoch >= structural_floor
            and self._seen_significant_learning
            and low_relative_progress
        ):
            self._low_progress_windows += 1
        else:
            self._low_progress_windows = 0

        no_learning_release_epoch = min(
            latest_allowed,
            max(
                structural_floor,
                int(math.ceil(structural_floor * self.warmup_no_learning_multiplier)),
            ),
        )
        no_learning_fallback = (
            observation.epoch >= no_learning_release_epoch
            and not self._seen_significant_learning
        )
        post_learning_release = (
            self._low_progress_windows
            >= self.warmup_required_low_progress_windows
        )
        self._warmup_released = (
            self._warmup_released
            or no_learning_fallback
            or post_learning_release
        )

        if observation.epoch < structural_floor:
            effective_min_epochs = structural_floor
        elif self._warmup_released:
            effective_min_epochs = structural_floor
        else:
            effective_min_epochs = min(latest_allowed, observation.epoch + 1)

        self._last_warmup_diagnostics = {
            "rapec_pf_structural_history_floor": structural_floor,
            "rapec_pf_effective_min_epochs": effective_min_epochs,
            "rapec_pf_recent_gain": recent_gain,
            "rapec_pf_per_epoch_noise": per_epoch_noise,
            "rapec_pf_horizon_noise": horizon_noise,
            "rapec_pf_progress_z": progress_z,
            "rapec_pf_progress_reference_quantile": progress_quantile,
            "rapec_pf_seen_significant_learning": int(
                self._seen_significant_learning
            ),
            "rapec_pf_low_relative_progress": int(low_relative_progress),
            "rapec_pf_low_progress_windows": self._low_progress_windows,
            "rapec_pf_no_learning_release_epoch": no_learning_release_epoch,
            "rapec_pf_no_learning_fallback": int(no_learning_fallback),
            "rapec_pf_warmup_released": int(self._warmup_released),
        }
        for horizon, item in progress.items():
            self._last_warmup_diagnostics.update(
                {
                    f"rapec_pf_h{horizon}_gain": item["gain"],
                    f"rapec_pf_h{horizon}_progress_z": item["z"],
                    f"rapec_pf_h{horizon}_reference_quantile": item["quantile"],
                    f"rapec_pf_h{horizon}_low_relative_progress": int(
                        bool(item["low"])
                    ),
                }
            )
        return effective_min_epochs

    def _structural_history_floor(self, observation: EpochObservation) -> int:
        # k-NN needs k completed horizon transitions, not merely k epochs.
        transition_floor = (
            self.min_fit_points + self.horizon_epochs + self.knn_neighbors
        )
        longest_horizon = max(self._confirmation_horizons())
        trend_floor = self.trend_window + longest_horizon
        return min(
            max(1, observation.max_epochs - self.horizon_epochs),
            max(self.min_epochs_floor, transition_floor, trend_floor),
        )

    def _confirmation_horizons(self) -> tuple[int, ...]:
        maximum = max(
            self.horizon_epochs,
            self.context.max_epochs - self.horizon_epochs,
        )
        return tuple(
            sorted(
                {
                    min(maximum, self.horizon_epochs * multiplier)
                    for multiplier in self.warmup_confirmation_horizon_multipliers
                }
            )
        )

    def _multi_horizon_progress(
        self,
        best: list[float],
        per_epoch_noise: float,
    ) -> dict[int, dict[str, float | bool | None]]:
        progress: dict[int, dict[str, float | bool | None]] = {}
        for horizon in self._confirmation_horizons():
            if len(best) <= horizon:
                continue
            gain = max(0.0, best[-1] - best[-1 - horizon])
            horizon_noise = per_epoch_noise * math.sqrt(horizon)
            progress_z = gain / max(horizon_noise, self.min_meaningful_gain)
            reference = self._normalized_progress_history.setdefault(horizon, [])
            reference_quantile = (
                _quantile(reference, self.warmup_low_progress_quantile)
                if len(reference) >= self.warmup_min_reference_points
                else None
            )
            low_relative_progress = (
                reference_quantile is not None
                and progress_z <= reference_quantile
            )
            if math.isfinite(progress_z):
                reference.append(progress_z)
            progress[horizon] = {
                "gain": gain,
                "noise": horizon_noise,
                "z": progress_z,
                "quantile": reference_quantile,
                "low": low_relative_progress,
            }
        return progress
