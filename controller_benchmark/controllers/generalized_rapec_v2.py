from __future__ import annotations

import math
import statistics
from dataclasses import dataclass, replace
from typing import Any, Mapping

from controller_benchmark.api import ControllerDecision, EpochObservation

from .rapec_v3_pf import ProfileFreeRapecV3Controller
from .risk_aware_predictive_energy import _best_so_far, _mad, _number, _quantile


@dataclass(frozen=True)
class CurveRegimeEvidence:
    regime: str
    short_gain: float
    long_gain: float
    short_gain_z: float
    long_gain_z: float
    local_probability: float
    local_upper_gain: float
    dynamic_patience: int
    non_improving_epochs: int
    low_relative_progress: bool
    low_energy_utility: bool
    late_learning_guard: bool


class GeneralizedRapecV2Controller(ProfileFreeRapecV3Controller):
    """Profile-free RAPEC with learning-curve regime adaptation.

    RAPEC-G v2 keeps the original RAPEC-v3-PF predictive decision and adds a
    second route for early plateaus. The route is calibrated only from the
    current run: robust validation noise, short/long horizon progress, local
    probability of meaningful gain, energy utility, loss, gradient and
    learning-rate evidence. No task, dataset, model or initialization name is
    inspected.
    """

    REGIME_CODES = {
        "insufficient_history": 0,
        "undertrained": 1,
        "late_learning": 2,
        "noisy_learning": 3,
        "early_plateau": 4,
        "overfit_risk": 5,
    }

    def __init__(self, context):
        super().__init__(context)
        params = context.parameters
        raw_horizons = params.get("regime_horizon_multipliers", (1, 2))
        if isinstance(raw_horizons, (int, float, str)):
            raw_horizons = (raw_horizons,)
        self.regime_horizon_multipliers = tuple(
            sorted({max(1, int(value)) for value in raw_horizons})
        )
        self.plateau_noise_z = max(
            0.0, float(params.get("plateau_noise_z", 1.96))
        )
        self.plateau_relative_quantile = min(
            0.5, max(0.0, float(params.get("plateau_relative_quantile", 0.25)))
        )
        self.plateau_probability_ceiling = min(
            1.0, max(0.0, float(params.get("plateau_probability_ceiling", 0.35)))
        )
        self.plateau_utility_quantile = min(
            0.5, max(0.0, float(params.get("plateau_utility_quantile", 0.35)))
        )
        self.plateau_confirmation_windows = max(
            1, int(params.get("plateau_confirmation_windows", 3))
        )
        self.plateau_patience_min = max(
            1, int(params.get("plateau_patience_min", 3))
        )
        self.plateau_patience_max = max(
            self.plateau_patience_min,
            int(params.get("plateau_patience_max", 10)),
        )
        self.loss_recovery_z = max(
            0.0, float(params.get("loss_recovery_z", 1.96))
        )
        self._plateau_streak = 0
        self._non_improving_epochs = 0
        self._comparison_best: float | None = None

    def evaluate(self, observation: EpochObservation) -> ControllerDecision:
        base_decision = super().evaluate(observation)
        rows = [*observation.history, observation.raw_metrics]
        qualities = self._quality_values(rows, observation)
        thresholds = self._dynamic_thresholds(rows, observation)
        evidence = self._curve_regime(rows, qualities, observation, thresholds)
        state_features = self._state_features(rows, observation)

        current_quality = (
            observation.best_quality
            if observation.best_quality is not None
            else observation.quality
        )
        structural_ready = (
            observation.epoch >= self._structural_history_floor(observation)
            and len(qualities) >= self.min_fit_points
        )
        quality_guard = (
            current_quality is not None
            and float(current_quality) >= self._minimum_quality(thresholds, observation)
        )
        predictive_probability = _number(
            base_decision.diagnostics.get(
                "rapec_prob_gain_gt_dynamic_threshold"
            )
        )
        predictive_probability_ok = (
            predictive_probability is None
            or predictive_probability <= self.plateau_probability_ceiling
        )
        plateau_candidate = (
            structural_ready
            and evidence.regime in {"early_plateau", "overfit_risk"}
            and evidence.non_improving_epochs >= evidence.dynamic_patience
            and evidence.low_relative_progress
            and evidence.low_energy_utility
            and evidence.local_probability <= self.plateau_probability_ceiling
            and predictive_probability_ok
            and not evidence.late_learning_guard
            and quality_guard
        )
        self._plateau_streak = self._plateau_streak + 1 if plateau_candidate else 0
        plateau_stop = self._plateau_streak >= self.plateau_confirmation_windows
        base_stop_accepted = (
            base_decision.stop
            and evidence.regime in {"early_plateau", "overfit_risk"}
            and not evidence.late_learning_guard
        )
        stop = base_stop_accepted or plateau_stop

        recent_epoch_energy = self._recent_epoch_energy(rows)
        remaining_epochs = max(0, observation.max_epochs - observation.epoch)
        projected_remaining_energy = (
            recent_epoch_energy * remaining_epochs
            if recent_epoch_energy is not None
            else None
        )
        projected_full_energy = (
            observation.cumulative_energy_wh + projected_remaining_energy
            if projected_remaining_energy is not None
            else None
        )
        predicted_saving = (
            projected_remaining_energy / projected_full_energy
            if projected_remaining_energy is not None
            and projected_full_energy is not None
            and projected_full_energy > 0.0
            else None
        )
        confidence = base_decision.confidence
        if plateau_stop:
            probability_confidence = 1.0 - evidence.local_probability
            streak_confidence = min(
                1.0,
                self._plateau_streak / max(1, self.plateau_confirmation_windows),
            )
            confidence = max(
                float(confidence or 0.0),
                0.7 * probability_confidence + 0.3 * streak_confidence,
            )

        diagnostics = {
            **base_decision.diagnostics,
            "rapec_g_v2_enabled": 1,
            "rapec_g_v2_task_profile_used": 0,
            "rapec_g_v2_current_run_only": 1,
            "rapec_g_v2_regime_code": self.REGIME_CODES[evidence.regime],
            "rapec_g_v2_short_gain": evidence.short_gain,
            "rapec_g_v2_long_gain": evidence.long_gain,
            "rapec_g_v2_short_gain_z": evidence.short_gain_z,
            "rapec_g_v2_long_gain_z": evidence.long_gain_z,
            "rapec_g_v2_local_probability_gain": evidence.local_probability,
            "rapec_g_v2_local_upper_gain": evidence.local_upper_gain,
            "rapec_g_v2_dynamic_patience": evidence.dynamic_patience,
            "rapec_g_v2_non_improving_epochs": evidence.non_improving_epochs,
            "rapec_g_v2_low_relative_progress": int(
                evidence.low_relative_progress
            ),
            "rapec_g_v2_low_energy_utility": int(evidence.low_energy_utility),
            "rapec_g_v2_late_learning_guard": int(evidence.late_learning_guard),
            "rapec_g_v2_predictive_probability_ok": int(
                predictive_probability_ok
            ),
            "rapec_g_v2_plateau_candidate": int(plateau_candidate),
            "rapec_g_v2_plateau_streak": self._plateau_streak,
            "rapec_g_v2_plateau_stop": int(plateau_stop),
            "rapec_g_v2_base_stop": int(base_decision.stop),
            "rapec_g_v2_base_stop_accepted": int(base_stop_accepted),
            "rapec_g_v2_gradient_norm": state_features.get("gradient_norm"),
            "rapec_g_v2_learning_rate": state_features.get("learning_rate"),
            "rapec_g_v2_epoch_duration_s": state_features.get("duration_seconds"),
            "rapec_g_v2_epoch_energy_wh": state_features.get("epoch_energy_wh"),
            "rapec_g_v2_gpu_utilization_pct": state_features.get(
                "gpu_utilization_pct"
            ),
            "rapec_g_v2_gpu_memory_used_mb": state_features.get(
                "gpu_memory_used_mb"
            ),
            "rapec_g_v2_gpu_power_avg_w": state_features.get("gpu_power_avg_w"),
            "rapec_g_v2_log_model_parameters": state_features.get(
                "log_model_parameters"
            ),
            "rapec_g_v2_log_model_flops": state_features.get("log_model_flops"),
        }
        reason = base_decision.reason
        if plateau_stop:
            reason = (
                "rapec_g_v2_plateau_stop: "
                f"epoch={observation.epoch}, regime={evidence.regime}, "
                f"p_gain={evidence.local_probability:.4f}, "
                f"upper_gain={evidence.local_upper_gain:.6f}, "
                f"non_improving={evidence.non_improving_epochs}"
            )
        elif base_stop_accepted:
            reason = base_decision.reason
        elif not stop:
            reason = f"continue:{evidence.regime}"
        return replace(
            base_decision,
            stop=stop,
            reason=reason,
            confidence=confidence,
            predicted_energy_saving_fraction=(
                predicted_saving
                if predicted_saving is not None
                else base_decision.predicted_energy_saving_fraction
            ),
            predicted_quality_regret=max(
                float(base_decision.predicted_quality_regret or 0.0),
                evidence.local_upper_gain,
            ),
            diagnostics=diagnostics,
        )

    def _curve_regime(
        self,
        rows: list[Mapping[str, Any]],
        qualities: list[float],
        observation: EpochObservation,
        thresholds,
    ) -> CurveRegimeEvidence:
        best = _best_so_far(qualities)
        short_horizon = min(
            max(1, self.horizon_epochs * self.regime_horizon_multipliers[0]),
            max(1, len(best) - 1),
        )
        long_horizon = min(
            max(
                short_horizon,
                self.horizon_epochs * self.regime_horizon_multipliers[-1],
            ),
            max(1, len(best) - 1),
        )
        deltas = [right - left for left, right in zip(qualities, qualities[1:])]
        recent_deltas = deltas[-max(self.trend_window, long_horizon) :]
        delta_noise = max(_mad(recent_deltas), self.min_meaningful_gain)
        short_noise = max(
            delta_noise * math.sqrt(short_horizon), self.min_meaningful_gain
        )
        long_noise = max(
            delta_noise * math.sqrt(long_horizon), self.min_meaningful_gain
        )
        short_gain = (
            max(0.0, best[-1] - best[-1 - short_horizon])
            if len(best) > short_horizon
            else 0.0
        )
        long_gain = (
            max(0.0, best[-1] - best[-1 - long_horizon])
            if len(best) > long_horizon
            else short_gain
        )
        short_gain_z = short_gain / short_noise
        long_gain_z = long_gain / long_noise

        historical_gains = self._completed_horizon_gains(
            best[:-short_horizon] if len(best) > short_horizon else [],
            short_horizon,
        )
        historical_floor = _quantile(
            historical_gains, self.plateau_relative_quantile
        )
        low_relative_progress = short_gain <= max(
            thresholds.meaningful_gain,
            float(historical_floor or 0.0),
            self.plateau_noise_z * short_noise,
        ) and long_gain <= max(
            thresholds.meaningful_gain * (long_horizon / short_horizon),
            self.plateau_noise_z * long_noise,
        )

        dynamic_delta = max(
            self.min_meaningful_gain,
            self.plateau_noise_z * delta_noise,
            thresholds.meaningful_gain / max(1, short_horizon),
        )
        current_best = best[-1] if best else None
        if self._comparison_best is None and current_best is not None:
            self._comparison_best = current_best
            self._non_improving_epochs = 0
        elif (
            current_best is not None
            and self._comparison_best is not None
            and current_best > self._comparison_best + dynamic_delta
        ):
            self._comparison_best = current_best
            self._non_improving_epochs = 0
        else:
            self._non_improving_epochs += 1

        signal = statistics.median(abs(value) for value in recent_deltas) if recent_deltas else 0.0
        noise_fraction = delta_noise / max(
            delta_noise + signal, self.min_meaningful_gain
        )
        dynamic_patience = int(
            round(short_horizon * (0.6 + noise_fraction))
        )
        dynamic_patience = min(
            self.plateau_patience_max,
            max(self.plateau_patience_min, dynamic_patience),
        )

        local_expected_gain = max(
            0.0, (statistics.median(recent_deltas) if recent_deltas else 0.0)
        ) * short_horizon
        local_upper_gain = max(
            0.0,
            local_expected_gain + self.plateau_noise_z * short_noise,
        )
        local_probability = self._normal_probability_above(
            thresholds.meaningful_gain,
            local_expected_gain,
            short_noise,
        )

        utility_values = self._historical_utilities(rows, observation)
        utility_threshold = _quantile(
            utility_values[:-1] if len(utility_values) > 1 else utility_values,
            self.plateau_utility_quantile,
        )
        recent_energy = sum(
            self._energy_wh(row) for row in rows[-short_horizon:]
        )
        recent_utility = short_gain / recent_energy if recent_energy > 0.0 else None
        low_energy_utility = (
            recent_utility is not None
            and (
                utility_threshold is None
                or recent_utility <= max(0.0, utility_threshold)
            )
        )

        base_state = self._learning_state(rows, observation, thresholds)
        loss_recovery = self._loss_recovery_signal(rows)
        gradient_recovery = self._gradient_recovery_signal(rows)
        late_learning_guard = (
            short_gain > thresholds.meaningful_gain
            or long_gain
            > thresholds.meaningful_gain * (long_horizon / short_horizon)
            or (loss_recovery and gradient_recovery)
        )
        if len(best) <= long_horizon:
            regime = "insufficient_history"
        elif base_state == "unstable":
            regime = "noisy_learning"
        elif late_learning_guard:
            regime = "late_learning"
        elif base_state == "overfit_risk":
            regime = "overfit_risk"
        elif low_relative_progress and (
            self._seen_significant_learning or self._warmup_released
        ):
            regime = "early_plateau"
        else:
            regime = "undertrained"
        return CurveRegimeEvidence(
            regime=regime,
            short_gain=short_gain,
            long_gain=long_gain,
            short_gain_z=short_gain_z,
            long_gain_z=long_gain_z,
            local_probability=local_probability,
            local_upper_gain=local_upper_gain,
            dynamic_patience=dynamic_patience,
            non_improving_epochs=self._non_improving_epochs,
            low_relative_progress=low_relative_progress,
            low_energy_utility=low_energy_utility,
            late_learning_guard=late_learning_guard,
        )

    @staticmethod
    def _completed_horizon_gains(
        best: list[float], horizon: int
    ) -> list[float]:
        return [
            max(0.0, best[end] - best[end - horizon])
            for end in range(horizon, len(best))
        ]

    @staticmethod
    def _normal_probability_above(
        threshold: float, mean: float, standard_deviation: float
    ) -> float:
        if standard_deviation <= 0.0:
            return 1.0 if mean > threshold else 0.0
        z_value = (threshold - mean) / standard_deviation
        return max(
            0.0,
            min(1.0, 0.5 * math.erfc(z_value / math.sqrt(2.0))),
        )

    def _loss_recovery_signal(self, rows: list[Mapping[str, Any]]) -> bool:
        losses = [
            value
            for row in rows[-self.trend_window :]
            if (value := _number(row.get("train_loss"))) is not None
        ]
        if len(losses) < 4:
            return False
        improvements = [
            left - right for left, right in zip(losses, losses[1:])
        ]
        typical_improvement = statistics.median(improvements)
        return typical_improvement > self.loss_recovery_z * max(
            _mad(improvements), self.min_meaningful_gain
        )

    def _gradient_recovery_signal(self, rows: list[Mapping[str, Any]]) -> bool:
        gradients = [
            value
            for row in rows[-self.trend_window :]
            if (value := _number(row.get("gradient_norm"))) is not None
            and value >= 0.0
        ]
        if len(gradients) < 4:
            return False
        gradient_reference = _quantile(gradients[:-1], 0.5)
        return (
            gradient_reference is not None
            and gradients[-1] >= gradient_reference
        )
