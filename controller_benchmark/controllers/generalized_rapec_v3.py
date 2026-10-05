from __future__ import annotations

import math
from dataclasses import dataclass, replace

from controller_benchmark.api import ControllerDecision, EpochObservation

from .generalized_rapec_v2 import GeneralizedRapecV2Controller
from .risk_aware_predictive_energy import _best_so_far, _mad, _number, _quantile


@dataclass(frozen=True)
class StationarityEvidence:
    window_epochs: int
    meaningful_record_gains: int
    probability_any_gain_next_horizon: float
    recent_volatility: float
    historical_volatility_limit: float
    volatility_limit: float
    volatility_ok: bool


class GeneralizedRapecV3Controller(GeneralizedRapecV2Controller):
    """RAPEC-G v2 plus profile-free online stationarity detection.

    The additional route targets long, expensive runs whose best quality has
    stopped moving even though ordinary prediction remains conservative. A
    Jeffreys-prior Beta-Binomial posterior estimates the probability of at
    least one meaningful record gain in the next prediction horizon. A robust
    volatility guard suppresses this route during unusually noisy regimes.
    Only measurements from the current run are used.
    """

    def __init__(self, context):
        super().__init__(context)
        params = context.parameters
        self.stationarity_window_horizon_multiplier = max(
            1,
            int(params.get("stationarity_window_horizon_multiplier", 3)),
        )
        self.stationarity_volatility_quantile = min(
            0.95,
            max(0.5, float(params.get("stationarity_volatility_quantile", 0.75))),
        )
        self.stationarity_confirmation_windows = max(
            1,
            int(
                params.get(
                    "stationarity_confirmation_windows",
                    self.plateau_confirmation_windows,
                )
            ),
        )
        self._stationarity_streak = 0

    def evaluate(self, observation: EpochObservation) -> ControllerDecision:
        base_decision = super().evaluate(observation)
        rows = [*observation.history, observation.raw_metrics]
        qualities = self._quality_values(rows, observation)
        diagnostics = base_decision.diagnostics
        meaningful_gain = max(
            self.min_meaningful_gain,
            float(
                _number(
                    diagnostics.get("rapec_dynamic_meaningful_gain"),
                    self.min_meaningful_gain,
                )
                or self.min_meaningful_gain
            ),
        )
        evidence = self._stationarity_evidence(
            qualities,
            observation,
            meaningful_gain,
        )

        current_quality = (
            observation.best_quality
            if observation.best_quality is not None
            else observation.quality
        )
        validation_noise = max(
            0.0,
            float(_number(diagnostics.get("rapec_validation_noise"), 0.0) or 0.0),
        )
        epoch_fraction = max(
            0.0,
            min(1.0, observation.epoch / max(1, observation.max_epochs)),
        )
        minimum_quality = max(
            self.minimum_quality_floor,
            min(0.95, epoch_fraction * 0.10 - validation_noise),
        )
        structural_ready = (
            observation.epoch >= self._structural_history_floor(observation)
            and len(qualities) >= evidence.window_epochs + 1
        )
        quality_guard = (
            current_quality is not None
            and float(current_quality) >= minimum_quality
        )
        low_energy_utility = bool(
            diagnostics.get("rapec_g_v2_low_energy_utility", 0)
        )
        late_learning_guard = bool(
            diagnostics.get("rapec_g_v2_late_learning_guard", 0)
        )
        stationarity_candidate = (
            structural_ready
            and evidence.probability_any_gain_next_horizon
            <= self.plateau_probability_ceiling
            and evidence.volatility_ok
            and low_energy_utility
            and not late_learning_guard
            and quality_guard
        )
        self._stationarity_streak = (
            self._stationarity_streak + 1 if stationarity_candidate else 0
        )
        stationarity_stop = (
            self._stationarity_streak >= self.stationarity_confirmation_windows
        )
        stop = base_decision.stop or stationarity_stop

        confidence = base_decision.confidence
        if stationarity_stop:
            probability_confidence = (
                1.0 - evidence.probability_any_gain_next_horizon
            )
            streak_confidence = min(
                1.0,
                self._stationarity_streak
                / max(1, self.stationarity_confirmation_windows),
            )
            confidence = max(
                float(confidence or 0.0),
                0.75 * probability_confidence + 0.25 * streak_confidence,
            )

        updated_diagnostics = {
            **diagnostics,
            "rapec_g_v3_enabled": 1,
            "rapec_g_v3_task_profile_used": 0,
            "rapec_g_v3_current_run_only": 1,
            "rapec_g_v3_stationarity_window_epochs": evidence.window_epochs,
            "rapec_g_v3_meaningful_record_gains": (
                evidence.meaningful_record_gains
            ),
            "rapec_g_v3_probability_any_gain_next_horizon": (
                evidence.probability_any_gain_next_horizon
            ),
            "rapec_g_v3_recent_volatility": evidence.recent_volatility,
            "rapec_g_v3_historical_volatility_limit": (
                evidence.historical_volatility_limit
            ),
            "rapec_g_v3_volatility_limit": evidence.volatility_limit,
            "rapec_g_v3_volatility_ok": int(evidence.volatility_ok),
            "rapec_g_v3_structural_ready": int(structural_ready),
            "rapec_g_v3_quality_guard_ok": int(quality_guard),
            "rapec_g_v3_stationarity_candidate": int(
                stationarity_candidate
            ),
            "rapec_g_v3_stationarity_streak": self._stationarity_streak,
            "rapec_g_v3_stationarity_stop": int(stationarity_stop),
            "rapec_g_v3_base_stop": int(base_decision.stop),
        }
        reason = base_decision.reason
        if stationarity_stop and not base_decision.stop:
            reason = (
                "rapec_g_v3_stationarity_stop: "
                f"epoch={observation.epoch}, "
                "p_any_gain="
                f"{evidence.probability_any_gain_next_horizon:.4f}, "
                f"record_gains={evidence.meaningful_record_gains}, "
                f"volatility={evidence.recent_volatility:.6f}"
            )
        elif not stop:
            reason = base_decision.reason
        return replace(
            base_decision,
            stop=stop,
            reason=reason,
            confidence=confidence,
            diagnostics=updated_diagnostics,
        )

    def _stationarity_evidence(
        self,
        qualities: list[float],
        observation: EpochObservation,
        meaningful_gain: float,
    ) -> StationarityEvidence:
        window_epochs = max(
            self.trend_window,
            self.horizon_epochs * self.stationarity_window_horizon_multiplier,
            int(math.ceil(math.sqrt(max(1, observation.max_epochs)))),
        )
        if len(qualities) < window_epochs + 1:
            return StationarityEvidence(
                window_epochs=window_epochs,
                meaningful_record_gains=window_epochs,
                probability_any_gain_next_horizon=1.0,
                recent_volatility=math.inf,
                historical_volatility_limit=0.0,
                volatility_limit=0.0,
                volatility_ok=False,
            )

        best = _best_so_far(qualities)
        recent_best = best[-(window_epochs + 1) :]
        meaningful_record_gains = sum(
            1
            for left, right in zip(recent_best, recent_best[1:])
            if right - left > self.min_meaningful_gain
        )

        # Jeffreys prior Beta(1/2, 1/2) avoids zero-probability conclusions.
        posterior_alpha = 0.5 + meaningful_record_gains
        posterior_beta = 0.5 + window_epochs - meaningful_record_gains
        probability_no_gain = 1.0
        for offset in range(self.horizon_epochs):
            probability_no_gain *= (
                posterior_beta + offset
            ) / (posterior_alpha + posterior_beta + offset)
        probability_any_gain = 1.0 - probability_no_gain

        volatility_window = min(self.trend_window, len(qualities))
        recent_volatility = _mad(qualities[-volatility_window:])
        historical_volatilities = [
            _mad(qualities[end - volatility_window : end])
            for end in range(volatility_window, len(qualities))
        ]
        historical_limit = float(
            _quantile(
                historical_volatilities,
                self.stationarity_volatility_quantile,
            )
            or 0.0
        )
        volatility_limit = max(
            self.min_meaningful_gain,
            min(
                meaningful_gain,
                historical_limit if historical_limit > 0.0 else meaningful_gain,
            ),
        )
        volatility_ok = recent_volatility <= volatility_limit
        return StationarityEvidence(
            window_epochs=window_epochs,
            meaningful_record_gains=meaningful_record_gains,
            probability_any_gain_next_horizon=max(
                0.0,
                min(1.0, probability_any_gain),
            ),
            recent_volatility=recent_volatility,
            historical_volatility_limit=historical_limit,
            volatility_limit=volatility_limit,
            volatility_ok=volatility_ok,
        )
