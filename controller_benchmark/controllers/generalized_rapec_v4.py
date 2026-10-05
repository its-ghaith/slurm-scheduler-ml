from __future__ import annotations

import math
from dataclasses import dataclass, replace

from controller_benchmark.api import (
    ControllerContext,
    EpochObservation,
)

from .generalized_rapec_v3 import GeneralizedRapecV3Controller
from .risk_aware_predictive_energy import _best_so_far, _mad, _number
from .standard_early_stopping import StandardEarlyStoppingController


@dataclass(frozen=True)
class BurstEvidence:
    lookback_epochs: int
    largest_record_gain: float
    recent_volatility: float
    current_drawdown: float
    jump_to_gain_ratio: float
    volatility_to_gain_ratio: float
    drawdown_to_volatility_ratio: float
    burst_risk: bool


class GeneralizedRapecV4Controller(GeneralizedRapecV3Controller):
    """RAPEC-G v3 with a task-independent continuation-dominance gate.

    A conventional early-stopping controller runs in shadow mode. Once that
    conservative reference says stop, RAPEC may continue only when its online
    evidence supports useful energy-normalized progress or when the current
    trace exhibits delayed, step-like quality jumps. The latter is protected
    for an energy-adaptive observation horizon. No task, dataset, model, or
    initialization profile is consulted.
    """

    def __init__(self, context: ControllerContext):
        super().__init__(context)
        params = context.parameters
        reference_context = replace(
            context,
            controller_id=f"{context.controller_id}-standard-es-reference",
            parameters={
                "start_epoch_fraction": float(
                    params.get("dominance_es_start_epoch_fraction", 0.20)
                ),
                "patience_fraction": float(
                    params.get("dominance_es_patience_fraction", 0.10)
                ),
                "min_delta": float(
                    params.get("dominance_es_min_delta", 0.001)
                ),
            },
        )
        self._reference_es = StandardEarlyStoppingController(reference_context)
        self.burst_jump_ratio = max(
            1.0, float(params.get("burst_jump_ratio", 1.5))
        )
        self.burst_volatility_ratio = max(
            0.0, float(params.get("burst_volatility_ratio", 0.65))
        )
        self.burst_drawdown_ratio = max(
            0.0, float(params.get("burst_drawdown_ratio", 2.5))
        )
        self._burst_protected_until = 0
        self._burst_armed_best: float | None = None

    def evaluate(self, observation: EpochObservation) -> ControllerDecision:
        base_decision = super().evaluate(observation)
        reference_decision = self._reference_es.evaluate(observation)
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
        burst = self._burst_evidence(qualities, meaningful_gain)
        current_best = max(qualities) if qualities else None
        new_meaningful_record = (
            current_best is not None
            and self._burst_armed_best is not None
            and current_best > self._burst_armed_best + meaningful_gain
        )
        hold_epochs = self._energy_adaptive_hold_epochs(
            rows,
            observation,
            meaningful_gain,
            diagnostics,
        )

        if new_meaningful_record and observation.epoch <= self._burst_protected_until:
            self._burst_armed_best = current_best
            self._burst_protected_until = observation.epoch + hold_epochs

        can_arm_burst = (
            self._burst_armed_best is None
            or (
                current_best is not None
                and current_best > self._burst_armed_best + meaningful_gain
            )
        )
        if (
            reference_decision.stop
            and burst.burst_risk
            and can_arm_burst
        ):
            self._burst_armed_best = current_best
            self._burst_protected_until = observation.epoch + hold_epochs

        burst_protected = observation.epoch <= self._burst_protected_until
        continuation_justified = self._continuation_is_energy_justified(
            diagnostics
        )
        dominance_stop = (
            reference_decision.stop
            and not burst_protected
            and not continuation_justified
        )
        stop = base_decision.stop or dominance_stop

        confidence = base_decision.confidence
        reason = base_decision.reason
        if dominance_stop and not base_decision.stop:
            confidence = max(float(confidence or 0.0), 0.90)
            reason = (
                "rapec_g_v4_reference_dominance_stop: "
                f"epoch={observation.epoch}, "
                f"reference_best_epoch="
                f"{reference_decision.diagnostics.get('es_best_epoch')}, "
                f"burst_risk={int(burst.burst_risk)}"
            )
        elif not stop and reference_decision.stop and burst_protected:
            reason = (
                "continue:rapec_g_v4_delayed_burst_protection: "
                f"until={self._burst_protected_until}"
            )

        updated_diagnostics = {
            **diagnostics,
            "rapec_g_v4_enabled": 1,
            "rapec_g_v4_task_profile_used": 0,
            "rapec_g_v4_current_run_only": 1,
            "rapec_g_v4_reference_es_stop": int(reference_decision.stop),
            "rapec_g_v4_reference_es_best_epoch": (
                reference_decision.diagnostics.get("es_best_epoch")
            ),
            "rapec_g_v4_reference_es_non_improving_epochs": (
                reference_decision.diagnostics.get("es_non_improving_epochs")
            ),
            "rapec_g_v4_burst_lookback_epochs": burst.lookback_epochs,
            "rapec_g_v4_largest_record_gain": burst.largest_record_gain,
            "rapec_g_v4_recent_volatility": burst.recent_volatility,
            "rapec_g_v4_current_drawdown": burst.current_drawdown,
            "rapec_g_v4_jump_to_gain_ratio": burst.jump_to_gain_ratio,
            "rapec_g_v4_volatility_to_gain_ratio": (
                burst.volatility_to_gain_ratio
            ),
            "rapec_g_v4_drawdown_to_volatility_ratio": (
                burst.drawdown_to_volatility_ratio
            ),
            "rapec_g_v4_burst_risk": int(burst.burst_risk),
            "rapec_g_v4_burst_hold_epochs": hold_epochs,
            "rapec_g_v4_burst_protected_until": self._burst_protected_until,
            "rapec_g_v4_burst_protected": int(burst_protected),
            "rapec_g_v4_continuation_energy_justified": int(
                continuation_justified
            ),
            "rapec_g_v4_dominance_stop": int(dominance_stop),
            "rapec_g_v4_base_stop": int(base_decision.stop),
        }
        return replace(
            base_decision,
            stop=stop,
            reason=reason,
            confidence=confidence,
            diagnostics=updated_diagnostics,
        )

    def close(self) -> None:
        self._reference_es.close()
        super().close()

    def _burst_evidence(
        self,
        qualities: list[float],
        meaningful_gain: float,
    ) -> BurstEvidence:
        lookback = max(self.trend_window * 2, self.horizon_epochs * 4)
        recent = qualities[-(lookback + 1) :]
        best = _best_so_far(recent)
        record_gains = [
            max(0.0, right - left)
            for left, right in zip(best, best[1:])
            if right > left
        ]
        largest_record_gain = max(record_gains, default=0.0)
        volatility_window = min(self.trend_window, len(qualities))
        recent_volatility = (
            _mad(qualities[-volatility_window:])
            if volatility_window > 1
            else 0.0
        )
        current_drawdown = (
            max(0.0, max(qualities) - qualities[-1]) if qualities else 0.0
        )
        gain_scale = max(meaningful_gain, self.min_meaningful_gain)
        volatility_scale = max(recent_volatility, self.min_meaningful_gain)
        jump_ratio = largest_record_gain / gain_scale
        volatility_ratio = recent_volatility / gain_scale
        drawdown_ratio = current_drawdown / volatility_scale
        burst_risk = (
            len(qualities) >= self.trend_window + 1
            and jump_ratio >= self.burst_jump_ratio
            and volatility_ratio >= self.burst_volatility_ratio
            and drawdown_ratio >= self.burst_drawdown_ratio
        )
        return BurstEvidence(
            lookback_epochs=lookback,
            largest_record_gain=largest_record_gain,
            recent_volatility=recent_volatility,
            current_drawdown=current_drawdown,
            jump_to_gain_ratio=jump_ratio,
            volatility_to_gain_ratio=volatility_ratio,
            drawdown_to_volatility_ratio=drawdown_ratio,
            burst_risk=burst_risk,
        )

    def _energy_adaptive_hold_epochs(
        self,
        rows,
        observation: EpochObservation,
        meaningful_gain: float,
        diagnostics,
    ) -> int:
        recent_energy = self._recent_epoch_energy(rows)
        utility_threshold = _number(
            diagnostics.get("rapec_dynamic_utility_threshold")
        )
        if (
            recent_energy is not None
            and recent_energy > 0.0
            and utility_threshold is not None
            and utility_threshold > 0.0
        ):
            estimated = math.ceil(
                meaningful_gain / (recent_energy * utility_threshold)
            )
        else:
            estimated = max(self.trend_window, 2 * self.horizon_epochs)
        return min(
            max(1, observation.max_epochs - observation.epoch),
            max(
                self.horizon_epochs,
                min(4 * self.horizon_epochs, estimated),
            ),
        )

    def _continuation_is_energy_justified(self, diagnostics) -> bool:
        probability = _number(
            diagnostics.get("rapec_prob_gain_gt_dynamic_threshold")
        )
        utility = _number(diagnostics.get("rapec_utility_quality_per_wh"))
        utility_threshold = _number(
            diagnostics.get("rapec_dynamic_utility_threshold")
        )
        return bool(
            probability is not None
            and probability > self.max_probability_gain_gt_threshold
            and utility is not None
            and utility_threshold is not None
            and utility > max(0.0, utility_threshold)
        )
