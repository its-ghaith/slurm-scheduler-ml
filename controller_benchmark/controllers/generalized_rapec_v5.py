from __future__ import annotations

import math
from dataclasses import dataclass, replace

from controller_benchmark.api import ControllerDecision, EpochObservation

from .generalized_rapec_v4 import GeneralizedRapecV4Controller
from .risk_aware_predictive_energy import _best_so_far, _mad, _number


@dataclass(frozen=True)
class ReadinessEvidence:
    window_epochs: int
    total_meaningful_record_gains: int
    recent_meaningful_record_gains: int
    dry_spell_epochs: int
    dry_spell_required_epochs: int
    recent_record_gain: float
    probability_any_gain_next_horizon: float
    recent_volatility: float
    mature_plateau: bool


class GeneralizedRapecV5Controller(GeneralizedRapecV4Controller):
    """RAPEC-G v4 with a profile-free dynamic readiness gate.

    RAPEC-G v4 can develop preferred early stop zones because enough history is
    often available around similar epochs. V5 keeps the same predictive and
    energy-aware decision logic, but accepts a stop only after the current run
    itself shows that the learning curve is mature: at least one meaningful
    record improvement has happened, recent record progress is low, and the
    dry spell since the last meaningful record is long relative to the observed
    curve noise. No task name, dataset name, scratch/pretrained flag, or model
    profile is used.
    """

    def __init__(self, context):
        super().__init__(context)
        params = context.parameters
        self.readiness_min_record_gains = max(
            1,
            int(params.get("readiness_min_record_gains", 1)),
        )
        self.readiness_dry_spell_multiplier = max(
            0.5,
            float(params.get("readiness_dry_spell_multiplier", 1.0)),
        )
        self.readiness_probability_ceiling = min(
            1.0,
            max(
                0.0,
                float(
                    params.get(
                        "readiness_probability_ceiling",
                        self.plateau_probability_ceiling,
                    )
                ),
            ),
        )
        self.readiness_noise_multiplier = max(
            0.0,
            float(params.get("readiness_noise_multiplier", 1.0)),
        )

    def evaluate(self, observation: EpochObservation) -> ControllerDecision:
        decision = super().evaluate(observation)
        rows = [*observation.history, observation.raw_metrics]
        qualities = self._quality_values(rows, observation)
        meaningful_gain = max(
            self.min_meaningful_gain,
            float(
                _number(
                    decision.diagnostics.get("rapec_dynamic_meaningful_gain"),
                    self.min_meaningful_gain,
                )
                or self.min_meaningful_gain
            ),
        )
        readiness = self._readiness_evidence(
            qualities,
            observation,
            meaningful_gain,
            decision.diagnostics,
        )
        accepted_stop = bool(decision.stop and readiness.mature_plateau)
        deferred_stop = bool(decision.stop and not readiness.mature_plateau)
        diagnostics = {
            **decision.diagnostics,
            "rapec_g_v5_enabled": 1,
            "rapec_g_v5_task_profile_used": 0,
            "rapec_g_v5_current_run_only": 1,
            "rapec_g_v5_readiness_window_epochs": readiness.window_epochs,
            "rapec_g_v5_total_meaningful_record_gains": (
                readiness.total_meaningful_record_gains
            ),
            "rapec_g_v5_recent_meaningful_record_gains": (
                readiness.recent_meaningful_record_gains
            ),
            "rapec_g_v5_dry_spell_epochs": readiness.dry_spell_epochs,
            "rapec_g_v5_dry_spell_required_epochs": (
                readiness.dry_spell_required_epochs
            ),
            "rapec_g_v5_recent_record_gain": readiness.recent_record_gain,
            "rapec_g_v5_probability_any_gain_next_horizon": (
                readiness.probability_any_gain_next_horizon
            ),
            "rapec_g_v5_recent_volatility": readiness.recent_volatility,
            "rapec_g_v5_mature_plateau": int(readiness.mature_plateau),
            "rapec_g_v5_deferred_stop": int(deferred_stop),
            "rapec_g_v5_accepted_stop": int(accepted_stop),
        }
        if not deferred_stop:
            return replace(
                decision,
                diagnostics=diagnostics,
            )
        confidence = min(float(decision.confidence or 0.0), 0.49)
        return replace(
            decision,
            stop=False,
            reason=(
                "continue:rapec_g_v5_dynamic_readiness_gate: "
                f"dry_spell={readiness.dry_spell_epochs}/"
                f"{readiness.dry_spell_required_epochs}, "
                f"p_any_gain={readiness.probability_any_gain_next_horizon:.4f}, "
                f"record_gains={readiness.total_meaningful_record_gains}"
            ),
            confidence=confidence,
            diagnostics=diagnostics,
        )

    def _readiness_evidence(
        self,
        qualities: list[float],
        observation: EpochObservation,
        meaningful_gain: float,
        diagnostics,
    ) -> ReadinessEvidence:
        window_epochs = max(
            self.trend_window,
            self.horizon_epochs * 2,
            int(math.ceil(math.sqrt(max(1, observation.max_epochs)))),
        )
        if len(qualities) < 2:
            return ReadinessEvidence(
                window_epochs=window_epochs,
                total_meaningful_record_gains=0,
                recent_meaningful_record_gains=0,
                dry_spell_epochs=0,
                dry_spell_required_epochs=window_epochs,
                recent_record_gain=math.inf,
                probability_any_gain_next_horizon=1.0,
                recent_volatility=math.inf,
                mature_plateau=False,
            )
        best = _best_so_far(qualities)
        record_gains = [
            max(0.0, right - left)
            for left, right in zip(best, best[1:])
        ]
        significant = [gain > meaningful_gain for gain in record_gains]
        total_meaningful = sum(1 for value in significant if value)
        recent = record_gains[-window_epochs:]
        recent_meaningful = sum(1 for gain in recent if gain > meaningful_gain)
        recent_record_gain = sum(recent)
        last_index = None
        for index, value in enumerate(significant, start=2):
            if value:
                last_index = index
        dry_spell = (
            observation.epoch - last_index
            if last_index is not None
            else observation.epoch - 1
        )
        deltas = [
            right - left for left, right in zip(qualities, qualities[1:])
        ]
        recent_volatility = _mad(deltas[-window_epochs:])
        dynamic_patience = int(
            _number(
                diagnostics.get("rapec_g_v2_dynamic_patience"),
                max(1, window_epochs // 2),
            )
            or max(1, window_epochs // 2)
        )
        dry_required = max(
            self.horizon_epochs,
            int(math.ceil(dynamic_patience * self.readiness_dry_spell_multiplier)),
        )

        posterior_alpha = 0.5 + recent_meaningful
        posterior_beta = 0.5 + max(0, len(recent) - recent_meaningful)
        probability_no_gain = 1.0
        for offset in range(self.horizon_epochs):
            probability_no_gain *= (
                posterior_beta + offset
            ) / (posterior_alpha + posterior_beta + offset)
        probability_any_gain = max(0.0, min(1.0, 1.0 - probability_no_gain))

        reference_es_stop = bool(
            diagnostics.get("rapec_g_v4_reference_es_stop", 0)
        )
        late_learning_guard = bool(
            diagnostics.get("rapec_g_v2_late_learning_guard", 0)
        )
        low_recent_progress = recent_record_gain <= max(
            meaningful_gain,
            self.readiness_noise_multiplier * recent_volatility,
            self.min_meaningful_gain,
        )
        enough_learning_seen = (
            total_meaningful >= self.readiness_min_record_gains
            or reference_es_stop
        )
        mature_plateau = (
            enough_learning_seen
            and recent_meaningful == 0
            and dry_spell >= dry_required
            and low_recent_progress
            and probability_any_gain <= self.readiness_probability_ceiling
            and not late_learning_guard
        )
        return ReadinessEvidence(
            window_epochs=window_epochs,
            total_meaningful_record_gains=total_meaningful,
            recent_meaningful_record_gains=recent_meaningful,
            dry_spell_epochs=max(0, dry_spell),
            dry_spell_required_epochs=dry_required,
            recent_record_gain=recent_record_gain,
            probability_any_gain_next_horizon=probability_any_gain,
            recent_volatility=recent_volatility,
            mature_plateau=mature_plateau,
        )
