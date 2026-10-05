from __future__ import annotations

import math
import statistics
from dataclasses import dataclass, replace
from typing import Any, Mapping

from controller_benchmark.api import ControllerDecision, EpochObservation

from .rapec_v3_pf import ProfileFreeRapecV3Controller
from .risk_aware_predictive_energy import _best_so_far, _mad, _median, _number


@dataclass(frozen=True)
class PreferenceWarmupProfile:
    energy_preference: float
    automatic_difficulty: float
    effective_difficulty: float
    difficulty_confidence: float
    difficulty_prior_weight: float
    aggressiveness: float
    significance_z: float
    low_progress_quantile: float
    required_low_progress_windows: int
    minimum_reference_points: int
    no_learning_multiplier: float


class PreferenceConditionedRapecV3PfController(ProfileFreeRapecV3Controller):
    """Profile-free RAPEC-v3 with user preference and online difficulty.

    The inherited RAPEC-v3 prediction and stop rule are unchanged. Only the
    profile-free warm-up gate is conditioned on a 1..10 energy preference and
    an optional 1..10 task-difficulty prior. Difficulty evidence is derived
    from the current run without inspecting task, dataset, scenario, or model
    labels. All progress confirmation uses exactly one five-epoch horizon.
    """

    def __init__(self, context):
        super().__init__(context)
        params = context.parameters
        self.energy_priority_level = self._required_level(
            params.get("energy_priority_level", 5), "energy_priority_level"
        )
        self.task_difficulty_level = self._optional_level(
            params.get("task_difficulty_level")
        )
        self.preference_energy_weight = max(
            0.0, float(params.get("preference_energy_weight", 0.75))
        )
        self.preference_difficulty_weight = max(
            0.0, float(params.get("preference_difficulty_weight", 0.25))
        )
        if self.preference_energy_weight + self.preference_difficulty_weight <= 0:
            raise ValueError("Preference weights must have a positive sum.")

        self.difficulty_estimation_window = max(
            2, int(params.get("difficulty_estimation_window", 5))
        )
        self.difficulty_min_observations = max(
            2, int(params.get("difficulty_min_observations", 8))
        )
        self.difficulty_update_rate = self._clamp(
            float(params.get("difficulty_update_rate", 0.25))
        )
        self.difficulty_prior_strength = max(
            0.0, float(params.get("difficulty_prior_strength", 12.0))
        )
        self.difficulty_minimum_prior_weight = self._clamp(
            float(params.get("difficulty_minimum_prior_weight", 0.35))
        )

        self.protective_significance_z = max(
            0.0, float(params.get("protective_significance_z", 2.58))
        )
        self.energy_significance_z = max(
            0.0, float(params.get("energy_significance_z", 0.84))
        )
        self.protective_low_progress_quantile = self._clamp_quantile(
            params.get("protective_low_progress_quantile", 0.10)
        )
        self.energy_low_progress_quantile = self._clamp_quantile(
            params.get("energy_low_progress_quantile", 0.50)
        )
        self.protective_required_windows = max(
            1, int(params.get("protective_required_windows", 4))
        )
        self.energy_required_windows = max(
            1, int(params.get("energy_required_windows", 1))
        )
        self.protective_min_reference_points = max(
            2, int(params.get("protective_min_reference_points", 8))
        )
        self.energy_min_reference_points = max(
            2, int(params.get("energy_min_reference_points", 2))
        )
        self.protective_no_learning_multiplier = max(
            1.0, float(params.get("protective_no_learning_multiplier", 2.50))
        )
        self.energy_no_learning_multiplier = max(
            1.0, float(params.get("energy_no_learning_multiplier", 1.25))
        )

        # Keep the scientific comparison fixed to the requested five epochs.
        self.warmup_confirmation_horizon_multipliers = (1,)
        self._automatic_difficulty = 0.5
        self._last_difficulty_epoch = 0
        self._last_preference_diagnostics: dict[str, float | int | None] = {}

    def evaluate(self, observation: EpochObservation) -> ControllerDecision:
        rows = [*observation.history, observation.raw_metrics]
        profile = self._preference_profile(rows, observation)
        self._apply_preference_profile(profile)
        decision = super().evaluate(observation)
        diagnostics = {
            **decision.diagnostics,
            **self._last_preference_diagnostics,
            "rapec_pc_task_independent": 1,
            "rapec_pc_uses_task_or_scenario_profile": 0,
            "rapec_pc_confirmation_horizon_epochs": self.horizon_epochs,
        }
        reason = decision.reason
        if decision.stop and reason.startswith("rapec_stop:"):
            reason = reason.replace("rapec_stop:", "rapec_pc_stop:", 1)
        return replace(decision, reason=reason, diagnostics=diagnostics)

    def _preference_profile(
        self,
        rows: list[Mapping[str, Any]],
        observation: EpochObservation,
    ) -> PreferenceWarmupProfile:
        automatic, confidence, components = self._estimate_difficulty(
            rows, observation
        )
        evidence = max(0, len(self._quality_values(rows, observation)) - 1)
        prior_weight = 0.0
        effective_difficulty = automatic
        if self.task_difficulty_level is not None:
            manual = self._normalize_level(self.task_difficulty_level)
            decay = self.difficulty_prior_strength / max(
                self.difficulty_prior_strength + evidence, 1e-12
            )
            prior_weight = self.difficulty_minimum_prior_weight + (
                1.0 - self.difficulty_minimum_prior_weight
            ) * decay
            effective_difficulty = (
                prior_weight * manual + (1.0 - prior_weight) * automatic
            )

        energy_preference = self._normalize_level(self.energy_priority_level)
        weight_sum = (
            self.preference_energy_weight + self.preference_difficulty_weight
        )
        aggressiveness = self._clamp(
            (
                self.preference_energy_weight * energy_preference
                + self.preference_difficulty_weight
                * (1.0 - effective_difficulty)
            )
            / weight_sum
        )
        profile = PreferenceWarmupProfile(
            energy_preference=energy_preference,
            automatic_difficulty=automatic,
            effective_difficulty=effective_difficulty,
            difficulty_confidence=confidence,
            difficulty_prior_weight=prior_weight,
            aggressiveness=aggressiveness,
            significance_z=self._lerp(
                self.protective_significance_z,
                self.energy_significance_z,
                aggressiveness,
            ),
            low_progress_quantile=self._lerp(
                self.protective_low_progress_quantile,
                self.energy_low_progress_quantile,
                aggressiveness,
            ),
            required_low_progress_windows=self._lerp_int(
                self.protective_required_windows,
                self.energy_required_windows,
                aggressiveness,
            ),
            minimum_reference_points=self._lerp_int(
                self.protective_min_reference_points,
                self.energy_min_reference_points,
                aggressiveness,
            ),
            no_learning_multiplier=self._lerp(
                self.protective_no_learning_multiplier,
                self.energy_no_learning_multiplier,
                aggressiveness,
            ),
        )
        self._last_preference_diagnostics = {
            "rapec_pc_energy_priority_level": self.energy_priority_level,
            "rapec_pc_task_difficulty_input_level": self.task_difficulty_level,
            "rapec_pc_energy_preference": energy_preference,
            "rapec_pc_automatic_difficulty": automatic,
            "rapec_pc_effective_difficulty": effective_difficulty,
            "rapec_pc_effective_difficulty_level": 1.0
            + 9.0 * effective_difficulty,
            "rapec_pc_difficulty_confidence": confidence,
            "rapec_pc_difficulty_prior_weight": prior_weight,
            "rapec_pc_aggressiveness": aggressiveness,
            "rapec_pc_significance_z": profile.significance_z,
            "rapec_pc_low_progress_quantile": profile.low_progress_quantile,
            "rapec_pc_required_low_progress_windows": (
                profile.required_low_progress_windows
            ),
            "rapec_pc_min_reference_points": profile.minimum_reference_points,
            "rapec_pc_no_learning_multiplier": profile.no_learning_multiplier,
            **components,
        }
        return profile

    def _estimate_difficulty(
        self,
        rows: list[Mapping[str, Any]],
        observation: EpochObservation,
    ) -> tuple[float, float, dict[str, float]]:
        qualities = self._quality_values(rows, observation)
        if not qualities:
            return 0.5, 0.0, self._difficulty_components(0.5, 0.5, 0.5, 0.5)
        best = _best_so_far(qualities)
        observations = len(qualities)
        confidence = self._clamp(
            (observations - 1) / max(1, self.difficulty_min_observations - 1)
        )
        epoch_fraction = self._clamp(
            observation.epoch / max(1, observation.max_epochs)
        )
        current_best = best[-1]
        attainment_difficulty = 1.0 - current_best
        reference_path = max(0.10, math.sqrt(max(epoch_fraction, 1e-9)))
        trajectory_difficulty = 1.0 - self._clamp(
            current_best / reference_path
        )

        window = min(self.difficulty_estimation_window, max(1, observations - 1))
        raw_deltas = [right - left for left, right in zip(qualities, qualities[1:])]
        best_deltas = [right - left for left, right in zip(best, best[1:])]
        recent_raw = raw_deltas[-window:]
        recent_best = best_deltas[-window:]
        positive_progress = _median(
            [max(0.0, value) for value in recent_best]
        ) or 0.0
        noise = _mad(recent_raw)
        noise_difficulty = noise / max(
            noise + positive_progress + self.min_meaningful_gain, 1e-12
        )

        losses = [
            value
            for row in rows
            if (value := _number(row.get("train_loss"))) is not None
            and value >= 0.0
        ]
        loss_difficulty = 0.5
        if len(losses) >= 2:
            reference_loss = statistics.median(losses[: min(3, len(losses))])
            recent_loss = statistics.median(losses[-min(3, len(losses)) :])
            if reference_loss > 0.0:
                loss_difficulty = self._clamp(recent_loss / reference_loss)

        raw_difficulty = self._clamp(
            0.35 * attainment_difficulty
            + 0.35 * trajectory_difficulty
            + 0.20 * noise_difficulty
            + 0.10 * loss_difficulty
        )
        evidence_adjusted = 0.5 * (1.0 - confidence) + raw_difficulty * confidence
        if observation.epoch != self._last_difficulty_epoch:
            self._automatic_difficulty = (
                (1.0 - self.difficulty_update_rate) * self._automatic_difficulty
                + self.difficulty_update_rate * evidence_adjusted
            )
            self._last_difficulty_epoch = observation.epoch
        components = self._difficulty_components(
            attainment_difficulty,
            trajectory_difficulty,
            noise_difficulty,
            loss_difficulty,
        )
        return self._clamp(self._automatic_difficulty), confidence, components

    def _apply_preference_profile(self, profile: PreferenceWarmupProfile) -> None:
        self.warmup_significance_z = profile.significance_z
        self.warmup_low_progress_quantile = profile.low_progress_quantile
        self.warmup_required_low_progress_windows = (
            profile.required_low_progress_windows
        )
        self.warmup_min_reference_points = profile.minimum_reference_points
        self.warmup_no_learning_multiplier = profile.no_learning_multiplier
        self.warmup_confirmation_horizon_multipliers = (1,)

    @staticmethod
    def _difficulty_components(
        attainment: float,
        trajectory: float,
        noise: float,
        loss: float,
    ) -> dict[str, float]:
        return {
            "rapec_pc_difficulty_attainment_component": attainment,
            "rapec_pc_difficulty_trajectory_component": trajectory,
            "rapec_pc_difficulty_noise_component": noise,
            "rapec_pc_difficulty_loss_component": loss,
        }

    @staticmethod
    def _required_level(value: Any, name: str) -> int:
        try:
            level = int(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{name} must be an integer from 1 to 10.") from exc
        if level < 1 or level > 10:
            raise ValueError(f"{name} must be an integer from 1 to 10.")
        return level

    @classmethod
    def _optional_level(cls, value: Any) -> int | None:
        if value in (None, "", 0, "0", "auto"):
            return None
        return cls._required_level(value, "task_difficulty_level")

    @staticmethod
    def _normalize_level(level: int) -> float:
        return (level - 1) / 9.0

    @staticmethod
    def _clamp(value: float) -> float:
        return max(0.0, min(1.0, float(value)))

    @classmethod
    def _clamp_quantile(cls, value: Any) -> float:
        return min(0.50, cls._clamp(float(value)))

    @staticmethod
    def _lerp(protective: float, energy: float, weight: float) -> float:
        return protective + (energy - protective) * weight

    @classmethod
    def _lerp_int(cls, protective: int, energy: int, weight: float) -> int:
        value = cls._lerp(float(protective), float(energy), weight)
        return max(1, int(math.floor(value + 0.5)))
