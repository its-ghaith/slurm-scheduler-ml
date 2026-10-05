from __future__ import annotations

import math
import random
import statistics
from dataclasses import dataclass
from typing import Mapping

from controller_benchmark.api import ControllerDecision, EpochObservation
from controller_benchmark.controllers.risk_aware_predictive_energy import (
    RiskAwarePredictiveEnergyController,
    _best_so_far,
    _mad,
    _quantile,
)


@dataclass(frozen=True)
class BayesianHorizonPosterior:
    expected_gain: float | None
    lower_gain: float | None
    upper_gain: float | None
    probability_relevant_gain: float | None
    expected_energy_wh: float | None
    expected_utility_per_wh: float | None
    posterior_mean_delta: float | None
    posterior_delta_stddev: float | None
    observations: int
    samples: int


class BayesianGuardedRapecV3Controller(RiskAwarePredictiveEnergyController):
    """RAPEC-v3 with a lightweight online Bayesian stop guard.

    RAPEC-v3 remains the primary decision rule. A stop is released only when a
    Normal-Inverse-Gamma posterior, fitted to the current run's recent
    best-quality increments, also assigns low probability to a meaningful gain
    over the same horizon. No historical curves or task-specific priors are
    used by the Bayesian layer.
    """

    def __init__(self, context):
        super().__init__(context)
        params = context.parameters
        self.bayesian_window = max(
            self.min_fit_points,
            int(params.get("bayesian_window", max(20, self.trend_window * 2))),
        )
        self.bayesian_samples = max(64, int(params.get("bayesian_samples", 512)))
        self.bayesian_min_observations = max(
            4,
            int(params.get("bayesian_min_observations", 8)),
        )
        self.bayesian_probability_limit = min(
            1.0,
            max(
                0.0,
                float(
                    params.get(
                        "bayesian_probability_limit",
                        0.35,
                    )
                ),
            ),
        )
        self.bayesian_prior_strength = max(
            1e-6,
            float(params.get("bayesian_prior_strength", 0.5)),
        )
        self.bayesian_prior_alpha = max(
            1.01,
            float(params.get("bayesian_prior_alpha", 2.0)),
        )
        self.bayesian_confirmation_patience = max(
            1,
            int(params.get("bayesian_confirmation_patience", 1)),
        )
        self.bayesian_confirmation_streak = 0

    def evaluate(self, observation: EpochObservation) -> ControllerDecision:
        base_decision = super().evaluate(observation)
        rows = [*observation.history, observation.raw_metrics]
        relevant_gain = float(
            base_decision.diagnostics["rapec_dynamic_meaningful_gain"]
        )
        posterior = self._bayesian_horizon_posterior(
            rows,
            observation,
            relevant_gain,
        )

        posterior_ready = (
            posterior.observations >= self.bayesian_min_observations
            and posterior.probability_relevant_gain is not None
        )
        posterior_low_gain = (
            posterior_ready
            and float(posterior.probability_relevant_gain or 0.0)
            <= self.bayesian_probability_limit
        )
        bayesian_candidate = base_decision.stop and posterior_low_gain
        self.bayesian_confirmation_streak = (
            self.bayesian_confirmation_streak + 1 if bayesian_candidate else 0
        )
        stop = (
            bayesian_candidate
            and self.bayesian_confirmation_streak
            >= self.bayesian_confirmation_patience
        )

        probability = posterior.probability_relevant_gain
        bayesian_confidence = (
            max(
                0.0,
                min(
                    1.0,
                    1.0
                    - float(probability or 0.0)
                    / max(self.bayesian_probability_limit, 1e-9),
                ),
            )
            if posterior_ready
            else 0.0
        )
        base_confidence = float(base_decision.confidence or 0.0)
        confidence = (
            math.sqrt(base_confidence * bayesian_confidence) if stop else 0.0
        )

        diagnostics = dict(base_decision.diagnostics)
        diagnostics.update(
            {
                "rapec8_is_bayesian": 1,
                "rapec8_v3_primary_decision": 1,
                "rapec8_online_current_run_only": 1,
                "rapec8_uses_historical_curves": 0,
                "rapec8_uses_task_or_dataset_prior": 0,
                "rapec8_energy_scope_training_only": 1,
                "rapec8_horizon_epochs": self.horizon_epochs,
                "rapec8_dynamic_relevant_gain": relevant_gain,
                "rapec8_posterior_expected_gain": posterior.expected_gain,
                "rapec8_posterior_gain_lower": posterior.lower_gain,
                "rapec8_posterior_gain_upper": posterior.upper_gain,
                "rapec8_posterior_probability_relevant_gain": (
                    posterior.probability_relevant_gain
                ),
                "rapec8_probability_limit": self.bayesian_probability_limit,
                "rapec8_expected_energy_wh": posterior.expected_energy_wh,
                "rapec8_expected_utility_per_wh": (
                    posterior.expected_utility_per_wh
                ),
                "rapec8_posterior_mean_epoch_delta": (
                    posterior.posterior_mean_delta
                ),
                "rapec8_posterior_epoch_delta_stddev": (
                    posterior.posterior_delta_stddev
                ),
                "rapec8_posterior_observations": posterior.observations,
                "rapec8_posterior_samples": posterior.samples,
                "rapec8_posterior_ready": int(posterior_ready),
                "rapec8_base_v3_stop": int(base_decision.stop),
                "rapec8_bayesian_low_gain": int(posterior_low_gain),
                "rapec8_bayesian_veto": int(base_decision.stop and not stop),
                "rapec8_candidate_stop": int(bayesian_candidate),
                "rapec8_confirmation_streak": self.bayesian_confirmation_streak,
            }
        )

        if stop:
            reason = (
                "rapec_v8_bayesian_v3_stop: "
                f"epoch={observation.epoch}, horizon={self.horizon_epochs}, "
                f"v3_reason=({base_decision.reason}), "
                f"dynamic_gain={relevant_gain:.6f}, "
                f"posterior_expected_gain={float(posterior.expected_gain or 0.0):.6f}, "
                f"posterior_prob_relevant_gain={float(probability or 0.0):.4f}, "
                f"probability_limit={self.bayesian_probability_limit:.4f}"
            )
        else:
            reason = "continue"

        return ControllerDecision(
            stop=stop,
            reason=reason,
            confidence=confidence,
            predicted_energy_saving_fraction=(
                base_decision.predicted_energy_saving_fraction
            ),
            predicted_quality_regret=posterior.upper_gain,
            diagnostics=diagnostics,
        )

    def _bayesian_horizon_posterior(
        self,
        rows: list[Mapping[str, object]],
        observation: EpochObservation,
        relevant_gain: float,
    ) -> BayesianHorizonPosterior:
        qualities = self._quality_values(rows, observation)
        best = _best_so_far(qualities)
        increments = [
            max(0.0, right - left)
            for left, right in zip(best, best[1:])
        ][-self.bayesian_window :]
        n = len(increments)
        if n < 2:
            return BayesianHorizonPosterior(
                expected_gain=None,
                lower_gain=None,
                upper_gain=None,
                probability_relevant_gain=None,
                expected_energy_wh=self._recent_horizon_energy(rows),
                expected_utility_per_wh=None,
                posterior_mean_delta=None,
                posterior_delta_stddev=None,
                observations=n,
                samples=0,
            )

        sample_mean = statistics.fmean(increments)
        centered_sum_squares = sum(
            (value - sample_mean) ** 2 for value in increments
        )
        robust_scale = _mad(increments)
        numerical_scale = max(
            relevant_gain / max(2.0 * self.horizon_epochs, 1.0),
            self.min_meaningful_gain / max(self.horizon_epochs, 1),
            1e-7,
        )

        # This weak prior contains no dataset, task, or Full100 information.
        prior_mean = 0.0
        prior_kappa = self.bayesian_prior_strength
        prior_alpha = self.bayesian_prior_alpha
        prior_scale = max(robust_scale, numerical_scale)
        prior_beta = prior_scale**2 * (prior_alpha - 1.0)

        posterior_kappa = prior_kappa + n
        posterior_mean = (
            prior_kappa * prior_mean + n * sample_mean
        ) / posterior_kappa
        posterior_alpha = prior_alpha + n / 2.0
        posterior_beta = (
            prior_beta
            + 0.5 * centered_sum_squares
            + (
                prior_kappa
                * n
                * (sample_mean - prior_mean) ** 2
                / (2.0 * posterior_kappa)
            )
        )
        posterior_beta = max(posterior_beta, 1e-18)

        seed = (
            observation.epoch * 1_000_003
            + int((best[-1] if best else 0.0) * 1_000_000)
            + sum(ord(char) for char in self.context.controller_id)
        )
        rng = random.Random(seed)
        room = max(0.0, 1.0 - (best[-1] if best else 0.0))
        gain_samples: list[float] = []
        variance_samples: list[float] = []
        for _ in range(self.bayesian_samples):
            precision = rng.gammavariate(
                posterior_alpha,
                1.0 / posterior_beta,
            )
            variance = 1.0 / max(precision, 1e-18)
            mean_delta = rng.gauss(
                posterior_mean,
                math.sqrt(variance / posterior_kappa),
            )
            horizon_gain = rng.gauss(
                self.horizon_epochs * mean_delta,
                math.sqrt(self.horizon_epochs * variance),
            )
            gain_samples.append(min(room, max(0.0, horizon_gain)))
            variance_samples.append(variance)

        expected_gain = statistics.fmean(gain_samples)
        lower_gain = _quantile(gain_samples, 0.10)
        upper_gain = _quantile(gain_samples, 0.90)
        probability = sum(
            gain > relevant_gain for gain in gain_samples
        ) / len(gain_samples)
        expected_energy = self._recent_horizon_energy(rows)
        utility = (
            expected_gain / expected_energy
            if expected_energy is not None and expected_energy > 0.0
            else None
        )
        posterior_variance_mean = statistics.fmean(variance_samples)

        return BayesianHorizonPosterior(
            expected_gain=expected_gain,
            lower_gain=lower_gain,
            upper_gain=upper_gain,
            probability_relevant_gain=probability,
            expected_energy_wh=expected_energy,
            expected_utility_per_wh=utility,
            posterior_mean_delta=posterior_mean,
            posterior_delta_stddev=math.sqrt(posterior_variance_mean),
            observations=n,
            samples=len(gain_samples),
        )
