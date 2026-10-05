from __future__ import annotations

import math
import statistics
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np

from controller_benchmark.api import ControllerDecision, EpochObservation
from controller_benchmark.controllers.online_bayesian_rapec import (
    BayesianEnsemble,
    OnlineBayesianRapecController,
    _best_so_far,
    _clamp,
    _mad,
    _number,
)


def _finite_sample_quantile(values: Sequence[float], coverage: float) -> float:
    """Return the split-conformal quantile with finite-sample correction."""
    clean = sorted(float(value) for value in values if math.isfinite(float(value)))
    if not clean:
        return 0.0
    rank = int(math.ceil((len(clean) + 1) * _clamp(coverage, 0.0, 1.0)))
    return clean[min(len(clean), max(1, rank)) - 1]


@dataclass(frozen=True)
class RiskConstrainedForecast:
    horizon: int
    expected_gain: float
    upper_gain: float
    probability_relevant_gain: float
    expected_energy_wh: float
    expected_duration_seconds: float
    expected_excess_utility_per_wh: float
    upper_excess_utility_per_wh: float
    calibration_radius: float
    samples: int


class RiskConstrainedBayesianMultiHorizonController(OnlineBayesianRapecController):
    """RAPEC-v9: online risk-constrained Bayesian optimal stopping.

    The controller uses only the current run. It predicts quality, training
    energy, and duration over several horizons, calibrates forecast uncertainty
    prequentially, and treats quality non-inferiority as a hard probabilistic
    constraint. Energy is optimized only inside the quality-feasible region.
    """

    def __init__(self, context):
        super().__init__(context)
        parameters = context.parameters
        self.quality_risk_alpha = _clamp(
            float(parameters.get("quality_risk_alpha", 0.05)),
            0.005,
            0.25,
        )
        self.minimum_calibration_points = max(
            4,
            int(parameters.get("minimum_calibration_points", 8)),
        )
        self.calibration_window = max(
            self.minimum_calibration_points,
            int(parameters.get("calibration_window", 40)),
        )
        self.evidence_decay = _clamp(
            float(parameters.get("evidence_decay", 0.50)),
            0.10,
            0.99,
        )
        self.evidence_stop_probability = _clamp(
            float(parameters.get("evidence_stop_probability", 0.95)),
            0.80,
            0.999,
        )
        self.quality_noise_quantile = _clamp(
            float(parameters.get("quality_noise_quantile", 0.75)),
            0.50,
            0.95,
        )
        self.log_stop_evidence = 0.0
        self.quality_noise_history: list[float] = []
        self.instantaneous_quality_epsilon = self.numerical_quality_floor

    def _allow_quality_telemetry(self) -> bool:
        # RAPEC-v9 learns predictive associations from the current run only.
        # Static parameter/FLOP context is absorbed by the intercept; no causal
        # cross-model claim is made without historical runs.
        return True

    def evaluate(self, observation: EpochObservation) -> ControllerDecision:
        rows = [*observation.history, observation.raw_metrics]
        qualities = self._quality_values(rows, observation)
        best = _best_so_far(qualities)
        current_best = best[-1] if best else float(
            observation.best_quality or observation.quality or 0.0
        )
        remaining_epochs = max(0, observation.max_epochs - observation.epoch)
        active_horizons = self._active_horizons(remaining_epochs)
        maximum_horizon = max(active_horizons, default=0)
        effective_sample_size = self._effective_sample_size(qualities)
        dynamic_epsilon = self._dynamic_quality_equivalence(
            qualities,
            effective_sample_size,
        )
        preliminary_observations = max(
            self.minimum_observations_floor + self.minimum_calibration_points,
            2 * max(self.horizons, default=1),
        )
        look_alpha = self.quality_risk_alpha
        if len(qualities) < preliminary_observations:
            feature_coverage = self._feature_coverage(rows, observation)
            evidence_probability = self._update_stop_evidence(False, 1.0)
            diagnostics = self._v9_diagnostics(
                observation=observation,
                forecasts=[],
                quality_ensemble=None,
                energy_ensemble=None,
                duration_ensemble=None,
                current_best=current_best,
                dynamic_epsilon=dynamic_epsilon,
                utility_floor=0.0,
                effective_sample_size=effective_sample_size,
                calibration_points=len(self._gain_calibration_errors(best, 1)),
                minimum_observations=preliminary_observations,
                feature_coverage=feature_coverage,
                model_ready=False,
                quality_constraint_met=False,
                energy_low_value=False,
                lr_transition=False,
                regime_recovery=False,
                record_waiting_recovery=False,
                candidate=False,
                evidence_probability=evidence_probability,
                sequential_look_alpha=look_alpha,
            )
            return ControllerDecision(
                stop=False,
                reason="rapec_v9_dynamic_readiness",
                confidence=0.0,
                diagnostics=diagnostics,
            )

        quality_ensemble = (
            self._quality_ensemble(rows, observation, maximum_horizon)
            if maximum_horizon
            else None
        )
        energy_ensemble = (
            self._resource_ensemble(rows, observation, maximum_horizon, "energy")
            if maximum_horizon
            else None
        )
        duration_ensemble = (
            self._resource_ensemble(rows, observation, maximum_horizon, "duration")
            if maximum_horizon
            else None
        )

        calibration_residuals = self._gain_calibration_errors(best, 1)
        minimum_observations = self._readiness_observations(quality_ensemble)
        look_alpha = self.quality_risk_alpha
        forecasts = self._risk_forecasts(
            observation=observation,
            active_horizons=active_horizons,
            current_best=current_best,
            dynamic_epsilon=dynamic_epsilon,
            best=best,
            quality_ensemble=quality_ensemble,
            energy_ensemble=energy_ensemble,
            duration_ensemble=duration_ensemble,
            quality_risk_alpha=look_alpha,
        )
        utility_floor = self._dynamic_excess_utility_floor(
            rows,
            observation,
            dynamic_epsilon,
        )
        feature_coverage = self._feature_coverage(rows, observation)
        lr_transition = self._learning_rate_transition(rows)
        telemetry_prior_weight = (
            1.0 / len(quality_ensemble.weights)
            if quality_ensemble and quality_ensemble.weights
            else 1.0
        )
        telemetry_quality_model_available = bool(
            quality_ensemble
            and (
                feature_coverage >= 1.0 - self.numerical_quality_floor
                or quality_ensemble.weights.get("telemetry", 0.0)
                >= telemetry_prior_weight
            )
        )
        record_waiting_recovery = (
            not telemetry_quality_model_available
            and self._record_waiting_time_recovery(best)
        )
        regime_recovery = self._regime_recovery(
            best,
            min(dynamic_epsilon, self.instantaneous_quality_epsilon),
        ) or record_waiting_recovery

        model_ready = (
            len(qualities) >= minimum_observations
            and len(calibration_residuals) >= self.minimum_calibration_points
            and quality_ensemble is not None
            and energy_ensemble is not None
            and duration_ensemble is not None
            and bool(forecasts)
        )
        maximum_gain_probability = max(
            (forecast.probability_relevant_gain for forecast in forecasts),
            default=1.0,
        )
        maximum_expected_utility = max(
            (forecast.expected_excess_utility_per_wh for forecast in forecasts),
            default=math.inf,
        )
        quality_constraint_met = (
            bool(forecasts)
            and maximum_gain_probability <= look_alpha
        )
        energy_low_value = (
            bool(forecasts)
            and maximum_expected_utility <= utility_floor
        )
        candidate = (
            model_ready
            and quality_constraint_met
            and energy_low_value
            and not lr_transition
            and not regime_recovery
        )
        evidence_probability = self._update_stop_evidence(
            candidate,
            maximum_gain_probability,
        )
        stop = candidate and evidence_probability >= self.evidence_stop_probability

        diagnostics = self._v9_diagnostics(
            observation=observation,
            forecasts=forecasts,
            quality_ensemble=quality_ensemble,
            energy_ensemble=energy_ensemble,
            duration_ensemble=duration_ensemble,
            current_best=current_best,
            dynamic_epsilon=dynamic_epsilon,
            utility_floor=utility_floor,
            effective_sample_size=effective_sample_size,
            calibration_points=len(calibration_residuals),
            minimum_observations=minimum_observations,
            feature_coverage=feature_coverage,
            model_ready=model_ready,
            quality_constraint_met=quality_constraint_met,
            energy_low_value=energy_low_value,
            lr_transition=lr_transition,
            regime_recovery=regime_recovery,
            record_waiting_recovery=record_waiting_recovery,
            telemetry_quality_model_available=telemetry_quality_model_available,
            candidate=candidate,
            evidence_probability=evidence_probability,
            sequential_look_alpha=look_alpha,
        )
        predicted_saving = self._projected_training_energy_saving(
            observation,
            energy_ensemble,
        )
        predicted_regret = max(
            (forecast.upper_gain for forecast in forecasts),
            default=None,
        )
        confidence = evidence_probability if candidate else max(
            0.0,
            1.0 - maximum_gain_probability,
        )

        if not model_ready:
            reason = "rapec_v9_dynamic_readiness"
        elif stop:
            reason = (
                "rapec_v9_risk_constrained_stop: "
                f"epoch={observation.epoch}, epsilon={dynamic_epsilon:.6f}, "
                f"max_prob_relevant_gain={maximum_gain_probability:.4f}, "
                f"family_risk_alpha={self.quality_risk_alpha:.4f}, "
                f"look_risk_alpha={look_alpha:.6f}, "
                f"max_expected_excess_utility={maximum_expected_utility:.6f}, "
                f"utility_floor={utility_floor:.6f}, "
                f"stop_probability={evidence_probability:.4f}"
            )
        else:
            reason = "continue"

        return ControllerDecision(
            stop=stop,
            reason=reason,
            confidence=_clamp(confidence, 0.0, 1.0),
            predicted_energy_saving_fraction=predicted_saving,
            predicted_quality_regret=predicted_regret,
            diagnostics=diagnostics,
        )

    def _active_horizons(self, remaining_epochs: int) -> list[int]:
        if remaining_epochs <= 0:
            return []
        horizons = {
            horizon for horizon in self.horizons if horizon <= remaining_epochs
        }
        # The endpoint horizon protects against a delayed improvement that is
        # invisible in short five- or ten-epoch forecasts.
        horizons.add(remaining_epochs)
        return sorted(horizons)

    def _risk_forecasts(
        self,
        *,
        observation: EpochObservation,
        active_horizons: list[int],
        current_best: float,
        dynamic_epsilon: float,
        best: list[float],
        quality_ensemble: BayesianEnsemble | None,
        energy_ensemble: BayesianEnsemble | None,
        duration_ensemble: BayesianEnsemble | None,
        quality_risk_alpha: float,
    ) -> list[RiskConstrainedForecast]:
        if not quality_ensemble or not energy_ensemble or not duration_ensemble:
            return []
        forecasts: list[RiskConstrainedForecast] = []
        coverage = 1.0 - quality_risk_alpha
        room = max(0.0, 1.0 - current_best)
        for horizon in active_horizons:
            seed = (
                observation.epoch * 1_000_003
                + horizon * 10_007
                + sum(ord(char) for char in self.context.controller_id)
            )
            quality_draws = self._draw_ensemble(
                quality_ensemble,
                horizon,
                seed,
                cumulative=False,
            )
            future_best = np.max(np.clip(quality_draws, 0.0, 1.0), axis=1)
            gains = np.maximum(0.0, future_best - current_best)
            calibration_radius = self._horizon_calibration_radius(
                best,
                horizon,
                coverage,
            )
            calibrated_gains = self._calibrated_gain_samples(
                gains,
                best=best,
                horizon=horizon,
                seed=seed + 97,
                room=room,
            )

            energy_draws = self._draw_ensemble(
                energy_ensemble,
                horizon,
                seed + 31,
                cumulative=True,
            )
            duration_draws = self._draw_ensemble(
                duration_ensemble,
                horizon,
                seed + 67,
                cumulative=True,
            )
            energy = np.maximum(
                energy_draws[:, -1],
                self.numerical_quality_floor,
            )
            duration = np.maximum(duration_draws[:, -1], 0.0)
            excess_utility = np.maximum(0.0, gains - dynamic_epsilon) / energy
            forecasts.append(
                RiskConstrainedForecast(
                    horizon=horizon,
                    expected_gain=float(np.median(gains)),
                    upper_gain=float(np.quantile(calibrated_gains, coverage)),
                    probability_relevant_gain=float(
                        np.mean(calibrated_gains > dynamic_epsilon)
                    ),
                    expected_energy_wh=float(np.median(energy)),
                    expected_duration_seconds=float(np.median(duration)),
                    expected_excess_utility_per_wh=float(
                        np.median(excess_utility)
                    ),
                    upper_excess_utility_per_wh=float(
                        np.quantile(excess_utility, coverage)
                    ),
                    calibration_radius=calibration_radius,
                    samples=int(gains.size),
                )
            )
        return forecasts

    def _dynamic_quality_equivalence(
        self,
        qualities: list[float],
        effective_sample_size: float,
    ) -> float:
        if len(qualities) < 3:
            return self.numerical_quality_floor
        values = np.asarray(qualities[-self.model_window :], dtype=float)
        differences = np.diff(values)
        innovation_noise = _mad(differences) / math.sqrt(2.0)
        late_count = max(5, int(math.ceil(math.sqrt(len(values)))))
        late_uncertainty = _mad(values[-late_count:])

        # This is the current-run counterpart of the benchmark's dynamic
        # non-inferiority tolerance. Robust scale estimates prevent the large
        # gains during warm-up from being mistaken for validation noise.
        # The equivalence margin is a two-sided 95% measurement tolerance.
        # It is intentionally independent from the controller's one-sided
        # posterior risk appetite so tuning risk cannot silently redefine what
        # counts as an acceptable quality loss.
        confidence = statistics.NormalDist().inv_cdf(0.975)
        # Adjacent epochs are not independent validation replicates. A
        # sqrt(n)-sized block correction gives a conservative effective sample
        # count between one observation and the invalid IID assumption.
        blocked_effective_size = math.sqrt(max(1.0, effective_sample_size))
        blocked_late_size = math.sqrt(max(1, late_count))
        innovation_standard_error = innovation_noise / math.sqrt(
            blocked_effective_size
        )
        late_standard_error = late_uncertainty / math.sqrt(blocked_late_size)
        detectable_noise = confidence * math.sqrt(
            innovation_standard_error**2 + late_standard_error**2
        )
        instantaneous = max(
            self.numerical_quality_floor,
            detectable_noise,
        )
        self.instantaneous_quality_epsilon = instantaneous
        readiness = max(
            self.minimum_observations_floor + self.minimum_calibration_points,
            2 * max(self.horizons, default=1),
        )
        if len(qualities) >= readiness:
            self.quality_noise_history.append(instantaneous)
            self.quality_noise_history = self.quality_noise_history[
                -self.calibration_window :
            ]
        if not self.quality_noise_history:
            return instantaneous
        return max(
            instantaneous,
            float(
                np.quantile(
                    self.quality_noise_history,
                    self.quality_noise_quantile,
                )
            ),
        )

    def _gain_calibration_errors(
        self,
        best: list[float],
        horizon: int,
    ) -> list[float]:
        if len(best) < self.minimum_observations_floor + horizon + 1:
            return []
        residuals = []
        for target in range(
            self.minimum_observations_floor + horizon,
            len(best),
        ):
            anchor = target - horizon
            prefix = best[: anchor + 1]
            increments = np.diff(
                np.asarray(prefix[-min(len(prefix), self.model_window) :], dtype=float)
            )
            center = float(np.median(np.maximum(increments, 0.0))) if increments.size else 0.0
            predicted_gain = max(0.0, center * horizon)
            observed_gain = max(0.0, best[target] - best[anchor])
            residuals.append(observed_gain - predicted_gain)
        return residuals[-self.calibration_window :]

    def _horizon_calibration_radius(
        self,
        best: list[float],
        horizon: int,
        coverage: float,
    ) -> float:
        errors = self._gain_calibration_errors(best, horizon)
        direct = (
            max(0.0, _finite_sample_quantile(errors, coverage))
            if len(errors) >= self.minimum_calibration_points
            else 0.0
        )
        one_step = self._gain_calibration_errors(best, 1)
        base = max(0.0, _finite_sample_quantile(one_step, coverage))
        # Forecast uncertainty grows sub-linearly because epoch errors are
        # strongly dependent. sqrt(log(1+h)) is more conservative than no
        # growth without exploding at the endpoint horizon.
        propagated = base * math.sqrt(math.log1p(max(1.0, float(horizon))))
        # A long-horizon residual history can contain only over-predictions.
        # Its one-sided radius would then collapse to zero even though delayed
        # improvement remains possible. Never let it undercut the empirically
        # calibrated one-step innovation bound.
        return min(1.0, max(direct, propagated))

    def _calibrated_gain_samples(
        self,
        gains: np.ndarray,
        *,
        best: list[float],
        horizon: int,
        seed: int,
        room: float,
    ) -> np.ndarray:
        """Augment posterior gains with online under-prediction residuals.

        A finite-sample conformal radius remains available as the conservative
        upper bound. For probabilities, sampling the observed residual
        distribution is better calibrated than shifting every posterior draw
        by that worst-case radius, which would force the stop probability to
        zero even on a stable plateau.
        """
        direct = self._gain_calibration_errors(best, horizon)
        one_step = self._gain_calibration_errors(best, 1)
        scale = math.sqrt(math.log1p(max(1.0, float(horizon))))
        rng = np.random.default_rng(seed)

        direct_pool = np.asarray(
            [max(0.0, value) for value in direct],
            dtype=float,
        )
        propagated_pool = np.asarray(
            [max(0.0, value) * scale for value in one_step],
            dtype=float,
        )
        correction = np.zeros(gains.size, dtype=float)
        if direct_pool.size >= self.minimum_calibration_points:
            correction = rng.choice(direct_pool, size=gains.size, replace=True)
        if propagated_pool.size >= self.minimum_calibration_points:
            propagated = rng.choice(
                propagated_pool,
                size=gains.size,
                replace=True,
            )
            correction = np.maximum(correction, propagated)
        return np.minimum(room, gains + correction)

    def _readiness_observations(
        self,
        ensemble: BayesianEnsemble | None,
    ) -> int:
        identifiable = (
            int(math.ceil(2.0 * ensemble.weighted_dimension + 2.0))
            if ensemble is not None
            else self.minimum_observations_floor
        )
        calibration = self.minimum_observations_floor + self.minimum_calibration_points
        horizon_evidence = 2 * max(self.horizons, default=1)
        return max(
            self.minimum_observations_floor,
            identifiable,
            calibration,
            horizon_evidence,
        )

    def _dynamic_excess_utility_floor(
        self,
        rows: list[Mapping[str, Any]],
        observation: EpochObservation,
        epsilon: float,
    ) -> float:
        qualities = _best_so_far(self._quality_values(rows, observation))
        utilities = []
        for left, right, row in zip(qualities, qualities[1:], rows[1:]):
            energy = self._energy_wh(row)
            gain = max(0.0, right - left - epsilon)
            if gain > 0.0 and energy > 0.0:
                utilities.append(gain / energy)
        if not utilities:
            return 0.0
        probability = max(
            self.quality_risk_alpha,
            1.0 / (len(utilities) + 1.0),
        )
        return max(0.0, float(np.quantile(utilities, probability)))

    def _learning_rate_transition(self, rows: list[Mapping[str, Any]]) -> bool:
        rates = [
            value
            for row in rows[-self.calibration_window :]
            if (
                value := _number(
                    row.get("learning_rate"),
                    _number(row.get("lr")),
                )
            )
            is not None
            and value > 0.0
        ]
        if len(rates) < 3:
            return False
        # A restart or warm-up increase is a genuine regime change. Smooth
        # monotone decay schedules are deliberately not blocked.
        return rates[-1] > rates[-2] and rates[-2] <= rates[-3]

    def _regime_recovery(self, best: list[float], epsilon: float) -> bool:
        if len(best) < 8:
            return False
        width = max(2, min(5, len(best) // 4))
        previous_gain = best[-width - 1] - best[-2 * width - 1]
        recent_gain = best[-1] - best[-width - 1]
        increments = np.diff(np.asarray(best[-2 * width - 1 :], dtype=float))
        noise = _mad(increments)
        short_recovery = (
            recent_gain > epsilon
            and recent_gain > previous_gain + noise
        )
        return short_recovery

    def _record_waiting_time_recovery(self, best: list[float]) -> bool:
        """Guard plateaus shorter than the run's empirical record intervals."""
        if len(best) < 4:
            return False
        record_epochs = [0]
        previous = best[0]
        for index, value in enumerate(best[1:], 1):
            if value > previous + self.numerical_quality_floor:
                record_epochs.append(index)
                previous = value
        if len(record_epochs) < 4:
            return False
        waiting_times = np.diff(np.asarray(record_epochs, dtype=float))
        expected_wait = _finite_sample_quantile(
            waiting_times.tolist(),
            1.0 - self.quality_risk_alpha,
        )
        current_wait = len(best) - 1 - record_epochs[-1]
        return current_wait <= expected_wait

    def _update_stop_evidence(
        self,
        candidate: bool,
        probability_relevant_gain: float,
    ) -> float:
        if candidate:
            resolution = 1.0 / (self.posterior_samples + 1.0)
            probability = _clamp(
                probability_relevant_gain,
                resolution,
                1.0 - resolution,
            )
            null_odds = (1.0 - self.quality_risk_alpha) / self.quality_risk_alpha
            observed_odds = (1.0 - probability) / probability
            log_bayes_factor = max(
                0.0,
                math.log(observed_odds) - math.log(null_odds),
            )
            target_log_odds = math.log(
                self.evidence_stop_probability
                / (1.0 - self.evidence_stop_probability)
            )
            # Bound one epoch's influence. The cap remains slightly above the
            # final threshold so persistent evidence can cross it, while the
            # decay lower bound guarantees that one candidate epoch cannot.
            log_bayes_factor = min(
                log_bayes_factor,
                1.05 * target_log_odds,
            )
            # Exponential averaging avoids treating highly correlated adjacent
            # epochs as independent evidence. Even a very strong single epoch
            # therefore cannot trigger a stop by itself.
            self.log_stop_evidence = (
                self.evidence_decay * self.log_stop_evidence
                + (1.0 - self.evidence_decay) * log_bayes_factor
            )
        else:
            self.log_stop_evidence *= self.evidence_decay
        return 1.0 / (1.0 + math.exp(-min(50.0, self.log_stop_evidence)))

    def _v9_diagnostics(
        self,
        *,
        observation: EpochObservation,
        forecasts: list[RiskConstrainedForecast],
        quality_ensemble: BayesianEnsemble | None,
        energy_ensemble: BayesianEnsemble | None,
        duration_ensemble: BayesianEnsemble | None,
        current_best: float,
        dynamic_epsilon: float,
        utility_floor: float,
        effective_sample_size: float,
        calibration_points: int,
        minimum_observations: int,
        feature_coverage: float,
        model_ready: bool,
        quality_constraint_met: bool,
        energy_low_value: bool,
        lr_transition: bool,
        regime_recovery: bool,
        record_waiting_recovery: bool,
        telemetry_quality_model_available: bool = False,
        candidate: bool,
        evidence_probability: float,
        sequential_look_alpha: float,
    ) -> dict[str, float | int | bool | None]:
        maximum_probability = max(
            (forecast.probability_relevant_gain for forecast in forecasts),
            default=None,
        )
        diagnostics: dict[str, float | int | bool | None] = {
            "rapec9_is_bayesian": 1,
            "rapec9_is_risk_constrained": 1,
            "rapec9_online_current_run_only": 1,
            "rapec9_uses_historical_curves": 0,
            "rapec9_uses_task_or_dataset_profile": 0,
            "rapec9_energy_scope_training_only": 1,
            "rapec9_current_best_quality": current_best,
            "rapec9_dynamic_quality_epsilon": dynamic_epsilon,
            "rapec9_instantaneous_quality_epsilon": (
                self.instantaneous_quality_epsilon
            ),
            "rapec9_quality_noise_quantile": self.quality_noise_quantile,
            "rapec9_quality_noise_history_points": len(self.quality_noise_history),
            "rapec9_quality_risk_alpha": self.quality_risk_alpha,
            "rapec9_quality_risk_scope": 1,
            "rapec9_sequential_look_alpha": sequential_look_alpha,
            "rapec9_dynamic_utility_floor": utility_floor,
            "rapec9_effective_sample_size": effective_sample_size,
            "rapec9_calibration_points": calibration_points,
            "rapec9_dynamic_minimum_observations": minimum_observations,
            "rapec9_feature_coverage_fraction": feature_coverage,
            "rapec9_model_ready": int(model_ready),
            "rapec9_quality_constraint_met": int(quality_constraint_met),
            "rapec9_energy_low_value": int(energy_low_value),
            "rapec9_learning_rate_transition": int(lr_transition),
            "rapec9_regime_recovery": int(regime_recovery),
            "rapec9_record_waiting_recovery": int(record_waiting_recovery),
            "rapec9_telemetry_quality_model_available": int(
                telemetry_quality_model_available
            ),
            "rapec9_candidate_stop": int(candidate),
            "rapec9_log_stop_evidence": self.log_stop_evidence,
            "rapec9_stop_evidence_probability": evidence_probability,
            "rapec9_evidence_stop_probability": self.evidence_stop_probability,
            "rapec9_max_probability_relevant_gain": maximum_probability,
            "rapec9_active_horizon_count": len(forecasts),
            "rapec9_remaining_horizon": max(
                0,
                observation.max_epochs - observation.epoch,
            ),
            "rapec9_quality_model_entropy": (
                quality_ensemble.entropy if quality_ensemble else None
            ),
            "rapec9_energy_model_entropy": (
                energy_ensemble.entropy if energy_ensemble else None
            ),
            "rapec9_duration_model_entropy": (
                duration_ensemble.entropy if duration_ensemble else None
            ),
        }
        for name in (
            "linear",
            "logarithmic",
            "saturation",
            "exponential",
            "change_point",
            "local_trend",
            "telemetry",
        ):
            diagnostics[f"rapec9_quality_model_weight_{name}"] = (
                quality_ensemble.weights.get(name, 0.0)
                if quality_ensemble
                else None
            )
        by_horizon = {forecast.horizon: forecast for forecast in forecasts}
        for horizon in self.horizons:
            forecast = by_horizon.get(horizon)
            prefix = f"rapec9_h{horizon}"
            diagnostics.update(
                {
                    f"{prefix}_expected_gain": (
                        forecast.expected_gain if forecast else None
                    ),
                    f"{prefix}_gain_upper": (
                        forecast.upper_gain if forecast else None
                    ),
                    f"{prefix}_prob_relevant_gain": (
                        forecast.probability_relevant_gain if forecast else None
                    ),
                    f"{prefix}_expected_energy_wh": (
                        forecast.expected_energy_wh if forecast else None
                    ),
                    f"{prefix}_expected_duration_seconds": (
                        forecast.expected_duration_seconds if forecast else None
                    ),
                    f"{prefix}_expected_excess_utility_per_wh": (
                        forecast.expected_excess_utility_per_wh
                        if forecast
                        else None
                    ),
                    f"{prefix}_calibration_radius": (
                        forecast.calibration_radius if forecast else None
                    ),
                }
            )
        remaining = max(0, observation.max_epochs - observation.epoch)
        endpoint = by_horizon.get(remaining)
        diagnostics.update(
            {
                "rapec9_endpoint_expected_gain": (
                    endpoint.expected_gain if endpoint else None
                ),
                "rapec9_endpoint_gain_upper": (
                    endpoint.upper_gain if endpoint else None
                ),
                "rapec9_endpoint_prob_relevant_gain": (
                    endpoint.probability_relevant_gain if endpoint else None
                ),
                "rapec9_endpoint_expected_energy_wh": (
                    endpoint.expected_energy_wh if endpoint else None
                ),
                "rapec9_endpoint_calibration_radius": (
                    endpoint.calibration_radius if endpoint else None
                ),
            }
        )
        return diagnostics
