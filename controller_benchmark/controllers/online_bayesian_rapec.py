from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Callable, Iterable, Mapping

import numpy as np

from controller_benchmark.api import Controller, ControllerDecision, EpochObservation


def _number(value: Any, default: float | None = None) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return default
    return result if math.isfinite(result) else default


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def _mad(values: Iterable[float]) -> float:
    clean = np.asarray([value for value in values if math.isfinite(value)], dtype=float)
    if clean.size < 2:
        return 0.0
    center = float(np.median(clean))
    return float(1.4826 * np.median(np.abs(clean - center)))


def _best_so_far(values: Iterable[float]) -> list[float]:
    result: list[float] = []
    best = 0.0
    for value in values:
        best = max(best, _clamp(float(value), 0.0, 1.0))
        result.append(best)
    return result


def _normalized_entropy(weights: Mapping[str, float]) -> float:
    values = np.asarray([value for value in weights.values() if value > 0.0], dtype=float)
    if values.size <= 1:
        return 0.0
    entropy = -float(np.sum(values * np.log(values)))
    return _clamp(entropy / math.log(values.size), 0.0, 1.0)


@dataclass(frozen=True)
class BayesianPosterior:
    name: str
    mean: np.ndarray
    precision_inverse: np.ndarray
    shape: float
    scale: float
    log_evidence: float
    scale_inflation: float
    sample_count: int

    @property
    def dimension(self) -> int:
        return int(self.mean.size)

    def draw(
        self,
        design: np.ndarray,
        sample_count: int,
        rng: np.random.Generator,
        *,
        include_observation_noise: bool = True,
    ) -> np.ndarray:
        matrix = np.atleast_2d(np.asarray(design, dtype=float))
        gamma = rng.gamma(self.shape, 1.0 / max(self.scale, 1e-12), size=sample_count)
        sigma_squared = 1.0 / np.maximum(gamma, 1e-12)
        covariance = self.precision_inverse
        try:
            root = np.linalg.cholesky(covariance + np.eye(covariance.shape[0]) * 1e-12)
        except np.linalg.LinAlgError:
            root = np.linalg.cholesky(np.linalg.pinv(np.linalg.pinv(covariance)) + np.eye(covariance.shape[0]) * 1e-9)
        coefficient_noise = rng.normal(size=(sample_count, self.dimension)) @ root.T
        coefficients = self.mean + coefficient_noise * np.sqrt(sigma_squared)[:, None]
        mean_prediction = coefficients @ matrix.T
        if not include_observation_noise:
            return mean_prediction
        observation_noise = rng.normal(
            0.0,
            np.sqrt(sigma_squared)[:, None] * self.scale_inflation,
            size=mean_prediction.shape,
        )
        return mean_prediction + observation_noise


@dataclass(frozen=True)
class BayesianEnsemble:
    posteriors: Mapping[str, BayesianPosterior]
    future_designs: Mapping[str, np.ndarray]
    weights: Mapping[str, float]
    target_transform: str

    @property
    def entropy(self) -> float:
        return _normalized_entropy(self.weights)

    @property
    def weighted_dimension(self) -> float:
        return sum(
            self.weights.get(name, 0.0) * posterior.dimension
            for name, posterior in self.posteriors.items()
        )

    @property
    def sample_count(self) -> int:
        return max((posterior.sample_count for posterior in self.posteriors.values()), default=0)


@dataclass(frozen=True)
class BayesianForecast:
    horizon: int
    relevant_gain: float
    expected_quality: float
    expected_gain: float
    lower_gain: float
    upper_gain: float
    probability_relevant_gain: float
    expected_energy_wh: float
    lower_energy_wh: float
    upper_energy_wh: float
    expected_duration_seconds: float
    lower_duration_seconds: float
    upper_duration_seconds: float
    expected_utility: float
    conservative_utility: float
    quality_interval_width: float


@dataclass(frozen=True)
class DynamicDecisionState:
    epsilon_reference: float
    risk_probability_limit: float
    utility_floor: float
    minimum_observations: int
    patience: int
    effective_sample_size: float
    validation_noise: float
    model_disagreement: float
    feature_coverage: float
    feature_recovery_probability: float
    posterior_recovery_probability: float


class OnlineBayesianRapecController(Controller):
    """RAPEC-v7: current-run-only Bayesian energy-aware early stopping.

    Quality, energy, and duration models are fitted after every epoch using
    conjugate Bayesian regression and Bayesian model averaging. No previous
    jobs, Full100 curves, task profiles, dataset names, or pretrained
    meta-models are consumed.
    """

    FEATURE_NAMES = (
        "quality",
        "train_loss",
        "gradient_norm",
        "learning_rate",
        "epoch_duration_seconds",
        "gpu_utilization_pct",
        "gpu_memory_used_mb",
        "gpu_power_avg_w",
        "epoch_energy_wh",
        "model_parameter_count",
        "model_flops",
    )

    TELEMETRY_NAMES = (
        "train_loss",
        "gradient_norm",
        "learning_rate",
        "epoch_duration_seconds",
        "gpu_utilization_pct",
        "gpu_memory_used_mb",
        "gpu_power_avg_w",
        "epoch_energy_wh",
        "model_parameter_count",
        "model_flops",
    )

    def __init__(self, context):
        super().__init__(context)
        parameters = context.parameters
        raw_horizons = parameters.get("horizons", [1, 3, 5, 10, 20])
        if not isinstance(raw_horizons, (list, tuple)):
            raise ValueError("RAPEC-v7 parameter 'horizons' must be a JSON array.")
        self.horizons = tuple(sorted({max(1, int(value)) for value in raw_horizons}))
        self.posterior_samples = max(512, int(parameters.get("posterior_samples", 1024)))
        self.model_window = max(16, int(parameters.get("model_window", 60)))
        self.minimum_observations_floor = max(8, int(parameters.get("minimum_observations_floor", 10)))
        self.maximum_confirmation_epochs = max(
            2,
            int(parameters.get("maximum_confirmation_epochs", math.ceil(math.sqrt(context.max_epochs)))),
        )
        self.numerical_quality_floor = max(
            np.finfo(float).eps,
            float(parameters.get("numerical_quality_floor", 1e-6)),
        )
        self.low_value_streak = 0

    def evaluate(self, observation: EpochObservation) -> ControllerDecision:
        rows = [*observation.history, observation.raw_metrics]
        qualities = self._quality_values(rows, observation)
        best = _best_so_far(qualities)
        current_best = best[-1] if best else float(observation.best_quality or observation.quality or 0.0)
        remaining_epochs = max(0, observation.max_epochs - observation.epoch)
        active_horizons = [horizon for horizon in self.horizons if horizon <= remaining_epochs]
        max_horizon = max(active_horizons, default=0)

        feature_coverage = self._feature_coverage(rows, observation)
        quality_ensemble = self._quality_ensemble(rows, observation, max_horizon) if max_horizon else None
        energy_ensemble = self._resource_ensemble(rows, observation, max_horizon, "energy") if max_horizon else None
        duration_ensemble = self._resource_ensemble(rows, observation, max_horizon, "duration") if max_horizon else None

        effective_sample_size = self._effective_sample_size(qualities)
        validation_noise = self._validation_noise(qualities)
        model_disagreement = quality_ensemble.entropy if quality_ensemble else 1.0
        epsilon_reference = self._dynamic_epsilon(qualities, effective_sample_size)
        risk_limit = self._dynamic_risk_limit(
            effective_sample_size,
            model_disagreement,
            quality_ensemble.sample_count if quality_ensemble else len(qualities),
        )
        utility_floor = self._dynamic_utility_floor(rows, observation, risk_limit)
        minimum_observations = self._dynamic_minimum_observations(quality_ensemble)
        patience = self._dynamic_patience(qualities)
        feature_recovery = self._feature_recovery_probability(rows, observation)

        forecasts = self._forecasts(
            observation=observation,
            active_horizons=active_horizons,
            current_best=current_best,
            epsilon_reference=epsilon_reference,
            risk_limit=risk_limit,
            quality_ensemble=quality_ensemble,
            energy_ensemble=energy_ensemble,
            duration_ensemble=duration_ensemble,
        )
        pareto_front = self._pareto_front(forecasts)
        forecast_recovery = max(
            (forecast.probability_relevant_gain for forecast in forecasts),
            default=1.0,
        )
        same_run_recovery = self._same_run_recovery_probability(
            best,
            epsilon_reference,
            max(active_horizons, default=1),
        )
        # Any independent recovery evidence is sufficient to continue. This
        # conservative union prevents a weak curve fit from overruling active
        # loss, gradient, or learning-rate signals.
        posterior_recovery = max(
            forecast_recovery,
            same_run_recovery,
            feature_recovery,
        )
        valuable_horizons = [
            forecast
            for forecast in pareto_front
            if (
                forecast.probability_relevant_gain > risk_limit
                or forecast.conservative_utility > utility_floor
            )
        ]

        state = DynamicDecisionState(
            epsilon_reference=epsilon_reference,
            risk_probability_limit=risk_limit,
            utility_floor=utility_floor,
            minimum_observations=minimum_observations,
            patience=patience,
            effective_sample_size=effective_sample_size,
            validation_noise=validation_noise,
            model_disagreement=model_disagreement,
            feature_coverage=feature_coverage,
            feature_recovery_probability=feature_recovery,
            posterior_recovery_probability=posterior_recovery,
        )
        model_ready = (
            len(qualities) >= minimum_observations
            and observation.epoch >= minimum_observations
            and bool(forecasts)
            and quality_ensemble is not None
            and energy_ensemble is not None
            and duration_ensemble is not None
        )
        candidate = (
            model_ready
            and not valuable_horizons
            and posterior_recovery <= risk_limit
        )
        self.low_value_streak = self.low_value_streak + 1 if candidate else 0
        stop = self.low_value_streak >= patience

        diagnostics = self._diagnostics(
            observation=observation,
            state=state,
            forecasts=forecasts,
            quality_ensemble=quality_ensemble,
            energy_ensemble=energy_ensemble,
            duration_ensemble=duration_ensemble,
            current_best=current_best,
            model_ready=model_ready,
            candidate=candidate,
            pareto_front_size=len(pareto_front),
            valuable_horizon_count=len(valuable_horizons),
        )
        projected_saving = self._projected_training_energy_saving(
            observation,
            energy_ensemble,
        )
        predicted_regret = max((forecast.upper_gain for forecast in forecasts), default=None)
        confidence = self._confidence(state, candidate)

        if not model_ready:
            self.low_value_streak = 0
            diagnostics["rapec7_low_value_streak"] = 0
            diagnostics["rapec7_candidate_stop"] = 0
            return ControllerDecision(
                stop=False,
                reason="rapec7_bayesian_warmup",
                confidence=0.0,
                predicted_energy_saving_fraction=projected_saving,
                predicted_quality_regret=predicted_regret,
                diagnostics=diagnostics,
            )

        reason = (
            "rapec7_bayesian_stop: "
            f"epoch={observation.epoch}, epsilon={epsilon_reference:.6f}, "
            f"posterior_recovery={posterior_recovery:.4f}, "
            f"risk_limit={risk_limit:.4f}, valuable_horizons={len(valuable_horizons)}, "
            f"confirmation={self.low_value_streak}/{patience}"
        )
        return ControllerDecision(
            stop=stop,
            reason=reason if stop else "continue",
            confidence=confidence,
            predicted_energy_saving_fraction=projected_saving,
            predicted_quality_regret=predicted_regret,
            diagnostics=diagnostics,
        )

    def _forecasts(
        self,
        *,
        observation: EpochObservation,
        active_horizons: list[int],
        current_best: float,
        epsilon_reference: float,
        risk_limit: float,
        quality_ensemble: BayesianEnsemble | None,
        energy_ensemble: BayesianEnsemble | None,
        duration_ensemble: BayesianEnsemble | None,
    ) -> list[BayesianForecast]:
        if not quality_ensemble or not energy_ensemble or not duration_ensemble:
            return []
        forecasts = []
        for horizon in active_horizons:
            # Epsilon is the smallest quality change distinguishable in this
            # run. Its meaning is independent of the selected forecast horizon.
            relevant_gain = epsilon_reference
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
            future_quality = np.clip(quality_draws[:, -1], 0.0, 1.0)
            gains = np.maximum(0.0, future_quality - current_best)

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
            energy = np.maximum(energy_draws[:, -1], self.numerical_quality_floor)
            duration = np.maximum(duration_draws[:, -1], 0.0)
            utility = gains / energy
            lower_quantile = _clamp(risk_limit, 1.0 / (self.posterior_samples + 1), 0.25)
            upper_quantile = 1.0 - lower_quantile
            expected_gain = float(np.median(gains))
            lower_gain = float(np.quantile(gains, lower_quantile))
            upper_gain = float(np.quantile(gains, upper_quantile))
            expected_energy = float(np.median(energy))
            forecasts.append(
                BayesianForecast(
                    horizon=horizon,
                    relevant_gain=relevant_gain,
                    expected_quality=_clamp(current_best + expected_gain, 0.0, 1.0),
                    expected_gain=expected_gain,
                    lower_gain=lower_gain,
                    upper_gain=upper_gain,
                    probability_relevant_gain=float(np.mean(gains > relevant_gain)),
                    expected_energy_wh=expected_energy,
                    lower_energy_wh=float(np.quantile(energy, lower_quantile)),
                    upper_energy_wh=float(np.quantile(energy, upper_quantile)),
                    expected_duration_seconds=float(np.median(duration)),
                    lower_duration_seconds=float(np.quantile(duration, lower_quantile)),
                    upper_duration_seconds=float(np.quantile(duration, upper_quantile)),
                    expected_utility=float(np.median(utility)),
                    conservative_utility=float(np.quantile(utility, lower_quantile)),
                    quality_interval_width=max(0.0, upper_gain - lower_gain),
                )
            )
        return forecasts

    def _draw_ensemble(
        self,
        ensemble: BayesianEnsemble,
        horizon: int,
        seed: int,
        *,
        cumulative: bool,
    ) -> np.ndarray:
        rng = np.random.default_rng(seed)
        names = list(ensemble.posteriors)
        probabilities = np.asarray([ensemble.weights[name] for name in names], dtype=float)
        probabilities = probabilities / probabilities.sum()
        choices = rng.choice(len(names), size=self.posterior_samples, p=probabilities)
        output = np.empty((self.posterior_samples, horizon), dtype=float)
        for index, name in enumerate(names):
            locations = np.flatnonzero(choices == index)
            if locations.size == 0:
                continue
            design = ensemble.future_designs[name][:horizon]
            draws = ensemble.posteriors[name].draw(
                design,
                int(locations.size),
                rng,
                include_observation_noise=ensemble.target_transform != "identity",
            )
            if ensemble.target_transform == "log":
                draws = np.exp(np.clip(draws, -30.0, 30.0))
            if cumulative:
                draws = np.cumsum(np.maximum(draws, 0.0), axis=1)
            output[locations] = draws
        return output

    def _quality_ensemble(
        self,
        rows: list[Mapping[str, Any]],
        observation: EpochObservation,
        max_horizon: int,
    ) -> BayesianEnsemble | None:
        entries: list[tuple[Mapping[str, Any], float]] = []
        running_best = 0.0
        for row in rows:
            value = self._row_quality(row, observation)
            if value is None:
                continue
            running_best = max(running_best, value)
            entries.append((row, running_best))
        entries = entries[-self.model_window :]
        if len(entries) < 4:
            return None
        model_rows = [row for row, _ in entries]
        targets = np.asarray([value for _, value in entries], dtype=float)
        return self._build_ensemble(
            model_rows,
            targets,
            observation,
            max_horizon,
            target_transform="identity",
            target_kind="quality",
        )

    def _resource_ensemble(
        self,
        rows: list[Mapping[str, Any]],
        observation: EpochObservation,
        max_horizon: int,
        resource: str,
    ) -> BayesianEnsemble | None:
        entries = []
        for row in rows:
            value = self._energy_wh(row) if resource == "energy" else self._duration_seconds(row)
            if value > 0.0:
                entries.append((row, math.log(value)))
        entries = entries[-self.model_window :]
        if len(entries) < 4:
            return None
        return self._build_ensemble(
            [row for row, _ in entries],
            np.asarray([value for _, value in entries], dtype=float),
            observation,
            max_horizon,
            target_transform="log",
            target_kind=resource,
        )

    def _build_ensemble(
        self,
        rows: list[Mapping[str, Any]],
        targets: np.ndarray,
        observation: EpochObservation,
        max_horizon: int,
        *,
        target_transform: str,
        target_kind: str,
    ) -> BayesianEnsemble | None:
        epochs = np.asarray(
            [int(row.get("epoch_index", row.get("epoch", index + 1))) for index, row in enumerate(rows)],
            dtype=float,
        )
        future_epochs = np.arange(
            int(epochs[-1]) + 1,
            int(epochs[-1]) + max_horizon + 1,
            dtype=float,
        )
        designs, future_designs = self._model_designs(
            rows,
            epochs,
            future_epochs,
            observation.max_epochs,
            excluded_feature=target_kind,
        )
        posteriors: dict[str, BayesianPosterior] = {}
        predictive_scores: dict[str, float] = {}
        for name, design in designs.items():
            # RAPEC-v7 avoids a causal interpretation of within-run hardware
            # telemetry. Newer subclasses may opt in for prediction while
            # still treating the association as non-causal.
            if (
                target_kind == "quality"
                and name == "telemetry"
                and not self._allow_quality_telemetry()
            ):
                continue
            if design.shape[0] < design.shape[1] + 2:
                continue
            posterior = self._fit_bayesian_model(
                name=name,
                design=design,
                targets=targets,
                target_transform=target_transform,
            )
            if posterior is not None:
                posteriors[name] = posterior
                predictive_scores[name] = self._multi_horizon_prequential_score(
                    name,
                    rows,
                    targets,
                    observation,
                    target_kind,
                    target_transform,
                )
        if target_kind == "quality":
            local_window = min(
                len(rows),
                max(
                    self.minimum_observations_floor,
                    int(math.ceil(2.0 * math.sqrt(len(rows)))),
                ),
            )
            local_design, local_future = self._local_trend_design(
                local_window,
                max_horizon,
            )
            local_posterior = self._fit_bayesian_model(
                name="local_trend",
                design=local_design,
                targets=targets[-local_window:],
                target_transform=target_transform,
            )
            if local_posterior is not None:
                posteriors["local_trend"] = local_posterior
                future_designs["local_trend"] = local_future
                predictive_scores["local_trend"] = self._local_prequential_log_score(
                    targets,
                    target_transform,
                )
        if not posteriors:
            return None
        log_scores = np.asarray([predictive_scores[name] for name in posteriors], dtype=float)
        log_scores -= float(np.max(log_scores))
        raw_weights = np.exp(np.clip(log_scores, -700.0, 0.0))
        raw_weights /= raw_weights.sum()
        weights = {
            name: float(weight)
            for name, weight in zip(posteriors, raw_weights)
        }
        return BayesianEnsemble(
            posteriors=posteriors,
            future_designs={name: future_designs[name] for name in posteriors},
            weights=weights,
            target_transform=target_transform,
        )

    def _allow_quality_telemetry(self) -> bool:
        """Keep the historical RAPEC-v7 behavior unless a subclass opts in."""
        return False

    def _local_trend_design(
        self,
        observed_count: int,
        future_count: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        denominator = max(1.0, float(observed_count - 1))
        observed = np.arange(observed_count, dtype=float) / denominator
        future = np.arange(
            observed_count,
            observed_count + future_count,
            dtype=float,
        ) / denominator
        return (
            np.column_stack([np.ones_like(observed), observed]),
            np.column_stack([np.ones_like(future), future]),
        )

    def _local_prequential_log_score(
        self,
        targets: np.ndarray,
        target_transform: str,
    ) -> float:
        scores = []
        for horizon in (1, 5, 10):
            if len(targets) < 2 * horizon + self.minimum_observations_floor:
                continue
            horizon_scores = []
            for target_index in range(max(horizon, len(targets) - 3), len(targets)):
                anchor = target_index - horizon + 1
                if anchor < self.minimum_observations_floor:
                    continue
                local_window = min(
                    anchor,
                    max(
                        self.minimum_observations_floor,
                        int(math.ceil(2.0 * math.sqrt(anchor))),
                    ),
                )
                design, future = self._local_trend_design(local_window, horizon)
                posterior = self._fit_bayesian_model(
                    name="local_prequential",
                    design=design,
                    targets=targets[anchor - local_window : anchor],
                    target_transform=target_transform,
                )
                if posterior is None:
                    continue
                horizon_scores.append(
                    self._posterior_predictive_log_density(
                        posterior,
                        future[horizon - 1],
                        float(targets[target_index]),
                    )
                )
            if horizon_scores:
                scores.append(float(np.mean(horizon_scores)))
        return float(sum(scores)) if scores else -math.inf

    def _multi_horizon_prequential_score(
        self,
        model_name: str,
        rows: list[Mapping[str, Any]],
        targets: np.ndarray,
        observation: EpochObservation,
        target_kind: str,
        target_transform: str,
    ) -> float:
        scores = []
        available_horizons = sorted(
            {
                horizon
                for horizon in (1, 5, 10)
                if len(rows) >= 2 * horizon + self.minimum_observations_floor
            }
        )
        for horizon in available_horizons:
            horizon_scores = []
            final_target_index = len(rows) - 1
            first_target_index = max(
                horizon + self.minimum_observations_floor,
                final_target_index - 2,
            )
            for target_index in range(first_target_index, final_target_index + 1):
                anchor = target_index - horizon + 1
                if anchor < self.minimum_observations_floor:
                    continue
                prefix_rows = rows[:anchor]
                prefix_targets = targets[:anchor]
                prefix_epochs = np.asarray(
                    [
                        int(row.get("epoch_index", row.get("epoch", index + 1)))
                        for index, row in enumerate(prefix_rows)
                    ],
                    dtype=float,
                )
                future_epochs = np.arange(
                    int(prefix_epochs[-1]) + 1,
                    int(prefix_epochs[-1]) + horizon + 1,
                    dtype=float,
                )
                designs, future_designs = self._model_designs(
                    prefix_rows,
                    prefix_epochs,
                    future_epochs,
                    observation.max_epochs,
                    excluded_feature=target_kind,
                )
                if model_name not in designs or model_name not in future_designs:
                    continue
                posterior = self._fit_bayesian_model(
                    name="prequential",
                    design=designs[model_name],
                    targets=prefix_targets,
                    target_transform=target_transform,
                )
                if posterior is None:
                    continue
                vector = future_designs[model_name][horizon - 1]
                horizon_scores.append(
                    self._posterior_predictive_log_density(
                        posterior,
                        vector,
                        float(targets[target_index]),
                    )
                )
            if horizon_scores:
                # Give each horizon equal influence regardless of the number of
                # technically available hindcast anchors.
                scores.append(float(np.mean(horizon_scores)))
        if scores:
            return float(sum(scores))
        # Before a prequential window exists, marginal evidence remains the
        # mathematically defined Bayesian fallback.
        epochs = np.asarray(
            [
                int(row.get("epoch_index", row.get("epoch", index + 1)))
                for index, row in enumerate(rows)
            ],
            dtype=float,
        )
        designs, _ = self._model_designs(
            rows,
            epochs,
            np.asarray([epochs[-1] + 1.0]),
            observation.max_epochs,
            excluded_feature=target_kind,
        )
        design = designs.get(model_name)
        if design is None:
            return -math.inf
        posterior = self._fit_bayesian_model(
            name="fallback",
            design=design,
            targets=targets,
            target_transform=target_transform,
        )
        return posterior.log_evidence if posterior is not None else -math.inf

    def _posterior_predictive_log_density(
        self,
        posterior: BayesianPosterior,
        vector: np.ndarray,
        observed: float,
    ) -> float:
        mean = float(vector @ posterior.mean)
        variance = (
            posterior.scale
            / posterior.shape
            * (1.0 + float(vector @ posterior.precision_inverse @ vector))
            * posterior.scale_inflation**2
        )
        variance = max(variance, 1e-12)
        degrees_of_freedom = 2.0 * posterior.shape
        standardized = (observed - mean) / math.sqrt(variance)
        return (
            math.lgamma((degrees_of_freedom + 1.0) / 2.0)
            - math.lgamma(degrees_of_freedom / 2.0)
            - 0.5 * math.log(degrees_of_freedom * math.pi * variance)
            - (degrees_of_freedom + 1.0)
            / 2.0
            * math.log1p(standardized**2 / degrees_of_freedom)
        )

    def _fit_bayesian_model(
        self,
        *,
        name: str,
        design: np.ndarray,
        targets: np.ndarray,
        target_transform: str,
    ) -> BayesianPosterior | None:
        try:
            sample_count, dimension = design.shape
            prior_mean = np.zeros(dimension, dtype=float)
            # Empirical-Bayes location centering uses only the visible prefix
            # and avoids imposing an artificial quality or resource scale.
            prior_mean[0] = float(np.median(targets))
            prior_std = np.full(dimension, 2.0 if target_transform == "identity" else 5.0)
            prior_std[0] = 0.75 if target_transform == "identity" else 8.0
            prior_precision = np.diag(1.0 / np.square(prior_std))

            # Keep a broad generic variance prior, but assign only a very small
            # pseudo-sample mass so the current run can calibrate it quickly.
            prior_shape = 1.001
            prior_noise = 0.10 if target_transform == "identity" else 0.50
            prior_scale = (prior_shape - 1.0) * prior_noise**2
            posterior_precision = prior_precision + design.T @ design
            precision_inverse = np.linalg.inv(posterior_precision)
            posterior_mean = precision_inverse @ (
                prior_precision @ prior_mean + design.T @ targets
            )
            posterior_shape = prior_shape + sample_count / 2.0
            quadratic = (
                targets @ targets
                + prior_mean @ prior_precision @ prior_mean
                - posterior_mean @ posterior_precision @ posterior_mean
            )
            posterior_scale = max(1e-12, prior_scale + 0.5 * float(quadratic))
            sign_prior, logdet_prior = np.linalg.slogdet(prior_precision)
            sign_post, logdet_post = np.linalg.slogdet(posterior_precision)
            if sign_prior <= 0 or sign_post <= 0:
                return None
            log_evidence = (
                -0.5 * sample_count * math.log(2.0 * math.pi)
                + 0.5 * (logdet_prior - logdet_post)
                + prior_shape * math.log(prior_scale)
                - posterior_shape * math.log(posterior_scale)
                + math.lgamma(posterior_shape)
                - math.lgamma(prior_shape)
            )
            residuals = targets - design @ posterior_mean
            empirical_noise = _mad(residuals)
            posterior_noise = math.sqrt(posterior_scale / posterior_shape)
            scale_inflation = _clamp(
                empirical_noise / max(posterior_noise, 1e-12),
                1.0,
                10.0,
            )
            return BayesianPosterior(
                name=name,
                mean=posterior_mean,
                precision_inverse=precision_inverse,
                shape=posterior_shape,
                scale=posterior_scale,
                log_evidence=float(log_evidence),
                scale_inflation=scale_inflation,
                sample_count=sample_count,
            )
        except (ValueError, FloatingPointError, np.linalg.LinAlgError):
            return None

    def _model_designs(
        self,
        rows: list[Mapping[str, Any]],
        epochs: np.ndarray,
        future_epochs: np.ndarray,
        max_epochs: int,
        *,
        excluded_feature: str,
    ) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
        scale = max(1.0, float(max_epochs))
        normalized = epochs / scale
        future_normalized = future_epochs / scale
        log_scale = max(1.0, math.log1p(scale))
        tau_fast = max(2.0, 0.15 * scale)
        tau_slow = max(3.0, 0.50 * scale)
        change_one = float(np.quantile(normalized, 0.50))
        change_two = float(np.quantile(normalized, 0.75))
        designs = {
            "linear": np.column_stack([np.ones_like(normalized), normalized]),
            "logarithmic": np.column_stack(
                [np.ones_like(normalized), np.log1p(epochs) / log_scale]
            ),
            "saturation": np.column_stack(
                [np.ones_like(normalized), 1.0 / np.sqrt(epochs), 1.0 / epochs]
            ),
            "exponential": np.column_stack(
                [
                    np.ones_like(normalized),
                    np.exp(-epochs / tau_fast),
                    np.exp(-epochs / tau_slow),
                ]
            ),
            "change_point": np.column_stack(
                [
                    np.ones_like(normalized),
                    normalized,
                    np.maximum(0.0, normalized - change_one),
                    np.maximum(0.0, normalized - change_two),
                ]
            ),
        }
        future = {
            "linear": np.column_stack([np.ones_like(future_normalized), future_normalized]),
            "logarithmic": np.column_stack(
                [np.ones_like(future_normalized), np.log1p(future_epochs) / log_scale]
            ),
            "saturation": np.column_stack(
                [
                    np.ones_like(future_normalized),
                    1.0 / np.sqrt(future_epochs),
                    1.0 / future_epochs,
                ]
            ),
            "exponential": np.column_stack(
                [
                    np.ones_like(future_normalized),
                    np.exp(-future_epochs / tau_fast),
                    np.exp(-future_epochs / tau_slow),
                ]
            ),
            "change_point": np.column_stack(
                [
                    np.ones_like(future_normalized),
                    future_normalized,
                    np.maximum(0.0, future_normalized - change_one),
                    np.maximum(0.0, future_normalized - change_two),
                ]
            ),
        }
        telemetry_observed, telemetry_future = self._telemetry_columns(
            rows,
            len(future_epochs),
            excluded_feature,
        )
        if telemetry_observed:
            maximum_columns = max(0, len(rows) - 5)
            selected_names = list(telemetry_observed)[:maximum_columns]
            if selected_names:
                designs["telemetry"] = np.column_stack(
                    [
                        np.ones_like(normalized),
                        normalized,
                        np.log1p(epochs) / log_scale,
                        *[telemetry_observed[name] for name in selected_names],
                    ]
                )
                future["telemetry"] = np.column_stack(
                    [
                        np.ones_like(future_normalized),
                        future_normalized,
                        np.log1p(future_epochs) / log_scale,
                        *[telemetry_future[name] for name in selected_names],
                    ]
                )
        return designs, future

    def _telemetry_columns(
        self,
        rows: list[Mapping[str, Any]],
        future_count: int,
        excluded_feature: str,
    ) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
        observed_columns: dict[str, np.ndarray] = {}
        future_columns: dict[str, np.ndarray] = {}
        excluded_names = {
            "energy": {"epoch_energy_wh"},
            "duration": {"epoch_duration_seconds"},
            "quality": set(),
        }.get(excluded_feature, set())
        for name in self.TELEMETRY_NAMES:
            if name in excluded_names:
                continue
            raw = [self._feature_value(row, name) for row in rows]
            available = [value for value in raw if value is not None]
            if len(available) < max(4, len(rows) // 2):
                continue
            transformed_available = [self._feature_transform(name, value) for value in available]
            fill = float(np.median(transformed_available))
            observed = np.asarray(
                [
                    self._feature_transform(name, value) if value is not None else fill
                    for value in raw
                ],
                dtype=float,
            )
            center = float(np.median(observed))
            spread = _mad(observed)
            if spread <= 1e-12:
                spread = float(np.std(observed))
            # Constant parameter/FLOP metadata are tracked, but their causal
            # effect cannot be identified from a single run.
            if spread <= 1e-12:
                continue
            recent = observed[-min(12, len(observed)) :]
            slopes = np.diff(recent)
            slope = float(np.median(slopes)) if slopes.size else 0.0
            lower = float(np.quantile(observed, 0.05) - 2.0 * spread)
            upper = float(np.quantile(observed, 0.95) + 2.0 * spread)
            future_raw = np.asarray(
                [
                    _clamp(float(observed[-1] + slope * step), lower, upper)
                    for step in range(1, future_count + 1)
                ],
                dtype=float,
            )
            observed_columns[name] = (observed - center) / spread
            future_columns[name] = (future_raw - center) / spread
        return observed_columns, future_columns

    def _feature_transform(self, name: str, value: float) -> float:
        if name == "learning_rate":
            return math.log(max(value, 1e-12))
        if name in {
            "train_loss",
            "gradient_norm",
            "epoch_duration_seconds",
            "gpu_memory_used_mb",
            "gpu_power_avg_w",
            "epoch_energy_wh",
            "model_parameter_count",
            "model_flops",
        }:
            return math.log1p(max(0.0, value))
        return value / 100.0 if name == "gpu_utilization_pct" else value

    def _dynamic_epsilon(self, qualities: list[float], effective_sample_size: float) -> float:
        if len(qualities) < 2:
            return self.numerical_quality_floor
        recent = np.asarray(qualities[-self.model_window :], dtype=float)
        differences = np.diff(recent)
        noise = _mad(differences - np.median(differences))
        nonzero_resolution = np.abs(differences[np.abs(differences) > self.numerical_quality_floor])
        resolution = float(np.min(nonzero_resolution)) if nonzero_resolution.size else 0.0
        detectable_noise = noise / math.sqrt(max(1.0, effective_sample_size))
        detectable_resolution = resolution / math.sqrt(max(1.0, effective_sample_size))
        return max(self.numerical_quality_floor, detectable_noise, detectable_resolution)

    def _dynamic_risk_limit(
        self,
        effective_sample_size: float,
        model_disagreement: float,
        posterior_sample_count: int,
    ) -> float:
        sample_resolution = 1.0 / (max(1, posterior_sample_count) + 1.0)
        evidence_tail = 1.0 / math.sqrt(max(1.0, effective_sample_size))
        disagreement_penalty = 1.0 - 0.75 * _clamp(model_disagreement, 0.0, 1.0)
        return _clamp(evidence_tail * disagreement_penalty, sample_resolution, 0.25)

    def _dynamic_utility_floor(
        self,
        rows: list[Mapping[str, Any]],
        observation: EpochObservation,
        risk_limit: float,
    ) -> float:
        qualities = _best_so_far(self._quality_values(rows, observation))
        utilities = []
        for left, right, row in zip(qualities, qualities[1:], rows[1:]):
            gain = max(0.0, right - left)
            energy = self._energy_wh(row)
            if gain > 0.0 and energy > 0.0:
                utilities.append(gain / energy)
        if not utilities:
            return 0.0
        quantile = _clamp(risk_limit, 1.0 / (len(utilities) + 1.0), 0.50)
        return max(0.0, float(np.quantile(utilities, quantile)))

    def _dynamic_minimum_observations(
        self,
        ensemble: BayesianEnsemble | None,
    ) -> int:
        if ensemble is None:
            return self.minimum_observations_floor
        identifiable = int(math.ceil(2.0 * ensemble.weighted_dimension + 2.0))
        # A horizon h needs at least two completed h-length windows before a
        # current-run-only controller can distinguish a temporary plateau from
        # delayed improvement without external learning curves.
        horizon_evidence = 2 * max(self.horizons, default=1)
        return max(self.minimum_observations_floor, identifiable, horizon_evidence)

    def _dynamic_patience(self, qualities: list[float]) -> int:
        if len(qualities) < 4:
            return 2
        regime_window = max(8, self.model_window // 3)
        differences = np.diff(np.asarray(qualities[-regime_window:], dtype=float))
        if differences.size < 3 or float(np.std(differences)) <= 1e-12:
            return 2
        left = differences[:-1]
        right = differences[1:]
        if float(np.std(left)) <= 1e-12 or float(np.std(right)) <= 1e-12:
            correlation = 0.0
        else:
            correlation = float(np.corrcoef(left, right)[0, 1])
        if not math.isfinite(correlation):
            correlation = 0.0
        correlation = _clamp(correlation, 0.0, 0.95)
        correlation_time = (1.0 + correlation) / max(1e-6, 1.0 - correlation)
        return int(_clamp(math.ceil(correlation_time), 2, self.maximum_confirmation_epochs))

    def _effective_sample_size(self, qualities: list[float]) -> float:
        if len(qualities) < 4:
            return float(max(1, len(qualities)))
        regime_window = max(8, self.model_window // 3)
        differences = np.diff(np.asarray(qualities[-regime_window:], dtype=float))
        if differences.size < 3 or float(np.std(differences)) <= 1e-12:
            return float(max(1, differences.size))
        left = differences[:-1]
        right = differences[1:]
        if float(np.std(left)) <= 1e-12 or float(np.std(right)) <= 1e-12:
            correlation = 0.0
        else:
            correlation = float(np.corrcoef(left, right)[0, 1])
        if not math.isfinite(correlation):
            correlation = 0.0
        correlation = _clamp(correlation, -0.95, 0.95)
        effective = differences.size * (1.0 - correlation) / (1.0 + correlation)
        return _clamp(effective, 1.0, float(differences.size))

    def _validation_noise(self, qualities: list[float]) -> float:
        if len(qualities) < 3:
            return 0.0
        differences = np.diff(np.asarray(qualities[-self.model_window :], dtype=float))
        return _mad(differences - np.median(differences))

    def _same_run_recovery_probability(
        self,
        best: list[float],
        epsilon: float,
        maximum_horizon: int,
    ) -> float:
        horizon = max(2, min(maximum_horizon, max(2, len(best) // 5)))
        if len(best) < 2 * horizon + 2:
            return 0.0
        stalls = 0
        recoveries = 0
        for anchor in range(horizon, len(best) - horizon):
            past_gain = best[anchor] - best[anchor - horizon]
            future_gain = best[anchor + horizon] - best[anchor]
            if past_gain <= epsilon:
                stalls += 1
                if future_gain > epsilon:
                    recoveries += 1
        if stalls == 0:
            return 0.0
        # Jeffreys Beta(1/2, 1/2) prior, updated from this run only.
        return (recoveries + 0.5) / (stalls + 1.0)

    def _feature_recovery_probability(
        self,
        rows: list[Mapping[str, Any]],
        observation: EpochObservation,
    ) -> float:
        signals = []
        losses = [self._row_loss(row) for row in rows if self._row_loss(row) is not None]
        if len(losses) >= 4:
            split = max(2, len(losses) // 4)
            earlier = float(np.median(losses[-2 * split : -split]))
            recent = float(np.median(losses[-split:]))
            signals.append(_clamp((earlier - recent) / max(abs(earlier), 1e-12), 0.0, 1.0))
        gradients = [
            value
            for row in rows
            if (value := _number(row.get("gradient_norm"))) is not None and value >= 0.0
        ]
        if len(gradients) >= 4:
            rank = sum(value <= gradients[-1] for value in gradients) / len(gradients)
            signals.append(rank)
        learning_rates = [
            value
            for row in rows
            if (value := _number(row.get("learning_rate"), _number(row.get("lr")))) is not None
            and value >= 0.0
        ]
        if len(learning_rates) >= 2:
            signals.append(_clamp(learning_rates[-1] / max(learning_rates), 0.0, 1.0))
        qualities = self._quality_values(rows, observation)
        if len(qualities) >= 5:
            differences = np.diff(np.asarray(qualities, dtype=float))
            recent = float(np.median(differences[-min(5, differences.size) :]))
            historical = float(np.quantile(np.maximum(differences, 0.0), 0.75))
            signals.append(_clamp(recent / max(historical, self.numerical_quality_floor), 0.0, 1.0))
        return float(np.mean(signals)) if signals else 0.0

    def _pareto_front(self, forecasts: list[BayesianForecast]) -> list[BayesianForecast]:
        front = []
        for candidate in forecasts:
            dominated = any(
                other.horizon != candidate.horizon
                and other.expected_energy_wh <= candidate.expected_energy_wh
                and other.expected_gain >= candidate.expected_gain
                and (
                    other.expected_energy_wh < candidate.expected_energy_wh
                    or other.expected_gain > candidate.expected_gain
                )
                for other in forecasts
            )
            if not dominated:
                front.append(candidate)
        return sorted(front, key=lambda forecast: forecast.horizon)

    def _projected_training_energy_saving(
        self,
        observation: EpochObservation,
        ensemble: BayesianEnsemble | None,
    ) -> float | None:
        remaining = max(0, observation.max_epochs - observation.epoch)
        if not ensemble or remaining <= 0:
            return 0.0
        horizon = min(remaining, next(iter(ensemble.future_designs.values())).shape[0])
        if horizon <= 0:
            return 0.0
        draws = self._draw_ensemble(
            ensemble,
            horizon,
            observation.epoch * 7919 + 101,
            cumulative=True,
        )
        future = float(np.median(draws[:, -1]))
        total = observation.cumulative_energy_wh + future
        return _clamp(future / total, 0.0, 1.0) if total > 0.0 else None

    def _confidence(self, state: DynamicDecisionState, candidate: bool) -> float:
        if not candidate:
            return _clamp(1.0 - state.model_disagreement, 0.0, 1.0)
        probability_margin = (
            state.risk_probability_limit - state.posterior_recovery_probability
        ) / max(state.risk_probability_limit, 1e-12)
        confirmation = self.low_value_streak / max(1, state.patience)
        return _clamp(
            0.60 * probability_margin
            + 0.25 * confirmation
            + 0.15 * (1.0 - state.model_disagreement),
            0.0,
            1.0,
        )

    def _diagnostics(
        self,
        *,
        observation: EpochObservation,
        state: DynamicDecisionState,
        forecasts: list[BayesianForecast],
        quality_ensemble: BayesianEnsemble | None,
        energy_ensemble: BayesianEnsemble | None,
        duration_ensemble: BayesianEnsemble | None,
        current_best: float,
        model_ready: bool,
        candidate: bool,
        pareto_front_size: int,
        valuable_horizon_count: int,
    ) -> dict[str, float | int | bool | None]:
        diagnostics: dict[str, float | int | bool | None] = {
            "rapec7_is_bayesian": 1,
            "rapec7_online_current_run_only": 1,
            "rapec7_uses_historical_curves": 0,
            "rapec7_uses_task_or_dataset_profile": 0,
            "rapec7_energy_scope_training_only": 1,
            "rapec7_dynamic_epsilon": state.epsilon_reference,
            "rapec7_dynamic_risk_limit": state.risk_probability_limit,
            "rapec7_dynamic_utility_floor": state.utility_floor,
            "rapec7_dynamic_minimum_observations": state.minimum_observations,
            "rapec7_dynamic_patience": state.patience,
            "rapec7_effective_sample_size": state.effective_sample_size,
            "rapec7_validation_noise": state.validation_noise,
            "rapec7_model_disagreement": state.model_disagreement,
            "rapec7_feature_coverage_fraction": state.feature_coverage,
            "rapec7_feature_recovery_probability": state.feature_recovery_probability,
            "rapec7_posterior_recovery_probability": state.posterior_recovery_probability,
            "rapec7_model_ready": int(model_ready),
            "rapec7_candidate_stop": int(candidate),
            "rapec7_low_value_streak": self.low_value_streak,
            "rapec7_pareto_front_size": pareto_front_size,
            "rapec7_valuable_horizon_count": valuable_horizon_count,
            "rapec7_current_best_quality": current_best,
            "rapec7_quality_model_entropy": quality_ensemble.entropy if quality_ensemble else None,
            "rapec7_energy_model_entropy": energy_ensemble.entropy if energy_ensemble else None,
            "rapec7_duration_model_entropy": duration_ensemble.entropy if duration_ensemble else None,
            "rapec7_quality_weighted_dimension": (
                quality_ensemble.weighted_dimension if quality_ensemble else None
            ),
            "rapec7_model_parameter_count_available": int(
                self._feature_value(observation.raw_metrics, "model_parameter_count") is not None
            ),
            "rapec7_model_flops_available": int(
                self._feature_value(observation.raw_metrics, "model_flops") is not None
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
            diagnostics[f"rapec7_quality_model_weight_{name}"] = (
                quality_ensemble.weights.get(name, 0.0) if quality_ensemble else None
            )
        for horizon in self.horizons:
            forecast = next((item for item in forecasts if item.horizon == horizon), None)
            prefix = f"rapec7_h{horizon}"
            diagnostics.update(
                {
                    f"{prefix}_relevant_gain": forecast.relevant_gain if forecast else None,
                    f"{prefix}_expected_quality": forecast.expected_quality if forecast else None,
                    f"{prefix}_expected_gain": forecast.expected_gain if forecast else None,
                    f"{prefix}_gain_lower": forecast.lower_gain if forecast else None,
                    f"{prefix}_gain_upper": forecast.upper_gain if forecast else None,
                    f"{prefix}_prob_relevant_gain": (
                        forecast.probability_relevant_gain if forecast else None
                    ),
                    f"{prefix}_expected_energy_wh": forecast.expected_energy_wh if forecast else None,
                    f"{prefix}_energy_lower_wh": forecast.lower_energy_wh if forecast else None,
                    f"{prefix}_energy_upper_wh": forecast.upper_energy_wh if forecast else None,
                    f"{prefix}_expected_duration_seconds": (
                        forecast.expected_duration_seconds if forecast else None
                    ),
                    f"{prefix}_duration_lower_seconds": (
                        forecast.lower_duration_seconds if forecast else None
                    ),
                    f"{prefix}_duration_upper_seconds": (
                        forecast.upper_duration_seconds if forecast else None
                    ),
                    f"{prefix}_expected_utility": forecast.expected_utility if forecast else None,
                    f"{prefix}_conservative_utility": (
                        forecast.conservative_utility if forecast else None
                    ),
                    f"{prefix}_quality_interval_width": (
                        forecast.quality_interval_width if forecast else None
                    ),
                }
            )
        return diagnostics

    def _feature_coverage(
        self,
        rows: list[Mapping[str, Any]],
        observation: EpochObservation,
    ) -> float:
        current = rows[-1] if rows else {}
        values = {
            "quality": self._row_quality(current, observation),
            **{name: self._feature_value(current, name) for name in self.TELEMETRY_NAMES},
        }
        return sum(value is not None for value in values.values()) / len(self.FEATURE_NAMES)

    def _quality_values(
        self,
        rows: list[Mapping[str, Any]],
        observation: EpochObservation,
    ) -> list[float]:
        return [
            value
            for row in rows
            if (value := self._row_quality(row, observation)) is not None
        ]

    def _row_quality(
        self,
        row: Mapping[str, Any],
        observation: EpochObservation,
    ) -> float | None:
        value = _number(
            row.get(observation.quality_metric),
            _number(row.get("quality_score"), _number(row.get("map50_95"))),
        )
        return _clamp(value, 0.0, 1.0) if value is not None else None

    def _feature_value(self, row: Mapping[str, Any], name: str) -> float | None:
        getters: dict[str, Callable[[Mapping[str, Any]], float | None]] = {
            "train_loss": self._row_loss,
            "gradient_norm": lambda item: _number(item.get("gradient_norm")),
            "learning_rate": lambda item: _number(
                item.get("learning_rate"),
                _number(item.get("lr")),
            ),
            "epoch_duration_seconds": lambda item: _number(item.get("duration_seconds")),
            "gpu_utilization_pct": lambda item: _number(
                item.get("gpu_util_avg_pct"),
                _number(item.get("gpu_utilization_pct")),
            ),
            "gpu_memory_used_mb": lambda item: _number(
                item.get("gpu_memory_used_mb"),
                _number(item.get("gpu_mem_used_avg_mb")),
            ),
            "gpu_power_avg_w": lambda item: _number(item.get("gpu_power_avg_w")),
            "epoch_energy_wh": lambda item: self._energy_wh(item),
            "model_parameter_count": lambda item: _number(
                item.get("model_parameter_count"),
                _number(item.get("model_parameters")),
            ),
            "model_flops": lambda item: _number(
                item.get("model_flops"),
                _number(item.get("model_flops_estimated")),
            ),
        }
        return getters[name](row)

    def _energy_wh(self, row: Mapping[str, Any]) -> float:
        direct = _number(row.get("epoch_energy_wh"))
        if direct is not None:
            return max(0.0, direct)
        kwh = _number(row.get("total_energy_kwh"), _number(row.get("gpu_energy_kwh")))
        if kwh is not None:
            return max(0.0, kwh * 1000.0)
        power = _number(row.get("gpu_power_avg_w"))
        duration = _number(row.get("duration_seconds"))
        if power is not None and duration is not None:
            return max(0.0, power * duration / 3600.0)
        return 0.0

    def _duration_seconds(self, row: Mapping[str, Any]) -> float:
        return max(0.0, _number(row.get("duration_seconds"), 0.0) or 0.0)

    def _row_loss(self, row: Mapping[str, Any]) -> float | None:
        direct = _number(row.get("train_loss"), _number(row.get("loss")))
        if direct is not None:
            return direct
        components = [
            _number(row.get("val_box_loss")),
            _number(row.get("val_cls_loss")),
            _number(row.get("val_dfl_loss")),
        ]
        available = [value for value in components if value is not None]
        return sum(available) if available else _number(row.get("val_loss"))
