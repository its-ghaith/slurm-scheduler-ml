from __future__ import annotations

import math
import random
import statistics
from dataclasses import dataclass
from typing import Any, Iterable, Mapping

from controller_benchmark.api import Controller, ControllerDecision, EpochObservation


def _number(value: Any, default: float | None = None) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return default
    return result if math.isfinite(result) else default


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def _median(values: Iterable[float]) -> float | None:
    clean = [value for value in values if math.isfinite(value)]
    return statistics.median(clean) if clean else None


def _quantile(values: Iterable[float], q: float) -> float | None:
    clean = sorted(value for value in values if math.isfinite(value))
    if not clean:
        return None
    if q <= 0.0:
        return clean[0]
    if q >= 1.0:
        return clean[-1]
    position = (len(clean) - 1) * q
    low = int(math.floor(position))
    high = int(math.ceil(position))
    if low == high:
        return clean[low]
    weight = position - low
    return clean[low] * (1.0 - weight) + clean[high] * weight


def _mad(values: Iterable[float]) -> float:
    clean = [value for value in values if math.isfinite(value)]
    if len(clean) < 2:
        return 0.0
    center = statistics.median(clean)
    return 1.4826 * statistics.median(abs(value - center) for value in clean)


def _best_so_far(values: list[float]) -> list[float]:
    result: list[float] = []
    best = 0.0
    for value in values:
        best = max(best, _clamp(value, 0.0, 1.0))
        result.append(best)
    return result


@dataclass(frozen=True)
class MultiHorizonForecast:
    horizon: int
    expected_quality: float
    expected_gain: float
    lower_gain: float
    upper_gain: float
    probability_gain_gt_epsilon: float
    expected_energy_wh: float
    energy_upper_wh: float
    expected_duration_seconds: float
    utility_quality_per_wh: float
    conservative_utility_per_wh: float
    uncertainty_width: float
    calibration_radius: float
    calibration_samples: int
    prediction_samples: int


@dataclass(frozen=True)
class OnlineThresholds:
    epsilon: float
    utility_floor: float
    risk_alpha: float
    minimum_epochs: int
    patience: int
    recovery_probability: float
    recovery_limit: float
    validation_noise: float
    validation_standard_error: float
    quality_scale_tolerance: float
    curve_stability: float
    recent_velocity: float
    historical_velocity: float
    remaining_gain_estimate: float


class OnlineMultiHorizonParetoController(Controller):
    """RAPEC-v6: online probabilistic multi-horizon energy controller.

    The controller is task-independent and does not consume historical Full100
    runs. It predicts future quality, epoch energy, and duration from the
    current run only. Prediction residuals from earlier prefixes of that same
    run provide online uncertainty calibration.
    """

    FEATURE_KEYS = (
        "quality",
        "best_quality",
        "quality_slope",
        "quality_noise",
        "train_loss",
        "loss_slope",
        "gradient_norm",
        "gradient_activity",
        "learning_rate",
        "epoch_duration_seconds",
        "gpu_utilization_pct",
        "gpu_memory_used_mb",
        "gpu_power_avg_w",
        "epoch_energy_wh",
        "model_parameter_count",
        "model_flops",
    )

    STOP_STATES = {"slow_learning", "plateau", "overfit_risk"}

    def __init__(self, context):
        super().__init__(context)
        params = context.parameters
        raw_horizons = params.get("horizons", [1, 3, 5, 10, 20])
        if not isinstance(raw_horizons, (list, tuple)):
            raise ValueError("RAPEC-v6 parameter 'horizons' must be a JSON array.")
        horizons = sorted({max(1, int(value)) for value in raw_horizons})
        self.horizons = tuple(horizons or [1, 3, 5, 10, 20])
        self.min_fit_points = max(6, int(params.get("min_fit_points", 12)))
        self.trend_window = max(6, int(params.get("trend_window", 20)))
        self.bootstrap_samples = max(16, int(params.get("bootstrap_samples", 128)))
        self.min_epochs_floor = max(self.min_fit_points, int(params.get("min_epochs_floor", 15)))
        self.min_patience = max(1, int(params.get("min_patience", 2)))
        self.max_patience = max(self.min_patience, int(params.get("max_patience", 6)))
        self.base_risk_alpha = _clamp(float(params.get("base_risk_alpha", 0.10)), 0.01, 0.49)
        self.epsilon_noise_multiplier = max(0.0, float(params.get("epsilon_noise_multiplier", 1.25)))
        self.epsilon_remaining_gain_fraction = _clamp(
            float(params.get("epsilon_remaining_gain_fraction", 0.10)), 0.0, 1.0
        )
        self.epsilon_floor = max(0.0, float(params.get("epsilon_floor", 1e-5)))
        self.epsilon_quality_fraction_min = _clamp(
            float(params.get("epsilon_quality_fraction_min", 0.005)), 0.0, 0.25
        )
        self.epsilon_quality_fraction_max = _clamp(
            float(params.get("epsilon_quality_fraction_max", 0.020)),
            self.epsilon_quality_fraction_min,
            0.25,
        )
        self.utility_quantile = _clamp(float(params.get("utility_quantile", 0.25)), 0.0, 1.0)
        self.minimum_feature_coverage = _clamp(
            float(params.get("minimum_feature_coverage", 0.75)), 0.0, 1.0
        )
        self.calibration_quantile = _clamp(float(params.get("calibration_quantile", 0.90)), 0.50, 0.99)
        self.low_value_streak = 0

    def evaluate(self, observation: EpochObservation) -> ControllerDecision:
        rows = [*observation.history, observation.raw_metrics]
        qualities = self._quality_values(rows, observation)
        current_best = max(qualities) if qualities else float(observation.best_quality or observation.quality or 0.0)
        remaining_epochs = max(0, observation.max_epochs - observation.epoch)
        active_horizons = [horizon for horizon in self.horizons if horizon <= remaining_epochs]
        recovery_probability = self._recovery_probability(rows, observation)

        provisional_epsilon = self._provisional_epsilon(qualities)
        forecasts = [
            self._forecast(rows, observation, horizon, provisional_epsilon, recovery_probability)
            for horizon in active_horizons
        ]
        thresholds = self._dynamic_thresholds(rows, observation, forecasts, recovery_probability)
        if abs(thresholds.epsilon - provisional_epsilon) > 1e-12:
            forecasts = [
                self._forecast(rows, observation, horizon, thresholds.epsilon, recovery_probability)
                for horizon in active_horizons
            ]

        feature_coverage = self._feature_coverage(rows, observation)
        learning_state = self._learning_state(rows, observation, thresholds)
        pareto_front = self._pareto_front(forecasts)
        valuable_horizons = [
            forecast
            for forecast in pareto_front
            if (
                forecast.probability_gain_gt_epsilon > thresholds.risk_alpha
                and (
                    forecast.utility_quality_per_wh > thresholds.utility_floor
                    or forecast.lower_gain > thresholds.epsilon
                )
            )
        ]
        risk_probability = max(
            (forecast.probability_gain_gt_epsilon for forecast in forecasts),
            default=1.0,
        )
        widest_relative_uncertainty = max(
            (
                forecast.uncertainty_width
                / max(thresholds.epsilon, forecast.expected_gain, self.epsilon_floor)
                for forecast in forecasts
            ),
            default=math.inf,
        )
        calibration_samples = max((forecast.calibration_samples for forecast in forecasts), default=0)
        uncertainty_actionable = (
            bool(forecasts)
            and feature_coverage >= self.minimum_feature_coverage
            and (
                calibration_samples >= 3
                or widest_relative_uncertainty <= 6.0
                or observation.epoch >= max(thresholds.minimum_epochs, 2 * self.min_fit_points)
            )
        )
        enough_history = (
            len(qualities) >= self.min_fit_points
            and observation.epoch >= thresholds.minimum_epochs
            and bool(forecasts)
        )
        risk_satisfied = risk_probability <= thresholds.risk_alpha
        recovery_satisfied = thresholds.recovery_probability <= thresholds.recovery_limit
        state_satisfied = learning_state in self.STOP_STATES
        candidate = (
            enough_history
            and risk_satisfied
            and not valuable_horizons
            and recovery_satisfied
            and state_satisfied
            and uncertainty_actionable
        )
        self.low_value_streak = self.low_value_streak + 1 if candidate else 0
        stop = self.low_value_streak >= thresholds.patience
        selected = self._select_horizon(forecasts, thresholds)
        training_energy_saving = self._projected_training_energy_saving(rows, observation)
        predicted_regret = max((forecast.upper_gain for forecast in forecasts), default=None)
        confidence = self._confidence(
            risk_probability,
            thresholds,
            self.low_value_streak,
            uncertainty_actionable,
        )

        diagnostics: dict[str, float | int | bool | None] = {
            "rapec6_dynamic_epsilon": thresholds.epsilon,
            "rapec6_dynamic_allowed_regret": thresholds.epsilon,
            "rapec6_dynamic_utility_floor": thresholds.utility_floor,
            "rapec6_dynamic_risk_alpha": thresholds.risk_alpha,
            "rapec6_dynamic_minimum_epochs": thresholds.minimum_epochs,
            "rapec6_dynamic_patience": thresholds.patience,
            "rapec6_validation_noise": thresholds.validation_noise,
            "rapec6_validation_standard_error": thresholds.validation_standard_error,
            "rapec6_quality_scale_tolerance": thresholds.quality_scale_tolerance,
            "rapec6_curve_stability": thresholds.curve_stability,
            "rapec6_recent_velocity": thresholds.recent_velocity,
            "rapec6_historical_velocity": thresholds.historical_velocity,
            "rapec6_remaining_gain_estimate": thresholds.remaining_gain_estimate,
            "rapec6_recovery_probability": thresholds.recovery_probability,
            "rapec6_recovery_limit": thresholds.recovery_limit,
            "rapec6_risk_probability": risk_probability,
            "rapec6_risk_constraint_satisfied": int(risk_satisfied),
            "rapec6_recovery_guard_satisfied": int(recovery_satisfied),
            "rapec6_uncertainty_actionable": int(uncertainty_actionable),
            "rapec6_feature_coverage_fraction": feature_coverage,
            "rapec6_calibration_samples": calibration_samples,
            "rapec6_relative_uncertainty": widest_relative_uncertainty
            if math.isfinite(widest_relative_uncertainty)
            else None,
            "rapec6_learning_state_code": self._learning_state_code(learning_state),
            "rapec6_pareto_front_size": len(pareto_front),
            "rapec6_valuable_horizon_count": len(valuable_horizons),
            "rapec6_selected_horizon": selected.horizon if selected else 0,
            "rapec6_candidate_stop": int(candidate),
            "rapec6_low_value_streak": self.low_value_streak,
            "rapec6_projected_training_energy_saving_fraction": training_energy_saving,
            "rapec6_cumulative_training_energy_wh": observation.cumulative_energy_wh,
            "rapec6_energy_scope_training_only": 1,
            "rapec6_uses_external_full100_history": 0,
            "rapec6_lcpfn_comparison_ready": int(len(qualities) >= self.min_fit_points),
            "rapec6_current_best_quality": current_best,
        }
        for horizon in self.horizons:
            forecast = next((item for item in forecasts if item.horizon == horizon), None)
            prefix = f"rapec6_h{horizon}"
            diagnostics.update(
                {
                    f"{prefix}_expected_quality": forecast.expected_quality if forecast else None,
                    f"{prefix}_expected_gain": forecast.expected_gain if forecast else None,
                    f"{prefix}_gain_lower": forecast.lower_gain if forecast else None,
                    f"{prefix}_gain_upper": forecast.upper_gain if forecast else None,
                    f"{prefix}_prob_gain_gt_epsilon": (
                        forecast.probability_gain_gt_epsilon if forecast else None
                    ),
                    f"{prefix}_expected_energy_wh": forecast.expected_energy_wh if forecast else None,
                    f"{prefix}_energy_upper_wh": forecast.energy_upper_wh if forecast else None,
                    f"{prefix}_expected_duration_seconds": (
                        forecast.expected_duration_seconds if forecast else None
                    ),
                    f"{prefix}_utility_quality_per_wh": (
                        forecast.utility_quality_per_wh if forecast else None
                    ),
                    f"{prefix}_conservative_utility_per_wh": (
                        forecast.conservative_utility_per_wh if forecast else None
                    ),
                    f"{prefix}_uncertainty_width": forecast.uncertainty_width if forecast else None,
                    f"{prefix}_calibration_radius": forecast.calibration_radius if forecast else None,
                }
            )

        if not enough_history:
            self.low_value_streak = 0
            diagnostics["rapec6_low_value_streak"] = 0
            diagnostics["rapec6_candidate_stop"] = 0
            return ControllerDecision(
                stop=False,
                reason="rapec6_warmup",
                confidence=0.0,
                predicted_energy_saving_fraction=training_energy_saving,
                predicted_quality_regret=predicted_regret,
                diagnostics=diagnostics,
            )

        reason = (
            "rapec6_stop: "
            f"epoch={observation.epoch}, state={learning_state}, "
            f"epsilon={thresholds.epsilon:.6f}, risk={risk_probability:.4f}, "
            f"alpha={thresholds.risk_alpha:.4f}, recovery={thresholds.recovery_probability:.4f}, "
            f"valuable_horizons={len(valuable_horizons)}"
        )
        return ControllerDecision(
            stop=stop,
            reason=reason if stop else "continue",
            confidence=confidence,
            predicted_energy_saving_fraction=training_energy_saving,
            predicted_quality_regret=predicted_regret,
            diagnostics=diagnostics,
        )

    def _forecast(
        self,
        rows: list[Mapping[str, Any]],
        observation: EpochObservation,
        horizon: int,
        epsilon: float,
        recovery_probability: float,
    ) -> MultiHorizonForecast:
        qualities = self._quality_values(rows, observation)
        best = _best_so_far(qualities)
        current_best = best[-1] if best else float(observation.best_quality or observation.quality or 0.0)
        gain_samples = self._gain_samples(best, rows, horizon, recovery_probability)
        calibration_residuals = self._online_calibration_residuals(best, horizon)
        if calibration_residuals:
            seed = observation.epoch * 1009 + horizon * 9173
            rng = random.Random(seed)
            gain_samples = [
                max(0.0, sample + rng.choice(calibration_residuals))
                for sample in gain_samples
            ]
        if not gain_samples:
            gain_samples = [0.0]

        calibration_radius = _quantile(
            [abs(value) for value in calibration_residuals],
            self.calibration_quantile,
        ) or 0.0
        expected_gain = _median(gain_samples) or 0.0
        lower_gain = max(0.0, (_quantile(gain_samples, 0.10) or 0.0) - calibration_radius)
        upper_gain = max(lower_gain, (_quantile(gain_samples, 0.90) or expected_gain) + calibration_radius)
        probability = sum(value > epsilon for value in gain_samples) / len(gain_samples)

        energy_samples = self._resource_sum_samples(rows, horizon, "energy")
        duration_samples = self._resource_sum_samples(rows, horizon, "duration")
        expected_energy = _median(energy_samples) or 0.0
        energy_upper = _quantile(energy_samples, 0.90) or expected_energy
        expected_duration = _median(duration_samples) or 0.0
        utility = expected_gain / expected_energy if expected_energy > 0.0 else 0.0
        conservative_utility = lower_gain / energy_upper if energy_upper > 0.0 else 0.0
        return MultiHorizonForecast(
            horizon=horizon,
            expected_quality=_clamp(current_best + expected_gain, 0.0, 1.0),
            expected_gain=expected_gain,
            lower_gain=lower_gain,
            upper_gain=upper_gain,
            probability_gain_gt_epsilon=probability,
            expected_energy_wh=expected_energy,
            energy_upper_wh=energy_upper,
            expected_duration_seconds=expected_duration,
            utility_quality_per_wh=utility,
            conservative_utility_per_wh=conservative_utility,
            uncertainty_width=max(0.0, upper_gain - lower_gain),
            calibration_radius=calibration_radius,
            calibration_samples=len(calibration_residuals),
            prediction_samples=len(gain_samples),
        )

    def _gain_samples(
        self,
        best: list[float],
        rows: list[Mapping[str, Any]],
        horizon: int,
        recovery_probability: float,
    ) -> list[float]:
        if len(best) < 2:
            return []
        increments = [max(0.0, right - left) for left, right in zip(best, best[1:])]
        recent = increments[-self.trend_window :]
        long_center = _median([value for value in increments if value > 0.0]) or 0.0
        short_center = _median(recent) or 0.0
        # Historical growth is represented only by explicit recovery samples.
        # Mixing it into every forecast would make a long, stable plateau look
        # profitable forever.
        center = short_center
        recent_positive = [value for value in recent if value > 0.0]
        recovery_gain = _quantile(recent_positive or [value for value in increments if value > 0.0], 0.65) or 0.0
        seed = len(best) * 10007 + horizon * 7919 + int(best[-1] * 1_000_000)
        rng = random.Random(seed)
        pool = recent or [0.0]
        samples = [max(0.0, center * horizon)]
        for _ in range(self.bootstrap_samples):
            gain = 0.0
            for step in range(horizon):
                sampled = rng.choice(pool)
                decay = 1.0 / (1.0 + 0.03 * step)
                gain += max(0.0, (0.55 * sampled + 0.45 * center) * decay)
            if recovery_gain > 0.0 and rng.random() < recovery_probability:
                gain += recovery_gain * horizon * rng.uniform(0.20, 0.80)
            samples.append(max(0.0, gain))

        samples.extend(self._same_run_analogs(best, rows, horizon))
        return samples

    def _same_run_analogs(
        self,
        best: list[float],
        rows: list[Mapping[str, Any]],
        horizon: int,
    ) -> list[float]:
        if len(best) <= self.min_fit_points + horizon:
            return []
        current_velocity = self._window_velocity(best, len(best))
        candidates: list[tuple[float, float]] = []
        for anchor in range(self.min_fit_points, len(best) - horizon):
            velocity = self._window_velocity(best, anchor)
            scale = max(abs(current_velocity), abs(velocity), 1e-6)
            distance = abs(current_velocity - velocity) / scale
            observed_gain = max(0.0, best[anchor + horizon] - best[anchor])
            candidates.append((distance, observed_gain))
        candidates.sort(key=lambda item: item[0])
        return [gain for _, gain in candidates[:8]]

    def _online_calibration_residuals(self, best: list[float], horizon: int) -> list[float]:
        residuals: list[tuple[int, float]] = []
        for anchor in range(self.min_fit_points, len(best) - horizon):
            increments = [
                max(0.0, right - left)
                for left, right in zip(best[max(0, anchor - self.trend_window) : anchor], best[max(0, anchor - self.trend_window) + 1 : anchor + 1])
            ]
            predicted = (_median(increments) or 0.0) * horizon
            actual = max(0.0, best[anchor + horizon] - best[anchor])
            residuals.append((anchor, actual - predicted))
        # Calibration must follow the current learning regime. Retaining early
        # high-growth residuals after a long plateau creates falsely wide
        # intervals and prevents any risk-constrained stop.
        residuals.sort(key=lambda item: item[0])
        limit = max(4, min(12, self.trend_window // 2))
        return [value for _, value in residuals[-limit:]]

    def _resource_sum_samples(
        self,
        rows: list[Mapping[str, Any]],
        horizon: int,
        resource: str,
    ) -> list[float]:
        if resource == "energy":
            values = [self._energy_wh(row) for row in rows[-self.trend_window :]]
        else:
            values = [
                max(0.0, _number(row.get("duration_seconds"), 0.0) or 0.0)
                for row in rows[-self.trend_window :]
            ]
        values = [value for value in values if value > 0.0]
        if not values:
            return [0.0]
        rng = random.Random(len(rows) * 3571 + horizon * 101 + (1 if resource == "energy" else 2))
        samples = [statistics.median(values) * horizon]
        for _ in range(self.bootstrap_samples):
            samples.append(sum(rng.choice(values) for _ in range(horizon)))
        return samples

    def _dynamic_thresholds(
        self,
        rows: list[Mapping[str, Any]],
        observation: EpochObservation,
        forecasts: list[MultiHorizonForecast],
        recovery_probability: float,
    ) -> OnlineThresholds:
        qualities = self._quality_values(rows, observation)
        best = _best_so_far(qualities)
        recent_raw = qualities[-self.trend_window :]
        validation_noise = _mad(recent_raw)
        increments = [max(0.0, right - left) for left, right in zip(best, best[1:])]
        recent_velocity = _median(increments[-max(3, self.trend_window // 3) :]) or 0.0
        historical_velocity = _quantile([value for value in increments if value > 0.0], 0.60) or recent_velocity
        velocity_reference = max(historical_velocity, 1e-8)
        curve_stability = _clamp(
            1.0 / (1.0 + validation_noise / max(velocity_reference, 1e-8)),
            0.0,
            1.0,
        )
        remaining_gain = max((forecast.expected_gain for forecast in forecasts), default=0.0)
        validation_standard_error = validation_noise / math.sqrt(max(2, len(recent_raw)))
        noise_multiplier = self.epsilon_noise_multiplier * (1.0 + 0.5 * (1.0 - curve_stability))
        gain_fraction = self.epsilon_remaining_gain_fraction * (0.75 + 0.5 * curve_stability)
        current_best = best[-1] if best else 0.0
        saturation_score = _clamp(
            1.0 - recent_velocity / max(historical_velocity, 1e-8),
            0.0,
            1.0,
        )
        quality_fraction = (
            self.epsilon_quality_fraction_min
            + (self.epsilon_quality_fraction_max - self.epsilon_quality_fraction_min)
            * curve_stability
        )
        quality_scale_tolerance = current_best * quality_fraction * saturation_score
        epsilon = max(
            self.epsilon_floor,
            noise_multiplier * validation_standard_error,
            gain_fraction * remaining_gain,
            quality_scale_tolerance,
        )

        historical_utilities = self._historical_utilities(rows, observation)
        utility_reference = _quantile(historical_utilities, self.utility_quantile) or 0.0
        utility_floor = max(0.0, utility_reference * (0.20 + 0.45 * curve_stability))
        risk_alpha = _clamp(
            self.base_risk_alpha
            * (0.55 + 0.90 * curve_stability)
            * (1.0 - 0.65 * recovery_probability),
            0.03,
            0.20,
        )
        epoch_fraction = _clamp(observation.epoch / max(1, observation.max_epochs), 0.0, 1.0)
        minimum_fraction = _clamp(
            0.12 + 0.30 * recovery_probability + 0.10 * (1.0 - curve_stability),
            0.12,
            0.70,
        )
        minimum_epochs = max(
            self.min_epochs_floor,
            int(math.ceil(minimum_fraction * observation.max_epochs)),
        )
        patience = int(
            round(
                _clamp(
                    self.min_patience
                    + 3.0 * recovery_probability
                    + 2.0 * (1.0 - curve_stability)
                    - (1.0 if epoch_fraction > 0.80 and curve_stability > 0.70 else 0.0),
                    self.min_patience,
                    self.max_patience,
                )
            )
        )
        recovery_limit = _clamp(0.12 + 0.18 * epoch_fraction + 0.10 * curve_stability, 0.10, 0.35)
        return OnlineThresholds(
            epsilon=epsilon,
            utility_floor=utility_floor,
            risk_alpha=risk_alpha,
            minimum_epochs=minimum_epochs,
            patience=patience,
            recovery_probability=recovery_probability,
            recovery_limit=recovery_limit,
            validation_noise=validation_noise,
            validation_standard_error=validation_standard_error,
            quality_scale_tolerance=quality_scale_tolerance,
            curve_stability=curve_stability,
            recent_velocity=recent_velocity,
            historical_velocity=historical_velocity,
            remaining_gain_estimate=remaining_gain,
        )

    def _provisional_epsilon(self, qualities: list[float]) -> float:
        if not qualities:
            return self.epsilon_floor
        best = _best_so_far(qualities)
        increments = [max(0.0, right - left) for left, right in zip(best, best[1:])]
        recent_gain = (_median(increments[-self.trend_window :]) or 0.0) * max(self.horizons)
        return max(
            self.epsilon_floor,
            self.epsilon_noise_multiplier
            * _mad(qualities[-self.trend_window :])
            / math.sqrt(max(2, min(len(qualities), self.trend_window))),
            self.epsilon_remaining_gain_fraction * recent_gain,
        )

    def _recovery_probability(
        self,
        rows: list[Mapping[str, Any]],
        observation: EpochObservation,
    ) -> float:
        qualities = self._quality_values(rows, observation)
        if len(qualities) < 4:
            return 1.0
        best = _best_so_far(qualities)
        increments = [max(0.0, right - left) for left, right in zip(best, best[1:])]
        recent_velocity = _median(increments[-max(3, self.trend_window // 3) :]) or 0.0
        historical_velocity = _quantile([value for value in increments if value > 0.0], 0.60) or recent_velocity
        stall = _clamp(1.0 - recent_velocity / max(historical_velocity, 1e-8), 0.0, 1.0)
        historical_recovery = self._historical_stall_recovery(best)
        loss_improving = self._loss_improvement_signal(rows)
        gradient_activity = self._relative_signal(rows, "gradient_norm")
        learning_rate_headroom = self._learning_rate_headroom(rows)
        progress = _clamp(observation.epoch / max(1, observation.max_epochs), 0.0, 1.0)
        early_stall_risk = stall * (1.0 - progress)
        return _clamp(
            0.35 * historical_recovery
            + 0.25 * loss_improving
            + 0.20 * gradient_activity
            + 0.10 * learning_rate_headroom
            + 0.10 * early_stall_risk,
            0.0,
            1.0,
        )

    def _historical_stall_recovery(self, best: list[float]) -> float:
        window = max(3, min(8, len(best) // 5))
        if len(best) < 2 * window + 2:
            return 0.0
        increments = [max(0.0, right - left) for left, right in zip(best, best[1:])]
        reference = _quantile([value for value in increments if value > 0.0], 0.50) or 0.0
        if reference <= 0.0:
            return 0.0
        stalls = 0
        recoveries = 0
        for anchor in range(window, len(best) - window):
            past_gain = best[anchor] - best[anchor - window]
            future_gain = best[anchor + window] - best[anchor]
            if past_gain <= reference * window * 0.30:
                stalls += 1
                if future_gain >= reference * window * 0.65:
                    recoveries += 1
        return recoveries / stalls if stalls else 0.0

    def _loss_improvement_signal(self, rows: list[Mapping[str, Any]]) -> float:
        values = [
            self._row_loss(row)
            for row in rows[-self.trend_window :]
            if self._row_loss(row) is not None
        ]
        if len(values) < 3:
            return 0.0
        half = max(1, len(values) // 2)
        earlier = _median(values[:half]) or 0.0
        recent = _median(values[half:]) or earlier
        if earlier <= 0.0:
            return 0.0
        return _clamp((earlier - recent) / earlier * 4.0, 0.0, 1.0)

    def _relative_signal(self, rows: list[Mapping[str, Any]], key: str) -> float:
        values = [
            _number(row.get(key))
            for row in rows
            if _number(row.get(key)) is not None and (_number(row.get(key)) or 0.0) >= 0.0
        ]
        if len(values) < 3:
            return 0.0
        historical = _median(values) or 0.0
        recent = _median(values[-max(3, len(values) // 4) :]) or 0.0
        return _clamp(recent / max(historical, 1e-12), 0.0, 1.0)

    def _learning_rate_headroom(self, rows: list[Mapping[str, Any]]) -> float:
        values = [
            _number(row.get("learning_rate"), _number(row.get("lr")))
            for row in rows
            if _number(row.get("learning_rate"), _number(row.get("lr"))) is not None
        ]
        if len(values) < 2:
            return 0.0
        current = max(0.0, values[-1])
        maximum = max(values)
        return _clamp(current / max(maximum, 1e-12), 0.0, 1.0)

    def _learning_state(
        self,
        rows: list[Mapping[str, Any]],
        observation: EpochObservation,
        thresholds: OnlineThresholds,
    ) -> str:
        qualities = self._quality_values(rows, observation)
        if len(qualities) < self.min_fit_points:
            return "insufficient_history"
        best = _best_so_far(qualities)
        recent_best = best[-max(4, self.trend_window // 2) :]
        recent_raw = qualities[-max(4, self.trend_window // 2) :]
        best_slope = _median(
            right - left for left, right in zip(recent_best, recent_best[1:])
        ) or 0.0
        raw_slope = _median(
            right - left for left, right in zip(recent_raw, recent_raw[1:])
        ) or 0.0
        horizon_gain = best_slope * min(5, max(self.horizons))
        if raw_slope < -max(thresholds.validation_noise, self.epsilon_floor):
            return "overfit_risk"
        if thresholds.validation_noise > max(thresholds.epsilon, abs(raw_slope) * 5.0) * 1.5:
            return "unstable"
        if horizon_gain > thresholds.epsilon:
            return "fast_learning"
        if horizon_gain > thresholds.epsilon * 0.5:
            return "stable_learning"
        if abs(horizon_gain) <= thresholds.epsilon * 0.20:
            return "plateau"
        return "slow_learning"

    def _pareto_front(self, forecasts: list[MultiHorizonForecast]) -> list[MultiHorizonForecast]:
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
        return sorted(front, key=lambda item: item.horizon)

    def _select_horizon(
        self,
        forecasts: list[MultiHorizonForecast],
        thresholds: OnlineThresholds,
    ) -> MultiHorizonForecast | None:
        if not forecasts:
            return None

        def score(forecast: MultiHorizonForecast) -> float:
            risk_penalty = max(0.0, forecast.probability_gain_gt_epsilon - thresholds.risk_alpha)
            uncertainty_penalty = forecast.uncertainty_width / max(
                thresholds.epsilon,
                self.epsilon_floor,
            )
            return (
                forecast.utility_quality_per_wh
                - thresholds.utility_floor
                - risk_penalty
                - 0.01 * uncertainty_penalty
            )

        return max(forecasts, key=score)

    def _historical_utilities(
        self,
        rows: list[Mapping[str, Any]],
        observation: EpochObservation,
    ) -> list[float]:
        qualities = self._quality_values(rows, observation)
        best = _best_so_far(qualities)
        utilities = []
        for horizon in self.horizons:
            for start in range(0, len(best) - horizon):
                gain = max(0.0, best[start + horizon] - best[start])
                energy = sum(self._energy_wh(row) for row in rows[start + 1 : start + horizon + 1])
                if gain > 0.0 and energy > 0.0:
                    utilities.append(gain / energy)
        return utilities

    def _projected_training_energy_saving(
        self,
        rows: list[Mapping[str, Any]],
        observation: EpochObservation,
    ) -> float | None:
        energies = [self._energy_wh(row) for row in rows[-self.trend_window :]]
        energies = [value for value in energies if value > 0.0]
        if not energies:
            return None
        future_training_wh = statistics.median(energies) * max(
            0,
            observation.max_epochs - observation.epoch,
        )
        projected_full_training_wh = observation.cumulative_energy_wh + future_training_wh
        if projected_full_training_wh <= 0.0:
            return None
        return _clamp(future_training_wh / projected_full_training_wh, 0.0, 1.0)

    def _feature_coverage(
        self,
        rows: list[Mapping[str, Any]],
        observation: EpochObservation,
    ) -> float:
        features = self._state_features(rows, observation)
        available = sum(features.get(key) is not None for key in self.FEATURE_KEYS)
        return available / len(self.FEATURE_KEYS)

    def _state_features(
        self,
        rows: list[Mapping[str, Any]],
        observation: EpochObservation,
    ) -> dict[str, float | None]:
        current = rows[-1] if rows else {}
        qualities = self._quality_values(rows, observation)
        best = _best_so_far(qualities)
        raw_recent = qualities[-self.trend_window :]
        quality_slope = _median(
            right - left for left, right in zip(raw_recent, raw_recent[1:])
        )
        losses = [
            self._row_loss(row)
            for row in rows[-self.trend_window :]
            if self._row_loss(row) is not None
        ]
        loss_slope = _median(
            right - left for left, right in zip(losses, losses[1:])
        )
        gradient = _number(current.get("gradient_norm"))
        return {
            "quality": qualities[-1] if qualities else None,
            "best_quality": best[-1] if best else None,
            "quality_slope": quality_slope,
            "quality_noise": _mad(raw_recent) if raw_recent else None,
            "train_loss": self._row_loss(current),
            "loss_slope": loss_slope,
            "gradient_norm": gradient,
            "gradient_activity": self._relative_signal(rows, "gradient_norm") if gradient is not None else None,
            "learning_rate": _number(current.get("learning_rate"), _number(current.get("lr"))),
            "epoch_duration_seconds": _number(current.get("duration_seconds")),
            "gpu_utilization_pct": _number(
                current.get("gpu_util_avg_pct"),
                _number(current.get("gpu_utilization_pct")),
            ),
            "gpu_memory_used_mb": _number(
                current.get("gpu_memory_used_mb"),
                _number(current.get("gpu_mem_used_avg_mb")),
            ),
            "gpu_power_avg_w": _number(current.get("gpu_power_avg_w")),
            "epoch_energy_wh": self._energy_wh(current),
            "model_parameter_count": _number(
                current.get("model_parameter_count"),
                _number(current.get("model_parameters")),
            ),
            "model_flops": _number(
                current.get("model_flops"),
                _number(current.get("model_flops_estimated")),
            ),
        }

    def _quality_values(
        self,
        rows: list[Mapping[str, Any]],
        observation: EpochObservation,
    ) -> list[float]:
        values = []
        for row in rows:
            value = _number(
                row.get(observation.quality_metric),
                _number(row.get("quality_score"), _number(row.get("map50_95"))),
            )
            if value is not None:
                values.append(_clamp(value, 0.0, 1.0))
        return values

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

    def _window_velocity(self, best: list[float], end: int) -> float:
        start = max(0, end - self.trend_window)
        values = best[start:end]
        return _median(
            right - left for left, right in zip(values, values[1:])
        ) or 0.0

    def _confidence(
        self,
        risk_probability: float,
        thresholds: OnlineThresholds,
        streak: int,
        uncertainty_actionable: bool,
    ) -> float:
        risk_margin = _clamp(
            (thresholds.risk_alpha - risk_probability)
            / max(thresholds.risk_alpha, 1e-9),
            0.0,
            1.0,
        )
        streak_fraction = _clamp(streak / max(1, thresholds.patience), 0.0, 1.0)
        return _clamp(
            0.55 * risk_margin
            + 0.25 * streak_fraction
            + 0.20 * int(uncertainty_actionable),
            0.0,
            1.0,
        )

    @staticmethod
    def _learning_state_code(state: str) -> int:
        return {
            "insufficient_history": 0,
            "fast_learning": 1,
            "stable_learning": 2,
            "slow_learning": 3,
            "plateau": 4,
            "unstable": 5,
            "overfit_risk": 6,
        }.get(state, -1)
