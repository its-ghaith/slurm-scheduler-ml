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
    pos = (len(clean) - 1) * q
    low = int(math.floor(pos))
    high = int(math.ceil(pos))
    if low == high:
        return clean[low]
    weight = pos - low
    return clean[low] * (1.0 - weight) + clean[high] * weight


def _mad(values: Iterable[float]) -> float:
    clean = [value for value in values if math.isfinite(value)]
    if len(clean) < 2:
        return 0.0
    median = statistics.median(clean)
    deviations = [abs(value - median) for value in clean]
    return 1.4826 * statistics.median(deviations)


def _best_so_far(values: list[float]) -> list[float]:
    result: list[float] = []
    best = 0.0
    for value in values:
        best = max(best, min(1.0, max(0.0, value)))
        result.append(best)
    return result


@dataclass(frozen=True)
class DynamicThresholds:
    meaningful_gain: float
    utility_threshold: float
    adaptive_min_epochs: int
    validation_noise: float
    recent_gain_horizon: float
    remaining_quality_room: float


@dataclass(frozen=True)
class HorizonPrediction:
    expected_quality: float | None
    expected_gain: float | None
    lower_gain: float | None
    upper_gain: float | None
    probability_gain_gt_dynamic_threshold: float | None
    expected_energy_wh: float | None
    expected_duration_seconds: float | None
    utility_quality_per_wh: float | None
    uncertainty_width: float | None
    feature_coverage_fraction: float
    samples: int


class RiskAwarePredictiveEnergyController(Controller):
    """RAPEC-v3: dynamic, risk-aware, energy-adaptive early stopping.

    The controller is intentionally task-independent: it consumes the abstract
    quality_score Q_t in [0, 1] and combines it with energy and runtime features.
    Instead of a fixed 0.2 percentage point threshold, it estimates a meaningful
    future gain from validation noise, recent learning speed, training stage and
    remaining quality room in the current run.
    """

    FEATURE_KEYS = (
        "epoch_fraction",
        "quality",
        "best_quality",
        "recent_slope",
        "recent_quality_noise",
        "recent_efficiency_per_wh",
        "epoch_energy_wh",
        "duration_seconds",
        "gpu_utilization_pct",
        "gpu_memory_used_mb",
        "gpu_power_avg_w",
        "learning_rate",
        "train_loss",
        "gradient_norm",
        "log_model_parameters",
        "log_model_flops",
    )

    STOP_STATES = {"slow_learning", "plateau", "overfit_risk"}

    def __init__(self, context):
        super().__init__(context)
        params = context.parameters
        self.horizon_epochs = max(1, int(params.get("horizon_epochs", 5)))
        self.min_fit_points = max(4, int(params.get("min_fit_points", 10)))
        self.patience = max(1, int(params.get("patience", 3)))
        self.trend_window = max(4, int(params.get("trend_window", 10)))
        self.bootstrap_samples = max(0, int(params.get("bootstrap_samples", 96)))
        self.knn_neighbors = max(1, int(params.get("knn_neighbors", 9)))
        self.min_epochs_floor = max(self.min_fit_points, int(params.get("min_epochs_floor", 12)))
        self.max_probability_gain_gt_threshold = float(params.get("max_probability_gain_gt_threshold", 0.25))
        self.max_uncertainty_to_gain_ratio = float(params.get("max_uncertainty_to_gain_ratio", 4.0))
        self.minimum_quality_floor = float(params.get("minimum_quality_floor", 0.0))
        self.noise_multiplier = float(params.get("noise_multiplier", 1.25))
        self.recent_gain_fraction = float(params.get("recent_gain_fraction", 0.35))
        self.remaining_room_fraction = float(params.get("remaining_room_fraction", 0.015))
        self.late_stage_relaxation = float(params.get("late_stage_relaxation", 0.55))
        self.min_meaningful_gain = float(params.get("min_meaningful_gain", 1e-4))
        self.utility_quantile = float(params.get("utility_quantile", 0.25))
        self.low_value_streak = 0

    def evaluate(self, observation: EpochObservation) -> ControllerDecision:
        rows = [*observation.history, observation.raw_metrics]
        qualities = self._quality_values(rows, observation)
        current_quality = observation.best_quality if observation.best_quality is not None else observation.quality
        thresholds = self._dynamic_thresholds(rows, observation)
        prediction = self._predict_horizon(rows, observation, thresholds.meaningful_gain)
        learning_state = self._learning_state(rows, observation, thresholds)
        diagnostics = {
            "rapec_horizon_epochs": self.horizon_epochs,
            "rapec_expected_quality_next_horizon": prediction.expected_quality,
            "rapec_expected_quality_gain_next_horizon": prediction.expected_gain,
            "rapec_quality_gain_lower": prediction.lower_gain,
            "rapec_quality_gain_upper": prediction.upper_gain,
            "rapec_prob_gain_gt_dynamic_threshold": prediction.probability_gain_gt_dynamic_threshold,
            "rapec_expected_energy_next_horizon_wh": prediction.expected_energy_wh,
            "rapec_expected_duration_next_horizon_s": prediction.expected_duration_seconds,
            "rapec_utility_quality_per_wh": prediction.utility_quality_per_wh,
            "rapec_uncertainty_width": prediction.uncertainty_width,
            "rapec_feature_coverage_fraction": prediction.feature_coverage_fraction,
            "rapec_prediction_samples": prediction.samples,
            "rapec_dynamic_meaningful_gain": thresholds.meaningful_gain,
            "rapec_dynamic_utility_threshold": thresholds.utility_threshold,
            "rapec_adaptive_min_epochs": thresholds.adaptive_min_epochs,
            "rapec_validation_noise": thresholds.validation_noise,
            "rapec_recent_gain_horizon": thresholds.recent_gain_horizon,
            "rapec_remaining_quality_room": thresholds.remaining_quality_room,
            "rapec_learning_state_code": self._learning_state_code(learning_state),
        }

        enough_history = (
            observation.epoch >= thresholds.adaptive_min_epochs
            and len(qualities) >= self.min_fit_points
            and current_quality is not None
            and prediction.expected_gain is not None
        )
        if not enough_history:
            self.low_value_streak = 0
            diagnostics.update(
                {
                    "rapec_candidate_stop": 0,
                    "rapec_low_value_streak": self.low_value_streak,
                    "rapec_enough_history": 0,
                    "rapec_low_probability": 0,
                    "rapec_low_efficiency": 0,
                    "rapec_stable_low_improvement_state": 0,
                    "rapec_uncertainty_ok": 0,
                    "rapec_quality_guard_ok": 0,
                }
            )
            return ControllerDecision(stop=False, reason="rapec_warmup_or_unstable_history", confidence=0.0, diagnostics=diagnostics)

        probability = prediction.probability_gain_gt_dynamic_threshold
        utility = prediction.utility_quality_per_wh
        uncertainty = prediction.uncertainty_width
        low_probability = probability is not None and probability < self.max_probability_gain_gt_threshold
        low_efficiency = utility is not None and utility < thresholds.utility_threshold
        state_ok = learning_state in self.STOP_STATES
        uncertainty_ok = self._uncertainty_is_actionable(uncertainty, thresholds.meaningful_gain, learning_state)
        quality_guard_ok = float(current_quality or 0.0) >= self._minimum_quality(thresholds, observation)
        candidate = low_probability and low_efficiency and state_ok and uncertainty_ok and quality_guard_ok
        self.low_value_streak = self.low_value_streak + 1 if candidate else 0
        stop = self.low_value_streak >= self.patience
        confidence = self._confidence(probability, utility, thresholds.utility_threshold, self.low_value_streak)
        diagnostics.update(
            {
                "rapec_candidate_stop": int(candidate),
                "rapec_low_value_streak": self.low_value_streak,
                "rapec_enough_history": 1,
                "rapec_low_probability": int(low_probability),
                "rapec_low_efficiency": int(low_efficiency),
                "rapec_stable_low_improvement_state": int(state_ok),
                "rapec_uncertainty_ok": int(uncertainty_ok),
                "rapec_quality_guard_ok": int(quality_guard_ok),
            }
        )
        reason = (
            "rapec_stop: "
            f"epoch={observation.epoch}, state={learning_state}, horizon={self.horizon_epochs}, "
            f"dynamic_gain={thresholds.meaningful_gain:.6f}, "
            f"expected_gain={float(prediction.expected_gain or 0.0):.6f}, "
            f"prob_gain={float(probability or 0.0):.4f}, "
            f"utility={float(utility or 0.0):.6f}, "
            f"utility_threshold={thresholds.utility_threshold:.6f}"
        )
        return ControllerDecision(
            stop=stop,
            reason=reason if stop else "continue",
            confidence=confidence,
            predicted_energy_saving_fraction=None,
            predicted_quality_regret=prediction.upper_gain,
            diagnostics=diagnostics,
        )

    def _dynamic_thresholds(self, rows: list[Mapping[str, Any]], observation: EpochObservation) -> DynamicThresholds:
        qualities = self._quality_values(rows, observation)
        best = _best_so_far(qualities)
        current_best = best[-1] if best else 0.0
        raw_recent = qualities[-self.trend_window :]
        best_recent = best[-self.trend_window :]
        validation_noise = _mad(raw_recent)
        horizon_gains = [
            max(0.0, best_recent[index] - best_recent[index - self.horizon_epochs])
            for index in range(self.horizon_epochs, len(best_recent))
        ]
        if not horizon_gains and len(best_recent) >= 2:
            per_epoch = [max(0.0, right - left) for left, right in zip(best_recent, best_recent[1:])]
            horizon_gains = [sum(per_epoch[-self.horizon_epochs :])]
        recent_gain_horizon = _median(horizon_gains) or 0.0
        remaining_room = max(0.0, 1.0 - current_best)
        epoch_fraction = max(0.0, min(1.0, observation.epoch / max(1, observation.max_epochs)))
        stage_relaxation = 1.0 - self.late_stage_relaxation * epoch_fraction
        meaningful_gain = max(
            self.min_meaningful_gain,
            self.noise_multiplier * validation_noise,
            self.recent_gain_fraction * recent_gain_horizon,
            self.remaining_room_fraction * remaining_room * stage_relaxation,
        )
        utility_values = self._historical_utilities(rows, observation)
        utility_threshold = _quantile(utility_values, self.utility_quantile)
        if utility_threshold is None:
            recent_energy = self._recent_horizon_energy(rows)
            utility_threshold = meaningful_gain / recent_energy if recent_energy and recent_energy > 0 else 0.0
        adaptive_min_epochs = self._adaptive_min_epochs(rows, observation, validation_noise)
        return DynamicThresholds(
            meaningful_gain=meaningful_gain,
            utility_threshold=max(0.0, utility_threshold),
            adaptive_min_epochs=adaptive_min_epochs,
            validation_noise=validation_noise,
            recent_gain_horizon=recent_gain_horizon,
            remaining_quality_room=remaining_room,
        )

    def _predict_horizon(
        self,
        rows: list[Mapping[str, Any]],
        observation: EpochObservation,
        dynamic_meaningful_gain: float,
    ) -> HorizonPrediction:
        current_best = observation.best_quality if observation.best_quality is not None else observation.quality
        current_best = float(current_best or 0.0)
        gain_distribution: list[float] = []
        energy_distribution: list[float] = []
        duration_distribution: list[float] = []

        memory = self._similar_state_memory(rows, observation)
        if memory:
            gain_distribution.extend(item["gain"] for item in memory)
            energy_distribution.extend(item["energy_wh"] for item in memory)
            duration_distribution.extend(item["duration_seconds"] for item in memory)

        trend_gains = self._bootstrap_trend_gains(rows, observation)
        if trend_gains:
            gain_distribution.extend(trend_gains)
        recent_energy = self._recent_horizon_energy(rows)
        if recent_energy is not None:
            energy_distribution.extend([recent_energy] * max(1, len(trend_gains)))
        recent_duration = self._recent_horizon_duration(rows)
        if recent_duration is not None:
            duration_distribution.extend([recent_duration] * max(1, len(trend_gains)))

        expected_gain = _median(gain_distribution)
        lower_gain = _quantile(gain_distribution, 0.10)
        upper_gain = _quantile(gain_distribution, 0.90)
        expected_energy = _median(energy_distribution)
        expected_duration = _median(duration_distribution)
        probability = None
        if gain_distribution:
            probability = sum(1 for gain in gain_distribution if gain > dynamic_meaningful_gain) / len(gain_distribution)
        utility = expected_gain / expected_energy if expected_gain is not None and expected_energy and expected_energy > 0 else None
        uncertainty = max(0.0, upper_gain - lower_gain) if lower_gain is not None and upper_gain is not None else None
        features = self._state_features(rows, observation)
        coverage = sum(1 for key in self.FEATURE_KEYS if features.get(key) is not None) / len(self.FEATURE_KEYS)
        expected_quality = min(1.0, max(0.0, current_best + expected_gain)) if expected_gain is not None else None
        return HorizonPrediction(
            expected_quality=expected_quality,
            expected_gain=expected_gain,
            lower_gain=lower_gain,
            upper_gain=upper_gain,
            probability_gain_gt_dynamic_threshold=probability,
            expected_energy_wh=expected_energy,
            expected_duration_seconds=expected_duration,
            utility_quality_per_wh=utility,
            uncertainty_width=uncertainty,
            feature_coverage_fraction=coverage,
            samples=len(gain_distribution),
        )

    def _learning_state(
        self,
        rows: list[Mapping[str, Any]],
        observation: EpochObservation,
        thresholds: DynamicThresholds,
    ) -> str:
        qualities = self._quality_values(rows, observation)
        if len(qualities) < max(4, self.min_fit_points // 2):
            return "insufficient_history"
        best = _best_so_far(qualities)
        recent_raw = qualities[-self.trend_window :]
        recent_best = best[-self.trend_window :]
        best_deltas = [right - left for left, right in zip(recent_best, recent_best[1:])]
        raw_deltas = [right - left for left, right in zip(recent_raw, recent_raw[1:])]
        median_best_delta = _median(best_deltas) or 0.0
        median_raw_delta = _median(raw_deltas) or 0.0
        volatility = _mad(recent_raw)
        horizon_slope = median_best_delta * self.horizon_epochs
        if volatility > max(thresholds.meaningful_gain, abs(median_raw_delta) * self.horizon_epochs) * 1.5:
            return "unstable"
        if median_raw_delta < -max(thresholds.validation_noise, self.min_meaningful_gain):
            return "overfit_risk"
        if horizon_slope > thresholds.meaningful_gain:
            return "fast_learning"
        if horizon_slope > thresholds.meaningful_gain * 0.5:
            return "stable_learning"
        if abs(horizon_slope) <= thresholds.meaningful_gain * 0.25:
            return "plateau"
        return "slow_learning"

    def _adaptive_min_epochs(self, rows: list[Mapping[str, Any]], observation: EpochObservation, noise: float) -> int:
        scenario = f"{self.context.scenario} {self.context.metadata.get('model_version', '')}".lower()
        task_type = self.context.task_type.lower()
        if "scratch" in scenario:
            fraction = 0.30
        elif "pretrained" in scenario:
            fraction = 0.18
        elif task_type == "semantic_segmentation":
            fraction = 0.25
        elif task_type == "image_classification":
            fraction = 0.18
        else:
            fraction = 0.22
        noise_penalty = 0.05 if noise > 0.01 else 0.0
        return min(
            max(1, observation.max_epochs - self.horizon_epochs),
            max(self.min_epochs_floor, int(math.ceil((fraction + noise_penalty) * observation.max_epochs))),
        )

    def _minimum_quality(self, thresholds: DynamicThresholds, observation: EpochObservation) -> float:
        epoch_fraction = max(0.0, min(1.0, observation.epoch / max(1, observation.max_epochs)))
        return max(self.minimum_quality_floor, min(0.95, epoch_fraction * 0.10 - thresholds.validation_noise))

    def _uncertainty_is_actionable(self, uncertainty: float | None, meaningful_gain: float, learning_state: str) -> bool:
        if learning_state == "overfit_risk":
            return True
        if uncertainty is None:
            return False
        return uncertainty <= max(meaningful_gain * self.max_uncertainty_to_gain_ratio, self.min_meaningful_gain)

    def _confidence(
        self,
        probability: float | None,
        utility: float | None,
        utility_threshold: float,
        streak: int,
    ) -> float:
        prob_component = 1.0 - max(0.0, min(1.0, float(probability or 0.0)))
        if utility is None or utility_threshold <= 0.0:
            utility_component = 0.5
        else:
            utility_component = max(0.0, min(1.0, 1.0 - utility / utility_threshold))
        streak_component = max(0.0, min(1.0, streak / self.patience))
        return max(0.0, min(1.0, 0.45 * prob_component + 0.35 * utility_component + 0.20 * streak_component))

    def _similar_state_memory(
        self, rows: list[Mapping[str, Any]], observation: EpochObservation
    ) -> list[dict[str, float]]:
        if len(rows) <= self.horizon_epochs + self.min_fit_points:
            return []
        current_features = self._state_features(rows, observation)
        scales = self._feature_scales(rows)
        samples = []
        for end_index in range(self.min_fit_points - 1, len(rows) - self.horizon_epochs):
            prefix = rows[: end_index + 1]
            horizon = rows[end_index + 1 : end_index + 1 + self.horizon_epochs]
            values = self._quality_values(prefix, observation)
            future_values = self._quality_values([*prefix, *horizon], observation)
            if not values or not future_values:
                continue
            start_best = max(values)
            end_best = max(future_values)
            gain = max(0.0, end_best - start_best)
            energy_wh = sum(self._energy_wh(row) for row in horizon)
            duration_seconds = sum(_number(row.get("duration_seconds"), 0.0) or 0.0 for row in horizon)
            if energy_wh <= 0:
                continue
            features = self._state_features(prefix, observation)
            distance = self._feature_distance(current_features, features, rows, scales=scales)
            if distance is None:
                continue
            samples.append({"distance": distance, "gain": gain, "energy_wh": energy_wh, "duration_seconds": duration_seconds})
        samples.sort(key=lambda item: item["distance"])
        return samples[: self.knn_neighbors]

    def _bootstrap_trend_gains(self, rows: list[Mapping[str, Any]], observation: EpochObservation) -> list[float]:
        values = self._quality_values(rows, observation)
        if len(values) < self.min_fit_points:
            return []
        best_values = _best_so_far(values)
        recent = best_values[-self.trend_window :]
        slopes = [max(0.0, right - left) for left, right in zip(recent, recent[1:])]
        if not slopes:
            return []
        center = statistics.median(slopes)
        residuals = [slope - center for slope in slopes]
        seed = len(rows) * 10007 + int(best_values[-1] * 1_000_000)
        rng = random.Random(seed)
        predictions = [max(0.0, center * self.horizon_epochs)]
        for _ in range(self.bootstrap_samples):
            sampled = [center + rng.choice(residuals or [0.0]) for _ in range(self.horizon_epochs)]
            predictions.append(max(0.0, sum(sampled)))
        return predictions

    def _state_features(self, rows: list[Mapping[str, Any]], observation: EpochObservation) -> dict[str, float | None]:
        current = rows[-1]
        values = self._quality_values(rows, observation)
        best_values = _best_so_far(values)
        recent = best_values[-self.trend_window :]
        recent_slope = _median([right - left for left, right in zip(recent, recent[1:])]) if len(recent) >= 2 else None
        recent_noise = _mad(values[-self.trend_window :])
        recent_energy = sum(self._energy_wh(row) for row in rows[-self.trend_window :])
        recent_gain = max(recent) - min(recent) if recent else None
        recent_efficiency = recent_gain / recent_energy if recent_gain is not None and recent_energy > 0 else None
        params = _number(current.get("model_parameter_count"), _number(current.get("model_parameters")))
        flops = _number(current.get("model_flops"), _number(current.get("model_flops_estimated")))
        return {
            "epoch_fraction": float(current.get("epoch_index", current.get("epoch", 0))) / max(1, observation.max_epochs),
            "quality": values[-1] if values else None,
            "best_quality": best_values[-1] if best_values else None,
            "recent_slope": recent_slope,
            "recent_quality_noise": recent_noise,
            "recent_efficiency_per_wh": recent_efficiency,
            "epoch_energy_wh": self._energy_wh(current),
            "duration_seconds": _number(current.get("duration_seconds")),
            "gpu_utilization_pct": _number(current.get("gpu_util_avg_pct"), _number(current.get("gpu_utilization_pct"))),
            "gpu_memory_used_mb": _number(current.get("gpu_memory_used_mb"), _number(current.get("gpu_mem_used_avg_mb"))),
            "gpu_power_avg_w": _number(current.get("gpu_power_avg_w")),
            "learning_rate": _number(current.get("learning_rate"), _number(current.get("lr"))),
            "train_loss": _number(current.get("train_loss")),
            "gradient_norm": _number(current.get("gradient_norm")),
            "log_model_parameters": math.log10(params) if params is not None and params > 0 else None,
            "log_model_flops": math.log10(flops) if flops is not None and flops > 0 else None,
        }

    def _feature_distance(
        self,
        current: Mapping[str, float | None],
        candidate: Mapping[str, float | None],
        rows: list[Mapping[str, Any]],
        *,
        scales: Mapping[str, float] | None = None,
    ) -> float | None:
        distance = 0.0
        used = 0
        scales = scales or self._feature_scales(rows)
        for key in self.FEATURE_KEYS:
            a, b = current.get(key), candidate.get(key)
            if a is None or b is None:
                continue
            scale = scales.get(key, 1.0) or 1.0
            distance += ((a - b) / scale) ** 2
            used += 1
        if used < 3:
            return None
        return math.sqrt(distance / used)

    def _feature_scales(self, rows: list[Mapping[str, Any]]) -> dict[str, float]:
        dummy = EpochObservation(
            epoch=len(rows),
            max_epochs=self.context.max_epochs,
            task_type=self.context.task_type,
            quality_metric=self.context.quality_metric,
            quality=None,
            best_quality=None,
            delta_quality=None,
            epoch_energy_wh=0.0,
            cumulative_energy_wh=0.0,
            epoch_duration_seconds=0.0,
            cumulative_duration_seconds=0.0,
            gpu_utilization_pct=None,
            history=(),
            raw_metrics={},
        )
        samples: dict[str, list[float]] = {}
        for index in range(max(1, self.min_fit_points - 1), len(rows)):
            features = self._state_features(rows[: index + 1], dummy)
            for key, value in features.items():
                if value is not None and math.isfinite(value):
                    samples.setdefault(key, []).append(value)
        result: dict[str, float] = {}
        for key, values in samples.items():
            result[key] = max(statistics.pstdev(values), 1e-9) if len(values) >= 2 else 1.0
        return result

    def _historical_utilities(self, rows: list[Mapping[str, Any]], observation: EpochObservation) -> list[float]:
        values = self._quality_values(rows, observation)
        if len(values) <= self.horizon_epochs:
            return []
        best = _best_so_far(values)
        utilities = []
        for start in range(0, len(rows) - self.horizon_epochs):
            end = start + self.horizon_epochs
            gain = max(0.0, best[end] - best[start])
            energy = sum(self._energy_wh(row) for row in rows[start + 1 : end + 1])
            if energy > 0:
                utilities.append(gain / energy)
        return utilities

    def _quality_values(self, rows: list[Mapping[str, Any]], observation: EpochObservation) -> list[float]:
        values = []
        for row in rows:
            value = _number(row.get(observation.quality_metric), _number(row.get("quality_score"), _number(row.get("map50_95"))))
            if value is not None:
                values.append(min(1.0, max(0.0, value)))
        return values

    def _energy_wh(self, row: Mapping[str, Any]) -> float:
        if row.get("epoch_energy_wh") is not None:
            return max(0.0, _number(row.get("epoch_energy_wh"), 0.0) or 0.0)
        total_kwh = _number(row.get("total_energy_kwh"), None)
        if total_kwh is None:
            total_kwh = _number(row.get("gpu_energy_kwh"), 0.0)
        return max(0.0, float(total_kwh or 0.0) * 1000.0)

    def _recent_epoch_energy(self, rows: list[Mapping[str, Any]]) -> float | None:
        values = [self._energy_wh(row) for row in rows[-self.trend_window :] if self._energy_wh(row) > 0]
        return statistics.median(values) if values else None

    def _recent_horizon_energy(self, rows: list[Mapping[str, Any]]) -> float | None:
        epoch_energy = self._recent_epoch_energy(rows)
        return epoch_energy * self.horizon_epochs if epoch_energy is not None else None

    def _recent_horizon_duration(self, rows: list[Mapping[str, Any]]) -> float | None:
        values = [
            _number(row.get("duration_seconds"), None)
            for row in rows[-self.trend_window :]
            if _number(row.get("duration_seconds"), None) is not None
        ]
        return statistics.median(values) * self.horizon_epochs if values else None

    def _learning_state_code(self, state: str) -> int:
        return {
            "insufficient_history": 0,
            "fast_learning": 1,
            "stable_learning": 2,
            "slow_learning": 3,
            "plateau": 4,
            "unstable": 5,
            "overfit_risk": 6,
        }.get(state, -1)
