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


def _quantile(values: Iterable[float], q: float) -> float | None:
    clean = sorted(value for value in values if math.isfinite(value))
    if not clean:
        return None
    if q <= 0:
        return clean[0]
    if q >= 1:
        return clean[-1]
    pos = (len(clean) - 1) * q
    low, high = int(math.floor(pos)), int(math.ceil(pos))
    if low == high:
        return clean[low]
    weight = pos - low
    return clean[low] * (1.0 - weight) + clean[high] * weight


def _best_so_far(values: list[float]) -> list[float]:
    result: list[float] = []
    best = 0.0
    for value in values:
        best = max(best, min(1.0, max(0.0, value)))
        result.append(best)
    return result


def _median(values: Iterable[float]) -> float | None:
    clean = [value for value in values if math.isfinite(value)]
    return statistics.median(clean) if clean else None


@dataclass(frozen=True)
class HorizonPrediction:
    expected_quality: float | None
    expected_gain: float | None
    lower_gain: float | None
    upper_gain: float | None
    probability_gain_gt_threshold: float | None
    expected_energy_wh: float | None
    expected_duration_seconds: float | None
    utility_quality_per_wh: float | None
    uncertainty_width: float | None
    projected_energy_saving_fraction: float | None
    next_horizon_energy_saving_fraction: float | None
    feature_coverage_fraction: float
    samples: int


class ProbabilisticEnergyParetoController(Controller):
    """Predictive energy-aware controller for task-independent quality_score.

    The controller predicts the value of continuing for a short horizon, by
    combining a local learning-curve bootstrap with a feature-similarity memory
    over previous prefixes of the same run. It intentionally has no hard
    dependency on LC-PFN so it can run inside the existing Rancher image; LC-PFN
    remains the quality-only external baseline for the thesis comparison.
    """

    FEATURE_KEYS = (
        "epoch_fraction",
        "quality",
        "best_quality",
        "recent_slope",
        "recent_quality_std",
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

    def __init__(self, context):
        super().__init__(context)
        params = context.parameters
        self.min_epochs = int(params.get("min_epochs", 30))
        self.horizon_epochs = max(1, int(params.get("horizon_epochs", 5)))
        self.min_fit_points = max(3, int(params.get("min_fit_points", 10)))
        self.patience = max(1, int(params.get("patience", 3)))
        self.trend_window = max(3, int(params.get("trend_window", 10)))
        self.bootstrap_samples = max(0, int(params.get("bootstrap_samples", 64)))
        self.knn_neighbors = max(1, int(params.get("knn_neighbors", 7)))
        self.useful_gain = float(params.get("useful_gain", 0.002))
        self.min_expected_gain = float(params.get("min_expected_gain", self.useful_gain))
        self.max_probability_gain_gt_threshold = float(params.get("max_probability_gain_gt_threshold", 0.20))
        self.min_quality_per_wh = float(params.get("min_quality_per_wh", 0.0002))
        self.minimum_quality = float(params.get("minimum_quality", 0.0))
        self.target_energy_saving_fraction = float(params.get("target_energy_saving_fraction", 0.20))
        self.post_training_reserve_wh = max(0.0, float(params.get("post_training_reserve_wh", 0.0)))
        self.energy_weight = float(params.get("energy_weight", 0.0))
        self.time_weight = float(params.get("time_weight", 0.0))
        self.uncertainty_weight = float(params.get("uncertainty_weight", 1.0))
        self.low_value_streak = 0

    def evaluate(self, observation: EpochObservation) -> ControllerDecision:
        rows = [*observation.history, observation.raw_metrics]
        current_quality = observation.best_quality if observation.best_quality is not None else observation.quality
        prediction = self._predict_horizon(rows, observation)
        diagnostics = {
            "pep_horizon_epochs": self.horizon_epochs,
            "pep_expected_quality_next_horizon": prediction.expected_quality,
            "pep_expected_quality_gain_next_horizon": prediction.expected_gain,
            "pep_quality_gain_lower": prediction.lower_gain,
            "pep_quality_gain_upper": prediction.upper_gain,
            "pep_prob_gain_gt_threshold": prediction.probability_gain_gt_threshold,
            "pep_expected_energy_next_horizon_wh": prediction.expected_energy_wh,
            "pep_expected_duration_next_horizon_s": prediction.expected_duration_seconds,
            "pep_utility_quality_per_wh": prediction.utility_quality_per_wh,
            "pep_uncertainty_width": prediction.uncertainty_width,
            "pep_projected_energy_saving_fraction": prediction.projected_energy_saving_fraction,
            "pep_next_horizon_energy_saving_fraction": prediction.next_horizon_energy_saving_fraction,
            "pep_feature_coverage_fraction": prediction.feature_coverage_fraction,
            "pep_prediction_samples": prediction.samples,
            "pep_useful_gain_threshold": self.useful_gain,
            "pep_target_energy_saving_fraction": self.target_energy_saving_fraction,
            "pep_min_quality": self.minimum_quality,
            "pep_lcpfn_comparison_ready": int(self._lcpfn_ready(rows, observation)),
        }

        enough_history = observation.epoch >= self.min_epochs and len(self._quality_values(rows, observation)) >= self.min_fit_points
        if not enough_history or current_quality is None or prediction.expected_gain is None:
            self.low_value_streak = 0
            diagnostics["pep_low_value_streak"] = self.low_value_streak
            diagnostics["pep_candidate_stop"] = 0
            diagnostics["pep_pareto_score"] = None
            return ControllerDecision(stop=False, reason="pep_warmup", confidence=0.0, diagnostics=diagnostics)

        low_expected_gain = prediction.expected_gain < self.min_expected_gain
        low_probability = (
            prediction.probability_gain_gt_threshold is not None
            and prediction.probability_gain_gt_threshold < self.max_probability_gain_gt_threshold
        )
        low_efficiency = (
            prediction.utility_quality_per_wh is not None
            and prediction.utility_quality_per_wh < self.min_quality_per_wh
        )
        quality_ok = current_quality >= self.minimum_quality
        saving_ok = (
            prediction.projected_energy_saving_fraction is not None
            and prediction.projected_energy_saving_fraction >= self.target_energy_saving_fraction
        )
        pareto_score = self._pareto_score(prediction)
        candidate = low_expected_gain and low_probability and low_efficiency and quality_ok and saving_ok
        self.low_value_streak = self.low_value_streak + 1 if candidate else 0
        stop = self.low_value_streak >= self.patience
        confidence = min(
            1.0,
            max(0.0, 1.0 - float(prediction.probability_gain_gt_threshold or 0.0))
            * (self.low_value_streak / self.patience),
        )
        diagnostics.update(
            {
                "pep_low_expected_gain": int(low_expected_gain),
                "pep_low_probability": int(low_probability),
                "pep_low_efficiency": int(low_efficiency),
                "pep_quality_ok": int(quality_ok),
                "pep_saving_ok": int(saving_ok),
                "pep_candidate_stop": int(candidate),
                "pep_low_value_streak": self.low_value_streak,
                "pep_pareto_score": pareto_score,
            }
        )
        reason = (
            "pep_stop: "
            f"epoch={observation.epoch}, horizon={self.horizon_epochs}, "
            f"expected_gain={prediction.expected_gain:.6f}, "
            f"prob_gain_gt_{self.useful_gain:.4f}={float(prediction.probability_gain_gt_threshold or 0.0):.4f}, "
            f"utility={float(prediction.utility_quality_per_wh or 0.0):.6f}, "
            f"projected_saving={float(prediction.projected_energy_saving_fraction or 0.0):.4f}"
        )
        return ControllerDecision(
            stop=stop,
            reason=reason if stop else "continue",
            confidence=confidence,
            predicted_energy_saving_fraction=prediction.projected_energy_saving_fraction,
            predicted_quality_regret=prediction.upper_gain,
            diagnostics=diagnostics,
        )

    def _predict_horizon(self, rows: list[Mapping[str, Any]], observation: EpochObservation) -> HorizonPrediction:
        current_best = observation.best_quality if observation.best_quality is not None else observation.quality
        if current_best is None:
            current_best = 0.0
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
            probability = sum(1 for gain in gain_distribution if gain > self.useful_gain) / len(gain_distribution)
        utility = expected_gain / expected_energy if expected_gain is not None and expected_energy and expected_energy > 0 else None
        uncertainty = (
            max(0.0, upper_gain - lower_gain)
            if lower_gain is not None and upper_gain is not None
            else None
        )
        saving, next_saving = self._project_energy_saving(rows, observation, expected_energy)
        features = self._state_features(rows, observation)
        coverage = sum(1 for key in self.FEATURE_KEYS if features.get(key) is not None) / len(self.FEATURE_KEYS)
        expected_quality = min(1.0, max(0.0, current_best + expected_gain)) if expected_gain is not None else None
        return HorizonPrediction(
            expected_quality=expected_quality,
            expected_gain=expected_gain,
            lower_gain=lower_gain,
            upper_gain=upper_gain,
            probability_gain_gt_threshold=probability,
            expected_energy_wh=expected_energy,
            expected_duration_seconds=expected_duration,
            utility_quality_per_wh=utility,
            uncertainty_width=uncertainty,
            projected_energy_saving_fraction=saving,
            next_horizon_energy_saving_fraction=next_saving,
            feature_coverage_fraction=coverage,
            samples=len(gain_distribution),
        )

    def _similar_state_memory(
        self, rows: list[Mapping[str, Any]], observation: EpochObservation
    ) -> list[dict[str, float]]:
        if len(rows) <= self.horizon_epochs + self.min_fit_points:
            return []
        current_features = self._state_features(rows, observation)
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
            distance = self._feature_distance(current_features, features, rows)
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

    def _project_energy_saving(
        self,
        rows: list[Mapping[str, Any]],
        observation: EpochObservation,
        horizon_energy_wh: float | None,
    ) -> tuple[float | None, float | None]:
        current_epoch = max(1, int(observation.epoch))
        remaining_epochs = max(0, int(observation.max_epochs) - current_epoch)
        recent_epoch_energy = self._recent_epoch_energy(rows)
        if recent_epoch_energy is None or recent_epoch_energy <= 0:
            return None, None
        current_wh = max(0.0, float(observation.cumulative_energy_wh))
        projected_full = current_wh + recent_epoch_energy * remaining_epochs + self.post_training_reserve_wh
        if projected_full <= 0:
            return None, None
        projected_stop = current_wh + self.post_training_reserve_wh
        saving = max(0.0, (projected_full - projected_stop) / projected_full)
        next_energy = horizon_energy_wh if horizon_energy_wh is not None and horizon_energy_wh > 0 else recent_epoch_energy * self.horizon_epochs
        projected_after_next = projected_stop + next_energy
        next_saving = max(0.0, (projected_full - projected_after_next) / projected_full)
        return saving, next_saving

    def _pareto_score(self, prediction: HorizonPrediction) -> float | None:
        if prediction.expected_gain is None:
            return None
        energy = prediction.expected_energy_wh or 0.0
        duration = prediction.expected_duration_seconds or 0.0
        uncertainty = prediction.uncertainty_width or 0.0
        return prediction.expected_gain - self.energy_weight * energy - self.time_weight * duration - self.uncertainty_weight * uncertainty

    def _state_features(self, rows: list[Mapping[str, Any]], observation: EpochObservation) -> dict[str, float | None]:
        current = rows[-1]
        values = self._quality_values(rows, observation)
        best_values = _best_so_far(values)
        recent = best_values[-self.trend_window :]
        recent_slope = _median([right - left for left, right in zip(recent, recent[1:])]) if len(recent) >= 2 else None
        recent_std = statistics.pstdev(recent) if len(recent) >= 2 else None
        recent_energy = sum(self._energy_wh(row) for row in rows[-self.trend_window :])
        recent_gain = max(recent) - min(recent) if recent else None
        recent_efficiency = (
            recent_gain / recent_energy
            if recent_gain is not None and recent_energy > 0
            else None
        )
        params = _number(current.get("model_parameter_count"), _number(current.get("model_parameters")))
        flops = _number(current.get("model_flops"), _number(current.get("model_flops_estimated")))
        return {
            "epoch_fraction": float(current.get("epoch_index", current.get("epoch", 0))) / max(1, observation.max_epochs),
            "quality": values[-1] if values else None,
            "best_quality": best_values[-1] if best_values else None,
            "recent_slope": recent_slope,
            "recent_quality_std": recent_std,
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
    ) -> float | None:
        distance = 0.0
        used = 0
        scales = self._feature_scales(rows)
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
        scales: dict[str, float] = {}
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
        for index in range(max(1, self.min_fit_points - 1), len(rows)):
            features = self._state_features(rows[: index + 1], dummy)
            for key, value in features.items():
                if value is not None and math.isfinite(value):
                    scales.setdefault(key, []).append(value)  # type: ignore[union-attr]
        result: dict[str, float] = {}
        for key, values in scales.items():
            if isinstance(values, list) and len(values) >= 2:
                result[key] = max(statistics.pstdev(values), 1e-9)
            else:
                result[key] = 1.0
        return result

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

    def _lcpfn_ready(self, rows: list[Mapping[str, Any]], observation: EpochObservation) -> bool:
        values = self._quality_values(rows, observation)
        return len(values) >= self.min_fit_points and all(0.0 <= value <= 1.0 for value in values)
