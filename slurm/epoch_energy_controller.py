from __future__ import annotations

import csv
import json
import math
import random
import statistics
import time
from dataclasses import dataclass
from pathlib import Path

try:
    from slurm.gpu_energy_utils import summarize_window, to_float
except ImportError:  # pragma: no cover - script execution from slurm/ directory
    from gpu_energy_utils import summarize_window, to_float

try:
    import mlflow
except Exception:  # pragma: no cover - optional for local validation
    mlflow = None


def _write_jsonl(path: Path, payload: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(payload) + "\n")


def _normalize_metric_key(name: str) -> str:
    return "".join(ch.lower() for ch in name if ch.isalnum())


def _mean(values: list[float]) -> float | None:
    clean = [v for v in values if v is not None and math.isfinite(v)]
    if not clean:
        return None
    return sum(clean) / len(clean)


def _quantile(values: list[float], q: float) -> float | None:
    clean = sorted(v for v in values if v is not None and math.isfinite(v))
    if not clean:
        return None
    if q <= 0:
        return clean[0]
    if q >= 1:
        return clean[-1]
    pos = (len(clean) - 1) * q
    low = int(math.floor(pos))
    high = int(math.ceil(pos))
    if low == high:
        return clean[low]
    weight = pos - low
    return clean[low] * (1.0 - weight) + clean[high] * weight


def _linear_regression(xs: list[float], ys: list[float]) -> tuple[float, float] | None:
    if len(xs) != len(ys) or len(xs) < 2:
        return None
    mean_x = sum(xs) / len(xs)
    mean_y = sum(ys) / len(ys)
    denom = sum((x - mean_x) ** 2 for x in xs)
    if denom <= 0:
        return None
    slope = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys)) / denom
    intercept = mean_y - slope * mean_x
    return intercept, slope


def _theil_sen_slope(xs: list[float], ys: list[float]) -> float | None:
    """Return a robust trend estimate that is insensitive to single noisy epochs."""
    if len(xs) != len(ys) or len(xs) < 2:
        return None
    slopes = [
        (ys[j] - ys[i]) / (xs[j] - xs[i])
        for i in range(len(xs) - 1)
        for j in range(i + 1, len(xs))
        if xs[j] != xs[i]
    ]
    return statistics.median(slopes) if slopes else None


def _bounded_map_value(value: float, current_best: float) -> float:
    return max(current_best, min(1.0, max(0.0, value)))


def _fit_asymptotic_exponential(epochs: list[int], values: list[float], target_epoch: int, current_best: float) -> float | None:
    max_value = max(values)
    lower_asymptote = max(max_value + 1e-4, current_best + 1e-4)
    upper_asymptote = 1.0
    if lower_asymptote >= upper_asymptote:
        return current_best

    best_pred = None
    best_sse = None
    for idx in range(24):
        asymptote = lower_asymptote + (upper_asymptote - lower_asymptote) * (idx / 23.0)
        transformed_x = []
        transformed_y = []
        for epoch, value in zip(epochs, values):
            residual = asymptote - value
            if residual <= 1e-8:
                break
            transformed_x.append(float(epoch))
            transformed_y.append(math.log(residual))
        else:
            fit = _linear_regression(transformed_x, transformed_y)
            if fit is None:
                continue
            intercept, slope = fit
            if slope >= 0:
                continue
            preds = []
            for epoch in epochs:
                pred = asymptote - math.exp(intercept + slope * float(epoch))
                preds.append(_bounded_map_value(pred, current_best))
            sse = sum((pred - value) ** 2 for pred, value in zip(preds, values))
            pred_target = asymptote - math.exp(intercept + slope * float(target_epoch))
            pred_target = _bounded_map_value(pred_target, current_best)
            if best_sse is None or sse < best_sse:
                best_sse = sse
                best_pred = pred_target
    return best_pred


def _fit_inverse_power(epochs: list[int], values: list[float], target_epoch: int, current_best: float) -> float | None:
    max_value = max(values)
    lower_asymptote = max(max_value + 1e-4, current_best + 1e-4)
    upper_asymptote = 1.0
    if lower_asymptote >= upper_asymptote:
        return current_best

    log_epochs = [math.log(max(float(epoch), 1.0)) for epoch in epochs]
    best_pred = None
    best_sse = None
    for idx in range(24):
        asymptote = lower_asymptote + (upper_asymptote - lower_asymptote) * (idx / 23.0)
        transformed_y = []
        valid = True
        for value in values:
            residual = asymptote - value
            if residual <= 1e-8:
                valid = False
                break
            transformed_y.append(math.log(residual))
        if not valid:
            continue

        fit = _linear_regression(log_epochs, transformed_y)
        if fit is None:
            continue
        intercept, slope = fit
        if slope >= 0:
            continue

        preds = []
        for log_epoch in log_epochs:
            pred = asymptote - math.exp(intercept + slope * log_epoch)
            preds.append(_bounded_map_value(pred, current_best))
        sse = sum((pred - value) ** 2 for pred, value in zip(preds, values))
        pred_target = asymptote - math.exp(intercept + slope * math.log(max(float(target_epoch), 1.0)))
        pred_target = _bounded_map_value(pred_target, current_best)
        if best_sse is None or sse < best_sse:
            best_sse = sse
            best_pred = pred_target
    return best_pred


def _bootstrap_prediction_distribution(
    *,
    epochs: list[int],
    values: list[float],
    target_epoch: int,
    current_best: float,
    bootstrap_samples: int,
    min_fit_points: int,
) -> list[float]:
    if len(epochs) < min_fit_points or len(values) < min_fit_points:
        return []

    preds: list[float] = []
    fitters = (_fit_asymptotic_exponential, _fit_inverse_power)
    for fitter in fitters:
        pred = fitter(epochs, values, target_epoch, current_best)
        if pred is not None and math.isfinite(pred):
            preds.append(_bounded_map_value(pred, current_best))

    seed = len(epochs) * 10007 + int(current_best * 1_000_000)
    rng = random.Random(seed)
    total_points = len(epochs)
    for _ in range(max(bootstrap_samples, 0)):
        sample_indices = sorted(set(rng.randrange(total_points) for _ in range(total_points)))
        if len(sample_indices) < min_fit_points:
            continue
        sample_epochs = [epochs[i] for i in sample_indices]
        sample_values = [values[i] for i in sample_indices]
        for fitter in fitters:
            pred = fitter(sample_epochs, sample_values, target_epoch, current_best)
            if pred is not None and math.isfinite(pred):
                preds.append(_bounded_map_value(pred, current_best))
    return preds


def _clip_metric(value: float) -> float:
    return min(1.0, max(0.0, value))


def _best_so_far(values: list[float]) -> list[float]:
    result: list[float] = []
    best = 0.0
    for value in values:
        best = max(best, _clip_metric(value))
        result.append(best)
    return result


def _fit_asymptotic_exponential_v2(
    epochs: list[int], values: list[float], target_epoch: int, current_best: float
) -> float | None:
    """Fit an asymptotic curve without clamping in-sample predictions to current best."""
    max_value = max(values)
    lower_asymptote = max_value + 1e-4
    if lower_asymptote >= 1.0:
        return current_best
    best_pred = None
    best_sse = None
    for idx in range(32):
        asymptote = lower_asymptote + (1.0 - lower_asymptote) * (idx / 31.0)
        transformed = [(float(epoch), math.log(asymptote - value)) for epoch, value in zip(epochs, values)]
        fit = _linear_regression([item[0] for item in transformed], [item[1] for item in transformed])
        if fit is None or fit[1] >= 0:
            continue
        intercept, slope = fit
        fitted = [_clip_metric(asymptote - math.exp(intercept + slope * float(epoch))) for epoch in epochs]
        sse = sum((pred - value) ** 2 for pred, value in zip(fitted, values))
        target = _bounded_map_value(
            asymptote - math.exp(intercept + slope * float(target_epoch)), current_best
        )
        if best_sse is None or sse < best_sse:
            best_sse = sse
            best_pred = target
    return best_pred


def _fit_inverse_power_v2(
    epochs: list[int], values: list[float], target_epoch: int, current_best: float
) -> float | None:
    """Fit an inverse-power learning curve with an unconstrained in-sample error."""
    max_value = max(values)
    lower_asymptote = max_value + 1e-4
    if lower_asymptote >= 1.0:
        return current_best
    log_epochs = [math.log(max(float(epoch), 1.0)) for epoch in epochs]
    best_pred = None
    best_sse = None
    for idx in range(32):
        asymptote = lower_asymptote + (1.0 - lower_asymptote) * (idx / 31.0)
        residuals = [asymptote - value for value in values]
        if any(value <= 1e-8 for value in residuals):
            continue
        fit = _linear_regression(log_epochs, [math.log(value) for value in residuals])
        if fit is None or fit[1] >= 0:
            continue
        intercept, slope = fit
        fitted = [_clip_metric(asymptote - math.exp(intercept + slope * x)) for x in log_epochs]
        sse = sum((pred - value) ** 2 for pred, value in zip(fitted, values))
        target = _bounded_map_value(
            asymptote - math.exp(intercept + slope * math.log(max(float(target_epoch), 1.0))),
            current_best,
        )
        if best_sse is None or sse < best_sse:
            best_sse = sse
            best_pred = target
    return best_pred


def _ensemble_prediction(
    epochs: list[int], values: list[float], target_epoch: int, current_best: float
) -> float | None:
    predictions = [
        fitter(epochs, values, target_epoch, current_best)
        for fitter in (_fit_asymptotic_exponential_v2, _fit_inverse_power_v2)
    ]
    for window in (5, 10, 20):
        local_epochs = epochs[-window:]
        local_values = values[-window:]
        fit = _linear_regression([float(value) for value in local_epochs], local_values)
        if fit is not None:
            intercept, slope = fit
            local_prediction = intercept + max(0.0, slope) * float(target_epoch)
            predictions.append(_bounded_map_value(local_prediction, current_best))
    predictions = [value for value in predictions if value is not None and math.isfinite(value)]
    # The lower ensemble quartile is the point predictor. Split-conformal
    # calibration then adds the empirically required one-sided safety margin.
    return _quantile(predictions, 0.25) if predictions else None


def bootstrap_prediction_distribution_v2(
    *,
    epochs: list[int],
    values: list[float],
    target_epoch: int,
    current_best: float,
    bootstrap_samples: int,
    min_fit_points: int,
    block_length: int | None = None,
) -> list[float]:
    """Moving-block residual bootstrap over a best-so-far learning curve."""
    if len(epochs) < min_fit_points or len(values) < min_fit_points:
        return []
    values = _best_so_far(values)
    base_prediction = _ensemble_prediction(epochs, values, target_epoch, current_best)
    if base_prediction is None:
        return []
    predictions = [base_prediction]

    # A short monotone smoother provides residuals without refitting both nonlinear
    # models for every historical epoch. This keeps the per-epoch controller cheap.
    fitted_history = []
    for index in range(len(values)):
        start = max(0, index - 2)
        end = min(len(values), index + 3)
        fitted_history.append(statistics.fmean(values[start:end]))
    fitted_history = _best_so_far(fitted_history)
    residuals = [actual - fitted for actual, fitted in zip(values, fitted_history)]
    n = len(epochs)
    block_length = max(2, min(n, block_length or int(round(math.sqrt(n)))))
    rng = random.Random(n * 10007 + int(current_best * 1_000_000) + target_epoch)
    for _ in range(max(bootstrap_samples, 0)):
        sampled_residuals: list[float] = []
        while len(sampled_residuals) < n:
            start = rng.randrange(max(1, n - block_length + 1))
            sampled_residuals.extend(residuals[start : start + block_length])
        sampled_values = _best_so_far(
            [_clip_metric(fitted + residual) for fitted, residual in zip(fitted_history, sampled_residuals[:n])]
        )
        prediction = _ensemble_prediction(epochs, sampled_values, target_epoch, current_best)
        if prediction is not None and math.isfinite(prediction):
            predictions.append(_bounded_map_value(prediction, current_best))
    return predictions


@dataclass
class EnergyAdaptiveConfig:
    enabled: bool = False
    controller_mode: str = "none"
    comparison_strategy: str = "unspecified"
    monitor_metric: str = "map50"
    min_epochs: int = 20
    patience: int = 3
    smoothing_window: int = 3
    min_delta_map50: float = 0.001
    min_mape_map50_per_wh: float = 0.0001
    standard_min_delta: float = 0.0005
    uncertainty_target_epoch: int = 100
    uncertainty_epsilon: float = 0.01
    uncertainty_alpha: float = 0.05
    uncertainty_bootstrap_samples: int = 24
    uncertainty_min_fit_points: int = 8
    uncertainty_min_quality: float = 0.55
    uncertainty_require_low_mape: bool = True
    scenario: str = "unspecified"
    conformal_calibration_path: str = ""
    conformal_regret_tolerance: float = 0.015
    conformal_future_efficiency_threshold: float = 0.05
    conformal_max_interval_width: float = 0.08
    conformal_energy_window: int = 5
    conformal_require_calibration: bool = True
    controller_evaluation_interval: int = 1
    hybrid_quality_target: float = 0.68
    hybrid_plateau_window: int = 12
    hybrid_plateau_slope_threshold: float = 0.0015
    hybrid_threshold_growth: float = 0.25
    hybrid_candidate_decay: float = 0.5
    hybrid_late_epoch_fraction: float = 0.90
    hybrid_late_patience: int = 2
    hybrid_max_epoch_budget: int = 95
    hybrid_max_net_energy_wh: float = 0.0
    hybrid_regret_weight: float = 1.0
    hybrid_energy_weight: float = 0.05
    energy_guard_target_saving_fraction: float = 0.20
    energy_guard_safety_margin_fraction: float = 0.01
    energy_guard_post_training_reserve_wh: float = 0.90
    energy_guard_window: int = 5
    energy_guard_fallback_epoch_fraction: float = 0.77
    energy_guard_strict: bool = True
    controller_plugin: str = ""
    controller_parameters_json: str = "{}"
    controller_id: str = "unspecified"
    benchmark_version: str = "unversioned"
    benchmark_run_id: str = "none"
    benchmark_case_id: str = "none"
    benchmark_stage: str = "none"
    task_type: str = "object_detection"
    quality_metric: str = "map50_95"


class EpochEnergyAdaptiveController:
    def __init__(
        self,
        *,
        gpu_csv_path: str | Path,
        epoch_timeline_path: str | Path,
        pue_factor: float = 1.0,
        price_eur_kwh: float = 0.30,
        co2_kg_kwh: float = 0.4,
        idle_power_w: float = 0.0,
        config: EnergyAdaptiveConfig | None = None,
        time_fn=time.time,
    ):
        self.gpu_csv_path = Path(gpu_csv_path)
        self.epoch_timeline_path = Path(epoch_timeline_path)
        self.pue_factor = pue_factor
        self.price_eur_kwh = price_eur_kwh
        self.co2_kg_kwh = co2_kg_kwh
        self.idle_power_w = max(0.0, float(idle_power_w))
        self.config = config or EnergyAdaptiveConfig()
        self.time_fn = time_fn
        self.current_epoch_start_ts: float | None = None
        self.current_epoch_number: int | None = None
        self.low_gain_streak = 0
        self.controller_evidence = 0.0
        self.history: list[dict] = []
        self.conformal_calibration = self._load_conformal_calibration()
        self.plugin_runtime = None
        self._plugin_closed = False
        if self.config.controller_mode == "plugin":
            if not self.config.controller_plugin:
                raise ValueError("controller_mode=plugin requires controller_plugin")
            from controller_benchmark.runtime import ControllerPluginRuntime

            self.plugin_runtime = ControllerPluginRuntime(
                plugin_path=self.config.controller_plugin,
                parameters_json=self.config.controller_parameters_json,
                controller_id=self.config.controller_id,
                benchmark_version=self.config.benchmark_version,
                task_type=self.config.task_type,
                quality_metric=self.config.quality_metric,
                scenario=self.config.scenario,
                max_epochs=self.config.uncertainty_target_epoch,
                metadata={
                    "benchmark_run_id": self.config.benchmark_run_id,
                    "benchmark_case_id": self.config.benchmark_case_id,
                    "benchmark_stage": self.config.benchmark_stage,
                },
            )

    def on_train_epoch_start(self, trainer):
        epoch = int(getattr(trainer, "epoch", len(self.history))) + 1
        start_ts = float(self.time_fn())
        self.current_epoch_start_ts = start_ts
        self.current_epoch_number = epoch
        _write_jsonl(
            self.epoch_timeline_path,
            {
                "epoch": epoch,
                "event": "start",
                "ts": start_ts,
                "controller_mode": self.config.controller_mode,
                "comparison_strategy": self.config.comparison_strategy,
            },
        )

    def on_fit_epoch_end(self, trainer):
        if self.current_epoch_start_ts is None:
            return
        epoch = self.current_epoch_number if self.current_epoch_number is not None else (len(self.history) + 1)
        if self.history and int(self.history[-1].get("epoch_index", 0)) >= epoch:
            return

        end_ts = float(self.time_fn())
        start_ts = self.current_epoch_start_ts
        epoch_gpu = summarize_window(self.gpu_csv_path, start_ts=start_ts, end_ts=end_ts)
        epoch_total_energy_kwh = to_float(epoch_gpu.get("gpu_energy_kwh")) * self.pue_factor
        epoch_duration_seconds = to_float(epoch_gpu.get("duration_seconds"), max(0.0, end_ts - start_ts))
        idle_energy_kwh = self.idle_power_w * epoch_duration_seconds / 3_600_000.0
        net_gpu_energy_kwh = max(0.0, to_float(epoch_gpu.get("gpu_energy_kwh")) - idle_energy_kwh)
        net_total_energy_kwh = net_gpu_energy_kwh * self.pue_factor
        epoch_metrics = self._extract_epoch_metrics(trainer, fallback_epoch=epoch)

        prev = self.history[-1] if self.history else None
        prev_map50 = to_float(prev.get("map50")) if prev else None
        prev_map50_95 = to_float(prev.get("map50_95")) if prev else None
        delta_map50 = (
            to_float(epoch_metrics.get("map50")) - prev_map50
            if prev_map50 is not None and epoch_metrics.get("map50") is not None
            else None
        )
        delta_map50_95 = (
            to_float(epoch_metrics.get("map50_95")) - prev_map50_95
            if prev_map50_95 is not None and epoch_metrics.get("map50_95") is not None
            else None
        )

        epoch_total_energy_wh = epoch_total_energy_kwh * 1000.0
        net_total_energy_wh = net_total_energy_kwh * 1000.0
        controller_energy_wh = net_total_energy_wh if net_total_energy_wh > 0 else epoch_total_energy_wh
        marginal_map50_per_wh = delta_map50 / controller_energy_wh if delta_map50 is not None and controller_energy_wh > 0 else None
        marginal_map50_95_per_wh = (
            delta_map50_95 / controller_energy_wh if delta_map50_95 is not None and controller_energy_wh > 0 else None
        )
        gross_marginal_map50_per_wh = delta_map50 / epoch_total_energy_wh if delta_map50 is not None and epoch_total_energy_wh > 0 else None
        gross_marginal_map50_95_per_wh = (
            delta_map50_95 / epoch_total_energy_wh if delta_map50_95 is not None and epoch_total_energy_wh > 0 else None
        )

        cumulative_total_energy_kwh = epoch_total_energy_kwh + sum(to_float(item.get("total_energy_kwh")) for item in self.history)
        cumulative_gpu_energy_kwh = to_float(epoch_gpu.get("gpu_energy_kwh")) + sum(
            to_float(item.get("gpu_energy_kwh")) for item in self.history
        )
        cumulative_net_gpu_energy_kwh = net_gpu_energy_kwh + sum(
            to_float(item.get("net_gpu_energy_kwh")) for item in self.history
        )
        cumulative_net_total_energy_kwh = net_total_energy_kwh + sum(
            to_float(item.get("net_total_energy_kwh")) for item in self.history
        )

        lr = self._current_learning_rate(trainer)
        gradient_norm = self._current_gradient_norm(trainer)
        parameter_count = self._model_parameter_count(trainer)
        model_flops = self._model_flops(trainer)
        best_map50 = max([to_float(item.get("best_map50"), 0.0) for item in self.history] + [to_float(epoch_metrics.get("map50"), 0.0)])
        best_map50_95 = max([to_float(item.get("best_map50_95"), 0.0) for item in self.history] + [to_float(epoch_metrics.get("map50_95"), 0.0)])

        event = {
            "epoch": epoch,
            "epoch_index": epoch,
            "event": "end",
            "ts": end_ts,
            "start_ts": start_ts,
            "end_ts": end_ts,
            "duration_seconds": epoch_duration_seconds,
            "samples": int(epoch_gpu.get("samples", 0)),
            "gpu_power_avg_w": to_float(epoch_gpu.get("gpu_power_avg_w")),
            "gpu_power_max_w": to_float(epoch_gpu.get("gpu_power_max_w")),
            "gpu_util_avg_pct": to_float(epoch_gpu.get("gpu_util_avg_pct")),
            "gpu_mem_used_avg_mb": to_float(epoch_gpu.get("gpu_mem_used_avg_mb")),
            "gpu_memory_used_mb": to_float(epoch_gpu.get("gpu_mem_used_avg_mb")),
            "gpu_temp_avg_c": to_float(epoch_gpu.get("gpu_temp_avg_c")),
            "gpu_energy_kwh": to_float(epoch_gpu.get("gpu_energy_kwh")),
            "total_energy_kwh": epoch_total_energy_kwh,
            "idle_power_w": self.idle_power_w,
            "idle_energy_kwh": idle_energy_kwh,
            "net_gpu_energy_kwh": net_gpu_energy_kwh,
            "net_total_energy_kwh": net_total_energy_kwh,
            "estimated_electricity_cost_eur": epoch_total_energy_kwh * self.price_eur_kwh,
            "estimated_co2_kg": epoch_total_energy_kwh * self.co2_kg_kwh,
            "cumulative_gpu_energy_kwh": cumulative_gpu_energy_kwh,
            "cumulative_total_energy_kwh": cumulative_total_energy_kwh,
            "cumulative_net_gpu_energy_kwh": cumulative_net_gpu_energy_kwh,
            "cumulative_net_total_energy_kwh": cumulative_net_total_energy_kwh,
            "map50": epoch_metrics.get("map50"),
            "map50_95": epoch_metrics.get("map50_95"),
            "quality_score": epoch_metrics.get("map50_95"),
            "precision": epoch_metrics.get("precision"),
            "recall": epoch_metrics.get("recall"),
            "val_box_loss": epoch_metrics.get("val_box_loss"),
            "val_cls_loss": epoch_metrics.get("val_cls_loss"),
            "val_dfl_loss": epoch_metrics.get("val_dfl_loss"),
            "learning_rate": lr,
            "gradient_norm": gradient_norm,
            "model_parameter_count": parameter_count,
            "model_flops": model_flops,
            "delta_map50": delta_map50,
            "delta_map50_95": delta_map50_95,
            "marginal_map50_per_wh": marginal_map50_per_wh,
            "marginal_map50_95_per_wh": marginal_map50_95_per_wh,
            "gross_marginal_map50_per_wh": gross_marginal_map50_per_wh,
            "gross_marginal_map50_95_per_wh": gross_marginal_map50_95_per_wh,
            "energy_efficiency_gain": marginal_map50_per_wh,
            "best_map50": best_map50,
            "best_map50_95": best_map50_95,
            "best_quality_score": best_map50_95,
            "adaptive_enabled": int(self.config.enabled),
            "controller_mode": self.config.controller_mode,
            "comparison_strategy": self.config.comparison_strategy,
            "controller_id": self.config.controller_id,
            "benchmark_version": self.config.benchmark_version,
            "benchmark_run_id": self.config.benchmark_run_id,
            "benchmark_case_id": self.config.benchmark_case_id,
            "benchmark_stage": self.config.benchmark_stage,
            "task_type": self.config.task_type,
            "quality_metric": self.config.quality_metric,
        }

        should_stop, stop_reason, diagnostics = self._evaluate_stop(event)
        event.update(diagnostics)
        event["low_gain_streak"] = self.low_gain_streak
        event["should_stop"] = int(should_stop)
        if stop_reason:
            event["stop_reason"] = stop_reason

        self.history.append(event)
        self._log_mlflow_metrics(event)
        _write_jsonl(self.epoch_timeline_path, event)

        if should_stop:
            setattr(trainer, "stop", True)
            self.close()

        self.current_epoch_start_ts = None
        self.current_epoch_number = None

    def on_train_end(self, trainer):
        self.close()

    def close(self):
        if self.plugin_runtime is not None and not self._plugin_closed:
            self.plugin_runtime.close()
            self._plugin_closed = True

    def _evaluate_stop(self, current_event: dict) -> tuple[bool, str | None, dict]:
        if self.config.controller_mode == "metric_early_stopping":
            return self._evaluate_metric_early_stopping(current_event)
        if self.config.controller_mode == "delta_mape":
            return self._evaluate_delta_mape_stop(current_event)
        if self.config.controller_mode == "uncertainty_aware":
            return self._evaluate_uncertainty_stop(current_event)
        if self.config.controller_mode == "conformal_energy_aware":
            return self._evaluate_conformal_energy_stop(current_event)
        if self.config.controller_mode == "hybrid_pareto_energy_aware":
            return self._evaluate_hybrid_pareto_energy_stop(current_event)
        if self.config.controller_mode == "generalized_energy_guard":
            return self._evaluate_generalized_energy_guard_stop(current_event)
        if self.config.controller_mode == "plugin":
            if self.plugin_runtime is None:
                raise RuntimeError("Plugin controller runtime was not initialized.")
            return self.plugin_runtime.evaluate(current_event, self.history)

        self.low_gain_streak = 0
        return False, None, self._default_diagnostics(current_event)

    def _is_controller_evaluation_epoch(self, current_epoch: int) -> bool:
        interval = max(1, int(self.config.controller_evaluation_interval))
        return (current_epoch - int(self.config.min_epochs)) % interval == 0

    def _evaluate_metric_early_stopping(self, current_event: dict) -> tuple[bool, str | None, dict]:
        diagnostics = self._default_diagnostics(current_event)
        metric_key = "map50_95" if self.config.monitor_metric == "map50_95" else "map50"
        current = to_float(current_event.get(metric_key), None)
        previous = [to_float(item.get(metric_key), None) for item in self.history]
        previous = [value for value in previous if value is not None]
        if not self.config.enabled or current is None or len(self.history) + 1 < self.config.min_epochs:
            self.low_gain_streak = 0
            return False, None, diagnostics

        previous_best = max(previous) if previous else None
        improved = previous_best is None or current > previous_best + self.config.standard_min_delta
        self.low_gain_streak = 0 if improved else self.low_gain_streak + 1
        if self.low_gain_streak >= self.config.patience:
            return (
                True,
                f"metric_early_stop[{metric_key}]: min_delta={self.config.standard_min_delta:.6f}",
                diagnostics,
            )
        return False, None, diagnostics

    def _default_diagnostics(self, current_event: dict) -> dict:
        monitor_metric = self.config.monitor_metric
        diagnostics = {
            "adaptive_monitor_metric": monitor_metric,
            "smoothed_delta_map50": None,
            "smoothed_delta_map50_95": None,
            "smoothed_mape_map50_per_wh": None,
            "smoothed_mape_map50_95_per_wh": None,
            "predicted_final_metric_mean": None,
            "predicted_final_metric_lower": None,
            "predicted_final_metric_upper": None,
            "predicted_remaining_gain_upper": None,
            "predicted_prob_gain_gt_epsilon": None,
            "uncertainty_interval_width": None,
            "conformal_correction": None,
            "conformal_remaining_gain_upper": None,
            "predicted_remaining_energy_wh": None,
            "predicted_future_efficiency_per_wh": None,
            "controller_decision_state": 0,
            "controller_decision": "continue",
            "calibration_available": 0,
            "plateau_slope_per_epoch": None,
            "recent_best_gain": None,
            "observed_efficiency_per_wh": None,
            "adaptive_efficiency_threshold": None,
            "pareto_continue_utility": None,
            "controller_evidence": self.controller_evidence,
            "dynamic_patience": self.config.patience,
            "quality_target_reached": 0,
            "budget_triggered": 0,
            "observed_job_energy_wh": None,
            "projected_full_job_energy_wh": None,
            "projected_stopped_job_energy_wh": None,
            "projected_energy_saving_fraction": None,
            "projected_next_energy_saving_fraction": None,
            "energy_guard_target_saving_fraction": None,
            "energy_guard_remaining_budget_wh": None,
            "energy_guard_boundary_reached": 0,
            "energy_guard_fallback_triggered": 0,
            "energy_guard_quality_conflict": 0,
            "controller_plugin_confidence": None,
            "controller_plugin_predicted_energy_saving_fraction": None,
            "controller_plugin_predicted_quality_regret": None,
            "controller_plugin_compute_seconds": None,
        }
        if monitor_metric == "map50_95":
            diagnostics["smoothed_delta_map50_95"] = current_event.get("delta_map50_95")
            diagnostics["smoothed_mape_map50_95_per_wh"] = current_event.get("marginal_map50_95_per_wh")
        else:
            diagnostics["smoothed_delta_map50"] = current_event.get("delta_map50")
            diagnostics["smoothed_mape_map50_per_wh"] = current_event.get("marginal_map50_per_wh")
        return diagnostics

    def _evaluate_delta_mape_stop(self, current_event: dict) -> tuple[bool, str | None, dict]:
        delta_key, mape_key = self._monitor_keys()
        valid_history = self.history + [current_event]
        recent = [
            item
            for item in valid_history
            if item.get(delta_key) is not None and item.get(mape_key) is not None
        ]
        window = self.config.smoothing_window
        smooth_delta = _mean([to_float(item.get(delta_key)) for item in recent[-window:]]) if recent else None
        smooth_mape = _mean([to_float(item.get(mape_key)) for item in recent[-window:]]) if recent else None

        diagnostics = self._default_diagnostics(current_event)
        if self.config.monitor_metric == "map50_95":
            diagnostics["smoothed_delta_map50_95"] = smooth_delta
            diagnostics["smoothed_mape_map50_95_per_wh"] = smooth_mape
        else:
            diagnostics["smoothed_delta_map50"] = smooth_delta
            diagnostics["smoothed_mape_map50_per_wh"] = smooth_mape

        if not self.config.enabled:
            self.low_gain_streak = 0
            return False, None, diagnostics

        if len(valid_history) < self.config.min_epochs or len(recent) < window:
            self.low_gain_streak = 0
            return False, None, diagnostics

        low_delta = smooth_delta is not None and smooth_delta < self.config.min_delta_map50
        low_mape = smooth_mape is not None and smooth_mape < self.config.min_mape_map50_per_wh
        if low_delta and low_mape:
            self.low_gain_streak += 1
        else:
            self.low_gain_streak = 0

        if self.low_gain_streak >= self.config.patience:
            smooth_mape_text = "unavailable" if smooth_mape is None else f"{smooth_mape:.6f}"
            return (
                True,
                f"energy_adaptive_stop[{self.config.monitor_metric}]: avg_delta={smooth_delta:.6f}, avg_mape_per_wh={smooth_mape:.6f}",
                diagnostics,
            )
        return False, None, diagnostics

    def _evaluate_generalized_energy_guard_stop(self, current_event: dict) -> tuple[bool, str | None, dict]:
        """Maximize quality while reserving a configured share of projected job energy.

        The projection uses total GPU energy observed since the job monitor started,
        not only callback-window energy. This captures validation and framework
        overhead that occurs between epoch callbacks. A conservative epoch fallback
        keeps the energy objective enforceable when the online projection is missing
        or unstable. Quality risk remains visible as a diagnostic instead of being
        silently presented as a joint accuracy guarantee.
        """
        diagnostics = self._default_diagnostics(current_event)
        if not self.config.enabled:
            return False, None, diagnostics

        history = self.history + [current_event]
        current_epoch = int(current_event.get("epoch_index", current_event.get("epoch", len(history))))
        target_epoch = max(1, int(self.config.uncertainty_target_epoch))
        if current_epoch < self.config.min_epochs:
            return False, None, diagnostics
        if not self._is_controller_evaluation_epoch(current_epoch):
            diagnostics.update({"controller_decision": "scheduled_wait", "controller_decision_state": 0})
            return False, None, diagnostics

        end_ts = to_float(current_event.get("end_ts"), None)
        observed_job_wh = None
        if end_ts is not None and self.gpu_csv_path.exists():
            observed = summarize_window(self.gpu_csv_path, end_ts=end_ts)
            observed_job_wh = to_float(observed.get("gpu_energy_kwh"), None)
            if observed_job_wh is not None:
                observed_job_wh *= 1000.0 * self.pue_factor

        previous_job_wh = [
            to_float(item.get("observed_job_energy_wh"), None)
            for item in self.history
            if item.get("observed_job_energy_wh") is not None
        ]
        all_job_wh = previous_job_wh + ([observed_job_wh] if observed_job_wh is not None else [])
        increments = [
            current - previous
            for previous, current in zip(all_job_wh, all_job_wh[1:])
            if current > previous and math.isfinite(current - previous)
        ]
        window = max(1, int(self.config.energy_guard_window))
        recent_increments = increments[-window:]
        epoch_energy_wh = statistics.median(recent_increments) if recent_increments else None

        reserve_wh = max(0.0, float(self.config.energy_guard_post_training_reserve_wh))
        projected_full_wh = None
        projected_stopped_wh = None
        saving_fraction = None
        next_saving_fraction = None
        remaining_budget_wh = None
        if observed_job_wh is not None and epoch_energy_wh is not None and epoch_energy_wh > 0:
            remaining_epochs = max(0, target_epoch - current_epoch)
            projected_full_wh = observed_job_wh + epoch_energy_wh * remaining_epochs + reserve_wh
            projected_stopped_wh = observed_job_wh + reserve_wh
            if projected_full_wh > 0:
                saving_fraction = max(0.0, (projected_full_wh - projected_stopped_wh) / projected_full_wh)
                next_stopped_wh = projected_stopped_wh + epoch_energy_wh
                next_saving_fraction = max(0.0, (projected_full_wh - next_stopped_wh) / projected_full_wh)
                allowed_wh = projected_full_wh * (
                    1.0 - max(0.0, min(0.95, self.config.energy_guard_target_saving_fraction))
                )
                remaining_budget_wh = allowed_wh - projected_stopped_wh

        target = max(0.0, min(0.95, float(self.config.energy_guard_target_saving_fraction)))
        margin = max(0.0, float(self.config.energy_guard_safety_margin_fraction))
        boundary_reached = (
            saving_fraction is not None
            and next_saving_fraction is not None
            and saving_fraction >= target
            and next_saving_fraction < target + margin
        )
        fallback_epoch = max(
            int(self.config.min_epochs),
            int(math.floor(target_epoch * max(0.0, min(1.0, self.config.energy_guard_fallback_epoch_fraction)))),
        )
        fallback_triggered = current_epoch >= fallback_epoch

        metric_key = "map50_95" if self.config.monitor_metric == "map50_95" else "map50"
        best_key = "best_map50_95" if self.config.monitor_metric == "map50_95" else "best_map50"
        usable = [item for item in history if item.get(metric_key) is not None]
        current_best = to_float(current_event.get(best_key), None)
        if current_best is None and usable:
            current_best = max(to_float(item.get(metric_key)) for item in usable)
        gain_upper = None
        interval_width = None
        if current_best is not None and len(usable) >= self.config.uncertainty_min_fit_points:
            epochs = [int(item.get("epoch_index", item.get("epoch", 0))) for item in usable]
            values = _best_so_far([to_float(item.get(metric_key)) for item in usable])
            predictions = bootstrap_prediction_distribution_v2(
                epochs=epochs,
                values=values,
                target_epoch=target_epoch,
                current_best=current_best,
                bootstrap_samples=self.config.uncertainty_bootstrap_samples,
                min_fit_points=self.config.uncertainty_min_fit_points,
            )
            if predictions:
                lower = _quantile(predictions, self.config.uncertainty_alpha)
                upper = _quantile(predictions, 1.0 - self.config.uncertainty_alpha)
                correction = self._conformal_correction(current_epoch) or 0.0
                gain_upper = min(1.0 - current_best, max(0.0, statistics.median(predictions) - current_best) + correction)
                interval_width = max(0.0, upper - lower) if lower is not None and upper is not None else None
                diagnostics.update(
                    {
                        "predicted_final_metric_mean": statistics.fmean(predictions),
                        "predicted_final_metric_lower": lower,
                        "predicted_final_metric_upper": upper,
                        "conformal_remaining_gain_upper": gain_upper,
                        "uncertainty_interval_width": interval_width,
                    }
                )

        quality_conflict = (
            gain_upper is not None and gain_upper > self.config.conformal_regret_tolerance
        ) or (
            current_best is not None
            and self.config.uncertainty_min_quality > 0
            and current_best < self.config.uncertainty_min_quality
        )
        energy_stop = boundary_reached or fallback_triggered
        should_stop = energy_stop and (self.config.energy_guard_strict or not quality_conflict)
        diagnostics.update(
            {
                "observed_job_energy_wh": observed_job_wh,
                "projected_full_job_energy_wh": projected_full_wh,
                "projected_stopped_job_energy_wh": projected_stopped_wh,
                "projected_energy_saving_fraction": saving_fraction,
                "projected_next_energy_saving_fraction": next_saving_fraction,
                "energy_guard_target_saving_fraction": target,
                "energy_guard_remaining_budget_wh": remaining_budget_wh,
                "energy_guard_boundary_reached": int(boundary_reached),
                "energy_guard_fallback_triggered": int(fallback_triggered),
                "energy_guard_quality_conflict": int(quality_conflict),
                "quality_target_reached": int(not quality_conflict),
                "budget_triggered": int(energy_stop),
                "controller_decision": "stop" if should_stop else ("quality_conflict" if energy_stop else "continue"),
                "controller_decision_state": 2 if should_stop else (1 if energy_stop else 0),
            }
        )
        if should_stop:
            trigger = "projection" if boundary_reached else "fallback"
            saving_text = float("nan") if saving_fraction is None else saving_fraction * 100.0
            return (
                True,
                (
                    f"generalized_energy_guard_stop[{self.config.monitor_metric}]: trigger={trigger}, "
                    f"epoch={current_epoch}, projected_saving_pct={saving_text:.3f}, "
                    f"target_saving_pct={target * 100.0:.3f}, quality_conflict={int(quality_conflict)}"
                ),
                diagnostics,
            )
        return False, None, diagnostics

    def _load_conformal_calibration(self) -> dict:
        path_value = str(self.config.conformal_calibration_path or "").strip()
        if not path_value:
            return {}
        path = Path(path_value)
        if not path.exists():
            return {}
        try:
            with path.open("r", encoding="utf-8") as handle:
                payload = json.load(handle)
            return payload if isinstance(payload, dict) else {}
        except (OSError, ValueError, TypeError):
            return {}

    def _conformal_correction(self, epoch: int) -> float | None:
        scenarios = self.conformal_calibration.get("scenarios", {})
        scenario = scenarios.get(self.config.scenario, {}) if isinstance(scenarios, dict) else {}
        bins = scenario.get("bins", []) if isinstance(scenario, dict) else []
        for item in bins:
            if int(item.get("start_epoch", 0)) <= epoch <= int(item.get("end_epoch", -1)):
                value = to_float(item.get("correction"), None)
                if value is not None and math.isfinite(value):
                    return max(0.0, value)
        fallback = to_float(scenario.get("global_correction"), None) if isinstance(scenario, dict) else None
        return max(0.0, fallback) if fallback is not None and math.isfinite(fallback) else None

    def _predict_remaining_energy_wh(self, history: list[dict], current_epoch: int) -> float | None:
        recent = history[-max(1, self.config.conformal_energy_window) :]
        energy_wh = []
        for item in recent:
            net = to_float(item.get("net_total_energy_kwh"), None)
            gross = to_float(item.get("total_energy_kwh"), None)
            value = net if net is not None and net > 0 else gross
            if value is not None and value > 0 and math.isfinite(value):
                energy_wh.append(value * 1000.0)
        if not energy_wh:
            return None
        remaining_epochs = max(0, self.config.uncertainty_target_epoch - current_epoch)
        return statistics.median(energy_wh) * remaining_epochs

    def _evaluate_conformal_energy_stop(self, current_event: dict) -> tuple[bool, str | None, dict]:
        diagnostics = self._default_diagnostics(current_event)
        history = self.history + [current_event]
        if not self.config.enabled:
            self.low_gain_streak = 0
            return False, None, diagnostics

        metric_key = "map50_95" if self.config.monitor_metric == "map50_95" else "map50"
        best_key = "best_map50_95" if self.config.monitor_metric == "map50_95" else "best_map50"
        usable = [item for item in history if item.get(metric_key) is not None]
        current_epoch = int(current_event.get("epoch_index", current_event.get("epoch", len(history))))
        current_best = to_float(current_event.get(best_key), None)
        if current_best is None and usable:
            current_best = max(to_float(item.get(metric_key)) for item in usable)
        if current_epoch < self.config.min_epochs:
            self.low_gain_streak = 0
            return False, None, diagnostics
        if current_best is None or len(usable) < self.config.uncertainty_min_fit_points:
            self.low_gain_streak = 0
            return False, None, diagnostics
        if not self._is_controller_evaluation_epoch(current_epoch):
            diagnostics.update({"controller_decision": "scheduled_wait", "controller_decision_state": 0})
            return False, None, diagnostics

        epochs = [int(item.get("epoch_index", item.get("epoch", 0))) for item in usable]
        values = _best_so_far([to_float(item.get(metric_key)) for item in usable])
        predictions = bootstrap_prediction_distribution_v2(
            epochs=epochs,
            values=values,
            target_epoch=self.config.uncertainty_target_epoch,
            current_best=current_best,
            bootstrap_samples=self.config.uncertainty_bootstrap_samples,
            min_fit_points=self.config.uncertainty_min_fit_points,
        )
        if not predictions:
            self.low_gain_streak = 0
            diagnostics.update({"controller_decision": "observe", "controller_decision_state": 1})
            return False, None, diagnostics

        pred_mean = statistics.fmean(predictions)
        pred_lower = _quantile(predictions, self.config.uncertainty_alpha)
        pred_upper = _quantile(predictions, 1.0 - self.config.uncertainty_alpha)
        if pred_lower is None or pred_upper is None:
            self.low_gain_streak = 0
            return False, None, diagnostics

        # The split-conformal correction is applied to the ensemble point forecast.
        # Bootstrap quantiles remain an uncertainty-width diagnostic.
        raw_gain_upper = max(0.0, statistics.median(predictions) - current_best)
        correction = self._conformal_correction(current_epoch)
        calibration_available = correction is not None
        correction = correction or 0.0
        conformal_gain_upper = min(1.0 - current_best, raw_gain_upper + correction)
        interval_width = max(0.0, pred_upper - pred_lower)
        remaining_energy_wh = self._predict_remaining_energy_wh(history, current_epoch)
        future_efficiency = (
            conformal_gain_upper / remaining_energy_wh
            if remaining_energy_wh is not None and remaining_energy_wh > 0
            else None
        )
        diagnostics.update(
            {
                "predicted_final_metric_mean": pred_mean,
                "predicted_final_metric_lower": pred_lower,
                "predicted_final_metric_upper": pred_upper,
                "predicted_remaining_gain_upper": raw_gain_upper,
                "predicted_prob_gain_gt_epsilon": sum(
                    1 for value in predictions if value - current_best > self.config.conformal_regret_tolerance
                )
                / len(predictions),
                "uncertainty_interval_width": interval_width,
                "conformal_correction": correction,
                "conformal_remaining_gain_upper": conformal_gain_upper,
                "predicted_remaining_energy_wh": remaining_energy_wh,
                "predicted_future_efficiency_per_wh": future_efficiency,
                "calibration_available": int(calibration_available),
            }
        )

        minimum_epochs_reached = current_epoch >= self.config.min_epochs
        minimum_quality_reached = current_best >= self.config.uncertainty_min_quality
        regret_is_bounded = conformal_gain_upper <= self.config.conformal_regret_tolerance
        uncertainty_is_bounded = interval_width <= self.config.conformal_max_interval_width
        efficiency_is_low = (
            future_efficiency is not None
            and future_efficiency <= self.config.conformal_future_efficiency_threshold
        )
        calibration_is_valid = calibration_available or not self.config.conformal_require_calibration

        prerequisites = minimum_epochs_reached and minimum_quality_reached and calibration_is_valid
        candidate = prerequisites and regret_is_bounded and uncertainty_is_bounded and efficiency_is_low
        if candidate:
            self.low_gain_streak += 1
            diagnostics.update({"controller_decision": "observe", "controller_decision_state": 1})
        else:
            self.low_gain_streak = 0
            decision = "observe" if prerequisites and (regret_is_bounded or uncertainty_is_bounded) else "continue"
            diagnostics.update(
                {"controller_decision": decision, "controller_decision_state": 1 if decision == "observe" else 0}
            )

        if self.low_gain_streak >= self.config.patience:
            diagnostics.update({"controller_decision": "stop", "controller_decision_state": 2})
            return (
                True,
                (
                    f"conformal_energy_stop[{self.config.monitor_metric}]: "
                    f"gain_upper={conformal_gain_upper:.6f}, "
                    f"regret_tolerance={self.config.conformal_regret_tolerance:.6f}, "
                    f"future_efficiency={future_efficiency:.6f}, "
                    f"efficiency_threshold={self.config.conformal_future_efficiency_threshold:.6f}, "
                    f"interval_width={interval_width:.6f}, quality={current_best:.6f}"
                ),
                diagnostics,
            )
        return False, None, diagnostics

    def _evaluate_hybrid_pareto_energy_stop(self, current_event: dict) -> tuple[bool, str | None, dict]:
        """Conservative two-stage controller with robust plateau and Pareto evidence.

        Stage one protects model quality. Stage two accumulates evidence instead of
        resetting after a single noisy evaluation. A hard budget is only allowed
        after the quality target has been reached.
        """
        diagnostics = self._default_diagnostics(current_event)
        history = self.history + [current_event]
        if not self.config.enabled:
            self.low_gain_streak = 0
            self.controller_evidence = 0.0
            return False, None, diagnostics

        metric_key = "map50_95" if self.config.monitor_metric == "map50_95" else "map50"
        best_key = "best_map50_95" if self.config.monitor_metric == "map50_95" else "best_map50"
        usable = [item for item in history if item.get(metric_key) is not None]
        current_epoch = int(current_event.get("epoch_index", current_event.get("epoch", len(history))))
        current_best = to_float(current_event.get(best_key), None)
        if current_best is None and usable:
            current_best = max(to_float(item.get(metric_key)) for item in usable)

        quality_target = max(self.config.uncertainty_min_quality, self.config.hybrid_quality_target)
        quality_reached = current_best is not None and current_best >= quality_target
        diagnostics["quality_target_reached"] = int(quality_reached)
        if current_epoch < self.config.min_epochs or current_best is None:
            self.low_gain_streak = 0
            self.controller_evidence = 0.0
            return False, None, diagnostics
        if len(usable) < self.config.uncertainty_min_fit_points:
            return False, None, diagnostics
        if not self._is_controller_evaluation_epoch(current_epoch):
            diagnostics.update(
                {
                    "controller_decision": "scheduled_wait",
                    "controller_decision_state": 0,
                    "controller_evidence": self.controller_evidence,
                }
            )
            return False, None, diagnostics

        epochs = [int(item.get("epoch_index", item.get("epoch", 0))) for item in usable]
        values = _best_so_far([to_float(item.get(metric_key)) for item in usable])
        predictions = bootstrap_prediction_distribution_v2(
            epochs=epochs,
            values=values,
            target_epoch=self.config.uncertainty_target_epoch,
            current_best=current_best,
            bootstrap_samples=self.config.uncertainty_bootstrap_samples,
            min_fit_points=self.config.uncertainty_min_fit_points,
        )
        if not predictions:
            diagnostics.update({"controller_decision": "observe", "controller_decision_state": 1})
            return False, None, diagnostics

        pred_mean = statistics.fmean(predictions)
        pred_lower = _quantile(predictions, self.config.uncertainty_alpha)
        pred_upper = _quantile(predictions, 1.0 - self.config.uncertainty_alpha)
        if pred_lower is None or pred_upper is None:
            return False, None, diagnostics

        raw_gain_upper = max(0.0, statistics.median(predictions) - current_best)
        correction = self._conformal_correction(current_epoch)
        calibration_available = correction is not None
        correction = correction or 0.0
        conformal_gain_upper = min(1.0 - current_best, raw_gain_upper + correction)
        interval_width = max(0.0, pred_upper - pred_lower)
        remaining_energy_wh = self._predict_remaining_energy_wh(history, current_epoch)
        future_efficiency = (
            conformal_gain_upper / remaining_energy_wh
            if remaining_energy_wh is not None and remaining_energy_wh > 0
            else None
        )

        window = max(3, min(self.config.hybrid_plateau_window, len(usable)))
        recent_epochs = epochs[-window:]
        recent_values = values[-window:]
        plateau_slope = _theil_sen_slope([float(value) for value in recent_epochs], recent_values)
        recent_gain = max(0.0, recent_values[-1] - recent_values[0])
        recent_energy_wh = 0.0
        for item in usable[-window:]:
            net = to_float(item.get("net_total_energy_kwh"), None)
            gross = to_float(item.get("total_energy_kwh"), None)
            energy = net if net is not None and net > 0 else gross
            if energy is not None and energy > 0:
                recent_energy_wh += energy * 1000.0
        observed_efficiency = recent_gain / recent_energy_wh if recent_energy_wh > 0 else None

        target_epoch = max(1, self.config.uncertainty_target_epoch)
        progress = min(1.0, current_epoch / target_epoch)
        adaptive_efficiency_threshold = self.config.conformal_future_efficiency_threshold * (
            1.0 + self.config.hybrid_threshold_growth * progress
        )
        adaptive_slope_threshold = self.config.hybrid_plateau_slope_threshold * (0.75 + 0.5 * progress)
        plateau_confirmed = (
            plateau_slope is not None
            and plateau_slope <= adaptive_slope_threshold
            and recent_gain <= max(self.config.conformal_regret_tolerance, adaptive_slope_threshold * window)
        )
        efficiency_is_low = (
            future_efficiency is not None and future_efficiency <= adaptive_efficiency_threshold
        ) or (
            observed_efficiency is not None and observed_efficiency <= adaptive_efficiency_threshold
        )
        uncertainty_is_bounded = interval_width <= self.config.conformal_max_interval_width
        regret_is_bounded = conformal_gain_upper <= self.config.conformal_regret_tolerance
        calibration_is_valid = calibration_available or not self.config.conformal_require_calibration
        pareto_continue_utility = None
        if remaining_energy_wh is not None:
            pareto_continue_utility = (
                self.config.hybrid_regret_weight * conformal_gain_upper
                - self.config.hybrid_energy_weight * remaining_energy_wh
            )
        pareto_dominated = pareto_continue_utility is not None and pareto_continue_utility <= 0.0

        cumulative_net_wh = to_float(current_event.get("cumulative_net_total_energy_kwh"), 0.0) * 1000.0
        epoch_budget_hit = self.config.hybrid_max_epoch_budget > 0 and current_epoch >= self.config.hybrid_max_epoch_budget
        energy_budget_hit = (
            self.config.hybrid_max_net_energy_wh > 0
            and cumulative_net_wh >= self.config.hybrid_max_net_energy_wh
        )
        budget_triggered = quality_reached and (epoch_budget_hit or energy_budget_hit)

        # At least three independent signals must agree. The calibrated regret
        # bound is a strong signal, while a late robust plateau can compensate
        # for a slightly conservative conformal upper bound.
        evidence_signals = sum(
            int(value)
            for value in (
                plateau_confirmed,
                efficiency_is_low,
                uncertainty_is_bounded,
                regret_is_bounded or pareto_dominated,
            )
        )
        candidate = (
            quality_reached
            and calibration_is_valid
            and plateau_confirmed
            and efficiency_is_low
            and uncertainty_is_bounded
            and evidence_signals >= 3
        )
        if candidate:
            self.controller_evidence += 1.0
        else:
            self.controller_evidence = max(
                0.0, self.controller_evidence - max(0.0, self.config.hybrid_candidate_decay)
            )

        dynamic_patience = self.config.patience
        if progress >= self.config.hybrid_late_epoch_fraction:
            dynamic_patience = min(dynamic_patience, max(1, self.config.hybrid_late_patience))
        self.low_gain_streak = int(math.floor(self.controller_evidence))

        diagnostics.update(
            {
                "predicted_final_metric_mean": pred_mean,
                "predicted_final_metric_lower": pred_lower,
                "predicted_final_metric_upper": pred_upper,
                "predicted_remaining_gain_upper": raw_gain_upper,
                "predicted_prob_gain_gt_epsilon": sum(
                    1 for value in predictions if value - current_best > self.config.conformal_regret_tolerance
                )
                / len(predictions),
                "uncertainty_interval_width": interval_width,
                "conformal_correction": correction,
                "conformal_remaining_gain_upper": conformal_gain_upper,
                "predicted_remaining_energy_wh": remaining_energy_wh,
                "predicted_future_efficiency_per_wh": future_efficiency,
                "calibration_available": int(calibration_available),
                "plateau_slope_per_epoch": plateau_slope,
                "recent_best_gain": recent_gain,
                "observed_efficiency_per_wh": observed_efficiency,
                "adaptive_efficiency_threshold": adaptive_efficiency_threshold,
                "pareto_continue_utility": pareto_continue_utility,
                "controller_evidence": self.controller_evidence,
                "dynamic_patience": dynamic_patience,
                "quality_target_reached": int(quality_reached),
                "budget_triggered": int(budget_triggered),
                "controller_decision": "candidate_stop" if candidate else ("observe" if quality_reached else "continue"),
                "controller_decision_state": 1 if quality_reached else 0,
            }
        )

        if budget_triggered or self.controller_evidence >= dynamic_patience:
            diagnostics.update({"controller_decision": "stop", "controller_decision_state": 2})
            reason_prefix = "hybrid_budget_stop" if budget_triggered else "hybrid_pareto_stop"
            return (
                True,
                (
                    f"{reason_prefix}[{self.config.monitor_metric}]: quality={current_best:.6f}, "
                    f"plateau_slope={plateau_slope:.6f}, gain_upper={conformal_gain_upper:.6f}, "
                    f"future_efficiency={future_efficiency if future_efficiency is not None else float('nan'):.6f}, "
                    f"evidence={self.controller_evidence:.2f}, patience={dynamic_patience}"
                ),
                diagnostics,
            )
        return False, None, diagnostics

    def _evaluate_uncertainty_stop(self, current_event: dict) -> tuple[bool, str | None, dict]:
        diagnostics = self._default_diagnostics(current_event)
        history = self.history + [current_event]
        if not self.config.enabled:
            self.low_gain_streak = 0
            return False, None, diagnostics

        if len(history) < self.config.min_epochs:
            self.low_gain_streak = 0
            return False, None, diagnostics

        current_epoch = int(current_event.get("epoch_index", current_event.get("epoch", len(history))))
        if not self._is_controller_evaluation_epoch(current_epoch):
            diagnostics.update({"controller_decision": "scheduled_wait", "controller_decision_state": 0})
            return False, None, diagnostics

        metric_key = "map50_95" if self.config.monitor_metric == "map50_95" else "map50"
        best_key = "best_map50_95" if self.config.monitor_metric == "map50_95" else "best_map50"
        epochs = [int(item.get("epoch_index", item.get("epoch", 0))) for item in history if item.get(metric_key) is not None]
        values = [to_float(item.get(metric_key)) for item in history if item.get(metric_key) is not None]
        current_best = to_float(current_event.get(best_key))
        _, mape_key = self._monitor_keys()
        recent_mape = [to_float(item.get(mape_key), None) for item in history[-self.config.smoothing_window :]]
        recent_mape = [value for value in recent_mape if value is not None]
        smooth_mape = _mean(recent_mape)
        if self.config.monitor_metric == "map50_95":
            diagnostics["smoothed_mape_map50_95_per_wh"] = smooth_mape
        else:
            diagnostics["smoothed_mape_map50_per_wh"] = smooth_mape
        predictions = _bootstrap_prediction_distribution(
            epochs=epochs,
            values=values,
            target_epoch=self.config.uncertainty_target_epoch,
            current_best=current_best,
            bootstrap_samples=self.config.uncertainty_bootstrap_samples,
            min_fit_points=self.config.uncertainty_min_fit_points,
        )
        if not predictions:
            self.low_gain_streak = 0
            return False, None, diagnostics

        pred_mean = statistics.fmean(predictions)
        pred_lower = _quantile(predictions, self.config.uncertainty_alpha)
        pred_upper = _quantile(predictions, 1.0 - self.config.uncertainty_alpha)
        if pred_lower is None or pred_upper is None:
            self.low_gain_streak = 0
            return False, None, diagnostics

        gain_upper = max(0.0, pred_upper - current_best)
        gain_risk = sum(1 for value in predictions if (value - current_best) > self.config.uncertainty_epsilon) / max(len(predictions), 1)
        diagnostics.update(
            {
                "predicted_final_metric_mean": pred_mean,
                "predicted_final_metric_lower": pred_lower,
                "predicted_final_metric_upper": pred_upper,
                "predicted_remaining_gain_upper": gain_upper,
                "predicted_prob_gain_gt_epsilon": gain_risk,
                "uncertainty_interval_width": max(0.0, pred_upper - pred_lower),
            }
        )

        minimum_quality_reached = current_best >= self.config.uncertainty_min_quality
        low_mape = smooth_mape is not None and smooth_mape < self.config.min_mape_map50_per_wh
        smooth_mape_text = "unavailable" if smooth_mape is None else f"{smooth_mape:.6f}"
        low_gain = (
            gain_upper < self.config.uncertainty_epsilon
            and gain_risk < self.config.uncertainty_alpha
            and minimum_quality_reached
            and (low_mape or not self.config.uncertainty_require_low_mape)
        )
        if low_gain:
            self.low_gain_streak += 1
        else:
            self.low_gain_streak = 0

        if self.low_gain_streak >= self.config.patience:
            return (
                True,
                (
                    f"uncertainty_stop[{self.config.monitor_metric}]: gain_upper={gain_upper:.6f}, "
                    f"prob_gain_gt_epsilon={gain_risk:.6f}, epsilon={self.config.uncertainty_epsilon:.6f}, "
                    f"alpha={self.config.uncertainty_alpha:.6f}, quality={current_best:.6f}, "
                    f"smoothed_mape={smooth_mape_text}"
                ),
                diagnostics,
            )
        return False, None, diagnostics

    def _monitor_keys(self) -> tuple[str, str]:
        if self.config.monitor_metric == "map50_95":
            return "delta_map50_95", "marginal_map50_95_per_wh"
        return "delta_map50", "marginal_map50_per_wh"

    def _extract_epoch_metrics(self, trainer, fallback_epoch: int) -> dict:
        csv_metrics = self._read_results_csv(trainer)
        if csv_metrics:
            return csv_metrics

        trainer_metrics = getattr(trainer, "metrics", None)
        normalized = {}
        if isinstance(trainer_metrics, dict):
            normalized = {_normalize_metric_key(str(k)): to_float(v, None) for k, v in trainer_metrics.items()}

        return {
            "epoch": fallback_epoch,
            "map50": self._first_metric(normalized, ["metricsmap50b", "metricsmap50"]),
            "map50_95": self._first_metric(normalized, ["metricsmap5095b", "metricsmap5095"]),
            "precision": self._first_metric(normalized, ["metricsprecisionb", "metricsprecision"]),
            "recall": self._first_metric(normalized, ["metricsrecallb", "metricsrecall"]),
            "val_box_loss": self._first_metric(normalized, ["valboxloss", "valbox_loss"]),
            "val_cls_loss": self._first_metric(normalized, ["valclsloss", "valcls_loss"]),
            "val_dfl_loss": self._first_metric(normalized, ["valdfl_loss", "valdfloss"]),
        }

    def _read_results_csv(self, trainer) -> dict | None:
        csv_path = self._resolve_results_csv(trainer)
        if not csv_path or not csv_path.exists():
            return None

        with csv_path.open("r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            rows = list(reader)
        if not rows:
            return None

        row = rows[-1]
        normalized = {_normalize_metric_key(str(k)): v for k, v in row.items()}
        return {
            "epoch": len(rows),
            "map50": self._first_metric(normalized, ["metricsmap50b", "metricsmap50"]),
            "map50_95": self._first_metric(normalized, ["metricsmap5095b", "metricsmap5095"]),
            "precision": self._first_metric(normalized, ["metricsprecisionb", "metricsprecision"]),
            "recall": self._first_metric(normalized, ["metricsrecallb", "metricsrecall"]),
            "val_box_loss": self._first_metric(normalized, ["valboxloss", "valbox_loss"]),
            "val_cls_loss": self._first_metric(normalized, ["valclsloss", "valcls_loss"]),
            "val_dfl_loss": self._first_metric(normalized, ["valdfl_loss", "valdfloss"]),
        }

    def _resolve_results_csv(self, trainer) -> Path | None:
        trainer_csv = getattr(trainer, "csv", None)
        if trainer_csv:
            return Path(trainer_csv)
        save_dir = getattr(trainer, "save_dir", None)
        if save_dir:
            return Path(save_dir) / "results.csv"
        return None

    def _current_learning_rate(self, trainer) -> float | None:
        optimizer = getattr(trainer, "optimizer", None)
        if optimizer is None:
            return None
        param_groups = getattr(optimizer, "param_groups", None) or []
        if not param_groups:
            return None
        return to_float(param_groups[0].get("lr"), None)

    def _current_gradient_norm(self, trainer) -> float | None:
        model = getattr(trainer, "model", None)
        parameters = getattr(model, "parameters", None)
        if parameters is None:
            return None
        total = 0.0
        count = 0
        try:
            for parameter in parameters():
                grad = getattr(parameter, "grad", None)
                if grad is None:
                    continue
                norm = float(grad.detach().float().norm(2).cpu())
                total += norm * norm
                count += 1
        except Exception:
            return None
        return math.sqrt(total) if count else None

    def _model_parameter_count(self, trainer) -> int | None:
        model = getattr(trainer, "model", None)
        parameters = getattr(model, "parameters", None)
        if parameters is None:
            return None
        try:
            return int(sum(parameter.numel() for parameter in parameters()))
        except Exception:
            return None

    def _model_flops(self, trainer) -> float | None:
        model = getattr(trainer, "model", None)
        if model is None:
            return None
        try:
            from ultralytics.utils.torch_utils import get_flops

            value = to_float(get_flops(model), None)
            if value is None:
                return None
            return value * 1e9 if value < 1e6 else value
        except Exception:
            return None

    @staticmethod
    def _first_metric(values: dict, keys: list[str]) -> float | None:
        for key in keys:
            if key in values and values[key] not in ("", None):
                return to_float(values[key], None)
        return None

    def _log_mlflow_metrics(self, event: dict):
        if mlflow is None:
            return
        try:
            if mlflow.active_run() is None:
                return
        except Exception:
            return

        step = int(event.get("epoch_index", event.get("epoch", 0)))
        metrics = {
            "epoch/slurm_gpu_energy_wh": to_float(event.get("gpu_energy_kwh")) * 1000.0,
            "epoch/slurm_total_energy_wh": to_float(event.get("total_energy_kwh")) * 1000.0,
            "epoch/slurm_duration_seconds": to_float(event.get("duration_seconds")),
            "epoch/slurm_gpu_util_avg_pct": to_float(event.get("gpu_util_avg_pct")),
            "epoch/slurm_cumulative_total_energy_wh": to_float(event.get("cumulative_total_energy_kwh")) * 1000.0,
            "epoch/energy_adaptive_should_stop": to_float(event.get("should_stop")),
            "epoch/energy_adaptive_low_gain_streak": to_float(event.get("low_gain_streak")),
            "epoch/controller_decision_state": to_float(event.get("controller_decision_state")),
            "epoch/calibration_available": to_float(event.get("calibration_available")),
            "epoch/plateau_slope_per_epoch": to_float(event.get("plateau_slope_per_epoch")),
            "epoch/recent_best_gain": to_float(event.get("recent_best_gain")),
            "epoch/observed_efficiency_per_wh": to_float(event.get("observed_efficiency_per_wh")),
            "epoch/adaptive_efficiency_threshold": to_float(event.get("adaptive_efficiency_threshold")),
            "epoch/pareto_continue_utility": to_float(event.get("pareto_continue_utility")),
            "epoch/controller_evidence": to_float(event.get("controller_evidence")),
            "epoch/dynamic_patience": to_float(event.get("dynamic_patience")),
            "epoch/quality_target_reached": to_float(event.get("quality_target_reached")),
            "epoch/budget_triggered": to_float(event.get("budget_triggered")),
            "epoch/controller_plugin_compute_seconds": to_float(event.get("controller_plugin_compute_seconds")),
            "epoch/energy_guard_boundary_reached": to_float(event.get("energy_guard_boundary_reached")),
            "epoch/energy_guard_fallback_triggered": to_float(event.get("energy_guard_fallback_triggered")),
            "epoch/energy_guard_quality_conflict": to_float(event.get("energy_guard_quality_conflict")),
        }
        optional_metrics = {
            "epoch/map50": event.get("map50"),
            "epoch/map50_95": event.get("map50_95"),
            "epoch/best_map50": event.get("best_map50"),
            "epoch/best_map50_95": event.get("best_map50_95"),
            "epoch/precision": event.get("precision"),
            "epoch/recall": event.get("recall"),
            "epoch/learning_rate": event.get("learning_rate"),
            "epoch/gradient_norm": event.get("gradient_norm"),
            "epoch/model_parameter_count": event.get("model_parameter_count"),
            "epoch/model_flops": event.get("model_flops"),
            "epoch/gpu_power_avg_w": event.get("gpu_power_avg_w"),
            "epoch/gpu_power_max_w": event.get("gpu_power_max_w"),
            "epoch/gpu_mem_used_avg_mb": event.get("gpu_mem_used_avg_mb"),
            "epoch/delta_map50": event.get("delta_map50"),
            "epoch/delta_map50_95": event.get("delta_map50_95"),
            "epoch/mape_map50_per_wh": event.get("marginal_map50_per_wh"),
            "epoch/mape_map50_95_per_wh": event.get("marginal_map50_95_per_wh"),
            "epoch/smoothed_delta_map50": event.get("smoothed_delta_map50"),
            "epoch/smoothed_mape_map50_per_wh": event.get("smoothed_mape_map50_per_wh"),
            "epoch/smoothed_delta_map50_95": event.get("smoothed_delta_map50_95"),
            "epoch/smoothed_mape_map50_95_per_wh": event.get("smoothed_mape_map50_95_per_wh"),
            "epoch/predicted_final_metric_mean": event.get("predicted_final_metric_mean"),
            "epoch/predicted_final_metric_lower": event.get("predicted_final_metric_lower"),
            "epoch/predicted_final_metric_upper": event.get("predicted_final_metric_upper"),
            "epoch/predicted_remaining_gain_upper": event.get("predicted_remaining_gain_upper"),
            "epoch/predicted_prob_gain_gt_epsilon": event.get("predicted_prob_gain_gt_epsilon"),
            "epoch/uncertainty_interval_width": event.get("uncertainty_interval_width"),
            "epoch/conformal_correction": event.get("conformal_correction"),
            "epoch/conformal_remaining_gain_upper": event.get("conformal_remaining_gain_upper"),
            "epoch/predicted_remaining_energy_wh": event.get("predicted_remaining_energy_wh"),
            "epoch/predicted_future_efficiency_per_wh": event.get("predicted_future_efficiency_per_wh"),
            "epoch/observed_job_energy_wh": event.get("observed_job_energy_wh"),
            "epoch/projected_full_job_energy_wh": event.get("projected_full_job_energy_wh"),
            "epoch/projected_stopped_job_energy_wh": event.get("projected_stopped_job_energy_wh"),
            "epoch/projected_energy_saving_fraction": event.get("projected_energy_saving_fraction"),
            "epoch/projected_next_energy_saving_fraction": event.get("projected_next_energy_saving_fraction"),
            "epoch/energy_guard_target_saving_fraction": event.get("energy_guard_target_saving_fraction"),
            "epoch/energy_guard_remaining_budget_wh": event.get("energy_guard_remaining_budget_wh"),
            "epoch/controller_plugin_confidence": event.get("controller_plugin_confidence"),
            "epoch/controller_plugin_predicted_energy_saving_fraction": event.get(
                "controller_plugin_predicted_energy_saving_fraction"
            ),
            "epoch/controller_plugin_predicted_quality_regret": event.get(
                "controller_plugin_predicted_quality_regret"
            ),
        }
        for name, value in optional_metrics.items():
            if value is None:
                continue
            value = to_float(value, None)
            if value is None or not math.isfinite(value):
                continue
            metrics[name] = value
        for key, value in event.items():
            if not key.startswith("controller_plugin_"):
                continue
            value = to_float(value, None)
            if value is None or not math.isfinite(value):
                continue
            metrics.setdefault(f"epoch/{key}", value)
        for name, value in metrics.items():
            mlflow.log_metric(name, value, step=step)
        if event.get("controller_mode"):
            mlflow.set_tag("controller_mode", str(event["controller_mode"]))
        if event.get("comparison_strategy"):
            mlflow.set_tag("comparison_strategy", str(event["comparison_strategy"]))
        if event.get("stop_reason"):
            mlflow.set_tag("energy_adaptive_stop_reason", str(event["stop_reason"]))
        if event.get("controller_decision"):
            mlflow.set_tag("controller_last_decision", str(event["controller_decision"]))


def load_epoch_events(path: Path) -> list[dict]:
    events = []
    if not path.exists():
        return events
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            events.append(json.loads(line))
    return events


def _dedupe_epoch_rows(epoch_rows: list[dict]) -> list[dict]:
    by_epoch: dict[int, dict] = {}
    for event in epoch_rows:
        try:
            epoch_index = int(event.get("epoch_index", event.get("epoch", 0)))
        except Exception:
            epoch_index = len(by_epoch) + 1

        current = by_epoch.get(epoch_index)
        if current is None:
            by_epoch[epoch_index] = event
            continue

        current_score = (
            to_float(current.get("duration_seconds")),
            to_float(current.get("samples")),
            to_float(current.get("ts")),
        )
        new_score = (
            to_float(event.get("duration_seconds")),
            to_float(event.get("samples")),
            to_float(event.get("ts")),
        )
        if new_score > current_score:
            by_epoch[epoch_index] = event

    return [by_epoch[idx] for idx in sorted(by_epoch)]


def build_epoch_summary(
    *,
    epoch_timeline_path: str | Path,
    output_path: str | Path,
    job_id: str,
    price_eur_kwh: float = 0.30,
    co2_kg_kwh: float = 0.4,
    adaptive_enabled: bool = False,
    adaptive_monitor_metric: str = "map50",
    controller_mode: str = "none",
    comparison_strategy: str = "unspecified",
    scenario: str = "unspecified",
    training_seed: int = 0,
    split_seed: int = 0,
    cache_policy: str = "ram",
    controller_id: str = "unspecified",
    benchmark_version: str = "unversioned",
    benchmark_run_id: str = "none",
    benchmark_case_id: str = "none",
    benchmark_stage: str = "none",
    task_type: str = "object_detection",
    quality_metric: str = "map50_95",
) -> dict:
    events = load_epoch_events(Path(epoch_timeline_path))
    epoch_rows = [event for event in events if event.get("event") == "end"]
    epoch_rows = _dedupe_epoch_rows(epoch_rows)

    if adaptive_monitor_metric == "map50_95":
        best_mape_key = "marginal_map50_95_per_wh"
    else:
        best_mape_key = "marginal_map50_per_wh"

    summary = {
        "job_id": str(job_id),
        "adaptive_enabled": bool(adaptive_enabled),
        "adaptive_monitor_metric": adaptive_monitor_metric,
        "controller_mode": controller_mode,
        "comparison_strategy": comparison_strategy,
        "scenario": scenario,
        "training_seed": int(training_seed),
        "split_seed": int(split_seed),
        "cache_policy": cache_policy,
        "controller_id": controller_id,
        "benchmark_version": benchmark_version,
        "benchmark_run_id": benchmark_run_id,
        "benchmark_case_id": benchmark_case_id,
        "benchmark_stage": benchmark_stage,
        "task_type": task_type,
        "quality_metric": quality_metric,
        "epochs": epoch_rows,
        "epochs_completed": len(epoch_rows),
        "adaptive_stopped": any(to_float(item.get("should_stop")) > 0 for item in epoch_rows),
        "stop_epoch": None,
        "stop_reason": None,
        "final_map50": None,
        "final_map50_95": None,
        "final_precision": None,
        "final_recall": None,
        "final_best_map50": None,
        "final_best_map50_95": None,
        "total_energy_kwh": 0.0,
        "total_gpu_energy_kwh": 0.0,
        "total_net_energy_kwh": 0.0,
        "total_net_gpu_energy_kwh": 0.0,
        "total_duration_seconds": 0.0,
        "estimated_electricity_cost_eur": 0.0,
        "estimated_co2_kg": 0.0,
        "best_mape_map50_per_wh": None,
        "best_mape_map50_95_per_wh": None,
        "predicted_final_metric_mean": None,
        "predicted_final_metric_lower": None,
        "predicted_final_metric_upper": None,
        "predicted_remaining_gain_upper": None,
        "predicted_prob_gain_gt_epsilon": None,
        "uncertainty_interval_width": None,
        "conformal_correction": None,
        "conformal_remaining_gain_upper": None,
        "predicted_remaining_energy_wh": None,
        "predicted_future_efficiency_per_wh": None,
        "controller_decision_state": None,
        "controller_decision": None,
        "plateau_slope_per_epoch": None,
        "recent_best_gain": None,
        "observed_efficiency_per_wh": None,
        "adaptive_efficiency_threshold": None,
        "pareto_continue_utility": None,
        "controller_evidence": None,
        "dynamic_patience": None,
        "quality_target_reached": None,
        "budget_triggered": None,
        "observed_job_energy_wh": None,
        "projected_full_job_energy_wh": None,
        "projected_stopped_job_energy_wh": None,
        "projected_energy_saving_fraction": None,
        "projected_next_energy_saving_fraction": None,
        "energy_guard_target_saving_fraction": None,
        "energy_guard_remaining_budget_wh": None,
        "energy_guard_boundary_reached": None,
        "energy_guard_fallback_triggered": None,
        "energy_guard_quality_conflict": None,
        "controller_plugin_confidence": None,
        "controller_plugin_predicted_energy_saving_fraction": None,
        "controller_plugin_predicted_quality_regret": None,
        "controller_plugin_compute_seconds": None,
    }
    if epoch_rows:
        last = epoch_rows[-1]
        summary["stop_epoch"] = int(last["epoch"]) if to_float(last.get("should_stop")) > 0 else None
        summary["stop_reason"] = last.get("stop_reason")
        summary["final_map50"] = last.get("map50")
        summary["final_map50_95"] = last.get("map50_95")
        summary["final_precision"] = last.get("precision")
        summary["final_recall"] = last.get("recall")
        summary["final_best_map50"] = last.get("best_map50")
        summary["final_best_map50_95"] = last.get("best_map50_95")
        summary["predicted_final_metric_mean"] = last.get("predicted_final_metric_mean")
        summary["predicted_final_metric_lower"] = last.get("predicted_final_metric_lower")
        summary["predicted_final_metric_upper"] = last.get("predicted_final_metric_upper")
        summary["predicted_remaining_gain_upper"] = last.get("predicted_remaining_gain_upper")
        summary["predicted_prob_gain_gt_epsilon"] = last.get("predicted_prob_gain_gt_epsilon")
        summary["uncertainty_interval_width"] = last.get("uncertainty_interval_width")
        summary["conformal_correction"] = last.get("conformal_correction")
        summary["conformal_remaining_gain_upper"] = last.get("conformal_remaining_gain_upper")
        summary["predicted_remaining_energy_wh"] = last.get("predicted_remaining_energy_wh")
        summary["predicted_future_efficiency_per_wh"] = last.get("predicted_future_efficiency_per_wh")
        summary["controller_decision_state"] = last.get("controller_decision_state")
        summary["controller_decision"] = last.get("controller_decision")
        for key in (
            "plateau_slope_per_epoch",
            "recent_best_gain",
            "observed_efficiency_per_wh",
            "adaptive_efficiency_threshold",
            "pareto_continue_utility",
            "controller_evidence",
            "dynamic_patience",
            "quality_target_reached",
            "budget_triggered",
            "observed_job_energy_wh",
            "projected_full_job_energy_wh",
            "projected_stopped_job_energy_wh",
            "projected_energy_saving_fraction",
            "projected_next_energy_saving_fraction",
            "energy_guard_target_saving_fraction",
            "energy_guard_remaining_budget_wh",
            "energy_guard_boundary_reached",
            "energy_guard_fallback_triggered",
            "energy_guard_quality_conflict",
            "controller_plugin_confidence",
            "controller_plugin_predicted_energy_saving_fraction",
            "controller_plugin_predicted_quality_regret",
            "controller_plugin_compute_seconds",
        ):
            summary[key] = last.get(key)
        for key, value in last.items():
            if key.startswith("controller_plugin_"):
                summary[key] = value
        summary["total_energy_kwh"] = sum(to_float(item.get("total_energy_kwh")) for item in epoch_rows)
        summary["total_gpu_energy_kwh"] = sum(to_float(item.get("gpu_energy_kwh")) for item in epoch_rows)
        summary["total_net_energy_kwh"] = sum(to_float(item.get("net_total_energy_kwh")) for item in epoch_rows)
        summary["total_net_gpu_energy_kwh"] = sum(to_float(item.get("net_gpu_energy_kwh")) for item in epoch_rows)
        summary["total_duration_seconds"] = sum(to_float(item.get("duration_seconds")) for item in epoch_rows)
        summary["estimated_electricity_cost_eur"] = summary["total_energy_kwh"] * price_eur_kwh
        summary["estimated_co2_kg"] = summary["total_energy_kwh"] * co2_kg_kwh
        valid_mape = [to_float(item.get(best_mape_key), None) for item in epoch_rows]
        valid_mape = [value for value in valid_mape if value is not None and math.isfinite(value)]
        if adaptive_monitor_metric == "map50_95":
            summary["best_mape_map50_95_per_wh"] = max(valid_mape) if valid_mape else None
        else:
            summary["best_mape_map50_per_wh"] = max(valid_mape) if valid_mape else None

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    return summary
