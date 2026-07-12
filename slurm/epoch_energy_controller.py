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
    uncertainty_target_epoch: int = 100
    uncertainty_epsilon: float = 0.01
    uncertainty_alpha: float = 0.05
    uncertainty_bootstrap_samples: int = 24
    uncertainty_min_fit_points: int = 8


class EpochEnergyAdaptiveController:
    def __init__(
        self,
        *,
        gpu_csv_path: str | Path,
        epoch_timeline_path: str | Path,
        pue_factor: float = 1.0,
        price_eur_kwh: float = 0.30,
        co2_kg_kwh: float = 0.4,
        config: EnergyAdaptiveConfig | None = None,
        time_fn=time.time,
    ):
        self.gpu_csv_path = Path(gpu_csv_path)
        self.epoch_timeline_path = Path(epoch_timeline_path)
        self.pue_factor = pue_factor
        self.price_eur_kwh = price_eur_kwh
        self.co2_kg_kwh = co2_kg_kwh
        self.config = config or EnergyAdaptiveConfig()
        self.time_fn = time_fn
        self.current_epoch_start_ts: float | None = None
        self.current_epoch_number: int | None = None
        self.low_gain_streak = 0
        self.history: list[dict] = []

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
        marginal_map50_per_wh = delta_map50 / epoch_total_energy_wh if delta_map50 is not None and epoch_total_energy_wh > 0 else None
        marginal_map50_95_per_wh = (
            delta_map50_95 / epoch_total_energy_wh if delta_map50_95 is not None and epoch_total_energy_wh > 0 else None
        )

        cumulative_total_energy_kwh = epoch_total_energy_kwh + sum(to_float(item.get("total_energy_kwh")) for item in self.history)
        cumulative_gpu_energy_kwh = to_float(epoch_gpu.get("gpu_energy_kwh")) + sum(
            to_float(item.get("gpu_energy_kwh")) for item in self.history
        )

        lr = self._current_learning_rate(trainer)
        best_map50 = max([to_float(item.get("best_map50"), 0.0) for item in self.history] + [to_float(epoch_metrics.get("map50"), 0.0)])
        best_map50_95 = max([to_float(item.get("best_map50_95"), 0.0) for item in self.history] + [to_float(epoch_metrics.get("map50_95"), 0.0)])

        event = {
            "epoch": epoch,
            "epoch_index": epoch,
            "event": "end",
            "ts": end_ts,
            "start_ts": start_ts,
            "end_ts": end_ts,
            "duration_seconds": to_float(epoch_gpu.get("duration_seconds"), max(0.0, end_ts - start_ts)),
            "samples": int(epoch_gpu.get("samples", 0)),
            "gpu_power_avg_w": to_float(epoch_gpu.get("gpu_power_avg_w")),
            "gpu_power_max_w": to_float(epoch_gpu.get("gpu_power_max_w")),
            "gpu_util_avg_pct": to_float(epoch_gpu.get("gpu_util_avg_pct")),
            "gpu_mem_used_avg_mb": to_float(epoch_gpu.get("gpu_mem_used_avg_mb")),
            "gpu_temp_avg_c": to_float(epoch_gpu.get("gpu_temp_avg_c")),
            "gpu_energy_kwh": to_float(epoch_gpu.get("gpu_energy_kwh")),
            "total_energy_kwh": epoch_total_energy_kwh,
            "estimated_electricity_cost_eur": epoch_total_energy_kwh * self.price_eur_kwh,
            "estimated_co2_kg": epoch_total_energy_kwh * self.co2_kg_kwh,
            "cumulative_gpu_energy_kwh": cumulative_gpu_energy_kwh,
            "cumulative_total_energy_kwh": cumulative_total_energy_kwh,
            "map50": epoch_metrics.get("map50"),
            "map50_95": epoch_metrics.get("map50_95"),
            "precision": epoch_metrics.get("precision"),
            "recall": epoch_metrics.get("recall"),
            "val_box_loss": epoch_metrics.get("val_box_loss"),
            "val_cls_loss": epoch_metrics.get("val_cls_loss"),
            "val_dfl_loss": epoch_metrics.get("val_dfl_loss"),
            "learning_rate": lr,
            "delta_map50": delta_map50,
            "delta_map50_95": delta_map50_95,
            "marginal_map50_per_wh": marginal_map50_per_wh,
            "marginal_map50_95_per_wh": marginal_map50_95_per_wh,
            "energy_efficiency_gain": marginal_map50_per_wh,
            "best_map50": best_map50,
            "best_map50_95": best_map50_95,
            "adaptive_enabled": int(self.config.enabled),
            "controller_mode": self.config.controller_mode,
            "comparison_strategy": self.config.comparison_strategy,
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

        self.current_epoch_start_ts = None
        self.current_epoch_number = None

    def _evaluate_stop(self, current_event: dict) -> tuple[bool, str | None, dict]:
        if self.config.controller_mode == "delta_mape":
            return self._evaluate_delta_mape_stop(current_event)
        if self.config.controller_mode == "uncertainty_aware":
            return self._evaluate_uncertainty_stop(current_event)

        self.low_gain_streak = 0
        return False, None, self._default_diagnostics(current_event)

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
            return (
                True,
                f"energy_adaptive_stop[{self.config.monitor_metric}]: avg_delta={smooth_delta:.6f}, avg_mape_per_wh={smooth_mape:.6f}",
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

        metric_key = "map50_95" if self.config.monitor_metric == "map50_95" else "map50"
        best_key = "best_map50_95" if self.config.monitor_metric == "map50_95" else "best_map50"
        epochs = [int(item.get("epoch_index", item.get("epoch", 0))) for item in history if item.get(metric_key) is not None]
        values = [to_float(item.get(metric_key)) for item in history if item.get(metric_key) is not None]
        current_best = to_float(current_event.get(best_key))
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

        low_gain = gain_upper < self.config.uncertainty_epsilon and gain_risk < self.config.uncertainty_alpha
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
                    f"alpha={self.config.uncertainty_alpha:.6f}"
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
        }
        optional_metrics = {
            "epoch/map50": event.get("map50"),
            "epoch/map50_95": event.get("map50_95"),
            "epoch/best_map50": event.get("best_map50"),
            "epoch/best_map50_95": event.get("best_map50_95"),
            "epoch/precision": event.get("precision"),
            "epoch/recall": event.get("recall"),
            "epoch/learning_rate": event.get("learning_rate"),
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
        }
        for name, value in optional_metrics.items():
            if value is None:
                continue
            value = to_float(value, None)
            if value is None or not math.isfinite(value):
                continue
            metrics[name] = value
        for name, value in metrics.items():
            mlflow.log_metric(name, value, step=step)
        if event.get("controller_mode"):
            mlflow.set_tag("controller_mode", str(event["controller_mode"]))
        if event.get("comparison_strategy"):
            mlflow.set_tag("comparison_strategy", str(event["comparison_strategy"]))
        if event.get("stop_reason"):
            mlflow.set_tag("energy_adaptive_stop_reason", str(event["stop_reason"]))


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
        summary["total_energy_kwh"] = sum(to_float(item.get("total_energy_kwh")) for item in epoch_rows)
        summary["total_gpu_energy_kwh"] = sum(to_float(item.get("gpu_energy_kwh")) for item in epoch_rows)
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
