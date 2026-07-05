from __future__ import annotations

import csv
import json
import math
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


@dataclass
class EnergyAdaptiveConfig:
    enabled: bool = False
    min_epochs: int = 20
    patience: int = 3
    smoothing_window: int = 3
    min_delta_map50: float = 0.001
    min_mape_map50_per_wh: float = 0.0001


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
        self.low_gain_streak = 0
        self.history: list[dict] = []

    def on_train_epoch_start(self, trainer):
        epoch = int(getattr(trainer, "epoch", len(self.history))) + 1
        start_ts = float(self.time_fn())
        self.current_epoch_start_ts = start_ts
        _write_jsonl(
            self.epoch_timeline_path,
            {
                "epoch": epoch,
                "event": "start",
                "ts": start_ts,
            },
        )

    def on_fit_epoch_end(self, trainer):
        epoch = int(getattr(trainer, "epoch", len(self.history))) + 1
        end_ts = float(self.time_fn())
        start_ts = self.current_epoch_start_ts if self.current_epoch_start_ts is not None else end_ts

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
        epoch_gpu_energy_wh = to_float(epoch_gpu.get("gpu_energy_kwh")) * 1000.0
        marginal_map50_per_wh = delta_map50 / epoch_total_energy_wh if delta_map50 is not None and epoch_total_energy_wh > 0 else None
        marginal_map50_95_per_wh = (
            delta_map50_95 / epoch_total_energy_wh if delta_map50_95 is not None and epoch_total_energy_wh > 0 else None
        )

        cumulative_total_energy_kwh = epoch_total_energy_kwh + sum(to_float(item.get("total_energy_kwh")) for item in self.history)
        cumulative_gpu_energy_kwh = to_float(epoch_gpu.get("gpu_energy_kwh")) + sum(
            to_float(item.get("gpu_energy_kwh")) for item in self.history
        )

        event = {
            "epoch": epoch_metrics.get("epoch", epoch),
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
            "delta_map50": delta_map50,
            "delta_map50_95": delta_map50_95,
            "marginal_map50_per_wh": marginal_map50_per_wh,
            "marginal_map50_95_per_wh": marginal_map50_95_per_wh,
            "energy_efficiency_gain": marginal_map50_per_wh,
            "adaptive_enabled": int(self.config.enabled),
        }

        should_stop, stop_reason, smooth_delta, smooth_mape = self._evaluate_stop(event)
        event["smoothed_delta_map50"] = smooth_delta
        event["smoothed_mape_map50_per_wh"] = smooth_mape
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

    def _evaluate_stop(self, current_event: dict) -> tuple[bool, str | None, float | None, float | None]:
        valid_history = self.history + [current_event]
        recent = [
            item
            for item in valid_history
            if item.get("delta_map50") is not None and item.get("marginal_map50_per_wh") is not None
        ]
        window = self.config.smoothing_window
        smooth_delta = _mean([to_float(item.get("delta_map50")) for item in recent[-window:]]) if recent else None
        smooth_mape = _mean([to_float(item.get("marginal_map50_per_wh")) for item in recent[-window:]]) if recent else None

        if not self.config.enabled:
            self.low_gain_streak = 0
            return False, None, smooth_delta, smooth_mape

        if len(valid_history) < self.config.min_epochs or len(recent) < window:
            self.low_gain_streak = 0
            return False, None, smooth_delta, smooth_mape

        low_delta = smooth_delta is not None and smooth_delta < self.config.min_delta_map50
        low_mape = smooth_mape is not None and smooth_mape < self.config.min_mape_map50_per_wh
        if low_delta and low_mape:
            self.low_gain_streak += 1
        else:
            self.low_gain_streak = 0

        if self.low_gain_streak >= self.config.patience:
            return (
                True,
                (
                    f"energy_adaptive_stop: avg_delta_map50={smooth_delta:.6f}, "
                    f"avg_mape_map50_per_wh={smooth_mape:.6f}"
                ),
                smooth_delta,
                smooth_mape,
            )
        return False, None, smooth_delta, smooth_mape

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
        epoch = int(to_float(row.get("epoch"), len(rows) - 1)) + 1
        return {
            "epoch": epoch,
            "map50": self._first_metric(normalized, ["metricsmap50b", "metricsmap50"]),
            "map50_95": self._first_metric(normalized, ["metricsmap5095b", "metricsmap5095"]),
            "precision": self._first_metric(normalized, ["metricsprecisionb", "metricsprecision"]),
            "recall": self._first_metric(normalized, ["metricsrecallb", "metricsrecall"]),
        }

    def _resolve_results_csv(self, trainer) -> Path | None:
        trainer_csv = getattr(trainer, "csv", None)
        if trainer_csv:
            return Path(trainer_csv)
        save_dir = getattr(trainer, "save_dir", None)
        if save_dir:
            return Path(save_dir) / "results.csv"
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
        }
        optional_metrics = {
            "epoch/map50": event.get("map50"),
            "epoch/map50_95": event.get("map50_95"),
            "epoch/precision": event.get("precision"),
            "epoch/recall": event.get("recall"),
            "epoch/delta_map50": event.get("delta_map50"),
            "epoch/delta_map50_95": event.get("delta_map50_95"),
            "epoch/mape_map50_per_wh": event.get("marginal_map50_per_wh"),
            "epoch/mape_map50_95_per_wh": event.get("marginal_map50_95_per_wh"),
            "epoch/smoothed_delta_map50": event.get("smoothed_delta_map50"),
            "epoch/smoothed_mape_map50_per_wh": event.get("smoothed_mape_map50_per_wh"),
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


def build_epoch_summary(
    *,
    epoch_timeline_path: str | Path,
    output_path: str | Path,
    job_id: str,
    price_eur_kwh: float = 0.30,
    co2_kg_kwh: float = 0.4,
    adaptive_enabled: bool = False,
) -> dict:
    events = load_epoch_events(Path(epoch_timeline_path))
    epoch_rows = [event for event in events if event.get("event") == "end"]
    epoch_rows.sort(key=lambda item: int(item.get("epoch_index", item.get("epoch", 0))))

    summary = {
        "job_id": str(job_id),
        "adaptive_enabled": bool(adaptive_enabled),
        "epochs": epoch_rows,
        "epochs_completed": len(epoch_rows),
        "adaptive_stopped": any(to_float(item.get("should_stop")) > 0 for item in epoch_rows),
        "stop_epoch": None,
        "final_map50": None,
        "final_map50_95": None,
        "final_precision": None,
        "final_recall": None,
        "total_energy_kwh": 0.0,
        "total_gpu_energy_kwh": 0.0,
        "total_duration_seconds": 0.0,
        "estimated_electricity_cost_eur": 0.0,
        "estimated_co2_kg": 0.0,
        "best_mape_map50_per_wh": None,
    }
    if epoch_rows:
        last = epoch_rows[-1]
        summary["stop_epoch"] = int(last["epoch"]) if to_float(last.get("should_stop")) > 0 else None
        summary["final_map50"] = last.get("map50")
        summary["final_map50_95"] = last.get("map50_95")
        summary["final_precision"] = last.get("precision")
        summary["final_recall"] = last.get("recall")
        summary["total_energy_kwh"] = sum(to_float(item.get("total_energy_kwh")) for item in epoch_rows)
        summary["total_gpu_energy_kwh"] = sum(to_float(item.get("gpu_energy_kwh")) for item in epoch_rows)
        summary["total_duration_seconds"] = sum(to_float(item.get("duration_seconds")) for item in epoch_rows)
        summary["estimated_electricity_cost_eur"] = summary["total_energy_kwh"] * price_eur_kwh
        summary["estimated_co2_kg"] = summary["total_energy_kwh"] * co2_kg_kwh
        valid_mape = [to_float(item.get("marginal_map50_per_wh"), None) for item in epoch_rows]
        valid_mape = [value for value in valid_mape if value is not None and math.isfinite(value)]
        summary["best_mape_map50_per_wh"] = max(valid_mape) if valid_mape else None

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    return summary
