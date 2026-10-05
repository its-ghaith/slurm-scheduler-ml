from __future__ import annotations

import math
import statistics
from typing import Any, Iterable, Mapping

from controller_benchmark.api import Controller, ControllerDecision, EpochObservation


def _number(value: Any, default: float | None = None) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return default
    return result if math.isfinite(result) else default


def _best_so_far(values: Iterable[float]) -> list[float]:
    result: list[float] = []
    best = 0.0
    for value in values:
        best = max(best, min(1.0, max(0.0, value)))
        result.append(best)
    return result


def _probability_above_from_quantiles(quantile_pairs: list[tuple[float, float]], threshold: float) -> float | None:
    clean = sorted(
        (float(q), float(value))
        for q, value in quantile_pairs
        if math.isfinite(float(q)) and math.isfinite(float(value))
    )
    if not clean:
        return None
    if threshold <= clean[0][1]:
        return max(0.0, min(1.0, 1.0 - clean[0][0]))
    if threshold >= clean[-1][1]:
        return max(0.0, min(1.0, 1.0 - clean[-1][0]))
    for (left_q, left_value), (right_q, right_value) in zip(clean, clean[1:]):
        if left_value <= threshold <= right_value:
            if right_value == left_value:
                cdf = (left_q + right_q) / 2.0
            else:
                weight = (threshold - left_value) / (right_value - left_value)
                cdf = left_q + weight * (right_q - left_q)
            return max(0.0, min(1.0, 1.0 - cdf))
    return None


class LcpfnQualityBaselineController(Controller):
    """Optional quality-only LC-PFN baseline controller.

    LC-PFN supports increasing curves with values in [0, 1]. Therefore this
    baseline uses the best-so-far task-independent quality curve Q_t. It is
    intentionally separated from the PEP controller because LC-PFN has its own
    runtime dependency constraints and does not model energy-related features.
    """

    QUANTILES = (0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95)

    def __init__(self, context):
        super().__init__(context)
        params = context.parameters
        self.min_epochs = int(params.get("min_epochs", 30))
        self.horizon_epochs = max(1, int(params.get("horizon_epochs", 5)))
        self.min_fit_points = max(3, int(params.get("min_fit_points", 10)))
        self.patience = max(1, int(params.get("patience", 3)))
        self.useful_gain = float(params.get("useful_gain", 0.002))
        self.min_expected_gain = float(params.get("min_expected_gain", self.useful_gain))
        self.max_probability_gain_gt_threshold = float(params.get("max_probability_gain_gt_threshold", 0.20))
        self.min_quality_per_wh = float(params.get("min_quality_per_wh", 0.0))
        self.model_name = str(params.get("model_name", "EMSIZE512_NLAYERS12_NBUCKETS1000"))
        self.low_value_streak = 0
        self._torch, self._model = self._load_lcpfn()

    def evaluate(self, observation: EpochObservation) -> ControllerDecision:
        rows = [*observation.history, observation.raw_metrics]
        qualities = self._quality_values(rows, observation)
        diagnostics: dict[str, float | int | bool | None] = {
            "lcpfn_horizon_epochs": self.horizon_epochs,
            "lcpfn_useful_gain_threshold": self.useful_gain,
            "lcpfn_quality_only_baseline": 1,
        }
        if observation.epoch < self.min_epochs or len(qualities) < self.min_fit_points:
            self.low_value_streak = 0
            diagnostics.update({"lcpfn_candidate_stop": 0, "lcpfn_low_value_streak": 0})
            return ControllerDecision(stop=False, reason="lcpfn_warmup", confidence=0.0, diagnostics=diagnostics)

        prediction = self._predict_horizon(qualities, rows, observation)
        diagnostics.update(prediction)
        expected_gain = _number(prediction.get("lcpfn_expected_quality_gain_next_horizon"))
        prob_gain = _number(prediction.get("lcpfn_prob_gain_gt_threshold"))
        utility = _number(prediction.get("lcpfn_utility_quality_per_wh"))
        low_expected_gain = expected_gain is not None and expected_gain < self.min_expected_gain
        low_probability = prob_gain is not None and prob_gain < self.max_probability_gain_gt_threshold
        low_efficiency = self.min_quality_per_wh <= 0.0 or (utility is not None and utility < self.min_quality_per_wh)
        candidate = low_expected_gain and low_probability and low_efficiency
        self.low_value_streak = self.low_value_streak + 1 if candidate else 0
        stop = self.low_value_streak >= self.patience
        diagnostics.update(
            {
                "lcpfn_low_expected_gain": int(low_expected_gain),
                "lcpfn_low_probability": int(low_probability),
                "lcpfn_low_efficiency": int(low_efficiency),
                "lcpfn_candidate_stop": int(candidate),
                "lcpfn_low_value_streak": self.low_value_streak,
            }
        )
        reason = (
            "lcpfn_stop: "
            f"epoch={observation.epoch}, horizon={self.horizon_epochs}, "
            f"expected_gain={float(expected_gain or 0.0):.6f}, "
            f"prob_gain_gt_{self.useful_gain:.4f}={float(prob_gain or 0.0):.4f}"
        )
        return ControllerDecision(
            stop=stop,
            reason=reason if stop else "continue",
            confidence=max(0.0, min(1.0, 1.0 - float(prob_gain or 0.0))) if prob_gain is not None else None,
            predicted_energy_saving_fraction=prediction.get("lcpfn_projected_energy_saving_fraction"),
            predicted_quality_regret=prediction.get("lcpfn_quality_gain_upper"),
            diagnostics=diagnostics,
        )

    def _predict_horizon(
        self,
        qualities: list[float],
        rows: list[Mapping[str, Any]],
        observation: EpochObservation,
    ) -> dict[str, float | int | None]:
        current_epoch = int(observation.epoch)
        target_epoch = min(int(observation.max_epochs), current_epoch + self.horizon_epochs)
        best_values = _best_so_far(qualities)
        current_best = best_values[-1]
        torch = self._torch
        x_train = torch.arange(1, len(best_values) + 1, dtype=torch.float32).unsqueeze(1)
        y_train = torch.tensor(best_values, dtype=torch.float32).unsqueeze(1)
        x_test = torch.tensor([target_epoch], dtype=torch.float32).unsqueeze(1)
        predicted = self._model.predict_quantiles(
            x_train=x_train,
            y_train=y_train,
            x_test=x_test,
            qs=list(self.QUANTILES),
        )
        values = [float(value) for value in predicted.detach().cpu().reshape(-1).tolist()]
        pairs = list(zip(self.QUANTILES, values))
        median_quality = self._quantile_value(pairs, 0.50)
        lower_quality = self._quantile_value(pairs, 0.10)
        upper_quality = self._quantile_value(pairs, 0.90)
        expected_gain = max(0.0, float(median_quality - current_best)) if median_quality is not None else None
        lower_gain = max(0.0, float(lower_quality - current_best)) if lower_quality is not None else None
        upper_gain = max(0.0, float(upper_quality - current_best)) if upper_quality is not None else None
        probability = _probability_above_from_quantiles(pairs, current_best + self.useful_gain)
        expected_energy = self._recent_horizon_energy(rows)
        utility = expected_gain / expected_energy if expected_gain is not None and expected_energy and expected_energy > 0 else None
        projected_saving = self._project_energy_saving(rows, observation)
        return {
            "lcpfn_expected_quality_next_horizon": median_quality,
            "lcpfn_expected_quality_gain_next_horizon": expected_gain,
            "lcpfn_quality_gain_lower": lower_gain,
            "lcpfn_quality_gain_upper": upper_gain,
            "lcpfn_prob_gain_gt_threshold": probability,
            "lcpfn_expected_energy_next_horizon_wh": expected_energy,
            "lcpfn_utility_quality_per_wh": utility,
            "lcpfn_projected_energy_saving_fraction": projected_saving,
        }

    def _load_lcpfn(self):
        try:
            import lcpfn  # type: ignore[import-not-found]
            import torch  # type: ignore[import-not-found]
        except ImportError as exc:
            raise RuntimeError(
                "LC-PFN is optional and not installed in this runtime. "
                "Use this baseline only in an environment compatible with lcpfn "
                "(Python >=3.9,<3.12 and its torch constraints), or install it in a separate image."
            ) from exc
        return torch, lcpfn.LCPFN(model_name=self.model_name)

    def _quality_values(self, rows: list[Mapping[str, Any]], observation: EpochObservation) -> list[float]:
        values = []
        for row in rows:
            value = _number(row.get(observation.quality_metric), _number(row.get("quality_score"), _number(row.get("map50_95"))))
            if value is not None:
                values.append(min(1.0, max(0.0, value)))
        return values

    def _recent_horizon_energy(self, rows: list[Mapping[str, Any]]) -> float | None:
        values = [self._energy_wh(row) for row in rows[-self.horizon_epochs :] if self._energy_wh(row) > 0]
        return sum(values) if values else None

    def _project_energy_saving(self, rows: list[Mapping[str, Any]], observation: EpochObservation) -> float | None:
        recent = [self._energy_wh(row) for row in rows[-10:] if self._energy_wh(row) > 0]
        if not recent:
            return None
        remaining = max(0, int(observation.max_epochs) - int(observation.epoch))
        full_projection = observation.cumulative_energy_wh + statistics.median(recent) * remaining
        if full_projection <= 0:
            return None
        return max(0.0, (full_projection - observation.cumulative_energy_wh) / full_projection)

    def _energy_wh(self, row: Mapping[str, Any]) -> float:
        if row.get("epoch_energy_wh") is not None:
            return max(0.0, _number(row.get("epoch_energy_wh"), 0.0) or 0.0)
        value = _number(row.get("total_energy_kwh"), _number(row.get("gpu_energy_kwh"), 0.0))
        return max(0.0, float(value or 0.0) * 1000.0)

    def _quantile_value(self, pairs: list[tuple[float, float]], q: float) -> float | None:
        for quantile, value in pairs:
            if abs(quantile - q) < 1e-9:
                return value
        return None
