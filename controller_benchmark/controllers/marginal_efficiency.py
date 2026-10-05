from __future__ import annotations

import statistics

from controller_benchmark.api import Controller, ControllerDecision, EpochObservation


class MarginalEfficiencyController(Controller):
    """Task-independent example using normalized quality gain per Wh."""

    def __init__(self, context):
        super().__init__(context)
        self.low_gain_streak = 0

    def evaluate(self, observation: EpochObservation) -> ControllerDecision:
        params = self.context.parameters
        min_epochs = int(params.get("min_epochs", 20))
        patience = int(params.get("patience", 3))
        window = max(2, int(params.get("window", 5)))
        min_delta = float(params.get("min_delta_quality", 0.0005))
        min_efficiency = float(params.get("min_quality_per_wh", 0.005))
        recent = [*observation.history[-(window - 1) :], observation.raw_metrics]
        deltas = []
        efficiencies = []
        previous = None
        for row in recent:
            quality = row.get(observation.quality_metric)
            if quality is None:
                continue
            quality = float(quality)
            if previous is not None:
                delta = quality - previous
                energy_wh = max(0.0, float(row.get("total_energy_kwh", 0.0)) * 1000.0)
                deltas.append(delta)
                if energy_wh > 0:
                    efficiencies.append(delta / energy_wh)
            previous = quality
        mean_delta = statistics.fmean(deltas) if deltas else None
        mean_efficiency = statistics.fmean(efficiencies) if efficiencies else None
        candidate = (
            observation.epoch >= min_epochs
            and mean_delta is not None
            and mean_efficiency is not None
            and mean_delta < min_delta
            and mean_efficiency < min_efficiency
        )
        self.low_gain_streak = self.low_gain_streak + 1 if candidate else 0
        stop = self.low_gain_streak >= patience
        return ControllerDecision(
            stop=stop,
            reason=(
                f"marginal_efficiency_stop: epoch={observation.epoch}, "
                f"mean_delta={mean_delta}, mean_quality_per_wh={mean_efficiency}"
            ),
            confidence=min(1.0, self.low_gain_streak / max(1, patience)),
            diagnostics={
                "mean_delta_quality": mean_delta,
                "mean_quality_per_wh": mean_efficiency,
                "low_gain_streak": self.low_gain_streak,
            },
        )
