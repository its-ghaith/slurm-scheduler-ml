from __future__ import annotations

import math

from controller_benchmark.api import Controller, ControllerDecision, EpochObservation


class StandardEarlyStoppingController(Controller):
    """Task-independent validation early stopping used as a simple baseline."""

    def __init__(self, context):
        super().__init__(context)
        parameters = context.parameters
        self.min_delta = float(parameters.get("min_delta", 0.001))
        self.start_epoch_fraction = float(parameters.get("start_epoch_fraction", 0.20))
        self.patience_fraction = float(parameters.get("patience_fraction", 0.10))
        self.start_epoch = int(
            parameters.get(
                "start_epoch",
                max(1, math.ceil(self.start_epoch_fraction * context.max_epochs)),
            )
        )
        self.patience = int(
            parameters.get(
                "patience",
                max(1, math.ceil(self.patience_fraction * context.max_epochs)),
            )
        )
        if self.min_delta < 0:
            raise ValueError("min_delta must be non-negative.")
        if not 0.0 <= self.start_epoch_fraction <= 1.0:
            raise ValueError("start_epoch_fraction must be in [0, 1].")
        if self.start_epoch < 1 or self.patience < 1:
            raise ValueError("start_epoch and patience must be positive.")
        self.best_quality: float | None = None
        self.best_epoch = 0
        self.non_improving_epochs = 0

    def evaluate(self, observation: EpochObservation) -> ControllerDecision:
        quality = observation.quality
        improved = quality is not None and (
            self.best_quality is None or quality > self.best_quality + self.min_delta
        )
        if improved:
            self.best_quality = float(quality)
            self.best_epoch = observation.epoch
            self.non_improving_epochs = 0
        elif observation.epoch >= self.start_epoch:
            self.non_improving_epochs += 1

        stop = (
            observation.epoch >= self.start_epoch
            and self.non_improving_epochs >= self.patience
        )
        return ControllerDecision(
            stop=stop,
            reason=(
                f"standard_early_stopping: no improvement greater than {self.min_delta:g} "
                f"for {self.non_improving_epochs} epochs"
                if stop
                else "continue"
            ),
            diagnostics={
                "es_best_quality": self.best_quality,
                "es_best_epoch": self.best_epoch,
                "es_non_improving_epochs": self.non_improving_epochs,
                "es_start_epoch": self.start_epoch,
                "es_patience": self.patience,
                "es_min_delta": self.min_delta,
                "es_candidate_stop": int(stop),
                "es_task_independent": 1,
            },
        )
