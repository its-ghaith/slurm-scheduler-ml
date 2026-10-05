from __future__ import annotations

from controller_benchmark.api import Controller, ControllerDecision, EpochObservation


class FixedEpochController(Controller):
    """Minimal reference plugin and fixed-budget benchmark baseline."""

    def evaluate(self, observation: EpochObservation) -> ControllerDecision:
        stop_epoch = int(self.context.parameters.get("stop_epoch", observation.max_epochs))
        stop = observation.epoch >= stop_epoch
        return ControllerDecision(
            stop=stop,
            reason=f"fixed_epoch_stop: epoch={observation.epoch}, limit={stop_epoch}",
            confidence=1.0,
            diagnostics={"configured_stop_epoch": stop_epoch},
        )
