from __future__ import annotations

from controller_benchmark.api import Controller, ControllerDecision, EpochObservation


class ExampleController(Controller):
    """Small authoring example; replace only the decision rule for a new study."""

    def evaluate(self, observation: EpochObservation) -> ControllerDecision:
        minimum_epochs = int(self.context.parameters.get("minimum_epochs", 30))
        minimum_gain = float(self.context.parameters.get("minimum_gain", 0.0005))
        gain = observation.delta_quality
        stop = observation.epoch >= minimum_epochs and gain is not None and gain < minimum_gain
        return ControllerDecision(
            stop=stop,
            reason=(
                f"example_low_gain_stop: epoch={observation.epoch}, "
                f"delta_quality={gain}, threshold={minimum_gain}"
            ),
            confidence=1.0 if stop else 0.0,
            diagnostics={
                "delta_quality": gain,
                "minimum_gain": minimum_gain,
                "minimum_epochs": minimum_epochs,
            },
        )
