from __future__ import annotations

from dataclasses import replace

from controller_benchmark.api import ControllerDecision, EpochObservation

from .rapec_v3_pf import ProfileFreeRapecV3Controller


class GeneralizedRapecEnergy5Controller(ProfileFreeRapecV3Controller):
    """Task-adapter form of RAPEC-v3-PF with identical decisions.

    Forecasts, probabilities, utilities, and confirmation logic are inherited
    without modification. The concrete thresholds come from the controller
    configuration. Generalisation is provided outside the decision rule by
    mapping each task metric to a maximized quality score in [0, 1] and by
    treating an epoch as a generic decision checkpoint.
    """

    def evaluate(self, observation: EpochObservation) -> ControllerDecision:
        decision = super().evaluate(observation)
        return replace(
            decision,
            diagnostics={
                **decision.diagnostics,
                "rapec_g_task_adapter_enabled": 1,
                "rapec_g_decision_rule_identical_to_rapec_v3_pf": 1,
                "rapec_g_uses_task_dataset_or_model_profile": 0,
                "rapec_g_decision_checkpoint": observation.checkpoint,
                "rapec_g_progress_fraction": observation.progress_fraction,
            },
        )
