from __future__ import annotations

import inspect
import json
import unittest
from pathlib import Path

from controller_benchmark.api import ControllerContext
from controller_benchmark.controllers.generalized_rapec_v5 import (
    GeneralizedRapecV5Controller,
)
from controller_benchmark.tests.test_generalized_rapec_v2 import observation, row


ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = (
    ROOT
    / "controller_benchmark/config/generalized-live-shadow-v5-offline-controllers.json"
)


def controller_parameters() -> dict:
    configuration = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    return next(
        item["controller_parameters"]
        for item in configuration["controllers"]
        if item["id"] == "rapec-g-v5"
    )


def context(task_type: str, scenario: str) -> ControllerContext:
    return ControllerContext(
        controller_id="rapec-g-v5-test",
        benchmark_version="rapec-g-v5-test",
        task_type=task_type,
        quality_metric="quality_score",
        scenario=scenario,
        max_epochs=100,
        parameters=controller_parameters(),
        metadata={"model_version": f"model-{task_type}"},
    )


def stop_epoch(controller_context: ControllerContext, qualities: list[float]):
    controller = GeneralizedRapecV5Controller(controller_context)
    history: list[dict] = []
    result = None
    diagnostics = {}
    for index, quality in enumerate(qualities, start=1):
        current = row(index, quality)
        decision = controller.evaluate(
            observation(controller_context, history, current)
        )
        diagnostics = decision.diagnostics
        history.append(current)
        if decision.stop:
            result = index
            break
    controller.close()
    return result, diagnostics


class GeneralizedRapecV5Tests(unittest.TestCase):
    def test_readiness_gate_source_has_no_task_or_dataset_profiles(self) -> None:
        source = inspect.getsource(GeneralizedRapecV5Controller)
        for forbidden in (
            "context.task_type",
            "context.scenario",
            "context.metadata",
            '"scratch"',
            '"pretrained"',
            '"classification"',
            '"segmentation"',
            '"detection"',
        ):
            self.assertNotIn(forbidden, source)

    def test_same_curve_is_independent_of_task_and_scenario_names(self) -> None:
        qualities = [
            0.10 + 0.03 * epoch if epoch <= 8 else 0.34
            for epoch in range(1, 55)
        ]
        outcomes = {
            stop_epoch(context(task, scenario), qualities)[0]
            for task, scenario in (
                ("object_detection", "vision"),
                ("language_modeling", "text"),
                ("reinforcement_learning", "control"),
            )
        }
        self.assertEqual(len(outcomes), 1)
        self.assertNotIn(None, outcomes)

    def test_accepted_stop_contains_current_run_readiness_evidence(self) -> None:
        qualities = [
            0.10 + 0.03 * epoch if epoch <= 8 else 0.34
            for epoch in range(1, 55)
        ]
        epoch, diagnostics = stop_epoch(context("unknown", "unknown"), qualities)
        self.assertIsNotNone(epoch)
        self.assertEqual(diagnostics["rapec_g_v5_task_profile_used"], 0)
        self.assertEqual(diagnostics["rapec_g_v5_current_run_only"], 1)
        self.assertEqual(diagnostics["rapec_g_v5_mature_plateau"], 1)
        self.assertEqual(diagnostics["rapec_g_v5_accepted_stop"], 1)


if __name__ == "__main__":
    unittest.main()
