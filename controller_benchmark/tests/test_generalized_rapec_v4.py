from __future__ import annotations

import inspect
import json
import unittest
from pathlib import Path

from controller_benchmark.api import ControllerContext
from controller_benchmark.controllers.generalized_rapec_v4 import (
    GeneralizedRapecV4Controller,
)
from controller_benchmark.tests.test_generalized_rapec_v2 import observation, row


ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = (
    ROOT
    / "controller_benchmark/config/generalized-live-shadow-v4-offline-controllers.json"
)


def controller_parameters() -> dict:
    configuration = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    return next(
        item["controller_parameters"]
        for item in configuration["controllers"]
        if item["id"] == "rapec-g-v4"
    )


def context(task_type: str, scenario: str) -> ControllerContext:
    return ControllerContext(
        controller_id="rapec-g-v4-test",
        benchmark_version="rapec-g-v4-test",
        task_type=task_type,
        quality_metric="quality_score",
        scenario=scenario,
        max_epochs=100,
        parameters=controller_parameters(),
        metadata={"model_version": f"model-{task_type}"},
    )


def stop_epoch(controller_context: ControllerContext, qualities: list[float]):
    controller = GeneralizedRapecV4Controller(controller_context)
    history: list[dict] = []
    result = None
    for index, quality in enumerate(qualities, start=1):
        current = row(index, quality)
        decision = controller.evaluate(
            observation(controller_context, history, current)
        )
        history.append(current)
        if decision.stop:
            result = index
            break
    controller.close()
    return result


class GeneralizedRapecV4Tests(unittest.TestCase):
    def test_controller_source_has_no_task_dataset_or_model_profiles(self) -> None:
        source = inspect.getsource(GeneralizedRapecV4Controller)
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
        epochs = {
            stop_epoch(context(task, scenario), qualities)
            for task, scenario in (
                ("object_detection", "vision"),
                ("language_modeling", "text"),
                ("reinforcement_learning", "control"),
            )
        }
        self.assertEqual(len(epochs), 1)
        self.assertNotIn(None, epochs)

    def test_step_like_curve_is_recognized_as_burst_risk(self) -> None:
        controller = GeneralizedRapecV4Controller(
            context("unknown", "unknown")
        )
        qualities = [
            0.10,
            0.18,
            0.26,
            0.35,
            0.44,
            0.28,
            0.22,
            0.31,
            0.20,
            0.29,
            0.19,
            0.30,
            0.18,
            0.27,
            0.21,
            0.26,
            0.20,
            0.25,
            0.19,
            0.24,
            0.18,
        ]
        evidence = controller._burst_evidence(qualities, 0.03)
        controller.close()
        self.assertTrue(evidence.burst_risk)
        self.assertGreaterEqual(evidence.jump_to_gain_ratio, 1.5)

    def test_smooth_plateau_is_not_misclassified_as_burst(self) -> None:
        controller = GeneralizedRapecV4Controller(
            context("unknown", "unknown")
        )
        qualities = [0.10, 0.20, 0.30, 0.40] + [0.399] * 20
        evidence = controller._burst_evidence(qualities, 0.01)
        controller.close()
        self.assertFalse(evidence.burst_risk)


if __name__ == "__main__":
    unittest.main()
