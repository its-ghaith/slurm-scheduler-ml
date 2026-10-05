from __future__ import annotations

import inspect
import json
import unittest
from pathlib import Path

from controller_benchmark.api import ControllerContext
from controller_benchmark.controllers.generalized_rapec_v2 import (
    GeneralizedRapecV2Controller,
)
from controller_benchmark.controllers.generalized_rapec_v3 import (
    GeneralizedRapecV3Controller,
)
from controller_benchmark.tests.test_generalized_rapec_v2 import observation, row


ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = (
    ROOT
    / "controller_benchmark/config/generalized-live-shadow-v3-small-controllers.json"
)


def controller_parameters(controller_id: str = "rapec-g-v3") -> dict:
    configuration = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    return next(
        item["controller_parameters"]
        for item in configuration["controllers"]
        if item["id"] == controller_id
    )


def context(
    task_type: str = "unknown",
    scenario: str = "unknown",
    model_version: str = "unknown",
    *,
    controller_id: str = "rapec-g-v3",
) -> ControllerContext:
    return ControllerContext(
        controller_id=f"{controller_id}-test",
        benchmark_version="rapec-g-v3-test",
        task_type=task_type,
        quality_metric="quality_score",
        scenario=scenario,
        max_epochs=100,
        parameters=controller_parameters(controller_id),
        metadata={"model_version": model_version},
    )


def evaluate_curve(controller_type, controller_context, qualities: list[float]):
    controller = controller_type(controller_context)
    history: list[dict] = []
    decisions = []
    for index, quality in enumerate(qualities, start=1):
        current = row(index, quality)
        decisions.append(
            controller.evaluate(
                observation(controller_context, history, current)
            )
        )
        history.append(current)
    controller.close()
    return decisions


class GeneralizedRapecV3Tests(unittest.TestCase):
    def test_controller_source_has_no_task_dataset_or_model_profiles(self) -> None:
        source = inspect.getsource(GeneralizedRapecV3Controller)
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

    def test_same_stationary_curve_is_task_name_independent(self) -> None:
        qualities = [
            0.10 + 0.03 * epoch if epoch <= 8 else 0.34
            for epoch in range(1, 50)
        ]
        contexts = (
            context("object_detection", "curve-a", "model-a"),
            context("language_modeling", "curve-b", "model-b"),
            context("reinforcement_learning", "curve-c", "model-c"),
        )
        traces = [
            evaluate_curve(GeneralizedRapecV3Controller, item, qualities)
            for item in contexts
        ]
        stop_epochs = []
        for decisions in traces:
            stops = [
                index
                for index, decision in enumerate(decisions, start=1)
                if decision.stop
            ]
            self.assertTrue(stops)
            stop_epochs.append(stops[0])
        self.assertEqual(len(set(stop_epochs)), 1)

    def test_stationarity_route_releases_a_flat_curve_earlier_than_v2(self) -> None:
        qualities = [
            0.12 + 0.04 * epoch if epoch <= 8 else 0.44
            for epoch in range(1, 55)
        ]
        v2 = evaluate_curve(
            GeneralizedRapecV2Controller,
            context(controller_id="rapec-g-v2"),
            qualities,
        )
        v3 = evaluate_curve(GeneralizedRapecV3Controller, context(), qualities)
        v2_stop = next(
            index for index, decision in enumerate(v2, start=1) if decision.stop
        )
        v3_stop = next(
            index for index, decision in enumerate(v3, start=1) if decision.stop
        )
        self.assertLessEqual(v3_stop, v2_stop)
        self.assertEqual(
            v3[v3_stop - 1].diagnostics["rapec_g_v3_stationarity_stop"],
            1,
        )

    def test_unusually_volatile_delayed_learning_is_guarded(self) -> None:
        qualities = []
        for epoch in range(1, 66):
            if epoch <= 18:
                quality = 0.10 + 0.015 * epoch
            elif epoch <= 50:
                quality = 0.25 + (0.08 if epoch % 2 else -0.08)
            else:
                quality = 0.38 + 0.012 * (epoch - 50)
            qualities.append(max(0.0, min(1.0, quality)))
        decisions = evaluate_curve(
            GeneralizedRapecV3Controller,
            context(),
            qualities,
        )
        self.assertFalse(
            any(
                decision.diagnostics["rapec_g_v3_stationarity_stop"]
                for decision in decisions[:50]
            )
        )


if __name__ == "__main__":
    unittest.main()
