from __future__ import annotations

import inspect
import json
import unittest
from pathlib import Path

from controller_benchmark.api import ControllerContext, EpochObservation
from controller_benchmark.controllers.generalized_rapec_v2 import (
    GeneralizedRapecV2Controller,
)
from controller_benchmark.live_shadow_plan import load_shadow_configuration
from controller_benchmark.manifest import load_manifest
from controller_benchmark.rapec_g_v2_small_manifest import (
    SELECTED_CASE_IDS,
    build_small_manifest,
)


ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = (
    ROOT
    / "controller_benchmark/config/generalized-live-shadow-v2-small-controllers.json"
)


def controller_parameters() -> dict:
    configuration = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    return next(
        item["controller_parameters"]
        for item in configuration["controllers"]
        if item["id"] == "rapec-g-v2"
    )


def context(
    task_type: str = "unknown",
    scenario: str = "unknown",
    model_version: str = "unknown",
) -> ControllerContext:
    return ControllerContext(
        controller_id="rapec-g-v2-test",
        benchmark_version="rapec-g-v2-test",
        task_type=task_type,
        quality_metric="quality_score",
        scenario=scenario,
        max_epochs=100,
        parameters=controller_parameters(),
        metadata={"model_version": model_version},
    )


def row(
    epoch: int,
    quality: float,
    *,
    train_loss: float | None = None,
    gradient_norm: float = 0.1,
) -> dict:
    return {
        "epoch": epoch,
        "epoch_index": epoch,
        "quality_score": quality,
        "epoch_energy_wh": 1.0,
        "duration_seconds": 2.0,
        "gpu_util_avg_pct": 70.0,
        "gpu_power_avg_w": 180.0,
        "gpu_mem_used_avg_mb": 4096.0,
        "learning_rate": max(1e-5, 0.001 * (1.0 - epoch / 120.0)),
        "gradient_norm": gradient_norm,
        "train_loss": max(0.01, 1.0 - quality) if train_loss is None else train_loss,
        "model_parameter_count": 4_000_000,
        "model_flops": 9_000_000_000,
    }


def observation(
    controller_context: ControllerContext,
    history: list[dict],
    current: dict,
) -> EpochObservation:
    rows = [*history, current]
    qualities = [float(item["quality_score"]) for item in rows]
    return EpochObservation(
        epoch=int(current["epoch"]),
        max_epochs=controller_context.max_epochs,
        task_type=controller_context.task_type,
        quality_metric=controller_context.quality_metric,
        quality=qualities[-1],
        best_quality=max(qualities),
        delta_quality=qualities[-1] - qualities[-2] if len(qualities) > 1 else None,
        epoch_energy_wh=float(current["epoch_energy_wh"]),
        cumulative_energy_wh=sum(float(item["epoch_energy_wh"]) for item in rows),
        epoch_duration_seconds=float(current["duration_seconds"]),
        cumulative_duration_seconds=sum(float(item["duration_seconds"]) for item in rows),
        gpu_utilization_pct=float(current["gpu_util_avg_pct"]),
        history=tuple(history),
        raw_metrics=current,
    )


def evaluate_curve(
    controller_context: ControllerContext,
    qualities: list[float],
    *,
    loss_curve: list[float] | None = None,
) -> list:
    controller = GeneralizedRapecV2Controller(controller_context)
    history: list[dict] = []
    decisions = []
    for index, quality in enumerate(qualities, start=1):
        current = row(
            index,
            quality,
            train_loss=(loss_curve[index - 1] if loss_curve else None),
        )
        decision = controller.evaluate(
            observation(controller_context, history, current)
        )
        decisions.append(decision)
        history.append(current)
    return decisions


class GeneralizedRapecV2Tests(unittest.TestCase):
    def test_controller_source_has_no_task_dataset_or_model_profiles(self) -> None:
        source = inspect.getsource(GeneralizedRapecV2Controller)
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

    def test_same_curve_has_identical_decisions_across_task_names(self) -> None:
        qualities = [
            0.10 + 0.02 * epoch if epoch <= 20 else 0.50
            for epoch in range(1, 55)
        ]
        contexts = (
            context("object_detection", "curve-a", "model-a"),
            context("language_modeling", "curve-b", "model-b"),
            context("reinforcement_learning", "curve-c", "model-c"),
        )
        traces = [evaluate_curve(item, qualities) for item in contexts]
        for decisions in zip(*traces):
            self.assertEqual({decision.stop for decision in decisions}, {decisions[0].stop})
            self.assertEqual({decision.reason for decision in decisions}, {decisions[0].reason})
            self.assertEqual(
                {json.dumps(decision.diagnostics, sort_keys=True) for decision in decisions},
                {json.dumps(decisions[0].diagnostics, sort_keys=True)},
            )

    def test_early_plateau_stops_before_full_budget(self) -> None:
        qualities = [
            0.08 + 0.021 * epoch if epoch <= 20 else 0.50
            for epoch in range(1, 61)
        ]
        decisions = evaluate_curve(context(), qualities)
        stop_epochs = [
            index
            for index, decision in enumerate(decisions, start=1)
            if decision.stop
        ]
        self.assertTrue(stop_epochs)
        self.assertLess(stop_epochs[0], 50)
        self.assertEqual(
            decisions[stop_epochs[0] - 1].diagnostics["rapec_g_v2_regime_code"],
            GeneralizedRapecV2Controller.REGIME_CODES["early_plateau"],
        )

    def test_sustained_slow_learning_is_not_stopped(self) -> None:
        qualities = [min(0.85, 0.08 + 0.006 * epoch) for epoch in range(1, 71)]
        decisions = evaluate_curve(context(), qualities)
        self.assertFalse(any(decision.stop for decision in decisions))
        self.assertEqual(
            decisions[-1].diagnostics["rapec_g_v2_late_learning_guard"], 1
        )

    def test_loss_and_gradient_recovery_guard_delayed_quality_gain(self) -> None:
        qualities = []
        for epoch in range(1, 51):
            if epoch <= 20:
                quality = 0.10 + 0.015 * epoch
            elif epoch <= 36:
                quality = 0.40
            else:
                quality = 0.40 + 0.012 * (epoch - 36)
            qualities.append(quality)
        loss_curve = [1.2 - 0.012 * epoch for epoch in range(1, 51)]
        decisions = evaluate_curve(context(), qualities, loss_curve=loss_curve)
        self.assertFalse(any(decision.stop for decision in decisions[:36]))

    def test_small_manifest_and_controller_configuration_are_loadable(self) -> None:
        vision = json.loads(
            (
                ROOT / "controller_benchmark/config/benchmark-v3-nine-dataset-fresh.json"
            ).read_text(encoding="utf-8")
        )
        classification = json.loads(
            (
                ROOT
                / "controller_benchmark/config/benchmark-live-shadow-classification-macro-f1.json"
            ).read_text(encoding="utf-8")
        )
        manifest = build_small_manifest(vision, classification)
        cases = [case for stage in manifest["stages"] for case in stage["cases"]]
        self.assertEqual([case["id"] for case in cases], list(SELECTED_CASE_IDS))
        self.assertEqual(len(cases), 6)
        self.assertTrue(all(not case.get("pretrained") for case in cases))
        with self.subTest("configuration plugin contract"):
            temporary_manifest = ROOT / "results/controller-benchmark/generated-manifests/rapec-g-v2-test.json"
            temporary_manifest.parent.mkdir(parents=True, exist_ok=True)
            temporary_manifest.write_text(
                json.dumps(manifest, indent=2), encoding="utf-8"
            )
            try:
                loaded = load_manifest(temporary_manifest)
                configuration = load_shadow_configuration(CONFIG_PATH, loaded)
                self.assertEqual(
                    [item["id"] for item in configuration["controllers"]],
                    ["standard-es-10", "rapec-g-v1", "rapec-g-v2"],
                )
            finally:
                temporary_manifest.unlink(missing_ok=True)


if __name__ == "__main__":
    unittest.main()
