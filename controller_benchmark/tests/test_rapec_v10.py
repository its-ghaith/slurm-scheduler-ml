from __future__ import annotations

import json
import unittest
from pathlib import Path

from controller_benchmark.api import ControllerContext, EpochObservation
from controller_benchmark.controllers.rapec_v10 import RAPECV10Controller
from controller_benchmark.controllers.risk_aware_predictive_energy import (
    RiskAwarePredictiveEnergyController,
)
from controller_benchmark.manifest import load_manifest


ROOT = Path(__file__).resolve().parents[2]


def _row(epoch: int, quality: float) -> dict:
    return {
        "epoch": epoch,
        "epoch_index": epoch,
        "quality_score": quality,
        "epoch_energy_wh": 1.0,
        "duration_seconds": 2.0,
        "gpu_util_avg_pct": 60.0,
        "gpu_power_avg_w": 150.0,
        "gpu_mem_used_avg_mb": 2048.0,
        "learning_rate": 0.001,
        "gradient_norm": 0.1,
        "train_loss": max(0.01, 1.0 - quality),
        "model_parameter_count": 3_000_000,
        "model_flops": 8_000_000_000,
    }


def _context(task_type: str, scenario: str) -> ControllerContext:
    return ControllerContext(
        controller_id="rapec-v10-test",
        benchmark_version="v10-test",
        task_type=task_type,
        quality_metric="quality_score",
        scenario=scenario,
        max_epochs=100,
        parameters={
            "horizon_epochs": 3,
            "min_fit_points": 5,
            "patience": 2,
            "trend_window": 6,
            "bootstrap_samples": 8,
            "min_epochs_floor": 6,
            "warmup_required_stable_windows": 2,
        },
        metadata={"model_version": "profile-name-must-not-matter"},
    )


def _observation(
    context: ControllerContext,
    history: list[dict],
    current: dict,
) -> EpochObservation:
    qualities = [float(row["quality_score"]) for row in [*history, current]]
    epoch = int(current["epoch_index"])
    return EpochObservation(
        epoch=epoch,
        max_epochs=context.max_epochs,
        task_type=context.task_type,
        quality_metric=context.quality_metric,
        quality=qualities[-1],
        best_quality=max(qualities),
        delta_quality=(qualities[-1] - qualities[-2]) if len(qualities) > 1 else None,
        epoch_energy_wh=1.0,
        cumulative_energy_wh=float(epoch),
        epoch_duration_seconds=2.0,
        cumulative_duration_seconds=float(epoch * 2),
        gpu_utilization_pct=60.0,
        history=tuple(history),
        raw_metrics=current,
    )


class RapecV10Tests(unittest.TestCase):
    def test_packaged_controller_configuration_targets_v10(self):
        config = json.loads(
            (ROOT / "controller_benchmark/config/rapec-v10-controller.json").read_text(
                encoding="utf-8"
            )
        )
        controller = config["controllers"][0]
        self.assertEqual(controller["id"], "rapec-v10")
        self.assertEqual(
            controller["controller_plugin"],
            "controller_benchmark.controllers.rapec_v10:RAPECV10Controller",
        )
        self.assertEqual(
            controller["controller_parameters"]["warmup_required_stable_windows"],
            2,
        )

    def test_v10_manifest_uses_single_controller_analysis(self):
        manifest = load_manifest(
            ROOT / "controller_benchmark/config/benchmark-v3-nine-dataset-v10.json"
        )
        self.assertEqual(
            manifest["benchmark_version"],
            "controller-benchmark-v3-nine-dataset-fresh",
        )
        self.assertEqual(
            manifest["execution"]["analysis_module"],
            "controller_benchmark.analyze",
        )
        self.assertEqual(manifest["execution"]["dashboard_files"], [])
        self.assertEqual(
            sum(len(stage["cases"]) for stage in manifest["stages"]),
            9,
        )

    def test_preserves_rapec_v3_quality_guard(self):
        self.assertIs(
            RAPECV10Controller._minimum_quality,
            RiskAwarePredictiveEnergyController._minimum_quality,
        )

    def test_decision_is_independent_of_task_scenario_and_model_names(self):
        contexts = [
            _context("object_detection", "carpk-full-scratch"),
            _context("semantic_segmentation", "unknown-pretrained"),
            _context("image_classification", "unseen-task"),
        ]
        histories = [
            [_row(epoch, min(0.65, 0.10 + 0.03 * epoch)) for epoch in range(1, 25)]
            for _ in contexts
        ]
        decisions = []
        for context, history in zip(contexts, histories):
            controller = RAPECV10Controller(context)
            current = _row(25, 0.65)
            decisions.append(controller.evaluate(_observation(context, history, current)))

        first = decisions[0]
        for decision in decisions[1:]:
            self.assertEqual(decision.stop, first.stop)
            self.assertEqual(
                decision.diagnostics["rapec_adaptive_min_epochs"],
                first.diagnostics["rapec_adaptive_min_epochs"],
            )
            self.assertEqual(
                decision.diagnostics["rapec10_warmup_released"],
                first.diagnostics["rapec10_warmup_released"],
            )
        self.assertEqual(first.diagnostics["rapec10_uses_task_or_scenario_profile"], 0)

    def test_active_learning_keeps_warmup_closed(self):
        context = _context("object_detection", "scratch")
        controller = RAPECV10Controller(context)
        history = [_row(epoch, 0.10 + 0.01 * epoch) for epoch in range(1, 20)]
        current = _row(20, 0.30)
        decision = controller.evaluate(_observation(context, history, current))

        self.assertFalse(decision.stop)
        self.assertEqual(decision.diagnostics["rapec10_learning_is_active"], 1)
        self.assertEqual(decision.diagnostics["rapec10_warmup_released"], 0)
        self.assertGreater(
            decision.diagnostics["rapec_adaptive_min_epochs"],
            current["epoch_index"],
        )

    def test_stable_plateau_releases_warmup_after_online_confirmation(self):
        context = _context("semantic_segmentation", "scratch")
        controller = RAPECV10Controller(context)
        history = [_row(epoch, 0.60) for epoch in range(1, 20)]

        first = _row(20, 0.60)
        first_decision = controller.evaluate(_observation(context, history, first))
        history.append(first)
        second = _row(21, 0.60)
        second_decision = controller.evaluate(_observation(context, history, second))

        self.assertEqual(first_decision.diagnostics["rapec10_warmup_released"], 0)
        self.assertEqual(second_decision.diagnostics["rapec10_warmup_released"], 1)
        self.assertEqual(
            second_decision.diagnostics["rapec_adaptive_min_epochs"],
            second_decision.diagnostics["rapec10_online_history_floor"],
        )
        self.assertAlmostEqual(
            second_decision.diagnostics["rapec10_dynamic_min_epoch_fraction"],
            second_decision.diagnostics["rapec10_online_history_floor"] / 100.0,
        )


if __name__ == "__main__":
    unittest.main()
