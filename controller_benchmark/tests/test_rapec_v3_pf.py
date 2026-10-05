from __future__ import annotations

import inspect
import json
import unittest
from pathlib import Path

from controller_benchmark.api import ControllerContext, EpochObservation
from controller_benchmark.controllers.rapec_v3_pf import (
    ProfileFreeRapecV3Controller,
)
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


def _context(task_type: str, scenario: str, model: str) -> ControllerContext:
    return ControllerContext(
        controller_id="rapec-v3-pf-test",
        benchmark_version="rapec-v3-pf-test",
        task_type=task_type,
        quality_metric="quality_score",
        scenario=scenario,
        max_epochs=100,
        parameters={
            "horizon_epochs": 5,
            "min_fit_points": 10,
            "patience": 3,
            "trend_window": 10,
            "bootstrap_samples": 16,
            "knn_neighbors": 9,
            "min_epochs_floor": 12,
            "max_probability_gain_gt_threshold": 0.25,
            "max_uncertainty_to_gain_ratio": 4.0,
            "minimum_quality_floor": 0.0,
            "noise_multiplier": 1.25,
            "recent_gain_fraction": 0.35,
            "remaining_room_fraction": 0.015,
            "late_stage_relaxation": 0.55,
            "min_meaningful_gain": 0.0001,
            "utility_quantile": 0.25,
            "warmup_significance_z": 1.28,
            "warmup_low_progress_quantile": 0.40,
            "warmup_required_low_progress_windows": 1,
            "warmup_min_reference_points": 3,
            "warmup_no_learning_multiplier": 1.5,
            "warmup_confirmation_horizon_multipliers": [1],
        },
        metadata={"model_version": model, "pretrained": "must-not-matter"},
    )


def _observation(
    context: ControllerContext,
    history: list[dict],
    current: dict,
) -> EpochObservation:
    rows = [*history, current]
    qualities = [float(row["quality_score"]) for row in rows]
    epoch = int(current["epoch_index"])
    return EpochObservation(
        epoch=epoch,
        max_epochs=context.max_epochs,
        task_type=context.task_type,
        quality_metric=context.quality_metric,
        quality=qualities[-1],
        best_quality=max(qualities),
        delta_quality=(qualities[-1] - qualities[-2]) if len(qualities) > 1 else None,
        epoch_energy_wh=float(current["epoch_energy_wh"]),
        cumulative_energy_wh=sum(float(row["epoch_energy_wh"]) for row in rows),
        epoch_duration_seconds=float(current["duration_seconds"]),
        cumulative_duration_seconds=sum(float(row["duration_seconds"]) for row in rows),
        gpu_utilization_pct=float(current["gpu_util_avg_pct"]),
        history=tuple(history),
        raw_metrics=current,
    )


def _evaluate_curve(
    controller: ProfileFreeRapecV3Controller,
    context: ControllerContext,
    qualities: list[float],
):
    history: list[dict] = []
    decisions = []
    for epoch, quality in enumerate(qualities, start=1):
        current = _row(epoch, quality)
        decisions.append(controller.evaluate(_observation(context, history, current)))
        history.append(current)
    return decisions


class ProfileFreeRapecV3Tests(unittest.TestCase):
    def test_packaged_configuration_targets_profile_free_v3(self):
        config = json.loads(
            (
                ROOT
                / "controller_benchmark/config/rapec-v3-pf-controller.json"
            ).read_text(encoding="utf-8")
        )
        controller = config["controllers"][0]
        self.assertEqual(controller["id"], "rapec-v3-pf")
        self.assertEqual(
            controller["controller_plugin"],
            "controller_benchmark.controllers.rapec_v3_pf:ProfileFreeRapecV3Controller",
        )
        self.assertEqual(
            controller["controller_parameters"]["warmup_significance_z"],
            1.28,
        )

    def test_manifest_covers_all_nine_cases(self):
        manifest = load_manifest(
            ROOT
            / "controller_benchmark/config/benchmark-v3-nine-dataset-v3-pf.json"
        )
        self.assertEqual(
            sum(len(stage["cases"]) for stage in manifest["stages"]),
            9,
        )
        self.assertEqual(
            manifest["execution"]["experiment_name"],
            "controller-benchmark-v3-nine-dataset-v3-pf",
        )

    def test_only_profile_dependent_minimum_epoch_logic_is_replaced(self):
        inherited_methods = (
            "_dynamic_thresholds",
            "_predict_horizon",
            "_learning_state",
            "_minimum_quality",
            "_uncertainty_is_actionable",
        )
        for name in inherited_methods:
            self.assertIs(
                getattr(ProfileFreeRapecV3Controller, name),
                getattr(RiskAwarePredictiveEnergyController, name),
            )
        self.assertIsNot(
            ProfileFreeRapecV3Controller._adaptive_min_epochs,
            RiskAwarePredictiveEnergyController._adaptive_min_epochs,
        )

    def test_source_does_not_read_task_scenario_or_model_profiles(self):
        source = inspect.getsource(ProfileFreeRapecV3Controller)
        for forbidden in (
            "context.task_type",
            "context.scenario",
            "context.metadata",
            '"scratch"',
            '"pretrained"',
        ):
            self.assertNotIn(forbidden, source)

    def test_structural_floor_follows_v3_predictor_data_requirements(self):
        context = _context("object_detection", "anything", "anything")
        controller = ProfileFreeRapecV3Controller(context)
        observation = _observation(context, [], _row(1, 0.1))
        self.assertEqual(controller._structural_history_floor(observation), 24)

    def test_decisions_are_invariant_to_task_dataset_and_model_names(self):
        contexts = [
            _context("object_detection", "carpk-full-scratch", "yolov8n.yaml"),
            _context(
                "image_classification",
                "cifar-pretrained-profile",
                "resnet-profile",
            ),
            _context(
                "semantic_segmentation",
                "unseen-segmentation",
                "unseen-model",
            ),
        ]
        qualities = [
            min(0.72, 0.05 + 0.025 * epoch)
            if epoch <= 26
            else 0.72 + (0.0001 if epoch % 2 else 0.0)
            for epoch in range(1, 41)
        ]
        decision_runs = [
            _evaluate_curve(ProfileFreeRapecV3Controller(context), context, qualities)
            for context in contexts
        ]
        for epoch_decisions in zip(*decision_runs):
            first = epoch_decisions[0]
            for decision in epoch_decisions[1:]:
                self.assertEqual(decision.stop, first.stop)
                self.assertEqual(decision.reason, first.reason)
                self.assertEqual(decision.diagnostics, first.diagnostics)

    def test_active_learning_then_relative_plateau_releases_warmup(self):
        context = _context("unknown", "unknown", "unknown")
        qualities = [
            0.10 + 0.02 * epoch if epoch <= 24 else 0.58
            for epoch in range(1, 52)
        ]
        decisions = _evaluate_curve(
            ProfileFreeRapecV3Controller(context), context, qualities
        )
        self.assertEqual(
            decisions[23].diagnostics["rapec_pf_seen_significant_learning"],
            1,
        )
        self.assertTrue(
            any(
                decision.diagnostics["rapec_pf_warmup_released"] == 1
                for decision in decisions[24:]
            )
        )

    def test_no_learning_trace_uses_profile_free_fallback(self):
        context = _context("unknown", "unknown", "unknown")
        decisions = _evaluate_curve(
            ProfileFreeRapecV3Controller(context), context, [0.1] * 36
        )
        final = decisions[-1]
        self.assertEqual(final.diagnostics["rapec_pf_no_learning_release_epoch"], 36)
        self.assertEqual(final.diagnostics["rapec_pf_no_learning_fallback"], 1)
        self.assertEqual(final.diagnostics["rapec_pf_warmup_released"], 1)


if __name__ == "__main__":
    unittest.main()
