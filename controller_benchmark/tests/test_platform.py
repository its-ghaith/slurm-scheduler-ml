from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from controller_benchmark.analysis import analyze_snapshot, write_analysis, write_prometheus
from controller_benchmark.api import ControllerContext
from controller_benchmark.dashboard.build_dashboard import main as build_dashboard_main
from controller_benchmark.loader import load_controller
from controller_benchmark.manifest import load_manifest
from controller_benchmark.planner import build_plan
from controller_benchmark.runtime import ControllerPluginRuntime
from controller_benchmark.api import EpochObservation
from controller_benchmark.controllers.bayesian_guarded_rapec_v3 import (
    BayesianGuardedRapecV3Controller,
)
from controller_benchmark.controllers.lcpfn_quality_baseline import _probability_above_from_quantiles
from controller_benchmark.controllers.online_bayesian_rapec import OnlineBayesianRapecController
from controller_benchmark.controllers.online_multi_horizon_pareto import OnlineMultiHorizonParetoController
from controller_benchmark.controllers.probabilistic_energy_pareto import ProbabilisticEnergyParetoController
from controller_benchmark.controllers.risk_aware_predictive_energy import RiskAwarePredictiveEnergyController
from controller_benchmark.controllers.self_calibrating_dynamic_threshold import SelfCalibratingDynamicThresholdController
from controller_benchmark.controllers.task_aware_dynamic_threshold import TaskAwareDynamicThresholdController
from controller_benchmark.runners.common import EpochTelemetry


ROOT = Path(__file__).resolve().parents[2]


class ControllerBenchmarkTests(unittest.TestCase):
    def test_dashboard_combines_task_independent_and_task_specific_views(self):
        with tempfile.TemporaryDirectory() as temp:
            output = Path(temp) / "dashboard.json"
            with mock.patch("sys.argv", ["build-dashboard", "--output", str(output)]):
                build_dashboard_main()
            dashboard = json.loads(output.read_text(encoding="utf-8"))
            variables = {item["name"]: item for item in dashboard["templating"]["list"]}
            self.assertEqual(list(variables), ["benchmark_run_id", "task_types", "stages", "cases", "phases"])
            self.assertIn('task_type=~"$task_types"', variables["stages"]["definition"])
            self.assertIn('task_type=~"$task_types"', variables["cases"]["definition"])
            self.assertIn('case_id=~"$cases"', variables["phases"]["definition"])
            panels = {panel["title"]: panel for panel in dashboard["panels"]}
            self.assertIn("Executive Summary", panels)
            self.assertIn("Task-Independent Quality Q_t", panels)
            self.assertIn("Object Detection: mAP50, mAP50-95, Precision, and Recall", panels)
            self.assertIn("Image Classification: Accuracy and Macro-F1", panels)
            self.assertIn("Semantic Segmentation: mIoU and Dice", panels)
            self.assertIn("Task-Independent Energy vs Normalized Quality by Job", panels)
            self.assertIn("Object Detection: GPU Energy vs mAP50-95 by Job", panels)
            self.assertIn("Image Classification: GPU Energy vs Macro-F1 by Job", panels)
            self.assertIn("Semantic Segmentation: GPU Energy vs mIoU by Job", panels)
            self.assertIn("SLURM vs CodeCarbon Training GPU Energy by Job (Mirrored Wh)", panels)
            self.assertIn("SLURM Phase: Energy, Cost, CO2, Quality, and Epoch Count by Job/Case", panels)
            self.assertIn("Full100 vs Candidate Duration by Case (s)", panels)
            self.assertIn("CodeCarbon GPU Energy by Case (Wh)", panels)
            self.assertIn("Epochs Saved by Case", panels)
            self.assertIn("Duration Saving by Case (%)", panels)
            self.assertIn("SLURM Global Total GPU Energy (Wh)", panels)
            self.assertIn("CodeCarbon Global Total Energy (Wh)", panels)
            self.assertIn("Phase Duration by Job/Case (s)", panels)
            self.assertIn("SLURM Phase Tabs", panels)
            self.assertIn("PEP Expected Gain and Probability by Epoch", panels)
            self.assertIn("PEP Expected Energy and Utility by Epoch", panels)
            self.assertIn("PEP Feature Coverage and Stop Evidence", panels)
            self.assertIn("RAPEC-v3 Expected Gain and Dynamic Threshold by Epoch", panels)
            self.assertIn("RAPEC-v3 Energy Utility and Adaptive Threshold by Epoch", panels)
            self.assertIn("RAPEC-v3 Learning State and Stop Evidence by Epoch", panels)
            self.assertIn("RAPEC-v4 Task-Aware Adapter Settings by Epoch", panels)
            self.assertIn("RAPEC-v4 Dynamic Risk and Utility Thresholds by Epoch", panels)
            self.assertIn("RAPEC-v5 Self-Calibration Settings by Epoch", panels)
            self.assertIn("RAPEC-v5 Curve Risk, Stability, and Energy Variability by Epoch", panels)
            self.assertIn("RAPEC-v6 Online Multi-Horizon Pareto Diagnostics", panels)
            self.assertIn("RAPEC-v6 Multi-Horizon Expected Quality Gain by Epoch", panels)
            self.assertIn("RAPEC-v6 Risk, Recovery, and Pareto Stop Evidence", panels)
            self.assertIn("RAPEC-v6 Predicted Training Energy by Horizon (Wh)", panels)
            self.assertIn("RAPEC-v6 Online Uncertainty Calibration", panels)
            self.assertIn("RAPEC-v7 Online Bayesian Diagnostics", panels)
            self.assertIn("RAPEC-v7 Posterior Quality Gain by Horizon", panels)
            self.assertIn("RAPEC-v7 Posterior Probability and Dynamic Stop Evidence", panels)
            self.assertIn("RAPEC-v7 Posterior Training Energy by Horizon (Wh)", panels)
            self.assertIn("RAPEC-v7 Posterior Training Duration by Horizon (s)", panels)
            self.assertIn("RAPEC-v7 Bayesian Model Weights and Uncertainty", panels)
            self.assertIn("RAPEC-v8 Bayesian-v3 Diagnostics", panels)
            self.assertIn("RAPEC-v8 v3 and Bayesian Stop Agreement", panels)
            self.assertIn("RAPEC-v8 Posterior Gain and Dynamic Threshold", panels)
            self.assertIn("RAPEC-v8 Posterior Energy Utility", panels)
            self.assertIn("RAPEC-v9 Risk-Constrained Bayesian Diagnostics", panels)
            self.assertIn("RAPEC-v9 Probability of a Relevant Future Quality Gain", panels)
            self.assertIn("RAPEC-v9 Dynamic Quality Equivalence and Predicted Gain", panels)
            self.assertIn("RAPEC-v9 Energy Utility and Sequential Stop Evidence", panels)
            self.assertIn("RAPEC-v9 Readiness, Constraints, and Regime Guards", panels)
            self.assertIn("RAPEC-v9 Predicted Training Energy by Horizon (Wh)", panels)
            self.assertIn("RAPEC-v10 Task-Independent Dynamic Warm-up", panels)
            self.assertIn("RAPEC-v10 Online Signal and Noise Calibration", panels)
            self.assertIn("RAPEC-v10 Dynamic Minimum Epoch and Release State", panels)
            self.assertIn(
                'task_type=~"$task_types"',
                panels["Full100 vs Candidate GPU Energy"]["targets"][0]["expr"],
            )

    def test_epoch_telemetry_ignores_duplicate_final_callback(self):
        spec = {
            "controller_mode": "none",
            "quality_metric": "map50_95",
            "strategy": "full100",
            "controller_id": "full100",
            "benchmark_version": "v1",
            "benchmark_run_id": "run",
            "benchmark_case_id": "case",
            "benchmark_stage": "stage",
            "task_type": "object_detection",
            "scenario": "test",
            "training_seed": 0,
            "execution": {"split_seed": 0, "cache_policy": "ram"},
        }
        with tempfile.TemporaryDirectory() as temp:
            telemetry = EpochTelemetry(spec, Path(temp))
            telemetry.history.append({"epoch": 2, "should_stop": 1})
            self.assertTrue(telemetry.finish_epoch(2, 0.5))
            self.assertEqual(len(telemetry.history), 1)

    def test_manifest_builds_shared_seed_baseline_matrix(self):
        manifest = load_manifest(ROOT / "controller_benchmark" / "config" / "benchmark-v2-shared-seed.json")
        plan = build_plan(
            manifest,
            controller_id="fixed77",
            controller_plugin="controller_benchmark.controllers.fixed_epoch:FixedEpochController",
            controller_parameters={"stop_epoch": 77},
        )
        self.assertEqual(len(plan["runs"]), 26)
        self.assertEqual(plan["planned_baseline_runs"], 11)
        self.assertEqual(plan["baseline_references"]["carpk-stability-seed-4"], "carpk-stability-seed-0")
        self.assertEqual(len({row["benchmark_case_id"] for row in plan["runs"]}), 15)
        self.assertEqual(plan["blocked_stages"], [])
        self.assertEqual(
            {row["runner"] for row in plan["runs"]},
            {"carpk_yolo", "vedai_yolo", "carpk_task_generalisation"},
        )
        self.assertEqual({row["quality_metric"] for row in plan["runs"]}, {"quality_score"})
        self.assertEqual(
            {row["task_type"] for row in plan["runs"]},
            {"object_detection", "image_classification", "semantic_segmentation"},
        )

    def test_complete_baseline_cache_plans_only_candidate_runs(self):
        manifest = load_manifest(ROOT / "controller_benchmark" / "config" / "benchmark-v2-shared-seed.json")
        case_ids = {
            case["id"]
            for stage in manifest["stages"]
            if stage.get("enabled")
            for case in stage["cases"]
        }
        plan = build_plan(
            manifest,
            controller_id="candidate",
            controller_plugin="controller_benchmark.controllers.fixed_epoch:FixedEpochController",
            controller_parameters={},
            baseline_catalog={"catalog_path": "catalog.json", "entries": {case_id: {} for case_id in case_ids}},
        )
        self.assertEqual(len(plan["runs"]), 15)
        self.assertEqual(plan["planned_baseline_runs"], 0)
        self.assertEqual(plan["planned_candidate_runs"], 15)

    def test_loader_rejects_non_controller_class(self):
        context = ControllerContext("bad", "v1", "task", "quality", "scenario", 10)
        with self.assertRaises(TypeError):
            load_controller("controller_benchmark.api:EpochObservation", context)

    def test_plan_rejects_baseline_as_controller_id(self):
        manifest = load_manifest(ROOT / "controller_benchmark" / "config" / "benchmark-v2-shared-seed.json")
        with self.assertRaises(ValueError):
            build_plan(
                manifest,
                controller_id="full100",
                controller_plugin="controller_benchmark.controllers.fixed_epoch:FixedEpochController",
                controller_parameters={},
            )

    def test_plugin_runtime_stops_at_configured_epoch(self):
        runtime = ControllerPluginRuntime(
            plugin_path="controller_benchmark.controllers.fixed_epoch:FixedEpochController",
            parameters_json=json.dumps({"stop_epoch": 2}),
            controller_id="fixed2",
            benchmark_version="v1",
            task_type="object_detection",
            quality_metric="map50_95",
            scenario="test",
            max_epochs=10,
        )
        first = {"epoch": 1, "epoch_index": 1, "map50_95": 0.5, "best_map50_95": 0.5, "total_energy_kwh": 0.001, "cumulative_total_energy_kwh": 0.001, "duration_seconds": 1.0}
        stop, _, _ = runtime.evaluate(first, [])
        self.assertFalse(stop)
        second = {"epoch": 2, "epoch_index": 2, "map50_95": 0.6, "best_map50_95": 0.6, "total_energy_kwh": 0.001, "cumulative_total_energy_kwh": 0.002, "duration_seconds": 1.0}
        stop, reason, diagnostics = runtime.evaluate(second, [first])
        self.assertTrue(stop)
        self.assertIn("fixed_epoch_stop", reason)
        self.assertEqual(diagnostics["controller_plugin_configured_stop_epoch"], 2)

    def test_runtime_exposes_arbitrary_numeric_diagnostics(self):
        runtime = ControllerPluginRuntime(
            plugin_path="controller_benchmark.controllers.example_controller:ExampleController",
            parameters_json=json.dumps({"minimum_epochs": 1, "minimum_gain": 0.1}),
            controller_id="example",
            benchmark_version="v1",
            task_type="object_detection",
            quality_metric="map50_95",
            scenario="test",
            max_epochs=10,
        )
        previous = {"epoch": 1, "map50_95": 0.5, "total_energy_kwh": 0.001, "duration_seconds": 1.0}
        current = {
            "epoch": 2,
            "epoch_index": 2,
            "map50_95": 0.51,
            "total_energy_kwh": 0.001,
            "cumulative_total_energy_kwh": 0.002,
            "duration_seconds": 1.0,
        }
        stop, _, diagnostics = runtime.evaluate(current, [previous])
        self.assertTrue(stop)
        self.assertAlmostEqual(diagnostics["controller_plugin_delta_quality"], 0.01)

    def test_probabilistic_energy_pareto_stops_on_low_value_plateau(self):
        context = ControllerContext(
            "pep-v1",
            "v1",
            "object_detection",
            "quality_score",
            "plateau",
            100,
            parameters={
                "min_epochs": 8,
                "min_fit_points": 5,
                "horizon_epochs": 5,
                "useful_gain": 0.01,
                "min_expected_gain": 0.01,
                "max_probability_gain_gt_threshold": 0.95,
                "min_quality_per_wh": 0.02,
                "target_energy_saving_fraction": 0.0,
                "minimum_quality": 0.20,
                "patience": 2,
                "bootstrap_samples": 16,
            },
        )
        controller = ProbabilisticEnergyParetoController(context)
        qualities = [0.10, 0.18, 0.24, 0.27, 0.285, 0.290, 0.292, 0.293, 0.2932, 0.2933, 0.29335]
        history = [self._pep_row(index + 1, quality) for index, quality in enumerate(qualities)]
        first = self._pep_observation(context, history, self._pep_row(12, 0.29337))
        first_decision = controller.evaluate(first)
        self.assertFalse(first_decision.stop)
        second_history = [*history, first.raw_metrics]
        second = self._pep_observation(context, second_history, self._pep_row(13, 0.29338))
        second_decision = controller.evaluate(second)
        self.assertFalse(second_decision.stop)
        third_history = [*second_history, second.raw_metrics]
        third = self._pep_observation(context, third_history, self._pep_row(14, 0.29339))
        third_decision = controller.evaluate(third)
        self.assertTrue(third_decision.stop)
        self.assertIn("pep_expected_quality_gain_next_horizon", third_decision.diagnostics)
        self.assertGreater(third_decision.diagnostics["pep_feature_coverage_fraction"], 0.5)

    def test_probabilistic_energy_pareto_continues_when_gain_is_likely(self):
        context = ControllerContext(
            "pep-v1",
            "v1",
            "object_detection",
            "quality_score",
            "learning",
            100,
            parameters={
                "min_epochs": 8,
                "min_fit_points": 5,
                "horizon_epochs": 5,
                "useful_gain": 0.01,
                "min_expected_gain": 0.01,
                "max_probability_gain_gt_threshold": 0.20,
                "min_quality_per_wh": 0.02,
                "target_energy_saving_fraction": 0.0,
                "minimum_quality": 0.20,
                "patience": 1,
                "bootstrap_samples": 16,
            },
        )
        controller = ProbabilisticEnergyParetoController(context)
        history = [self._pep_row(epoch, 0.05 + epoch * 0.02) for epoch in range(1, 12)]
        observation = self._pep_observation(context, history, self._pep_row(12, 0.29))
        decision = controller.evaluate(observation)
        self.assertFalse(decision.stop)
        self.assertGreater(decision.diagnostics["pep_prob_gain_gt_threshold"], 0.20)

    def test_rapec_v3_stops_on_dynamic_low_value_plateau(self):
        context = ControllerContext(
            "rapec-v3",
            "v1",
            "object_detection",
            "quality_score",
            "train128-pretrained-yolov8n",
            100,
            parameters={
                "min_epochs_floor": 8,
                "min_fit_points": 5,
                "horizon_epochs": 5,
                "max_probability_gain_gt_threshold": 0.95,
                "patience": 2,
                "bootstrap_samples": 16,
                "remaining_room_fraction": 0.001,
                "max_uncertainty_to_gain_ratio": 20.0,
            },
        )
        controller = RiskAwarePredictiveEnergyController(context)
        qualities = [
            0.10,
            0.18,
            0.24,
            0.27,
            0.285,
            0.290,
            0.292,
            0.293,
            0.2932,
            0.2933,
            0.29335,
            0.29336,
            0.29337,
            0.29337,
            0.29338,
            0.29338,
            0.29339,
            0.29339,
            0.29339,
        ]
        history = []
        for index, quality in enumerate(qualities):
            row = self._pep_row(index + 1, quality)
            if index >= 9:
                row["train_loss"] = 0.35
                row["gradient_norm"] = 0.0
                row["learning_rate"] = 0.00001
            history.append(row)
        first = self._pep_observation(context, history, self._pep_row(20, 0.29339))
        first_decision = controller.evaluate(first)
        self.assertFalse(first_decision.stop)
        second_history = [*history, first.raw_metrics]
        second = self._pep_observation(context, second_history, self._pep_row(21, 0.29339))
        second_decision = controller.evaluate(second)
        self.assertFalse(second_decision.stop)
        third_history = [*second_history, second.raw_metrics]
        third = self._pep_observation(context, third_history, self._pep_row(22, 0.29339))
        third_decision = controller.evaluate(third)
        self.assertTrue(third_decision.stop)
        self.assertIn("rapec_dynamic_meaningful_gain", third_decision.diagnostics)
        self.assertIn("rapec_dynamic_utility_threshold", third_decision.diagnostics)
        self.assertIn("rapec_learning_state_code", third_decision.diagnostics)
        self.assertEqual(third_decision.diagnostics["rapec_candidate_stop"], 1)

    def test_rapec_v3_continues_when_learning_state_is_fast(self):
        context = ControllerContext(
            "rapec-v3",
            "v1",
            "object_detection",
            "quality_score",
            "train128-scratch-yolov8n",
            100,
            parameters={
                "min_epochs_floor": 8,
                "min_fit_points": 5,
                "horizon_epochs": 5,
                "max_probability_gain_gt_threshold": 0.95,
                "patience": 1,
                "bootstrap_samples": 16,
                "remaining_room_fraction": 0.001,
            },
        )
        controller = RiskAwarePredictiveEnergyController(context)
        history = [self._pep_row(epoch, 0.05 + epoch * 0.02) for epoch in range(1, 31)]
        observation = self._pep_observation(context, history, self._pep_row(31, 0.67))
        decision = controller.evaluate(observation)
        self.assertFalse(decision.stop)
        self.assertEqual(decision.diagnostics["rapec_stable_low_improvement_state"], 0)
        self.assertGreater(decision.diagnostics["rapec_dynamic_meaningful_gain"], 0.0)

    def test_rapec_v4_adapts_thresholds_for_scratch_detection(self):
        context = ControllerContext(
            "rapec-v4",
            "v1",
            "object_detection",
            "quality_score",
            "train128-scratch-yolov8n",
            100,
            parameters={"min_fit_points": 5, "bootstrap_samples": 16},
        )
        controller = TaskAwareDynamicThresholdController(context)
        history = [self._pep_row(epoch, 0.05 + epoch * 0.01) for epoch in range(1, 60)]
        observation = self._pep_observation(context, history, self._pep_row(60, 0.65))
        decision = controller.evaluate(observation)
        self.assertFalse(decision.stop)
        self.assertGreaterEqual(decision.diagnostics["rapec4_adapter_horizon_epochs"], 15)
        self.assertGreaterEqual(decision.diagnostics["rapec4_adapter_patience"], 6)
        self.assertLessEqual(decision.diagnostics["rapec4_adapter_probability_threshold"], 0.12)

    def test_rapec_v4_keeps_short_horizon_for_classification(self):
        context = ControllerContext(
            "rapec-v4",
            "v1",
            "image_classification",
            "quality_score",
            "carpk-density-classification-train128-resnet18",
            100,
            parameters={"min_fit_points": 5, "bootstrap_samples": 16},
        )
        controller = TaskAwareDynamicThresholdController(context)
        history = [self._pep_row(epoch, min(0.75, 0.20 + epoch * 0.02)) for epoch in range(1, 25)]
        observation = self._pep_observation(context, history, self._pep_row(25, 0.7501))
        decision = controller.evaluate(observation)
        self.assertIn("rapec4_adapter_horizon_epochs", decision.diagnostics)
        self.assertLessEqual(decision.diagnostics["rapec4_adapter_horizon_epochs"], 5)
        self.assertLessEqual(decision.diagnostics["rapec4_adapter_patience"], 2)
        self.assertGreaterEqual(decision.diagnostics["rapec4_adapter_probability_threshold"], 0.30)

    def test_rapec_v5_self_calibrates_longer_horizon_for_delayed_curve(self):
        context = ControllerContext(
            "rapec-v5",
            "v1",
            "object_detection",
            "quality_score",
            "unknown-scenario",
            100,
            parameters={"min_fit_points": 5, "bootstrap_samples": 16},
        )
        controller = SelfCalibratingDynamicThresholdController(context)
        qualities = [0.05 + epoch * 0.012 for epoch in range(1, 26)]
        qualities.extend([qualities[-1] + 0.0002 * (epoch - 25) for epoch in range(26, 50)])
        history = [self._pep_row(index + 1, min(0.95, quality)) for index, quality in enumerate(qualities)]
        observation = self._pep_observation(context, history, self._pep_row(50, min(0.95, qualities[-1] + 0.0001)))
        decision = controller.evaluate(observation)
        self.assertIn("rapec5_delayed_learning_risk", decision.diagnostics)
        self.assertGreater(decision.diagnostics["rapec5_horizon_epochs"], 5)
        self.assertGreaterEqual(decision.diagnostics["rapec5_patience"], 3)
        self.assertLess(decision.diagnostics["rapec5_probability_threshold"], 0.45)

    def test_rapec_v5_stops_on_stable_plateau_without_task_profile(self):
        context = ControllerContext(
            "rapec-v5",
            "v1",
            "object_detection",
            "quality_score",
            "unknown-scenario",
            100,
            parameters={"min_fit_points": 5, "bootstrap_samples": 16, "min_epoch_fraction_floor": 0.10},
        )
        controller = SelfCalibratingDynamicThresholdController(context)
        qualities = [0.10, 0.25, 0.40, 0.55, 0.66, 0.72, 0.745, 0.750, 0.751]
        qualities.extend([0.751 for _ in range(30)])
        history = [self._pep_row(index + 1, quality) for index, quality in enumerate(qualities)]
        stopped = False
        for epoch in range(len(qualities) + 1, len(qualities) + 8):
            observation = self._pep_observation(context, history, self._pep_row(epoch, 0.751))
            decision = controller.evaluate(observation)
            history.append(observation.raw_metrics)
            stopped = stopped or decision.stop
            if stopped:
                self.assertIn("rapec5_curve_stability", decision.diagnostics)
                self.assertGreaterEqual(decision.diagnostics["rapec5_curve_stability"], 0.5)
                break
        self.assertTrue(stopped)

    def test_rapec_v6_stops_on_online_calibrated_plateau(self):
        context = ControllerContext(
            "rapec-v6",
            "v1",
            "object_detection",
            "quality_score",
            "unknown-scenario",
            100,
            parameters={
                "min_fit_points": 8,
                "trend_window": 12,
                "bootstrap_samples": 32,
                "min_epochs_floor": 12,
                "min_patience": 2,
                "max_patience": 3,
            },
        )
        controller = OnlineMultiHorizonParetoController(context)
        qualities = [0.10, 0.22, 0.35, 0.48, 0.59, 0.67, 0.72, 0.745, 0.752]
        qualities.extend([0.752 for _ in range(32)])
        history = [self._pep_row(index + 1, quality) for index, quality in enumerate(qualities)]
        stopped = False
        for epoch in range(len(history) + 1, len(history) + 8):
            current = self._pep_row(epoch, 0.752)
            current["train_loss"] = 0.35
            current["gradient_norm"] = 0.0
            current["learning_rate"] = 0.00001
            observation = self._pep_observation(context, history, current)
            decision = controller.evaluate(observation)
            history.append(current)
            if decision.stop:
                stopped = True
                self.assertEqual(decision.diagnostics["rapec6_uses_external_full100_history"], 0)
                self.assertEqual(decision.diagnostics["rapec6_energy_scope_training_only"], 1)
                self.assertLessEqual(
                    decision.diagnostics["rapec6_risk_probability"],
                    decision.diagnostics["rapec6_dynamic_risk_alpha"],
                )
                break
        self.assertTrue(stopped)

    def test_rapec_v6_continues_when_recovery_signals_are_active(self):
        context = ControllerContext(
            "rapec-v6",
            "v1",
            "semantic_segmentation",
            "quality_score",
            "another-unknown-scenario",
            100,
            parameters={
                "min_fit_points": 8,
                "trend_window": 12,
                "bootstrap_samples": 32,
                "min_epochs_floor": 10,
                "min_patience": 1,
            },
        )
        controller = OnlineMultiHorizonParetoController(context)
        qualities = [0.05 + epoch * 0.015 for epoch in range(1, 21)]
        qualities.extend([qualities[-1] + 0.0001 * (epoch - 20) for epoch in range(21, 34)])
        history = []
        for index, quality in enumerate(qualities):
            row = self._pep_row(index + 1, quality)
            row["train_loss"] = 1.0 - index * 0.015
            row["gradient_norm"] = 0.45
            row["learning_rate"] = 0.006
            history.append(row)
        current = self._pep_row(34, qualities[-1] + 0.0001)
        current["train_loss"] = 0.50
        current["gradient_norm"] = 0.44
        current["learning_rate"] = 0.006
        decision = controller.evaluate(self._pep_observation(context, history, current))
        self.assertFalse(decision.stop)
        self.assertGreater(
            decision.diagnostics["rapec6_recovery_probability"],
            decision.diagnostics["rapec6_recovery_limit"],
        )

    def test_rapec_v6_exports_all_multi_horizon_predictions(self):
        context = ControllerContext(
            "rapec-v6",
            "v1",
            "image_classification",
            "quality_score",
            "unknown",
            100,
            parameters={"min_fit_points": 6, "bootstrap_samples": 16},
        )
        controller = OnlineMultiHorizonParetoController(context)
        history = [self._pep_row(epoch, min(0.80, 0.10 + epoch * 0.02)) for epoch in range(1, 31)]
        decision = controller.evaluate(
            self._pep_observation(context, history, self._pep_row(31, 0.701))
        )
        for horizon in (1, 3, 5, 10, 20):
            self.assertIn(f"rapec6_h{horizon}_expected_gain", decision.diagnostics)
            self.assertIn(f"rapec6_h{horizon}_expected_energy_wh", decision.diagnostics)
            self.assertIn(f"rapec6_h{horizon}_prob_gain_gt_epsilon", decision.diagnostics)
        self.assertGreater(decision.diagnostics["rapec6_feature_coverage_fraction"], 0.75)

    def test_rapec_v6_is_independent_of_task_and_scenario_names(self):
        parameters = {
            "min_fit_points": 6,
            "bootstrap_samples": 16,
            "min_epochs_floor": 8,
            "min_patience": 1,
        }
        detection_context = ControllerContext(
            "rapec-v6",
            "v1",
            "object_detection",
            "quality_score",
            "carpk-pretrained",
            100,
            parameters=parameters,
        )
        segmentation_context = ControllerContext(
            "rapec-v6",
            "v1",
            "semantic_segmentation",
            "quality_score",
            "unseen-dataset-scratch",
            100,
            parameters=parameters,
        )
        history = [self._pep_row(epoch, min(0.75, 0.10 + epoch * 0.02)) for epoch in range(1, 31)]
        current = self._pep_row(31, 0.701)
        detection = OnlineMultiHorizonParetoController(detection_context).evaluate(
            self._pep_observation(detection_context, history, current)
        )
        segmentation = OnlineMultiHorizonParetoController(segmentation_context).evaluate(
            self._pep_observation(segmentation_context, history, current)
        )
        self.assertEqual(detection.stop, segmentation.stop)
        for key in (
            "rapec6_dynamic_epsilon",
            "rapec6_dynamic_risk_alpha",
            "rapec6_recovery_probability",
            "rapec6_selected_horizon",
        ):
            self.assertAlmostEqual(detection.diagnostics[key], segmentation.diagnostics[key])

    def test_rapec_v7_stops_on_bayesian_plateau(self):
        context = ControllerContext(
            "rapec-v7",
            "v1",
            "object_detection",
            "quality_score",
            "unknown-scenario",
            100,
            parameters={
                "posterior_samples": 256,
                "minimum_observations_floor": 8,
            },
        )
        controller = OnlineBayesianRapecController(context)
        qualities = [0.10, 0.22, 0.35, 0.48, 0.59, 0.67, 0.72, 0.745, 0.752]
        qualities.extend([0.752 for _ in range(32)])
        history = [self._pep_row(index + 1, quality) for index, quality in enumerate(qualities)]
        stopped = False
        for epoch in range(len(history) + 1, len(history) + 16):
            current = self._pep_row(epoch, 0.752)
            decision = controller.evaluate(self._pep_observation(context, history, current))
            history.append(current)
            if decision.stop:
                stopped = True
                self.assertEqual(decision.diagnostics["rapec7_is_bayesian"], 1)
                self.assertEqual(decision.diagnostics["rapec7_uses_historical_curves"], 0)
                self.assertEqual(decision.diagnostics["rapec7_uses_task_or_dataset_profile"], 0)
                self.assertLessEqual(
                    decision.diagnostics["rapec7_posterior_recovery_probability"],
                    decision.diagnostics["rapec7_dynamic_risk_limit"],
                )
                break
        self.assertTrue(stopped)

    def test_rapec_v7_continues_on_active_learning_curve(self):
        context = ControllerContext(
            "rapec-v7",
            "v1",
            "semantic_segmentation",
            "quality_score",
            "unseen-scenario",
            100,
            parameters={"posterior_samples": 128, "minimum_observations_floor": 8},
        )
        controller = OnlineBayesianRapecController(context)
        history = [
            self._pep_row(epoch, min(0.90, 0.05 + epoch * 0.015))
            for epoch in range(1, 42)
        ]
        current = self._pep_row(42, 0.68)
        decision = controller.evaluate(self._pep_observation(context, history, current))
        self.assertFalse(decision.stop)
        self.assertGreater(
            decision.diagnostics["rapec7_posterior_recovery_probability"],
            decision.diagnostics["rapec7_dynamic_risk_limit"],
        )

    def test_rapec_v7_exports_bayesian_horizons_and_model_weights(self):
        context = ControllerContext(
            "rapec-v7",
            "v1",
            "image_classification",
            "quality_score",
            "unknown",
            100,
            parameters={"posterior_samples": 128, "minimum_observations_floor": 8},
        )
        controller = OnlineBayesianRapecController(context)
        history = [
            self._pep_row(epoch, min(0.80, 0.10 + epoch * 0.02))
            for epoch in range(1, 31)
        ]
        decision = controller.evaluate(
            self._pep_observation(context, history, self._pep_row(31, 0.701))
        )
        for horizon in (1, 3, 5, 10, 20):
            self.assertIn(f"rapec7_h{horizon}_expected_gain", decision.diagnostics)
            self.assertIn(f"rapec7_h{horizon}_prob_relevant_gain", decision.diagnostics)
            self.assertIn(f"rapec7_h{horizon}_expected_energy_wh", decision.diagnostics)
            self.assertIn(f"rapec7_h{horizon}_expected_duration_seconds", decision.diagnostics)
        model_weight_sum = sum(
            decision.diagnostics[f"rapec7_quality_model_weight_{name}"]
            for name in (
                "linear",
                "logarithmic",
                "saturation",
                "exponential",
                "change_point",
                "local_trend",
                "telemetry",
            )
        )
        self.assertAlmostEqual(model_weight_sum, 1.0)

    def test_rapec_v7_is_task_and_scenario_independent(self):
        parameters = {"posterior_samples": 128, "minimum_observations_floor": 8}
        contexts = [
            ControllerContext(
                "rapec-v7",
                "v1",
                "object_detection",
                "quality_score",
                "carpk-pretrained",
                100,
                parameters=parameters,
            ),
            ControllerContext(
                "rapec-v7",
                "v1",
                "semantic_segmentation",
                "quality_score",
                "unseen-dataset-scratch",
                100,
                parameters=parameters,
            ),
        ]
        history = [
            self._pep_row(epoch, min(0.75, 0.10 + epoch * 0.02))
            for epoch in range(1, 31)
        ]
        current = self._pep_row(31, 0.701)
        decisions = [
            OnlineBayesianRapecController(context).evaluate(
                self._pep_observation(context, history, current)
            )
            for context in contexts
        ]
        self.assertEqual(decisions[0].stop, decisions[1].stop)
        for key in (
            "rapec7_dynamic_epsilon",
            "rapec7_dynamic_risk_limit",
            "rapec7_posterior_recovery_probability",
            "rapec7_dynamic_minimum_observations",
        ):
            self.assertAlmostEqual(
                decisions[0].diagnostics[key],
                decisions[1].diagnostics[key],
            )

    def test_rapec_v8_stops_when_v3_and_bayesian_guard_agree(self):
        context = ControllerContext(
            "rapec-v8",
            "v1",
            "object_detection",
            "quality_score",
            "train128-pretrained-yolov8n",
            100,
            parameters={
                "min_epochs_floor": 8,
                "min_fit_points": 5,
                "horizon_epochs": 5,
                "max_probability_gain_gt_threshold": 0.95,
                "patience": 2,
                "bootstrap_samples": 16,
                "remaining_room_fraction": 0.001,
                "max_uncertainty_to_gain_ratio": 20.0,
                "bayesian_window": 20,
                "bayesian_samples": 256,
                "bayesian_min_observations": 8,
                "bayesian_probability_limit": 0.35,
            },
        )
        controller = BayesianGuardedRapecV3Controller(context)
        qualities = [
            0.10,
            0.18,
            0.24,
            0.27,
            0.285,
            0.290,
            0.292,
            0.293,
            0.2932,
            0.2933,
            0.29335,
            0.29336,
            0.29337,
            0.29337,
            0.29338,
            0.29338,
            0.29339,
            0.29339,
            0.29339,
        ]
        history = [
            self._pep_row(index + 1, quality)
            for index, quality in enumerate(qualities)
        ]
        stopped = False
        for epoch in range(len(history) + 1, len(history) + 12):
            current = self._pep_row(epoch, 0.29339)
            current["gradient_norm"] = 0.0
            current["learning_rate"] = 0.00001
            decision = controller.evaluate(
                self._pep_observation(context, history, current)
            )
            history.append(current)
            if decision.stop:
                stopped = True
                self.assertEqual(decision.diagnostics["rapec8_is_bayesian"], 1)
                self.assertEqual(
                    decision.diagnostics["rapec8_v3_primary_decision"],
                    1,
                )
                self.assertEqual(
                    decision.diagnostics["rapec8_uses_historical_curves"],
                    0,
                )
                self.assertEqual(decision.diagnostics["rapec8_base_v3_stop"], 1)
                self.assertEqual(
                    decision.diagnostics["rapec8_bayesian_low_gain"],
                    1,
                )
                break
        self.assertTrue(stopped)

    def test_rapec_v8_bayesian_guard_allows_active_learning(self):
        context = ControllerContext(
            "rapec-v8",
            "v1",
            "semantic_segmentation",
            "quality_score",
            "unknown-scenario",
            100,
            parameters={
                "min_epochs_floor": 8,
                "min_fit_points": 5,
                "horizon_epochs": 5,
                "max_probability_gain_gt_threshold": 1.0,
                "patience": 1,
                "bootstrap_samples": 8,
                "remaining_room_fraction": 0.001,
                "max_uncertainty_to_gain_ratio": 100.0,
                "bayesian_window": 20,
                "bayesian_samples": 256,
                "bayesian_min_observations": 8,
                "bayesian_probability_limit": 0.25,
            },
        )
        controller = BayesianGuardedRapecV3Controller(context)
        qualities = [0.10 + epoch * 0.012 for epoch in range(1, 31)]
        history = [
            self._pep_row(index + 1, quality)
            for index, quality in enumerate(qualities)
        ]
        current = self._pep_row(31, qualities[-1] + 0.012)
        decision = controller.evaluate(
            self._pep_observation(context, history, current)
        )
        self.assertFalse(decision.stop)
        self.assertGreater(
            decision.diagnostics[
                "rapec8_posterior_probability_relevant_gain"
            ],
            decision.diagnostics["rapec8_probability_limit"],
        )

    def test_rapec_v8_bayesian_layer_is_task_and_scenario_independent(self):
        parameters = {
            "min_fit_points": 5,
            "bootstrap_samples": 8,
            "bayesian_samples": 128,
            "bayesian_min_observations": 8,
        }
        contexts = [
            ControllerContext(
                "rapec-v8",
                "v1",
                "object_detection",
                "quality_score",
                "carpk-pretrained",
                100,
                parameters=parameters,
            ),
            ControllerContext(
                "rapec-v8",
                "v1",
                "semantic_segmentation",
                "quality_score",
                "unseen-dataset-scratch",
                100,
                parameters=parameters,
            ),
        ]
        history = [
            self._pep_row(epoch, min(0.75, 0.10 + epoch * 0.02))
            for epoch in range(1, 31)
        ]
        current = self._pep_row(31, 0.701)
        decisions = [
            BayesianGuardedRapecV3Controller(context).evaluate(
                self._pep_observation(context, history, current)
            )
            for context in contexts
        ]
        for key in (
            "rapec8_dynamic_relevant_gain",
            "rapec8_posterior_expected_gain",
            "rapec8_posterior_probability_relevant_gain",
            "rapec8_posterior_mean_epoch_delta",
        ):
            self.assertAlmostEqual(
                decisions[0].diagnostics[key],
                decisions[1].diagnostics[key],
            )
        self.assertEqual(
            decisions[0].diagnostics["rapec8_uses_task_or_dataset_prior"],
            0,
        )

    def test_lcpfn_quantile_probability_interpolation(self):
        pairs = [(0.1, 0.40), (0.5, 0.50), (0.9, 0.70)]
        self.assertAlmostEqual(_probability_above_from_quantiles(pairs, 0.50), 0.5)
        self.assertAlmostEqual(_probability_above_from_quantiles(pairs, 0.35), 0.9)
        self.assertAlmostEqual(_probability_above_from_quantiles(pairs, 0.80), 0.1)

    def test_epoch_prometheus_exports_research_features(self):
        from slurm.export_job_epoch_metrics_prom import main as export_epoch_main

        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            summary = {
                "job_id": "42",
                "scenario": "feature-test",
                "training_seed": 0,
                "split_seed": 0,
                "controller_id": "pep-v1",
                "benchmark_version": "v1",
                "benchmark_run_id": "run",
                "benchmark_case_id": "case",
                "benchmark_stage": "stage",
                "task_type": "object_detection",
                "quality_metric": "quality_score",
                "epochs": [
                    {
                        "epoch": 1,
                        "quality_score": 0.1,
                        "total_energy_kwh": 0.001,
                        "gpu_power_avg_w": 120.0,
                        "gpu_power_max_w": 150.0,
                        "gpu_mem_used_avg_mb": 2048.0,
                        "learning_rate": 0.01,
                        "gradient_norm": 3.5,
                        "train_loss": 1.2,
                        "model_parameter_count": 3157200,
                        "model_flops": 8.7e9,
                    }
                ],
            }
            summary_path = root / "epoch_summary_job_42.json"
            output_path = root / "job_42_epochs.prom"
            summary_path.write_text(json.dumps(summary), encoding="utf-8")
            with mock.patch("sys.argv", ["export", "--summary-json", str(summary_path), "--output-prom", str(output_path)]):
                export_epoch_main()
            content = output_path.read_text(encoding="utf-8")
            self.assertIn("slurm_job_epoch_gradient_norm", content)
            self.assertIn("slurm_job_epoch_model_flops", content)
            self.assertIn("slurm_job_epoch_gpu_power_avg_w", content)

    @staticmethod
    def _pep_row(epoch: int, quality: float) -> dict:
        return {
            "epoch": epoch,
            "epoch_index": epoch,
            "quality_score": quality,
            "total_energy_kwh": 0.001,
            "duration_seconds": 10.0,
            "gpu_util_avg_pct": 70.0,
            "gpu_memory_used_mb": 4096.0,
            "gpu_power_avg_w": 180.0,
            "learning_rate": 0.01 / epoch,
            "train_loss": 1.0 / epoch,
            "gradient_norm": 0.5 / epoch,
            "model_parameter_count": 3_000_000,
            "model_flops": 8.0e9,
        }

    @staticmethod
    def _pep_observation(context: ControllerContext, history: list[dict], current: dict) -> EpochObservation:
        previous_quality = history[-1]["quality_score"] if history else None
        return EpochObservation(
            epoch=int(current["epoch"]),
            max_epochs=context.max_epochs,
            task_type=context.task_type,
            quality_metric=context.quality_metric,
            quality=float(current["quality_score"]),
            best_quality=max([float(row["quality_score"]) for row in [*history, current]]),
            delta_quality=(float(current["quality_score"]) - previous_quality) if previous_quality is not None else None,
            epoch_energy_wh=float(current["total_energy_kwh"]) * 1000.0,
            cumulative_energy_wh=sum(float(row["total_energy_kwh"]) * 1000.0 for row in [*history, current]),
            epoch_duration_seconds=float(current["duration_seconds"]),
            cumulative_duration_seconds=sum(float(row["duration_seconds"]) for row in [*history, current]),
            gpu_utilization_pct=float(current["gpu_util_avg_pct"]),
            history=tuple(history),
            raw_metrics=current,
        )

    def test_paired_analysis_and_prometheus_export(self):
        manifest = load_manifest(ROOT / "controller_benchmark" / "config" / "benchmark-v2-shared-seed.json")
        plan = build_plan(
            manifest,
            controller_id="candidate",
            controller_plugin="controller_benchmark.controllers.fixed_epoch:FixedEpochController",
            controller_parameters={"stop_epoch": 80},
        )
        case_id = plan["runs"][0]["benchmark_case_id"]
        stage = plan["runs"][0]["benchmark_stage"]
        plan["runs"] = [row for row in plan["runs"] if row["benchmark_case_id"] == case_id]
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            metrics = root / "data" / "energy_metrics"
            metrics.mkdir(parents=True)
            for job_id, controller_id, energy_kwh, quality, epochs in (
                ("1", "full100", 0.010, 0.70, 100),
                ("2", "candidate", 0.008, 0.69, 80),
            ):
                summary = {
                    "job_id": job_id,
                    "benchmark_run_id": plan["benchmark_run_id"],
                    "benchmark_case_id": case_id,
                    "benchmark_stage": stage,
                    "controller_id": controller_id,
                    "scenario": "test",
                    "task_type": "object_detection",
                    "quality_metric": "map50_95",
                    "training_seed": 0,
                    "epochs_completed": epochs,
                    "total_gpu_energy_kwh": energy_kwh * 0.6,
                    "epochs": [{"map50_95": quality}],
                }
                gpu = {
                    "gpu_energy_kwh": energy_kwh,
                    "codecarbon_job_total_gpu_energy_kwh": energy_kwh * 0.9,
                    "codecarbon_measurement_available": 1,
                    "phase_metrics": {
                        "training": {
                            "gpu_energy_kwh": energy_kwh * 0.8,
                            "codecarbon_gpu_energy_kwh": energy_kwh * 0.75,
                        }
                    },
                    "duration_seconds": epochs,
                    "run_metadata": {
                        "runtime_image_id": manifest["execution"]["runtime_image_id"],
                        "gpu": {
                            "name": manifest["execution"]["gpu_name"],
                            "power_limit_w": manifest["execution"]["gpu_power_limit_w"],
                        },
                    },
                }
                (metrics / f"epoch_summary_job_{job_id}.json").write_text(json.dumps(summary), encoding="utf-8")
                (metrics / f"gpu_summary_job_{job_id}.json").write_text(json.dumps(gpu), encoding="utf-8")
            plan_path = root / "run-matrix.json"
            plan_path.write_text(json.dumps(plan), encoding="utf-8")
            result = analyze_snapshot(root, plan_path)
            self.assertAlmostEqual(result["pairs"][0]["energy_saving_fraction"], 0.2)
            self.assertAlmostEqual(result["pairs"][0]["quality_regret"], 0.01)
            self.assertAlmostEqual(result["pairs"][0]["candidate_comparable_slurm_gpu_energy_wh"], 6.4)
            self.assertAlmostEqual(result["pairs"][0]["candidate_comparable_codecarbon_gpu_energy_wh"], 6.0)
            plan["execution"]["energy_comparison_scope"] = "epoch"
            plan_path.write_text(json.dumps(plan), encoding="utf-8")
            epoch_result = analyze_snapshot(root, plan_path)
            self.assertEqual(epoch_result["pairs"][0]["energy_comparison_scope"], "epoch")
            self.assertAlmostEqual(epoch_result["pairs"][0]["baseline_energy_wh"], 6.0)
            self.assertAlmostEqual(epoch_result["pairs"][0]["candidate_energy_wh"], 4.8)
            self.assertAlmostEqual(epoch_result["pairs"][0]["energy_saving_fraction"], 0.2)
            self.assertAlmostEqual(epoch_result["pairs"][0]["baseline_job_energy_wh"], 10.0)
            self.assertAlmostEqual(epoch_result["pairs"][0]["candidate_job_energy_wh"], 8.0)
            write_analysis(result, root / "controller_benchmark")
            prom = metrics / "node_exporter" / "controller_benchmark.prom"
            write_prometheus(result, prom)
            content = prom.read_text(encoding="utf-8")
            self.assertIn("controller_benchmark_energy_saving_fraction", content)
            self.assertIn("controller_benchmark_comparable_slurm_gpu_energy_wh", content)
            self.assertIn("controller_benchmark_comparable_codecarbon_gpu_energy_wh", content)
            self.assertIn("controller_benchmark_job_energy_wh", content)
            self.assertIn('job_id="2"', content)


if __name__ == "__main__":
    unittest.main()
