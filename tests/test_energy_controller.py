import tempfile
import unittest
import json
from pathlib import Path
from unittest.mock import patch

from slurm.epoch_energy_controller import EnergyAdaptiveConfig, EpochEnergyAdaptiveController


class EnergyControllerTests(unittest.TestCase):
    def make_controller(self, config):
        root = Path(tempfile.mkdtemp())
        return EpochEnergyAdaptiveController(
            gpu_csv_path=root / "gpu.csv",
            epoch_timeline_path=root / "epochs.jsonl",
            config=config,
        )

    def test_metric_early_stopping_uses_explicit_min_delta(self):
        controller = self.make_controller(
            EnergyAdaptiveConfig(
                enabled=True,
                controller_mode="metric_early_stopping",
                monitor_metric="map50_95",
                min_epochs=1,
                patience=2,
                standard_min_delta=0.001,
            )
        )
        controller.history = [{"map50_95": 0.6000}]
        stop, _, _ = controller._evaluate_metric_early_stopping({"map50_95": 0.6005})
        self.assertFalse(stop)
        controller.history.append({"map50_95": 0.6005})
        stop, reason, _ = controller._evaluate_metric_early_stopping({"map50_95": 0.6008})
        self.assertTrue(stop)
        self.assertIn("metric_early_stop", reason)

    def test_conformal_controller_stops_only_after_all_guards_pass(self):
        root = Path(tempfile.mkdtemp())
        calibration_path = root / "calibration.json"
        calibration_path.write_text(
            json.dumps(
                {
                    "scenarios": {
                        "test-scenario": {
                            "global_correction": 0.0,
                            "bins": [{"start_epoch": 1, "end_epoch": 19, "correction": 0.0}],
                        }
                    }
                }
            ),
            encoding="utf-8",
        )
        controller = self.make_controller(
            EnergyAdaptiveConfig(
                enabled=True,
                controller_mode="conformal_energy_aware",
                monitor_metric="map50_95",
                min_epochs=10,
                patience=1,
                uncertainty_target_epoch=20,
                uncertainty_min_fit_points=3,
                uncertainty_min_quality=0.60,
                scenario="test-scenario",
                conformal_calibration_path=str(calibration_path),
                conformal_regret_tolerance=0.015,
                conformal_future_efficiency_threshold=0.05,
            )
        )
        controller.history = [
            {
                "epoch": epoch,
                "epoch_index": epoch,
                "map50_95": 0.60 + epoch * 0.001,
                "best_map50_95": 0.60 + epoch * 0.001,
                "net_total_energy_kwh": 0.001,
            }
            for epoch in range(1, 10)
        ]
        current = {
            "epoch": 10,
            "epoch_index": 10,
            "map50_95": 0.61,
            "best_map50_95": 0.61,
            "net_total_energy_kwh": 0.001,
        }
        with patch(
            "slurm.epoch_energy_controller.bootstrap_prediction_distribution_v2",
            return_value=[0.614, 0.615, 0.616],
        ):
            stop, reason, diagnostics = controller._evaluate_conformal_energy_stop(current)
        self.assertTrue(stop)
        self.assertIn("conformal_energy_stop", reason)
        self.assertEqual(diagnostics["controller_decision"], "stop")
        self.assertEqual(diagnostics["controller_decision_state"], 2)

    def test_conformal_controller_observes_when_calibration_is_missing(self):
        controller = self.make_controller(
            EnergyAdaptiveConfig(
                enabled=True,
                controller_mode="conformal_energy_aware",
                monitor_metric="map50_95",
                min_epochs=3,
                patience=1,
                uncertainty_target_epoch=10,
                uncertainty_min_fit_points=3,
                uncertainty_min_quality=0.50,
                scenario="missing",
                conformal_calibration_path="",
                conformal_require_calibration=True,
            )
        )
        controller.history = [
            {"epoch": 1, "epoch_index": 1, "map50_95": 0.55, "best_map50_95": 0.55, "total_energy_kwh": 0.001},
            {"epoch": 2, "epoch_index": 2, "map50_95": 0.56, "best_map50_95": 0.56, "total_energy_kwh": 0.001},
        ]
        current = {"epoch": 3, "epoch_index": 3, "map50_95": 0.57, "best_map50_95": 0.57, "total_energy_kwh": 0.001}
        with patch(
            "slurm.epoch_energy_controller.bootstrap_prediction_distribution_v2",
            return_value=[0.571, 0.572, 0.573],
        ):
            stop, _, diagnostics = controller._evaluate_conformal_energy_stop(current)
        self.assertFalse(stop)
        self.assertEqual(diagnostics["calibration_available"], 0)

    def test_conformal_controller_skips_expensive_fit_between_scheduled_evaluations(self):
        controller = self.make_controller(
            EnergyAdaptiveConfig(
                enabled=True,
                controller_mode="conformal_energy_aware",
                monitor_metric="map50_95",
                min_epochs=40,
                uncertainty_min_fit_points=12,
                controller_evaluation_interval=3,
            )
        )
        controller.history = [
            {"epoch": epoch, "epoch_index": epoch, "map50_95": 0.60, "best_map50_95": 0.60}
            for epoch in range(1, 41)
        ]
        controller.low_gain_streak = 1
        current = {"epoch": 41, "epoch_index": 41, "map50_95": 0.60, "best_map50_95": 0.60}
        with patch("slurm.epoch_energy_controller.bootstrap_prediction_distribution_v2") as predictor:
            stop, _, diagnostics = controller._evaluate_conformal_energy_stop(current)
        self.assertFalse(stop)
        predictor.assert_not_called()
        self.assertEqual(controller.low_gain_streak, 1)
        self.assertEqual(diagnostics["controller_decision"], "scheduled_wait")

    def test_uncertainty_stop_requires_minimum_quality(self):
        config = EnergyAdaptiveConfig(
            enabled=True,
            controller_mode="uncertainty_aware",
            monitor_metric="map50_95",
            min_epochs=8,
            patience=1,
            smoothing_window=3,
            min_mape_map50_per_wh=0.001,
            uncertainty_target_epoch=100,
            uncertainty_epsilon=0.02,
            uncertainty_alpha=0.10,
            uncertainty_bootstrap_samples=16,
            uncertainty_min_fit_points=8,
            uncertainty_min_quality=0.55,
        )
        controller = self.make_controller(config)
        controller.history = [
            {
                "epoch": index,
                "epoch_index": index,
                "map50_95": 0.30 + index * 0.005,
                "best_map50_95": 0.30 + index * 0.005,
                "marginal_map50_95_per_wh": 0.0,
            }
            for index in range(1, 8)
        ]
        stop, _, _ = controller._evaluate_uncertainty_stop(
            {
                "epoch": 8,
                "epoch_index": 8,
                "map50_95": 0.34,
                "best_map50_95": 0.34,
                "marginal_map50_95_per_wh": 0.0,
            }
        )
        self.assertFalse(stop)

    def test_hybrid_controller_accumulates_persistent_pareto_evidence(self):
        root = Path(tempfile.mkdtemp())
        calibration_path = root / "calibration.json"
        calibration_path.write_text(
            json.dumps(
                {
                    "scenarios": {
                        "test-scenario": {
                            "global_correction": 0.0,
                            "bins": [{"start_epoch": 1, "end_epoch": 20, "correction": 0.0}],
                        }
                    }
                }
            ),
            encoding="utf-8",
        )
        controller = self.make_controller(
            EnergyAdaptiveConfig(
                enabled=True,
                controller_mode="hybrid_pareto_energy_aware",
                monitor_metric="map50_95",
                min_epochs=8,
                patience=3,
                uncertainty_target_epoch=20,
                uncertainty_min_fit_points=3,
                uncertainty_min_quality=0.60,
                scenario="test-scenario",
                conformal_calibration_path=str(calibration_path),
                conformal_regret_tolerance=0.015,
                conformal_future_efficiency_threshold=0.05,
                hybrid_quality_target=0.68,
                hybrid_plateau_window=5,
                hybrid_plateau_slope_threshold=0.0015,
                hybrid_max_epoch_budget=0,
            )
        )
        controller.history = [
            {
                "epoch": epoch,
                "epoch_index": epoch,
                "map50_95": 0.70,
                "best_map50_95": 0.70,
                "net_total_energy_kwh": 0.001,
            }
            for epoch in range(1, 8)
        ]
        with patch(
            "slurm.epoch_energy_controller.bootstrap_prediction_distribution_v2",
            return_value=[0.704, 0.705, 0.706],
        ):
            for epoch in (8, 9):
                current = {
                    "epoch": epoch,
                    "epoch_index": epoch,
                    "map50_95": 0.70,
                    "best_map50_95": 0.70,
                    "net_total_energy_kwh": 0.001,
                    "cumulative_net_total_energy_kwh": epoch * 0.001,
                }
                stop, _, diagnostics = controller._evaluate_hybrid_pareto_energy_stop(current)
                self.assertFalse(stop)
                self.assertEqual(diagnostics["controller_decision"], "candidate_stop")
                controller.history.append(current)

            current = {
                "epoch": 10,
                "epoch_index": 10,
                "map50_95": 0.70,
                "best_map50_95": 0.70,
                "net_total_energy_kwh": 0.001,
                "cumulative_net_total_energy_kwh": 0.010,
            }
            stop, reason, diagnostics = controller._evaluate_hybrid_pareto_energy_stop(current)
        self.assertTrue(stop)
        self.assertIn("hybrid_pareto_stop", reason)
        self.assertEqual(diagnostics["controller_decision"], "stop")

    def test_generalized_energy_guard_enforces_fallback_and_reports_quality_conflict(self):
        controller = self.make_controller(
            EnergyAdaptiveConfig(
                enabled=True,
                controller_mode="generalized_energy_guard",
                monitor_metric="map50_95",
                min_epochs=30,
                uncertainty_target_epoch=100,
                uncertainty_min_fit_points=200,
                uncertainty_min_quality=0.90,
                energy_guard_target_saving_fraction=0.20,
                energy_guard_fallback_epoch_fraction=0.77,
                energy_guard_strict=True,
            )
        )
        controller.gpu_csv_path.touch()
        controller.history = [
            {
                "epoch": epoch,
                "epoch_index": epoch,
                "map50_95": 0.60,
                "best_map50_95": 0.60,
                "observed_job_energy_wh": epoch * 0.1,
            }
            for epoch in range(1, 77)
        ]
        current = {
            "epoch": 77,
            "epoch_index": 77,
            "end_ts": 77.0,
            "map50_95": 0.60,
            "best_map50_95": 0.60,
        }
        with patch(
            "slurm.epoch_energy_controller.summarize_window",
            return_value={"gpu_energy_kwh": 0.0077},
        ):
            stop, reason, diagnostics = controller._evaluate_generalized_energy_guard_stop(current)
        self.assertTrue(stop)
        self.assertIn("generalized_energy_guard_stop", reason)
        self.assertEqual(diagnostics["energy_guard_fallback_triggered"], 1)
        self.assertEqual(diagnostics["energy_guard_quality_conflict"], 1)

    def test_generalized_energy_guard_can_prioritize_quality_in_non_strict_mode(self):
        controller = self.make_controller(
            EnergyAdaptiveConfig(
                enabled=True,
                controller_mode="generalized_energy_guard",
                monitor_metric="map50_95",
                min_epochs=30,
                uncertainty_target_epoch=100,
                uncertainty_min_fit_points=200,
                uncertainty_min_quality=0.90,
                energy_guard_fallback_epoch_fraction=0.77,
                energy_guard_strict=False,
            )
        )
        controller.gpu_csv_path.touch()
        controller.history = [
            {"epoch": epoch, "epoch_index": epoch, "map50_95": 0.60, "best_map50_95": 0.60}
            for epoch in range(1, 77)
        ]
        current = {
            "epoch": 77,
            "epoch_index": 77,
            "end_ts": 77.0,
            "map50_95": 0.60,
            "best_map50_95": 0.60,
        }
        with patch(
            "slurm.epoch_energy_controller.summarize_window",
            return_value={"gpu_energy_kwh": 0.0077},
        ):
            stop, _, diagnostics = controller._evaluate_generalized_energy_guard_stop(current)
        self.assertFalse(stop)
        self.assertEqual(diagnostics["controller_decision"], "quality_conflict")


if __name__ == "__main__":
    unittest.main()
