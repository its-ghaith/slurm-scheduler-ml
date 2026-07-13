import tempfile
import unittest
from pathlib import Path

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


if __name__ == "__main__":
    unittest.main()
