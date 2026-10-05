from __future__ import annotations

import unittest

from controller_benchmark.optimization.replay import (
    BaselineTrace,
    dynamic_quality_tolerance,
    replay_controller,
)


def _trace() -> BaselineTrace:
    qualities = [0.10, 0.20, 0.28, 0.34, 0.37, 0.39, 0.40, 0.405, 0.407, 0.408]
    epochs = []
    cumulative = 0.0
    for epoch, quality in enumerate(qualities, start=1):
        cumulative += 0.001
        epochs.append(
            {
                "epoch": epoch,
                "epoch_index": epoch,
                "quality_score": quality,
                "best_quality_score": quality,
                "total_energy_kwh": 0.001,
                "gpu_energy_kwh": 0.001,
                "cumulative_total_energy_kwh": cumulative,
                "duration_seconds": 1.0,
                "gpu_util_avg_pct": 50.0,
            }
        )
    return BaselineTrace(
        case_id="synthetic",
        stage="test",
        task_type="image_classification",
        scenario="synthetic-scratch",
        quality_metric="quality_score",
        max_epochs=10,
        metadata={},
        epochs=tuple(epochs),
        source_job_id="1",
    )


class OptimizationReplayTests(unittest.TestCase):
    def test_fixed_epoch_replay_uses_training_energy_and_best_quality(self):
        result = replay_controller(
            _trace(),
            controller_plugin="controller_benchmark.controllers.fixed_epoch:FixedEpochController",
            controller_id="fixed-five",
            parameters={"stop_epoch": 5},
        )
        self.assertEqual(result.stop_epoch, 5)
        self.assertAlmostEqual(result.baseline_energy_wh, 10.0)
        self.assertAlmostEqual(result.stopped_energy_wh, 5.0)
        self.assertAlmostEqual(result.energy_saving_fraction, 0.5)
        self.assertAlmostEqual(result.stopped_best_quality, 0.37)
        self.assertAlmostEqual(result.quality_regret, 0.038)

    def test_dynamic_tolerance_is_data_derived_and_positive_for_noisy_curve(self):
        info = dynamic_quality_tolerance(_trace())
        self.assertGreater(info["dynamic_quality_tolerance"], 0.0)
        self.assertGreaterEqual(info["dynamic_quality_tolerance"], info["late_stage_uncertainty"])


if __name__ == "__main__":
    unittest.main()
