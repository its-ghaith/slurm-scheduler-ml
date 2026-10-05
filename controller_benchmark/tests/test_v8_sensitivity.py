from __future__ import annotations

import json
import unittest
from pathlib import Path

import numpy as np

from controller_benchmark.optimization.replay import (
    BaselineTrace,
    ReplayResult,
    replay_controller,
)
from controller_benchmark.v8_sensitivity.analysis import (
    calculate_sobol_indices,
    first_order_ale,
)
from controller_benchmark.v8_sensitivity.evaluation import (
    EvaluatedTrace,
    ReplayDiagnostics,
    _scope_metrics,
)


def _row(epoch: int, quality: float) -> dict:
    return {
        "epoch": epoch,
        "epoch_index": epoch,
        "quality_score": quality,
        "epoch_energy_wh": 1.0,
        "duration_seconds": 1.0,
    }


def _trace(task_type: str = "image_classification") -> BaselineTrace:
    return BaselineTrace(
        case_id="case",
        stage="test",
        task_type=task_type,
        scenario="scenario",
        quality_metric="quality_score",
        max_epochs=4,
        metadata={"dataset_fingerprint": "dataset", "training_seed": 0},
        epochs=tuple(
            _row(index, quality)
            for index, quality in enumerate((0.10, 0.20, 0.25, 0.26), start=1)
        ),
        source_job_id="1",
    )


class _LinearModel:
    def predict(self, values):
        return np.column_stack((3.0 * values[:, 0],))


class V8SensitivityTests(unittest.TestCase):
    def test_search_space_contains_23_parameters_without_clipping_dependencies(self):
        path = Path("controller_benchmark/v8_sensitivity/search-space.json")
        document = json.loads(path.read_text(encoding="utf-8"))
        parameters = document["parameters"]
        self.assertEqual(len(parameters), 23)
        self.assertLessEqual(
            parameters["min_fit_points"]["high"],
            parameters["min_epochs_floor"]["low"],
        )
        self.assertLessEqual(
            parameters["min_fit_points"]["high"],
            parameters["bayesian_window"]["low"],
        )
        self.assertLessEqual(
            parameters["bayesian_min_observations"]["high"],
            parameters["bayesian_window"]["low"],
        )

    def test_focused_search_space_contains_only_dominant_parameters(self):
        path = Path("controller_benchmark/v8_sensitivity/search-space-focused.json")
        document = json.loads(path.read_text(encoding="utf-8"))
        self.assertEqual(
            list(document["parameters"]),
            [
                "min_epochs_floor",
                "noise_multiplier",
                "utility_quantile",
                "minimum_quality_floor",
                "trend_window",
            ],
        )

    def test_replay_observer_receives_each_decision_and_timing(self):
        observations = []

        def observer(observation, decision, decision_seconds):
            observations.append((observation.epoch, decision.stop, decision_seconds))

        result = replay_controller(
            _trace(),
            controller_plugin="controller_benchmark.controllers.fixed_epoch:FixedEpochController",
            controller_id="observer-test",
            parameters={"stop_epoch": 3},
            decision_observer=observer,
        )
        self.assertEqual(result.stop_epoch, 3)
        self.assertEqual([row[0] for row in observations], [1, 2, 3])
        self.assertTrue(all(row[2] >= 0.0 for row in observations))

    def test_scope_metrics_use_training_energy_and_quality_tolerance(self):
        trace = _trace()
        result = ReplayResult(
            case_id="case",
            stage="test",
            task_type=trace.task_type,
            stop_epoch=3,
            ideal_stop_epoch=3,
            baseline_best_quality=0.26,
            stopped_best_quality=0.25,
            quality_regret=0.01,
            dynamic_quality_tolerance=0.02,
            normalized_quality_regret=0.5,
            baseline_energy_wh=4.0,
            stopped_energy_wh=3.0,
            energy_saving_fraction=0.25,
            normalized_stop_error=0.0,
            stopped_early=True,
            stop_reason="test",
        )
        diagnostics = ReplayDiagnostics(
            decision_count=3,
            decision_seconds=0.003,
            bayesian_veto_count=1,
        )
        metrics = _scope_metrics(
            [EvaluatedTrace(trace, result, diagnostics)],
            {"dataset": 0.02},
        )
        self.assertAlmostEqual(metrics["mean_energy_saving_fraction"], 0.25)
        self.assertAlmostEqual(metrics["mean_quality_regret_pp"], 1.0)
        self.assertAlmostEqual(metrics["bayesian_veto_rate"], 1.0 / 3.0)
        self.assertAlmostEqual(metrics["controller_runtime_ms_per_epoch"], 1.0)
        self.assertEqual(metrics["joint_success_fraction"], 1.0)

    def test_sobol_indices_rank_known_dominant_parameter(self):
        from SALib.sample.sobol import sample

        problem = {
            "num_vars": 2,
            "names": ["dominant", "minor"],
            "bounds": [[0.0, 1.0], [0.0, 1.0]],
        }
        matrix = sample(
            problem,
            256,
            calc_second_order=True,
            scramble=True,
            seed=11,
        )
        values = (matrix[:, 0] + 0.1 * matrix[:, 1]).reshape(-1, 1)
        rows, _ = calculate_sobol_indices(
            problem,
            matrix,
            values,
            ["output"],
            bootstrap_resamples=64,
            seed=11,
        )
        by_name = {row["parameter"]: row for row in rows}
        self.assertGreater(by_name["dominant"]["st"], 0.95)
        self.assertLess(by_name["minor"]["st"], 0.05)

    def test_ale_curve_preserves_positive_effect_direction(self):
        reference = np.column_stack(
            (np.linspace(0.0, 1.0, 100), np.linspace(1.0, 0.0, 100))
        )
        _, effects, counts = first_order_ale(
            _LinearModel(),
            reference,
            feature_index=0,
            output_index=0,
            bins=5,
        )
        self.assertTrue(np.all(np.diff(effects) > 0.0))
        self.assertEqual(int(counts.sum()), len(reference))
        self.assertAlmostEqual(float(np.average(effects, weights=counts)), 0.0)


if __name__ == "__main__":
    unittest.main()
