from __future__ import annotations

import unittest
from pathlib import Path

from controller_benchmark.api import ControllerContext, EpochObservation
from controller_benchmark.controllers.risk_constrained_bayesian_multi_horizon import (
    RiskConstrainedBayesianMultiHorizonController,
)
from controller_benchmark.manifest import load_manifest
from controller_benchmark.scientific_optimization.metrics import (
    select_energy_under_quality_constraint,
)


def _row(epoch: int, quality: float, learning_rate: float = 0.001) -> dict:
    return {
        "epoch": epoch,
        "epoch_index": epoch,
        "quality_score": quality,
        "epoch_energy_wh": 1.0,
        "duration_seconds": 2.0,
        "gpu_util_avg_pct": 60.0,
        "gpu_power_avg_w": 150.0,
        "gpu_mem_used_avg_mb": 2048.0,
        "learning_rate": learning_rate,
        "gradient_norm": 0.1,
        "train_loss": max(0.01, 1.0 - quality),
        "model_parameter_count": 3_000_000,
        "model_flops": 8_000_000_000,
    }


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


def _context(task_type: str = "object_detection", scenario: str = "unknown"):
    return ControllerContext(
        controller_id="rapec-v9-test",
        benchmark_version="v9-test",
        task_type=task_type,
        quality_metric="quality_score",
        scenario=scenario,
        max_epochs=100,
        parameters={
            "horizons": [1, 5, 10, 20],
            "posterior_samples": 512,
            "model_window": 40,
            "minimum_observations_floor": 8,
            "minimum_calibration_points": 6,
            "calibration_window": 30,
            "quality_risk_alpha": 0.25,
            "evidence_decay": 0.2,
            "evidence_stop_probability": 0.8,
        },
    )


class RapecV9Tests(unittest.TestCase):
    def test_dynamic_equivalence_is_not_inflated_by_warmup_gains(self):
        controller = RiskConstrainedBayesianMultiHorizonController(_context())
        qualities = [0.0, 0.20, 0.38, 0.52, 0.60]
        qualities.extend(0.70 + 0.001 * index for index in range(35))
        epsilon = controller._dynamic_quality_equivalence(qualities, 20.0)
        self.assertLess(epsilon, 0.05)
        self.assertGreater(epsilon, 0.0)

    def test_decision_does_not_depend_on_task_or_scenario_name(self):
        contexts = [
            _context("object_detection", "carpk-pretrained"),
            _context("semantic_segmentation", "unseen-scratch"),
        ]
        history = [_row(epoch, min(0.75, 0.1 + 0.02 * epoch)) for epoch in range(1, 45)]
        current = _row(45, 0.75)
        decisions = [
            RiskConstrainedBayesianMultiHorizonController(context).evaluate(
                _observation(context, history, current)
            )
            for context in contexts
        ]
        self.assertEqual(decisions[0].stop, decisions[1].stop)
        for key in (
            "rapec9_dynamic_quality_epsilon",
            "rapec9_max_probability_relevant_gain",
            "rapec9_dynamic_minimum_observations",
        ):
            self.assertAlmostEqual(
                decisions[0].diagnostics[key], decisions[1].diagnostics[key]
            )
        self.assertEqual(decisions[0].diagnostics["rapec9_uses_task_or_dataset_profile"], 0)

    def test_active_learning_curve_is_not_stopped(self):
        context = _context()
        controller = RiskConstrainedBayesianMultiHorizonController(context)
        history = [_row(epoch, 0.10 + 0.007 * epoch) for epoch in range(1, 50)]
        current = _row(50, 0.45)
        decision = controller.evaluate(_observation(context, history, current))
        self.assertFalse(decision.stop)
        self.assertFalse(decision.diagnostics["rapec9_quality_constraint_met"])

    def test_one_candidate_epoch_cannot_stop_and_lr_restart_is_guarded(self):
        context = _context()
        controller = RiskConstrainedBayesianMultiHorizonController(context)
        first_evidence = controller._update_stop_evidence(True, 0.0)
        self.assertLess(first_evidence, controller.evidence_stop_probability)

        qualities = [min(0.72, 0.08 + 0.025 * epoch) for epoch in range(1, 31)]
        qualities.extend([0.72] * 30)
        history: list[dict] = []
        for epoch, quality in enumerate(qualities, 1):
            current = _row(epoch, quality)
            controller.evaluate(_observation(context, history, current))
            history.append(current)

        restart = _row(len(history) + 1, 0.72, learning_rate=0.01)
        history[-2]["learning_rate"] = 0.0002
        history[-1]["learning_rate"] = 0.0001
        guarded = controller.evaluate(_observation(context, history, restart))
        self.assertEqual(guarded.diagnostics["rapec9_learning_rate_transition"], 1)
        self.assertFalse(guarded.stop)

    def test_constraint_first_selection_never_trades_quality_for_energy(self):
        records = [
            {
                "trial_number": 1,
                "metrics": {
                    "quality_constraint": -0.01,
                    "energy_saving_q25": 0.20,
                    "mean_energy_saving_fraction": 0.25,
                    "quality_cvar90": 0.30,
                    "maximum_quality_regret": 0.01,
                },
            },
            {
                "trial_number": 2,
                "metrics": {
                    "quality_constraint": 0.20,
                    "energy_saving_q25": 0.80,
                    "mean_energy_saving_fraction": 0.85,
                    "quality_cvar90": 2.00,
                    "maximum_quality_regret": 0.20,
                },
            },
        ]
        selected = select_energy_under_quality_constraint(records)
        self.assertEqual(selected["trial_number"], 1)
        self.assertTrue(selected["quality_feasible"])

    def test_calibration_samples_residual_distribution_not_worst_case_shift(self):
        controller = RiskConstrainedBayesianMultiHorizonController(_context())
        best = [min(0.8, 0.05 + 0.02 * epoch) for epoch in range(1, 70)]
        best[45:] = [best[44]] * (len(best) - 45)
        gains = __import__("numpy").zeros(512)
        calibrated = controller._calibrated_gain_samples(
            gains,
            best=best,
            horizon=20,
            seed=7,
            room=1.0,
        )
        radius = controller._horizon_calibration_radius(best, 20, 0.99)
        self.assertGreaterEqual(radius, 0.0)
        self.assertLessEqual(float(calibrated.mean()), radius)

    def test_long_horizon_calibration_keeps_one_step_safety_floor(self):
        controller = RiskConstrainedBayesianMultiHorizonController(_context())
        best = [min(0.8, 0.05 + 0.02 * epoch) for epoch in range(1, 70)]
        best[45:] = [best[44]] * (len(best) - 45)
        one_step = controller._horizon_calibration_radius(best, 1, 0.99)
        long_horizon = controller._horizon_calibration_radius(best, 20, 0.99)
        self.assertGreaterEqual(long_horizon, one_step)

    def test_short_recovery_blocks_stop(self):
        controller = RiskConstrainedBayesianMultiHorizonController(_context())
        best = [0.10] * 10 + [0.10, 0.10, 0.12, 0.14, 0.16]
        self.assertTrue(controller._regime_recovery(best, epsilon=0.02))

    def test_record_waiting_guard_learns_patience_from_current_run(self):
        controller = RiskConstrainedBayesianMultiHorizonController(_context())
        best = [0.1, 0.2, 0.2, 0.21, 0.21, 0.21, 0.22, 0.22]
        self.assertTrue(controller._record_waiting_time_recovery(best))
        best.extend([best[-1]] * 10)
        self.assertFalse(controller._record_waiting_time_recovery(best))

    def test_v9_validation_uses_fresh_seed_and_epoch_energy(self):
        manifest = load_manifest(
            Path("controller_benchmark/config/benchmark-v4-rapec-v9-nine-dataset.json")
        )
        self.assertEqual(manifest["execution"]["energy_comparison_scope"], "epoch")
        self.assertEqual(
            manifest["acceptance"]["quality_tolerance_mode"],
            "full100_dynamic",
        )
        self.assertEqual(
            manifest["execution"]["measurement_protocol_version"],
            "rapec-v9-epoch-energy-v1",
        )
        seeds = {
            case["training_seed"]
            for stage in manifest["stages"]
            for case in stage["cases"]
        }
        self.assertEqual(seeds, {1})


if __name__ == "__main__":
    unittest.main()
