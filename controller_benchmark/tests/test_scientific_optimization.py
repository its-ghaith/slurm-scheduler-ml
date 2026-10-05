from __future__ import annotations

import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from controller_benchmark.api import ControllerContext, EpochObservation
from controller_benchmark.controllers.task_independent_rapec_v3 import (
    TaskIndependentRapecV3Controller,
)
from controller_benchmark.optimization.replay import BaselineTrace
from controller_benchmark.optimization.replay import ReplayResult
from controller_benchmark.scientific_optimization.build_validation_manifest import (
    build_validation_manifest,
)
from controller_benchmark.scientific_optimization.metrics import (
    calibrated_tolerances,
    evaluate_parameters,
    select_nash_compromise,
)
from controller_benchmark.scientific_optimization.optimize import (
    _base_parameters,
    _directed_extreme_parameters,
    _read_json,
)
from controller_benchmark.scientific_optimization.mobo import run_sobol_search
from controller_benchmark.scientific_optimization.oracle import build_oracle_report
from controller_benchmark.scientific_optimization.optimizer_benchmark import _aggregate
from controller_benchmark.scientific_optimization.parameters import ParameterCodec
from controller_benchmark.scientific_optimization.progress import ProgressReporter
from controller_benchmark.scientific_optimization.replay_cache import PersistentReplayCache
from controller_benchmark.scientific_optimization.rf_parego import run_rf_parego_search


def _row(epoch: int, quality: float) -> dict:
    return {
        "epoch": epoch,
        "epoch_index": epoch,
        "quality_score": quality,
        "total_energy_kwh": 0.001,
        "duration_seconds": 1.0,
        "gpu_util_avg_pct": 50.0,
        "gpu_power_avg_w": 100.0,
        "gpu_mem_used_avg_mb": 1024.0,
        "learning_rate": 0.001,
        "gradient_norm": 0.1,
        "train_loss": 0.5,
    }


def _trace(seed: int, quality_offset: float = 0.0) -> BaselineTrace:
    qualities = [0.1, 0.2, 0.3, 0.36, 0.39, 0.40, 0.401, 0.400, 0.401, 0.401]
    return BaselineTrace(
        case_id="same-dataset",
        stage="test",
        task_type="image_classification",
        scenario="ignored",
        quality_metric="quality_score",
        max_epochs=10,
        metadata={"dataset_fingerprint": "same-data", "training_seed": seed},
        epochs=tuple(_row(index, value + quality_offset) for index, value in enumerate(qualities, 1)),
        source_job_id=str(seed),
    )


class ScientificOptimizationTests(unittest.TestCase):
    def test_focused_v8_space_uses_four_sensitivity_supported_parameters(self):
        path = Path(
            "controller_benchmark/scientific_optimization/"
            "search-space-rapec-v8-focused.json"
        )
        search_space = _read_json(path)
        self.assertEqual(
            set(search_space["mandatory_parameters"]),
            {
                "min_epochs_floor",
                "utility_quantile",
                "noise_multiplier",
                "minimum_quality_floor",
            },
        )
        self.assertEqual(search_space["fixed_parameters"]["trend_window"], 10)

    def test_focused_v8_extremes_follow_ale_directions(self):
        base = _base_parameters(
            "controller_benchmark/config/rapec-v3-v8-controllers.json",
            "rapec-v8",
        )
        search_space = _read_json(
            "controller_benchmark/scientific_optimization/"
            "search-space-rapec-v8-focused.json"
        )
        safe = _directed_extreme_parameters(base, search_space, "safe")
        aggressive = _directed_extreme_parameters(base, search_space, "aggressive")
        self.assertGreater(safe["min_epochs_floor"], aggressive["min_epochs_floor"])
        self.assertLess(safe["utility_quantile"], aggressive["utility_quantile"])
        self.assertLess(safe["noise_multiplier"], aggressive["noise_multiplier"])
        self.assertGreater(
            safe["minimum_quality_floor"], aggressive["minimum_quality_floor"]
        )
        self.assertEqual(safe["trend_window"], 10)
        self.assertEqual(safe["bayesian_samples"], 512)

    def test_progress_reporter_writes_immediate_jsonl_records(self):
        with TemporaryDirectory() as directory:
            reporter = ProgressReporter(Path(directory))
            reporter.emit("optimization_trial_completed", {"trial": 4, "completed": 4, "total": 10})
            lines = (Path(directory) / "progress.jsonl").read_text(encoding="utf-8").splitlines()
        self.assertEqual(len(lines), 1)
        self.assertIn('"event": "optimization_trial_completed"', lines[0])
        self.assertIn('"trial": 4', lines[0])

    def test_sobol_search_reports_each_completed_trial(self):
        events = []

        def evaluate(parameters):
            return {
                "metrics": {
                    "quality_cvar90": parameters["value"],
                    "energy_saving_q25": 1.0 - parameters["value"],
                    "quality_constraint": -0.1,
                },
                "cases": [],
            }

        records = run_sobol_search(
            dimension=1,
            trials=3,
            seed=7,
            decode=lambda vector: {"value": float(vector[0])},
            evaluator=evaluate,
            progress=lambda event, details: events.append((event, details)),
        )
        self.assertEqual(len(records), 3)
        self.assertEqual(len(events), 3)
        self.assertEqual(events[-1][0], "optimization_trial_completed")
        self.assertEqual(events[-1][1]["completed"], 3)

    def test_parameter_codec_projects_mixed_search_space(self):
        codec = ParameterCodec.from_search_space(
            {"fixed_base": 3},
            {
                "parameters": {
                    "integer": {"type": "int", "low": 2, "high": 8, "step": 2},
                    "floating": {"type": "float", "low": 0.1, "high": 0.5},
                    "choice": {"type": "categorical", "choices": [32, 64, 96]},
                    "fixed": {"type": "fixed", "value": 0.0},
                }
            },
        )
        parameters = codec.decode([1.0, 0.5, 0.99])
        self.assertEqual(parameters["integer"], 8)
        self.assertAlmostEqual(parameters["floating"], 0.3)
        self.assertEqual(parameters["choice"], 96)
        self.assertEqual(parameters["fixed"], 0.0)
        round_trip = codec.decode(codec.encode(parameters))
        self.assertEqual(round_trip["integer"], 8)
        self.assertEqual(round_trip["choice"], 96)

    def test_repeated_full100_variation_increases_calibrated_tolerance(self):
        single = calibrated_tolerances([_trace(0)])["same-data"]
        repeated = calibrated_tolerances([_trace(0), _trace(1, 0.02), _trace(2, -0.01)])["same-data"]
        self.assertGreater(repeated, single)

    def test_oracle_report_exposes_energy_ceiling_and_replication_limit(self):
        report = build_oracle_report([_trace(0)])
        self.assertEqual(report["trace_count"], 1)
        self.assertGreater(report["oracle_energy_saving_q25"], 0.0)
        self.assertTrue(
            report["replication_audit"]["single_replica_probability_is_binary"]
        )
        self.assertFalse(
            report["replication_audit"]["probabilistic_noninferiority_estimable"]
        )

    def test_persistent_replay_cache_uses_exact_parameters(self):
        result = ReplayResult(
            case_id="case",
            stage="stage",
            task_type="image_classification",
            stop_epoch=8,
            ideal_stop_epoch=7,
            baseline_best_quality=0.9,
            stopped_best_quality=0.89,
            quality_regret=0.01,
            dynamic_quality_tolerance=0.02,
            normalized_quality_regret=0.5,
            baseline_energy_wh=10.0,
            stopped_energy_wh=8.0,
            energy_saving_fraction=0.2,
            normalized_stop_error=0.1,
            stopped_early=True,
            stop_reason="test",
        )
        plugin = (
            "controller_benchmark.controllers.task_independent_rapec_v3:"
            "TaskIndependentRapecV3Controller"
        )
        with TemporaryDirectory() as directory:
            cache = PersistentReplayCache(directory)
            key = cache.key(
                _trace(0),
                controller_plugin=plugin,
                controller_id="fixed",
                parameters={"threshold": 0.123456},
            )
            different = cache.key(
                _trace(0),
                controller_plugin=plugin,
                controller_id="fixed",
                parameters={"threshold": 0.123457},
            )
            cache.put(key, result)
            restored = cache.get(key)
        self.assertNotEqual(key, different)
        self.assertEqual(restored, result)

    def test_replay_workers_are_reusable_across_parameter_evaluations(self):
        plugin = (
            "controller_benchmark.controllers.task_independent_rapec_v3:"
            "TaskIndependentRapecV3Controller"
        )
        traces = [_trace(0), _trace(1, 0.01)]
        first = evaluate_parameters(
            traces,
            controller_plugin=plugin,
            controller_id="fixed-worker-test",
            parameters={"min_epochs_floor": 3, "min_fit_points": 3},
            replay_workers=2,
        )
        second = evaluate_parameters(
            traces,
            controller_plugin=plugin,
            controller_id="fixed-worker-test",
            parameters={"min_epochs_floor": 4, "min_fit_points": 3},
            replay_workers=2,
        )
        self.assertEqual(len(first.cases), 2)
        self.assertEqual(len(second.cases), 2)

    def test_rf_parego_deduplicates_decoded_integer_configurations(self):
        codec = ParameterCodec.from_search_space(
            {},
            {
                "parameters": {
                    "level": {"type": "int", "low": 0, "high": 4, "step": 2},
                    "threshold": {"type": "float", "low": 0.0, "high": 1.0},
                }
            },
        )

        def evaluate(parameters):
            level = int(parameters["level"])
            threshold = float(parameters["threshold"])
            violation = max(0.0, 0.25 - threshold) if level >= 2 else 0.25
            return {
                "metrics": {
                    "quality_cvar90": abs(0.75 - threshold) + 0.05 * (4 - level),
                    "energy_saving_q25": 0.10 * level + 0.20 * threshold,
                    "quality_constraint": violation,
                },
                "cases": [],
            }

        records = run_rf_parego_search(
            dimension=2,
            trials=12,
            initial_trials=6,
            seed=11,
            decode=codec.decode,
            encode=codec.encode,
            evaluator=evaluate,
            initial_vectors=([0.0, 0.0], [1.0, 1.0]),
            candidate_pool_size=128,
            trees=32,
            min_samples_leaf=1,
            random_design_probability=0.0,
        )
        keys = {
            tuple(sorted(record["parameters"].items())) for record in records
        }
        self.assertEqual(len(keys), len(records))
        self.assertTrue(
            any(record["metrics"]["quality_constraint"] <= 0.0 for record in records)
        )

    def test_optimizer_aggregate_keeps_constraint_first_selected_controller_shape(self):
        def trial(quality, energy, constraint=0.0):
            return {
                "metrics": {
                    "quality_cvar90": quality,
                    "energy_saving_q25": energy,
                    "quality_constraint": constraint,
                }
            }

        rows = []
        with TemporaryDirectory() as directory:
            root = Path(directory)
            for backend, energy in (("sobol", 0.10), ("rf-parego", 0.25)):
                output = root / backend
                output.mkdir()
                trials = [trial(0.2, 0.05), trial(0.1, energy)]
                (output / "trials.json").write_text(
                    json.dumps(trials), encoding="utf-8"
                )
                (output / "selected-controller.json").write_text(
                    json.dumps(
                        {
                            "controller_id": "fixed",
                            "controller_plugin": "plugin:Controller",
                            "parameters": {"x": energy},
                            "quality_feasible": True,
                            "metrics": trials[-1]["metrics"],
                        }
                    ),
                    encoding="utf-8",
                )
                rows.append(
                    {
                        "backend": backend,
                        "optimizer_seed": 0,
                        "scope": "final-fit",
                        "selected_quality_feasible": True,
                        "selected_quality_cvar90": 0.1,
                        "selected_energy_saving_q25": energy,
                        "elapsed_seconds": 1.0,
                        "output_dir": str(output),
                        "_trials": trials,
                    }
                )
            summaries, selected = _aggregate(
                rows, {"oracle_energy_saving_q25": 0.4}
            )
        self.assertEqual(
            selected["optimizer_benchmark_selection"]["winning_backend"],
            "rf-parego",
        )
        self.assertEqual(selected["controller_id"], "fixed")
        self.assertNotIn("controller", selected)
        self.assertFalse(
            selected["optimizer_benchmark_selection"]["scientific_selection_ready"]
        )
        self.assertEqual(len(summaries), 2)

    def test_nash_selection_rejects_both_zero_benefit_and_extreme_endpoint(self):
        def record(number, quality, energy):
            return {
                "trial_number": number,
                "metrics": {
                    "quality_cvar90": quality,
                    "energy_saving_q25": energy,
                    "quality_constraint": 0.0,
                },
            }

        selected = select_nash_compromise(
            [record(0, 0.0, 0.0), record(1, 0.1, 0.2), record(2, 1.0, 0.5)]
        )
        self.assertEqual(selected["trial_number"], 1)
        self.assertGreater(selected["nash_product"], 0.0)
        self.assertTrue(selected["quality_feasible"])

    def test_controller_decision_is_independent_of_task_and_scenario_names(self):
        parameters = {
            "min_fit_points": 5,
            "horizon_epochs": 3,
            "trend_window": 6,
            "min_epochs_floor": 6,
            "bootstrap_samples": 8,
            "warmup_required_stable_windows": 1,
        }
        contexts = [
            ControllerContext(
                "rapec-v3-ti",
                "v1",
                "object_detection",
                "quality_score",
                "carpk-full-scratch",
                100,
                parameters=parameters,
            ),
            ControllerContext(
                "rapec-v3-ti",
                "v1",
                "semantic_segmentation",
                "quality_score",
                "unknown-pretrained",
                100,
                parameters=parameters,
            ),
        ]
        history = [_row(epoch, min(0.60, 0.1 + 0.03 * epoch)) for epoch in range(1, 26)]
        current = _row(26, 0.60)
        decisions = []
        for context in contexts:
            observation = EpochObservation(
                epoch=26,
                max_epochs=100,
                task_type=context.task_type,
                quality_metric="quality_score",
                quality=0.60,
                best_quality=0.60,
                delta_quality=0.0,
                epoch_energy_wh=1.0,
                cumulative_energy_wh=26.0,
                epoch_duration_seconds=1.0,
                cumulative_duration_seconds=26.0,
                gpu_utilization_pct=50.0,
                history=tuple(history),
                raw_metrics=current,
            )
            decisions.append(TaskIndependentRapecV3Controller(context).evaluate(observation))
        self.assertEqual(decisions[0].stop, decisions[1].stop)
        self.assertEqual(
            decisions[0].diagnostics["rapec_adaptive_min_epochs"],
            decisions[1].diagnostics["rapec_adaptive_min_epochs"],
        )
        self.assertEqual(decisions[0].diagnostics["rapec_ti_uses_task_or_scenario_profile"], 0)

    def test_fresh_seed_manifest_preserves_source_and_expands_cases(self):
        source = {
            "benchmark_version": "v1",
            "description": "source",
            "execution": {"experiment_name": "v1", "dashboard_files": []},
            "stages": [
                {
                    "id": "stage",
                    "enabled": True,
                    "cases": [{"id": "case", "training_seed": 0}],
                }
            ],
        }
        result = build_validation_manifest(source, [1, 2, 3])
        self.assertEqual([case["id"] for case in result["stages"][0]["cases"]], [
            "case-seed-1", "case-seed-2", "case-seed-3"
        ])
        self.assertEqual(source["stages"][0]["cases"][0]["id"], "case")


if __name__ == "__main__":
    unittest.main()
