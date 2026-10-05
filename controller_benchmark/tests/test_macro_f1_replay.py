from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from controller_benchmark.replay_live_shadow_macro_f1 import replay_macro_f1


class MacroF1ReplayTests(unittest.TestCase):
    def test_replays_macro_f1_without_modifying_accuracy_trace(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            snapshot = root / "snapshot"
            metrics = snapshot / "data/energy_metrics"
            metrics.mkdir(parents=True)
            plan = {
                "benchmark_run_id": "source-run",
                "benchmark_version": "test-v1",
                "acceptance": {
                    "minimum_energy_saving_fraction": 0.0,
                    "maximum_quality_regret": 0.10,
                },
                "controllers": [
                    {
                        "id": "fixed-2",
                        "controller_plugin": (
                            "controller_benchmark.controllers.fixed_epoch:FixedEpochController"
                        ),
                        "controller_parameters": {"stop_epoch": 2},
                    }
                ],
                "runs": [
                    {
                        "sequence": 1,
                        "benchmark_case_id": "classification-test",
                        "benchmark_stage": "image_classification",
                        "task_type": "image_classification",
                        "scenario": "synthetic",
                        "controller_parameters": {"fallback_watts_per_cpu_second": 5.0},
                        "case": {},
                    }
                ],
            }
            plan_path = root / "run-matrix.json"
            plan_path.write_text(json.dumps(plan), encoding="utf-8")
            epochs = [
                {
                    "epoch": epoch,
                    "accuracy": accuracy,
                    "macro_f1": macro_f1,
                    "quality_score": accuracy,
                    "gpu_energy_kwh": 0.001,
                    "duration_seconds": 1.0,
                    "controller_plugin_old_accuracy_signal": 999.0,
                    "should_stop": 1,
                }
                for epoch, accuracy, macro_f1 in [
                    (1, 0.90, 0.40),
                    (2, 0.91, 0.50),
                    (3, 0.92, 0.80),
                ]
            ]
            summary = {
                "benchmark_run_id": "source-run",
                "benchmark_case_id": "classification-test",
                "task_type": "image_classification",
                "epochs": epochs,
            }
            summary_path = metrics / "epoch_summary_job_1.json"
            summary_path.write_text(json.dumps(summary), encoding="utf-8")
            report_dir = snapshot / "controller_benchmark"
            report_dir.mkdir(parents=True)
            original_pair = {
                "case_id": "classification-test",
                "controller_id": "fixed-2",
                "stage": "image_classification",
                "task_type": "image_classification",
                "energy_saving_fraction": 0.0,
                "quality_regret": 0.0,
                "energy_target_met": True,
                "quality_target_met": True,
                "joint_target_met": True,
                "controller_overhead_energy_wh": 0.0,
                "controller_compute_seconds": 0.0,
            }
            (report_dir / "live-shadow-report.json").write_text(
                json.dumps(
                    {
                        "training_jobs": 1,
                        "pairs": [original_pair],
                    }
                ),
                encoding="utf-8",
            )
            result = replay_macro_f1(snapshot, plan_path)
            pair = result["pairs"][0]
            self.assertEqual(pair["stop_epoch"], 2)
            self.assertAlmostEqual(pair["full100_best_quality"], 0.80)
            self.assertAlmostEqual(pair["best_quality_at_stop"], 0.50)
            self.assertEqual(json.loads(summary_path.read_text())["epochs"], epochs)
            self.assertTrue(
                (snapshot / "controller_benchmark/live-shadow-macro-f1-replay-report.json").is_file()
            )
            combined = json.loads(
                (
                    snapshot
                    / "controller_benchmark/live-shadow-mixed-quality-macro-f1-report.json"
                ).read_text()
            )
            self.assertEqual(combined["quality_metrics"]["image_classification"], "macro_f1")
            self.assertAlmostEqual(combined["pairs"][0]["best_quality_at_stop"], 0.50)


if __name__ == "__main__":
    unittest.main()
