from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from controller_benchmark.api import ControllerContext, EpochObservation
from controller_benchmark.analyze_live_shadow import _best_quality, analyze_live_shadow
from controller_benchmark.controllers.live_shadow_suite import LiveShadowSuiteController
from controller_benchmark.live_shadow_plan import (
    build_live_shadow_plan,
    load_shadow_configuration,
)
from controller_benchmark.manifest import load_manifest
from controller_benchmark.runners.classification_metrics import primary_quality


ROOT = Path(__file__).resolve().parents[2]


def observation(epoch: int, quality: float, history: list[dict]) -> EpochObservation:
    return EpochObservation(
        epoch=epoch,
        max_epochs=5,
        task_type="image_classification",
        quality_metric="quality_score",
        quality=quality,
        best_quality=max([quality, *[row["quality_score"] for row in history]]),
        delta_quality=(quality - history[-1]["quality_score"] if history else None),
        epoch_energy_wh=1.0,
        cumulative_energy_wh=float(epoch),
        epoch_duration_seconds=2.0,
        cumulative_duration_seconds=float(epoch * 2),
        gpu_utilization_pct=50.0,
        history=tuple(history),
        raw_metrics={"quality_score": quality},
    )


class StandardEarlyStoppingTests(unittest.TestCase):
    def test_standard_early_stopping_uses_no_task_profile(self) -> None:
        from controller_benchmark.controllers.standard_early_stopping import (
            StandardEarlyStoppingController,
        )

        controller = StandardEarlyStoppingController(
            ControllerContext(
                controller_id="es",
                benchmark_version="test",
                task_type="semantic_segmentation",
                quality_metric="quality_score",
                scenario="arbitrary",
                max_epochs=100,
                parameters={"start_epoch": 2, "patience": 2, "min_delta": 0.001},
            )
        )
        history: list[dict] = []
        decisions = []
        for epoch, quality in [(1, 0.5), (2, 0.5005), (3, 0.5006)]:
            decisions.append(controller.evaluate(observation(epoch, quality, history)))
            history.append({"quality_score": quality})
        self.assertFalse(decisions[0].stop)
        self.assertFalse(decisions[1].stop)
        self.assertTrue(decisions[2].stop)


class LiveShadowSuiteTests(unittest.TestCase):
    def test_suite_records_stop_but_never_stops_full100(self) -> None:
        with tempfile.TemporaryDirectory() as temporary, patch.dict(
            os.environ,
            {
                "BENCHMARK_METRICS_DIR": temporary,
                "SLURM_JOB_ID": "123",
                "BENCHMARK_RUN_ID": "shadow-test",
            },
        ):
            suite = LiveShadowSuiteController(
                ControllerContext(
                    controller_id="live-shadow-suite",
                    benchmark_version="shadow-test-v1",
                    task_type="image_classification",
                    quality_metric="quality_score",
                    scenario="synthetic",
                    max_epochs=5,
                    metadata={"benchmark_case_id": "synthetic", "benchmark_stage": "test"},
                    parameters={
                        "fallback_watts_per_cpu_second": 5.0,
                        "controllers": [
                            {
                                "id": "fixed-3",
                                "controller_plugin": "controller_benchmark.controllers.fixed_epoch:FixedEpochController",
                                "controller_parameters": {"stop_epoch": 3},
                            }
                        ],
                    },
                )
            )
            history: list[dict] = []
            for epoch in range(1, 6):
                decision = suite.evaluate(observation(epoch, 0.5, history))
                self.assertFalse(decision.stop)
                history.append({"quality_score": 0.5})
            document = json.loads(
                (Path(temporary) / "shadow_summary_job_123.json").read_text(encoding="utf-8")
            )
            result = document["controllers"][0]
            self.assertEqual(result["stop_epoch"], 3)
            self.assertEqual(result["training_energy_to_stop_wh"], 3.0)
            self.assertGreaterEqual(result["controller_overhead_energy_wh"], 0.0)
            self.assertGreaterEqual(
                result["counterfactual_total_energy_wh"],
                result["training_energy_to_stop_wh"],
            )
            self.assertEqual(result["controller_decisions"], 3)


class LiveShadowPlanTests(unittest.TestCase):
    def test_development_plan_has_nine_seed_zero_full100_jobs(self) -> None:
        manifest = load_manifest(
            ROOT / "controller_benchmark/config/benchmark-v3-nine-dataset-fresh.json"
        )
        configuration = load_shadow_configuration(
            ROOT / "controller_benchmark/config/live-shadow-controllers.json",
            manifest,
        )
        plan = build_live_shadow_plan(manifest, configuration, run_id="shadow-test")
        self.assertEqual(plan["plan_type"], "live_shadow_campaign")
        self.assertEqual(len(plan["runs"]), 9)
        self.assertEqual({run["training_seed"] for run in plan["runs"]}, {0})
        self.assertEqual(plan["planned_baseline_runs"], 0)
        self.assertEqual(plan["planned_candidate_runs"], 9)
        self.assertEqual(len(plan["controllers"]), 4)
        self.assertTrue(all(run["controller_mode"] == "plugin" for run in plan["runs"]))

    def test_plan_can_wait_for_a_pretraining_job(self) -> None:
        manifest = load_manifest(
            ROOT / "controller_benchmark/config/benchmark-v3-nine-dataset-fresh.json"
        )
        configuration = load_shadow_configuration(
            ROOT / "controller_benchmark/config/live-shadow-controllers.json",
            manifest,
        )
        plan = build_live_shadow_plan(
            manifest,
            configuration,
            run_id="shadow-pretraining-test",
            pretraining_job_id="153",
        )
        self.assertEqual(plan["execution"]["pretraining_job_id"], "153")
        self.assertEqual(plan["execution"]["pretraining_timeout_seconds"], 172800)
        with self.assertRaises(ValueError):
            build_live_shadow_plan(
                manifest,
                configuration,
                run_id="shadow-invalid-pretraining-test",
                pretraining_job_id="153; rm -rf /",
            )

    def test_best_quality_supports_carpk_legacy_epoch_summary(self) -> None:
        summary = {
            "quality_metric": "quality_score",
            "epochs": [
                {"quality_score": 0.51, "best_quality_score": 0.51},
                {"quality_score": 0.49, "best_quality_score": 0.51},
                {"quality_score": 0.53, "best_quality_score": 0.53},
            ],
        }
        self.assertEqual(_best_quality(summary), 0.53)

    def test_analyzer_exports_all_nine_by_four_shadow_pairs(self) -> None:
        manifest = load_manifest(
            ROOT / "controller_benchmark/config/benchmark-v3-nine-dataset-fresh.json"
        )
        configuration = load_shadow_configuration(
            ROOT / "controller_benchmark/config/live-shadow-controllers.json",
            manifest,
        )
        plan = build_live_shadow_plan(manifest, configuration, run_id="shadow-analysis-test")
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            plan_path = root / "run-matrix.json"
            plan_path.write_text(json.dumps(plan), encoding="utf-8")
            metrics = root / "snapshot/data/energy_metrics"
            metrics.mkdir(parents=True)
            for run in plan["runs"]:
                job_id = str(run["sequence"])
                epochs = [
                    {"quality_score": 0.80, "best_quality_score": 0.80},
                    {"quality_score": 0.90, "best_quality_score": 0.90},
                ]
                epoch_summary = {
                    "benchmark_run_id": plan["benchmark_run_id"],
                    "job_id": job_id,
                    "quality_metric": "quality_score",
                    "epochs_completed": 100,
                    "epochs": epochs,
                    "total_gpu_energy_kwh": 0.1,
                }
                (metrics / f"epoch_summary_job_{job_id}.json").write_text(
                    json.dumps(epoch_summary), encoding="utf-8"
                )
                controllers = []
                for definition in plan["controllers"]:
                    controllers.append(
                        {
                            "controller_id": definition["id"],
                            "controller_plugin": definition["controller_plugin"],
                            "stop_epoch": 50,
                            "stopped_early": True,
                            "stop_reason": "synthetic",
                            "best_quality_at_stop": 0.89,
                            "training_energy_to_stop_wh": 50.0,
                            "controller_overhead_energy_wh": 0.01,
                            "controller_compute_seconds": 0.02,
                            "controller_process_cpu_seconds": 0.01,
                            "controller_decisions": 50,
                            "training_time_to_stop_s": 100.0,
                            "energy_measurement_method": "process_cpu_time_estimate",
                        }
                    )
                shadow_summary = {
                    "benchmark_run_id": plan["benchmark_run_id"],
                    "benchmark_case_id": run["benchmark_case_id"],
                    "job_id": job_id,
                    "energy_scope": "epoch_gpu_plus_attributed_controller_cpu",
                    "lifecycle_energy_complete": False,
                    "controllers": controllers,
                }
                (metrics / f"shadow_summary_job_{job_id}.json").write_text(
                    json.dumps(shadow_summary), encoding="utf-8"
                )
            result = analyze_live_shadow(root / "snapshot", plan_path)
            self.assertEqual(result["training_jobs"], 9)
            self.assertEqual(result["controller_pairs"], 36)
            self.assertEqual(result["represented_jobs_total"], 45)
            self.assertEqual(result["real_full100_jobs"], 9)
            self.assertEqual(result["virtual_es_jobs"], 9)
            represented_ids = [row["represented_job_id"] for row in result["represented_jobs"]]
            self.assertEqual(len(represented_ids), len(set(represented_ids)))
            self.assertTrue(
                (root / "snapshot/controller_benchmark/paired-results.csv").is_file()
            )
            self.assertTrue(
                (metrics / "node_exporter/controller_benchmark.prom").is_file()
            )

    def test_classification_plan_uses_macro_f1_for_exactly_three_jobs(self) -> None:
        manifest = load_manifest(
            ROOT
            / "controller_benchmark/config/benchmark-live-shadow-classification-macro-f1.json"
        )
        configuration = load_shadow_configuration(
            ROOT / "controller_benchmark/config/live-shadow-controllers.json",
            manifest,
        )
        plan = build_live_shadow_plan(
            manifest,
            configuration,
            run_id="macro-f1-test",
            expected_cases=3,
            benchmark_version="macro-f1-test-v1",
            dashboard_file="macro-f1.json",
            experiment_name="macro-f1-test",
        )
        self.assertEqual(len(plan["runs"]), 3)
        self.assertEqual({run["training_seed"] for run in plan["runs"]}, {0})
        self.assertEqual({run["task_type"] for run in plan["runs"]}, {"image_classification"})
        self.assertEqual({run["case"]["primary_metric"] for run in plan["runs"]}, {"macro_f1"})
        self.assertEqual(plan["execution"]["dashboard_files"], ["macro-f1.json"])

    def test_classification_primary_quality_selects_macro_f1(self) -> None:
        self.assertEqual(primary_quality("accuracy", 0.8, 0.7), 0.8)
        self.assertEqual(primary_quality("macro_f1", 0.8, 0.7), 0.7)
        with self.assertRaises(ValueError):
            primary_quality("unsupported", 0.8, 0.7)


if __name__ == "__main__":
    unittest.main()
