from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from controller_benchmark.api import ControllerContext, EpochObservation
from controller_benchmark.controllers.generalized_rapec_energy5 import (
    GeneralizedRapecEnergy5Controller,
)
from controller_benchmark.controllers.rapec_v3_pf import ProfileFreeRapecV3Controller
from controller_benchmark.generalized_manifest import build_manifest
from controller_benchmark.task_adapter import QualityDefinition


ROOT = Path(__file__).resolve().parents[2]


PARAMETERS = {
    "horizon_epochs": 5,
    "min_fit_points": 6,
    "patience": 2,
    "trend_window": 6,
    "bootstrap_samples": 16,
    "knn_neighbors": 3,
    "min_epochs_floor": 8,
    "warmup_significance_z": 1.28,
    "warmup_low_progress_quantile": 0.4,
    "warmup_required_low_progress_windows": 1,
    "warmup_min_reference_points": 3,
    "warmup_no_learning_multiplier": 1.5,
    "warmup_confirmation_horizon_multipliers": [1],
    "conformal_min_residuals": 2,
    "maximum_dynamic_confirmations": 4,
    "optimization_window": 5,
}


def run_trace(
    task_type: str,
    scenario: str,
    controller_class=GeneralizedRapecEnergy5Controller,
) -> list[tuple[bool, str, int, int]]:
    context = ControllerContext(
        controller_id="rapec-g-test",
        benchmark_version="test",
        task_type=task_type,
        quality_metric="quality_score",
        scenario=scenario,
        max_epochs=60,
        parameters=PARAMETERS,
        metadata={"model_version": f"different-{task_type}"},
    )
    controller = controller_class(context)
    history = []
    output = []
    best = 0.0
    cumulative_energy = 0.0
    for checkpoint in range(1, 61):
        quality = min(0.78, 0.15 + 0.03 * checkpoint) if checkpoint <= 20 else 0.75 + (checkpoint % 2) * 0.00005
        best = max(best, quality)
        cumulative_energy += 1.5
        event = {
            "epoch": checkpoint,
            "epoch_index": checkpoint,
            "decision_checkpoint": checkpoint,
            "quality_score": quality,
            "best_quality_score": best,
            "total_energy_kwh": 0.0015,
            "cumulative_total_energy_kwh": cumulative_energy / 1000.0,
            "duration_seconds": 10.0,
            "gpu_util_avg_pct": 75.0,
            "gpu_mem_used_avg_mb": 2048.0,
            "gpu_memory_total_mb": 8192.0,
            "gpu_power_avg_w": 180.0,
            "train_loss": max(0.1, 1.0 / checkpoint),
            "gradient_norm": max(0.01, 1.0 / checkpoint),
            "learning_rate": 0.001 * (1.0 - checkpoint / 100.0),
            "model_parameter_count": 1000000,
            "model_flops": 100000000,
        }
        observation = EpochObservation(
            epoch=checkpoint,
            max_epochs=60,
            task_type=task_type,
            quality_metric="quality_score",
            quality=quality,
            best_quality=best,
            delta_quality=quality - history[-1]["quality_score"] if history else None,
            epoch_energy_wh=1.5,
            cumulative_energy_wh=cumulative_energy,
            epoch_duration_seconds=10.0,
            cumulative_duration_seconds=checkpoint * 10.0,
            gpu_utilization_pct=75.0,
            history=tuple(history),
            raw_metrics=event,
        )
        decision = controller.evaluate(observation)
        output.append(
            (
                decision.stop,
                decision.reason,
                int(decision.diagnostics.get("rapec_low_value_streak", 0)),
                int(decision.diagnostics.get("rapec_candidate_stop", 0)),
            )
        )
        history.append(event)
        if decision.stop:
            break
    return output


class GeneralizedRapecTests(unittest.TestCase):
    def test_quality_transforms_are_bounded_and_monotone(self):
        self.assertAlmostEqual(QualityDefinition("accuracy").normalize(0.8), 0.8)
        self.assertAlmostEqual(QualityDefinition("error", "complement").normalize(0.2), 0.8)
        self.assertGreater(
            QualityDefinition("rmse", "inverse_positive").normalize(0.5),
            QualityDefinition("rmse", "inverse_positive").normalize(2.0),
        )
        self.assertGreater(
            QualityDefinition("loss", "exp_negative").normalize(0.5),
            QualityDefinition("loss", "exp_negative").normalize(2.0),
        )

    def test_checkpoint_aliases_are_task_neutral(self):
        observation = EpochObservation(
            epoch=5,
            max_epochs=20,
            task_type="anything",
            quality_metric="quality_score",
            quality=0.5,
            best_quality=0.5,
            delta_quality=0.1,
            epoch_energy_wh=1.0,
            cumulative_energy_wh=5.0,
            epoch_duration_seconds=2.0,
            cumulative_duration_seconds=10.0,
            gpu_utilization_pct=None,
            history=(),
            raw_metrics={},
        )
        self.assertEqual(observation.checkpoint, 5)
        self.assertEqual(observation.max_checkpoints, 20)
        self.assertEqual(observation.progress_fraction, 0.25)

    def test_decisions_do_not_depend_on_task_or_scenario_labels(self):
        detection = run_trace("object_detection", "scratch-yolo")
        language = run_trace("language_modeling", "pretrained-transformer")
        self.assertEqual(detection, language)
        self.assertTrue(any(stop for stop, _, _, _ in detection))

    def test_decisions_are_identical_to_rapec_v3_pf(self):
        generalized = run_trace("recommendation", "generic-checkpoints")
        original = run_trace(
            "recommendation",
            "generic-checkpoints",
            ProfileFreeRapecV3Controller,
        )
        self.assertEqual(generalized, original)

    def test_generated_manifest_spans_cross_domain_cases(self):
        vision = json.loads((ROOT / "controller_benchmark/config/benchmark-v3-nine-dataset-fresh.json").read_text(encoding="utf-8"))
        classification = json.loads((ROOT / "controller_benchmark/config/benchmark-live-shadow-classification-macro-f1.json").read_text(encoding="utf-8"))
        pretrained = json.loads((ROOT / "controller_benchmark/config/benchmark-live-shadow-pretrained-nine-dataset.json").read_text(encoding="utf-8"))
        manifest = build_manifest(vision, classification, pretrained)
        cases = [case for stage in manifest["stages"] for case in stage["cases"]]
        self.assertEqual(len(cases), 48)
        self.assertEqual(len({case["id"] for case in cases}), 48)
        self.assertTrue(any(case.get("pretrained") for case in cases))
        self.assertTrue(any(not case.get("pretrained") for case in cases))
        self.assertNotIn("object_detection", {case["task_type"] for case in cases})
        self.assertNotIn("image_classification", {case["task_type"] for case in cases})
        self.assertNotIn("semantic_segmentation", {case["task_type"] for case in cases})
        self.assertTrue({"tabular_classification", "tabular_regression", "anomaly_detection", "text_classification", "language_modeling", "time_series_forecasting", "recommendation", "reinforcement_learning"}.issubset({case["task_type"] for case in cases}))
        self.assertTrue(all("quality_definition" in case for case in cases))
        by_id = {case["id"]: case for case in cases}
        for task_type in {case["task_type"] for case in cases}:
            task_cases = [case for case in cases if case["task_type"] == task_type]
            self.assertEqual(len(task_cases), 6)
            self.assertEqual(len({case["dataset_key"] for case in task_cases}), 3)
            self.assertEqual(sum(bool(case.get("pretrained")) for case in task_cases), 3)
            self.assertEqual(sum(not bool(case.get("pretrained")) for case in task_cases), 3)
        for case in cases:
            if not case.get("pretrained"):
                continue
            source = by_id[case["pretrained_source_case_id"]]
            self.assertFalse(source.get("pretrained"))
            self.assertEqual(source["task_type"], case["task_type"])
            self.assertNotEqual(source["dataset_key"], case["dataset_key"])

    def test_hard_manifest_replaces_easy_datasets_and_uses_isolated_checkpoints(self):
        vision = json.loads((ROOT / "controller_benchmark/config/benchmark-v3-nine-dataset-fresh.json").read_text(encoding="utf-8"))
        classification = json.loads((ROOT / "controller_benchmark/config/benchmark-live-shadow-classification-macro-f1.json").read_text(encoding="utf-8"))
        pretrained = json.loads((ROOT / "controller_benchmark/config/benchmark-live-shadow-pretrained-nine-dataset.json").read_text(encoding="utf-8"))
        manifest = build_manifest(vision, classification, pretrained, suite="hard")
        cases = [case for stage in manifest["stages"] for case in stage["cases"]]
        dataset_keys = {case["dataset_key"] for case in cases}

        self.assertEqual(manifest["benchmark_version"], "rapec-generalized-nonvision-hard-v1")
        self.assertEqual(len(cases), 48)
        self.assertTrue(
            {
                "sensorless_drive",
                "letter_recognition",
                "year_prediction_msd",
                "online_news_popularity",
                "superconductivity",
                "yelp_review_full",
                "wikitext103",
                "ettm2",
                "movielens20m",
                "cartpole_noisy",
            }.issubset(dataset_keys)
        )
        self.assertTrue(
            {"diabetes", "breast_cancer_anomaly", "wine_anomaly", "tiny_shakespeare", "movielens100k", "cartpole"}.isdisjoint(dataset_keys)
        )
        pretrained_cases = [case for case in cases if case.get("pretrained")]
        self.assertTrue(
            all(
                case["pretrained_checkpoint"].startswith(
                    "/workspace-cache/controller-pretrained-cross-domain-hard-v1/"
                )
                for case in pretrained_cases
            )
        )

    def test_expensive_manifest_requires_cuda_and_isolates_checkpoints(self):
        vision = json.loads((ROOT / "controller_benchmark/config/benchmark-v3-nine-dataset-fresh.json").read_text(encoding="utf-8"))
        classification = json.loads((ROOT / "controller_benchmark/config/benchmark-live-shadow-classification-macro-f1.json").read_text(encoding="utf-8"))
        pretrained = json.loads((ROOT / "controller_benchmark/config/benchmark-live-shadow-pretrained-nine-dataset.json").read_text(encoding="utf-8"))
        manifest = build_manifest(vision, classification, pretrained, suite="expensive")
        cases = [case for stage in manifest["stages"] for case in stage["cases"]]

        self.assertEqual(
            manifest["benchmark_version"],
            "rapec-generalized-nonvision-expensive-v1",
        )
        self.assertEqual(len(cases), 30)
        self.assertTrue(all(case.get("require_cuda") is True for case in cases))
        self.assertTrue(all(case.get("difficulty_tier") == "expensive" for case in cases))
        self.assertEqual(sum(bool(case.get("pretrained")) for case in cases), 15)
        self.assertEqual(sum(not bool(case.get("pretrained")) for case in cases), 15)
        self.assertTrue(
            all(
                case["pretrained_checkpoint"].startswith(
                    "/workspace-cache/controller-pretrained-cross-domain-expensive-v1/"
                )
                for case in cases
                if case.get("pretrained")
            )
        )

    def test_hardest_manifest_uses_frontier_datasets_without_changing_vision(self):
        vision = json.loads((ROOT / "controller_benchmark/config/benchmark-v3-nine-dataset-fresh.json").read_text(encoding="utf-8"))
        classification = json.loads((ROOT / "controller_benchmark/config/benchmark-live-shadow-classification-macro-f1.json").read_text(encoding="utf-8"))
        pretrained = json.loads((ROOT / "controller_benchmark/config/benchmark-live-shadow-pretrained-nine-dataset.json").read_text(encoding="utf-8"))
        manifest = build_manifest(vision, classification, pretrained, suite="hardest")
        cases = [case for stage in manifest["stages"] for case in stage["cases"]]
        dataset_keys = {case["dataset_key"] for case in cases}

        self.assertEqual(
            manifest["benchmark_version"],
            "rapec-generalized-nonvision-hardest-v1",
        )
        self.assertEqual(len(cases), 30)
        self.assertEqual(len(manifest["stages"]), 5)
        self.assertTrue(all(case.get("require_cuda") is True for case in cases))
        self.assertTrue(all(case.get("difficulty_tier") == "hardest_practical" for case in cases))
        self.assertTrue(
            {
                "go_emotions",
                "multi_eurlex",
                "marc_multilingual",
                "wikitext103",
                "lm1b_shard",
                "c4_en_shard",
                "monash_traffic_hourly",
                "monash_electricity_hourly",
                "monash_solar_10_minutes",
                "movielens25m",
                "amazon_books_5core",
                "amazon_electronics_5core",
                "minatar_breakout",
                "minatar_seaquest",
                "minatar_asterix",
            }.issubset(dataset_keys)
        )
        self.assertTrue(
            {"object_detection", "image_classification", "semantic_segmentation"}.isdisjoint(
                {case["task_type"] for case in cases}
            )
        )
        self.assertTrue(
            all(
                case["pretrained_checkpoint"].startswith(
                    "/workspace-cache/controller-pretrained-cross-domain-hardest-v1/"
                )
                for case in cases
                if case.get("pretrained")
            )
        )

    def test_live_shadow_uses_only_standard_es_and_rapec_g(self):
        generalized = json.loads(
            (ROOT / "controller_benchmark/config/generalized-live-shadow-controllers.json").read_text(
                encoding="utf-8"
            )
        )
        original = json.loads(
            (ROOT / "controller_benchmark/config/live-shadow-controllers.json").read_text(
                encoding="utf-8"
            )
        )
        self.assertEqual(
            [controller["id"] for controller in generalized["controllers"]],
            ["standard-es-10", "rapec-g-v1"],
        )
        profile_free = next(
            controller
            for controller in original["controllers"]
            if controller["id"] == "rapec-v3-pf"
        )
        rapec_g = generalized["controllers"][1]
        self.assertEqual(
            rapec_g["controller_parameters"],
            profile_free["controller_parameters"],
        )


if __name__ == "__main__":
    unittest.main()
