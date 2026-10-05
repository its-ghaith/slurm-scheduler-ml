from __future__ import annotations

import unittest
from pathlib import Path

from controller_benchmark.campaign import (
    build_campaign_plan,
    load_controller_specifications,
)
from controller_benchmark.dashboard.build_nine_dataset_dashboard import (
    build_dashboard,
)
from controller_benchmark.manifest import load_manifest


ROOT = Path(__file__).resolve().parents[2]
MANIFEST = ROOT / "controller_benchmark" / "config" / "benchmark-v3-nine-dataset-fresh.json"
CONTROLLERS = ROOT / "controller_benchmark" / "config" / "rapec-v3-v8-controllers.json"


class NineDatasetCampaignTests(unittest.TestCase):
    def setUp(self) -> None:
        self.manifest = load_manifest(MANIFEST)
        self.controllers = load_controller_specifications(CONTROLLERS, self.manifest)
        self.plan = build_campaign_plan(
            self.manifest,
            self.controllers,
            run_id="test-nine-dataset-campaign",
        )

    def test_campaign_has_three_datasets_per_task_and_no_stability_runs(self):
        enabled = [stage for stage in self.manifest["stages"] if stage["enabled"]]
        self.assertEqual(len(enabled), 3)
        self.assertEqual([len(stage["cases"]) for stage in enabled], [3, 3, 3])
        case_ids = [case["id"] for stage in enabled for case in stage["cases"]]
        self.assertEqual(len(case_ids), 9)
        self.assertFalse(any("stability" in case_id for case_id in case_ids))
        self.assertEqual(
            {case["task_type"] for stage in enabled for case in stage["cases"]},
            {"object_detection", "image_classification", "semantic_segmentation"},
        )

    def test_campaign_runs_fresh_full100_v3_and_v8_once_per_dataset(self):
        self.assertEqual(len(self.plan["runs"]), 27)
        self.assertEqual(self.plan["planned_baseline_runs"], 9)
        self.assertEqual(self.plan["planned_candidate_runs"], 18)
        self.assertEqual(self.plan["baseline_cache"]["entries"], {})
        self.assertEqual(self.plan["baseline_cache"]["cached_case_ids"], [])
        by_case: dict[str, list[str]] = {}
        for run in self.plan["runs"]:
            by_case.setdefault(run["benchmark_case_id"], []).append(run["strategy"])
        self.assertEqual(len(by_case), 9)
        for strategies in by_case.values():
            self.assertCountEqual(strategies, ["full100", "rapec-v3", "rapec-v8"])

    def test_controllers_use_the_existing_v3_and_v8_plugins(self):
        by_id = {item["id"]: item for item in self.controllers["controllers"]}
        self.assertEqual(
            by_id["rapec-v3"]["controller_plugin"],
            "controller_benchmark.controllers.risk_aware_predictive_energy:"
            "RiskAwarePredictiveEnergyController",
        )
        self.assertEqual(
            by_id["rapec-v8"]["controller_plugin"],
            "controller_benchmark.controllers.bayesian_guarded_rapec_v3:"
            "BayesianGuardedRapecV3Controller",
        )
        self.assertEqual(by_id["rapec-v3"]["controller_parameters"]["horizon_epochs"], 5)
        self.assertEqual(by_id["rapec-v8"]["controller_parameters"]["bayesian_samples"], 512)

    def test_dashboards_have_unique_uids_and_correct_controller_metrics(self):
        dashboards = [
            build_dashboard(
                title="RAPEC-v3 vs RAPEC-v8 - Nine-Dataset Fresh Benchmark",
                uid="rapec-v3-v8-nine-dataset",
                controller=None,
            ),
            build_dashboard(
                title="RAPEC-v3 - Nine-Dataset Fresh Benchmark",
                uid="rapec-v3-nine-dataset",
                controller="rapec-v3",
            ),
            build_dashboard(
                title="RAPEC-v8 - Nine-Dataset Fresh Benchmark",
                uid="rapec-v8-nine-dataset",
                controller="rapec-v8",
            ),
        ]
        self.assertEqual(len({dashboard["uid"] for dashboard in dashboards}), 3)
        combined = dashboards[0]
        panels = {panel["title"]: panel for panel in combined["panels"]}
        v3_query = " ".join(
            target["expr"]
            for target in panels["RAPEC-v3 Expected Gain and Dynamic Threshold"]["targets"]
        )
        v8_query = " ".join(
            target["expr"]
            for target in panels["RAPEC-v8 Bayesian Guard"]["targets"]
        )
        self.assertIn("rapec_expected_quality_gain_next_horizon", v3_query)
        self.assertIn("rapec8_posterior_probability_relevant_gain", v8_query)
        energy_query = panels["Full100 vs Controller GPU Energy by Dataset"]["targets"][0]["expr"]
        self.assertIn("label_join(", energy_query)
        self.assertEqual(
            panels["Full100 vs Controller GPU Energy by Dataset"]["transformations"][1]["options"][
                "columnField"
            ],
            "series",
        )

    def test_manifest_declares_separate_dashboard_files(self):
        files = self.manifest["execution"]["dashboard_files"]
        self.assertEqual(
            files,
            [
                "nine-dataset-comparison.json",
                "nine-dataset-rapec-v3.json",
                "nine-dataset-rapec-v8.json",
            ],
        )
        self.assertEqual(len(set(files)), 3)


if __name__ == "__main__":
    unittest.main()
