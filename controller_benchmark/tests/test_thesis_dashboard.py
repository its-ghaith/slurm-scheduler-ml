from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from controller_benchmark.dashboard.build_live_shadow_dashboard import build
from controller_benchmark.case_labels import CASE_LABELS
from controller_benchmark.dashboard.build_thesis_results_dashboard import (
    build as build_thesis_dashboard,
)
from controller_benchmark.export_thesis_live_shadow_metrics import build as build_metrics


class ThesisDashboardTests(unittest.TestCase):
    def test_complete_thesis_dashboard_contains_requested_sections(self) -> None:
        dashboard = build_thesis_dashboard(
            "20260812T143435Z-live-shadow-dev", "carpk-aerial-vehicle"
        )
        titles = {panel["title"] for panel in dashboard["panels"]}
        required = {
            "SLURM − CodeCarbon Training GPU Difference by Job",
            "SLURM Phase: Energy, Quality, and Epoch Count",
            "CodeCarbon Phase: Energy, Quality, and Epoch Count",
            "Stop Epoch by Dataset: RAPEC-G, ES, Full100",
            "Raw Quality by Dataset and Metric",
            "Full100 Quality–Energy Curve with RAPEC-G and ES Stop Points",
            "GPU Energy per +1 Normalized Quality Point by Dataset",
            "Energy Saving vs Quality Regret by Dataset",
            "RAPEC-G Generalisation by Task Type (Mean and 95% CI)",
        }
        self.assertTrue(required.issubset(titles))
        self.assertTrue(all(panel.get("description") for panel in dashboard["panels"]))
        self.assertEqual(dashboard["uid"], "rapec-g-v4-live-shadow-cv-nv")
        self.assertEqual(dashboard["title"], "Thesis Results – Energy-Adaptive MLOps")
        variables = {item["name"] for item in dashboard["templating"]["list"]}
        self.assertTrue(
            {
                "cases",
                "raw_quality_metrics",
                "slurm_phase",
                "codecarbon_phase",
            }.issubset(
                variables
            )
        )
        self.assertNotIn("op_job_ids", variables)
        self.assertNotIn("curve_case", variables)
        run_variable = next(
            item
            for item in dashboard["templating"]["list"]
            if item["name"] == "benchmark_run_id"
        )
        self.assertEqual(run_variable["type"], "constant")
        self.assertEqual(run_variable["hide"], 2)
        self.assertEqual(
            run_variable["current"]["value"], "20260812T143435Z-live-shadow-dev"
        )
        encoded = json.dumps(dashboard)
        self.assertNotIn("60_run", encoded)
        self.assertIn("slurm_job_codecarbon_job_total_energy_kwh", encoded)
        self.assertIn("slurm_job_codecarbon_job_total_gpu_energy_kwh", encoded)
        self.assertNotIn('sum(slurm_job_codecarbon_total_energy_kwh{', encoded)
        self.assertNotIn('sum(slurm_job_codecarbon_total_gpu_energy_kwh{', encoded)
        self.assertNotIn("ignoring(controller_id)", encoded)
        self.assertIn(
            "on(benchmark_run_id,benchmark_version,stage,case_id,case_code,"
            "case_name,case_order,task_type,energy_measurement_method)",
            encoded,
        )
        signed = next(
            panel
            for panel in dashboard["panels"]
            if panel["title"] == "SLURM − CodeCarbon Training GPU Difference by Job"
        )
        script = signed["options"]["getOption"]
        self.assertIn("#2E8B57", script)
        self.assertIn("#C23B4A", script)
        self.assertIn("{yAxis: 0", script)

        curve = next(
            panel
            for panel in dashboard["panels"]
            if panel["title"]
            == "Full100 Quality–Energy Curve with RAPEC-G and ES Stop Points"
        )
        self.assertNotIn("$curve_case", json.dumps(curve))
        self.assertTrue(
            all('case_id=~"$cases"' in target["expr"] for target in curve["targets"])
        )
        self.assertIn("curves.push", curve["options"]["getOption"])
        self.assertIn("symbolSize:8", curve["options"]["getOption"])
        self.assertIn("focus:'series'", curve["options"]["getOption"])
        self.assertEqual(len(curve["targets"]), 12)
        self.assertNotIn("Stop reason", curve["options"]["getOption"])
        self.assertNotIn("Total energy", curve["options"]["getOption"])
        self.assertIn("Overhead", curve["options"]["getOption"])
        self.assertIn("comparisonTable(dataset)", curve["options"]["getOption"])
        self.assertIn("RAPEC-G v4</th>", curve["options"]["getOption"])
        self.assertIn("Standard ES</th>", curve["options"]["getOption"])
        self.assertIn("max-width:740px", curve["options"]["getOption"])
        self.assertIn("enterable:true", curve["options"]["getOption"])

        panels = {panel["id"]: panel for panel in dashboard["panels"]}
        self.assertEqual(len(panels[201]["targets"]), 4)
        self.assertIn("Preprocessing & Initialization", panels[201]["options"]["getOption"])
        for panel_id in (202, 203):
            self.assertEqual(len(panels[panel_id]["targets"]), 3)
            self.assertIn("quality", " ".join(target["expr"] for target in panels[panel_id]["targets"]))
            self.assertIn("caseMeta", panels[panel_id]["options"]["getOption"])
        for panel_id in (205, 206):
            self.assertEqual(panels[panel_id]["type"], "volkovlabs-echarts-panel")
            self.assertIn("caseColor", panels[panel_id]["options"]["getOption"])
            self.assertIn("type:'scroll'", panels[panel_id]["options"]["getOption"])
        tradeoff_script = panels[503]["options"]["getOption"]
        self.assertIn("RAPEC-G v4 vs Full100", tradeoff_script)
        self.assertIn("Standard ES vs Full100", tradeoff_script)
        self.assertIn("RAPEC-G v4 vs Standard ES", tradeoff_script)
        self.assertIn("gridIndex:2", tradeoff_script)
        self.assertEqual(len(panels[503]["targets"]), 6)
        self.assertIn("rowSlot=86/rowsCount", panels[407]["options"]["getOption"])
        self.assertIn("gridTop=titleTop+5", panels[407]["options"]["getOption"])

    def test_all_thesis_cases_have_unique_three_character_labels(self) -> None:
        codes = [metadata[0] for metadata in CASE_LABELS.values()]
        self.assertEqual(len(codes), 48)
        self.assertEqual(len(set(codes)), 48)
        self.assertTrue(all(len(code) == 3 for code in codes))

    def test_dashboard_has_documented_energy_quality_panels(self) -> None:
        dashboard = build(
            default_run_id="run-1",
            default_curve_case="dataset-1",
        )
        panels = {panel["id"]: panel for panel in dashboard["panels"]}
        self.assertIn("Quality-Energy Curve", panels[17]["title"])
        self.assertIn("per +1 Normalized Quality Point", panels[18]["title"])
        self.assertTrue(all(panel.get("description") for panel in dashboard["panels"]))
        variables = {item["name"]: item for item in dashboard["templating"]["list"]}
        self.assertEqual(variables["benchmark_run_id"]["current"]["value"], "run-1")
        self.assertEqual(variables["curve_case"]["current"]["value"], "dataset-1")

    def test_compact_export_keeps_normalized_and_raw_quality(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            snapshot = Path(directory)
            metrics = snapshot / "data" / "energy_metrics"
            metrics.mkdir(parents=True)
            summary = {
                "job_id": "1",
                "benchmark_run_id": "run-1",
                "benchmark_version": "v1",
                "benchmark_case_id": "dataset-1",
                "benchmark_stage": "classification",
                "task_type": "image_classification",
                "epochs": [
                    {
                        "epoch": 1,
                        "gpu_energy_kwh": 0.001,
                        "quality_score": 0.75,
                        "accuracy": 0.75,
                        "macro_f1": 0.70,
                    }
                ],
            }
            (metrics / "epoch_summary_job_1.json").write_text(
                json.dumps(summary), encoding="utf-8"
            )
            output = build_metrics(snapshot)
            self.assertIn("thesis_live_shadow_epoch_gpu_energy_wh", output)
            self.assertIn(" 1\n", output)
            self.assertIn("thesis_live_shadow_epoch_quality_score", output)
            self.assertIn('raw_quality_metric="accuracy"', output)
            self.assertIn('raw_quality_metric="macro_f1"', output)


if __name__ == "__main__":
    unittest.main()
