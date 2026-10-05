from __future__ import annotations

import csv
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from controller_benchmark.server.orchestrate import measurement_validation_errors


ROOT = Path(__file__).resolve().parents[2]


class LifecycleMeasurementTests(unittest.TestCase):
    def test_carpk_phase_summary_exposes_common_lifecycle_and_passes_validation(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            metrics = Path(temporary)
            gpu_csv = metrics / "gpu.csv"
            with gpu_csv.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(
                    handle,
                    fieldnames=(
                        "ts",
                        "gpu_index",
                        "power_w",
                        "util_gpu_pct",
                        "util_mem_pct",
                        "mem_used_mb",
                        "temp_c",
                    ),
                )
                writer.writeheader()
                for timestamp in range(6):
                    writer.writerow(
                        {
                            "ts": timestamp,
                            "gpu_index": 0,
                            "power_w": 100,
                            "util_gpu_pct": 50,
                            "util_mem_pct": 10,
                            "mem_used_mb": 1000,
                            "temp_c": 50,
                        }
                    )

            timeline = metrics / "timeline.jsonl"
            events = []
            for name, start, end in (
                ("preprocessing_labels", 0, 1),
                ("preprocessing_split", 1, 2),
                ("training", 2, 4),
                ("test_evaluation", 4, 5),
            ):
                events.extend(
                    (
                        {"event": "start", "phase": name, "ts": start},
                        {
                            "event": "end",
                            "phase": name,
                            "ts": end,
                            "duration_seconds": end - start,
                            "codecarbon_energy_kwh": 0.00002,
                            "codecarbon_gpu_energy_kwh": 0.00001,
                            "codecarbon_cpu_energy_kwh": 0.00001,
                        },
                    )
                )
            timeline.write_text(
                "".join(json.dumps(event) + "\n" for event in events), encoding="utf-8"
            )

            base = metrics / "base.json"
            base.write_text(
                json.dumps(
                    {
                        "job_id": "42",
                        "duration_seconds": 5,
                        "gpu_energy_kwh": 0.00014,
                        "training_energy_kwh": 0.00014,
                    }
                ),
                encoding="utf-8",
            )
            codecarbon = metrics / "codecarbon.json"
            codecarbon.write_text(
                json.dumps(
                    {
                        "duration_seconds": 5,
                        "codecarbon_energy_kwh": 0.00010,
                        "codecarbon_gpu_energy_kwh": 0.00005,
                        "codecarbon_cpu_energy_kwh": 0.00005,
                    }
                ),
                encoding="utf-8",
            )
            output = metrics / "gpu_summary_job_42.json"
            subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "slurm/summarize_gpu_metrics_phases.py"),
                    "--gpu-csv",
                    str(gpu_csv),
                    "--timeline-jsonl",
                    str(timeline),
                    "--base-summary-json",
                    str(base),
                    "--output-json",
                    str(output),
                    "--job-codecarbon-json",
                    str(codecarbon),
                ],
                check=True,
                capture_output=True,
                text=True,
            )
            summary = json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual(summary["codecarbon_measurement_available"], 1)
            self.assertEqual(summary["lifecycle_energy_complete"], 1)
            self.assertEqual(
                summary["phase_metrics"]["preprocessing_initialization"]["duration_seconds"],
                2,
            )
            self.assertIn("finalization_evaluation", summary["phase_metrics"])

            epochs = [
                {
                    "epoch": epoch,
                    "quality_score": 0.5,
                    "duration_seconds": 1.0,
                    "gpu_energy_kwh": 0.00001,
                    "gpu_util_avg_pct": 50.0,
                }
                for epoch in range(1, 101)
            ]
            (metrics / "epoch_summary_job_42.json").write_text(
                json.dumps({"epochs": epochs}), encoding="utf-8"
            )
            controller = {
                "stop_epoch": 50,
                "best_quality_at_stop": 0.5,
                "training_energy_to_stop_wh": 0.5,
                "controller_overhead_energy_wh": 0.01,
                "controller_compute_seconds": 0.01,
            }
            (metrics / "shadow_summary_job_42.json").write_text(
                json.dumps(
                    {
                        "controllers": [
                            {"controller_id": "standard-es-10", **controller},
                            {"controller_id": "rapec-g-v4", **controller},
                        ]
                    }
                ),
                encoding="utf-8",
            )
            self.assertEqual(
                measurement_validation_errors(
                    "42", metrics, ["standard-es-10", "rapec-g-v4"], 100
                ),
                [],
            )


if __name__ == "__main__":
    unittest.main()
