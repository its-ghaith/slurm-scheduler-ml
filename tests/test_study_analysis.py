import json
import tempfile
import unittest
from pathlib import Path

from slurm.analyze_stop_policy_study import aggregate, enrich_regret_and_pareto, load_runs


class StudyAnalysisTests(unittest.TestCase):
    def write_run(self, root, job_id, strategy, quality, energy_kwh):
        epoch = {
            "job_id": str(job_id),
            "scenario": "train128-scratch",
            "comparison_strategy": strategy,
            "training_seed": 0,
            "split_seed": 42,
            "test_best_map50_95": quality,
            "epochs_completed": 2,
            "epochs": [
                {"epoch": 1, "map50_95": 0.50, "gpu_energy_kwh": 0.001},
                {"epoch": 2, "map50_95": quality, "gpu_energy_kwh": 0.001},
            ],
        }
        phase = {"job_id": str(job_id), "gpu_energy_kwh": energy_kwh, "duration_seconds": 10}
        (root / f"epoch_summary_job_{job_id}.json").write_text(json.dumps(epoch), encoding="utf-8")
        (root / f"gpu_summary_job_{job_id}_phases.json").write_text(json.dumps(phase), encoding="utf-8")

    def test_regret_energy_to_target_and_aggregation(self):
        root = Path(tempfile.mkdtemp())
        self.write_run(root, 1, "full100", 0.70, 0.010)
        self.write_run(root, 2, "delta_mape_controller", 0.68, 0.008)
        runs = load_runs(root)
        enrich_regret_and_pareto(runs)
        delta = next(row for row in runs if row["strategy"] == "delta_mape_controller")
        self.assertAlmostEqual(delta["accuracy_regret_map50_95_pp"], 2.0)
        self.assertAlmostEqual(delta["energy_to_map_0.50_wh"], 1.0)
        self.assertFalse(delta["pareto_dominated"])
        self.assertEqual(len(aggregate(runs)), 2)


if __name__ == "__main__":
    unittest.main()
