#!/usr/bin/env python3
import argparse
import tempfile
from pathlib import Path

try:
    from slurm.analyze_stop_policy_study import enrich_regret_and_pareto, load_runs
except ImportError:
    from analyze_stop_policy_study import enrich_regret_and_pareto, load_runs


def escape(value) -> str:
    return str(value).replace("\\", "\\\\").replace('"', '\\"')


def write_atomic(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", newline="\n", delete=False, dir=path.parent) as handle:
        handle.write(content)
        temporary = Path(handle.name)
    temporary.replace(path)
    path.chmod(0o644)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--metrics-dir", required=True, type=Path)
    parser.add_argument("--output-prom", required=True, type=Path)
    args = parser.parse_args()

    runs = load_runs(args.metrics_dir)
    enrich_regret_and_pareto(runs)
    lines = []
    for row in runs:
        labels = (
            f'job_id="{escape(row["job_id"])}",scenario="{escape(row["scenario"])}",'
            f'comparison_strategy="{escape(row["strategy"])}",training_seed="{row["training_seed"]}",'
            f'split_seed="{row["split_seed"]}"'
        )
        metrics = {
            "slurm_study_test_best_map50_95": row.get("test_best_map50_95"),
            "slurm_study_test_best_map50": row.get("test_best_map50"),
            "slurm_study_job_gpu_energy_wh": row.get("job_gpu_energy_wh"),
            "slurm_study_job_net_gpu_energy_wh": row.get("job_net_gpu_energy_wh"),
            "slurm_study_accuracy_regret_map50_95_pp": row.get("accuracy_regret_map50_95_pp"),
            "slurm_study_accuracy_per_wh": row.get("accuracy_per_wh"),
            "slurm_study_energy_to_map50_95_50_wh": row.get("energy_to_map_0.50_wh"),
            "slurm_study_energy_to_map50_95_60_wh": row.get("energy_to_map_0.60_wh"),
            "slurm_study_energy_to_map50_95_68_wh": row.get("energy_to_map_0.68_wh"),
            "slurm_study_pareto_dominated": 1 if row.get("pareto_dominated") else 0,
        }
        for name, value in metrics.items():
            if value is not None:
                lines.append(f"{name}{{{labels}}} {float(value):.12g}")
    write_atomic(args.output_prom, "\n".join(lines) + "\n")
    print(f"Wrote {args.output_prom} for {len(runs)} runs")


if __name__ == "__main__":
    main()
