import argparse
import json
import tempfile
from pathlib import Path


JOB_METRIC_KEYS = [
    "duration_seconds",
    "gpu_power_avg_w",
    "gpu_power_max_w",
    "gpu_util_avg_pct",
    "gpu_mem_used_avg_mb",
    "gpu_temp_avg_c",
    "gpu_energy_kwh",
    "training_energy_kwh",
    "estimated_electricity_cost_eur",
    "estimated_co2_kg",
]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary-json", required=True, help="Path to gpu_summary_job_<id>.json")
    parser.add_argument("--output-dir", required=True, help="Directory for .prom files")
    parser.add_argument(
        "--aggregate-dir",
        default=None,
        help="Directory containing gpu_summary_job_*.json for aggregate metrics (defaults to summary parent)",
    )
    return parser.parse_args()


def load_json(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def to_float(value, default=0.0):
    try:
        return float(value)
    except Exception:
        return default


def prom_escape(value: str) -> str:
    return value.replace("\\", "\\\\").replace('"', '\\"')


def write_atomic(path: Path, content: str):
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", newline="\n", delete=False, dir=str(path.parent)
    ) as tmp:
        tmp.write(content)
        tmp_path = Path(tmp.name)
    tmp_path.replace(path)
    path.chmod(0o644)


def format_job_metrics(summary):
    job_id = str(summary.get("job_id", "unknown"))
    label = prom_escape(job_id)
    lines = [
        "# HELP slurm_job_info Static metadata for a SLURM job summary.",
        "# TYPE slurm_job_info gauge",
        f'slurm_job_info{{job_id="{label}"}} 1',
    ]

    for key in JOB_METRIC_KEYS:
        value = to_float(summary.get(key, 0.0))
        metric_name = f"slurm_job_{key}"
        lines.append(f"# HELP {metric_name} Job metric derived from GPU summary JSON.")
        lines.append(f"# TYPE {metric_name} gauge")
        lines.append(f'{metric_name}{{job_id="{label}"}} {value:.12g}')

    return "\n".join(lines) + "\n"


def aggregate_summaries(summaries):
    count = len(summaries)
    total_duration = sum(to_float(s.get("duration_seconds", 0.0)) for s in summaries)
    total_gpu_energy = sum(to_float(s.get("gpu_energy_kwh", 0.0)) for s in summaries)
    total_training_energy = sum(to_float(s.get("training_energy_kwh", 0.0)) for s in summaries)
    total_cost = sum(to_float(s.get("estimated_electricity_cost_eur", 0.0)) for s in summaries)
    total_co2 = sum(to_float(s.get("estimated_co2_kg", 0.0)) for s in summaries)
    avg_gpu_util = (
        sum(to_float(s.get("gpu_util_avg_pct", 0.0)) for s in summaries) / count if count else 0.0
    )

    lines = [
        "# HELP slurm_jobs_total Number of summarized SLURM jobs.",
        "# TYPE slurm_jobs_total gauge",
        f"slurm_jobs_total {count}",
        "# HELP slurm_jobs_duration_seconds_total Total duration over all summarized jobs.",
        "# TYPE slurm_jobs_duration_seconds_total gauge",
        f"slurm_jobs_duration_seconds_total {total_duration:.12g}",
        "# HELP slurm_jobs_gpu_energy_kwh_total Total GPU-only energy in kWh over all summarized jobs.",
        "# TYPE slurm_jobs_gpu_energy_kwh_total gauge",
        f"slurm_jobs_gpu_energy_kwh_total {total_gpu_energy:.12g}",
        "# HELP slurm_jobs_training_energy_kwh_total Total training energy in kWh over all summarized jobs.",
        "# TYPE slurm_jobs_training_energy_kwh_total gauge",
        f"slurm_jobs_training_energy_kwh_total {total_training_energy:.12g}",
        "# HELP slurm_jobs_estimated_electricity_cost_eur_total Total estimated electricity cost in EUR.",
        "# TYPE slurm_jobs_estimated_electricity_cost_eur_total gauge",
        f"slurm_jobs_estimated_electricity_cost_eur_total {total_cost:.12g}",
        "# HELP slurm_jobs_estimated_co2_kg_total Total estimated CO2 in kg over all summarized jobs.",
        "# TYPE slurm_jobs_estimated_co2_kg_total gauge",
        f"slurm_jobs_estimated_co2_kg_total {total_co2:.12g}",
        "# HELP slurm_jobs_gpu_util_avg_pct_mean Mean GPU utilization percentage across job summaries.",
        "# TYPE slurm_jobs_gpu_util_avg_pct_mean gauge",
        f"slurm_jobs_gpu_util_avg_pct_mean {avg_gpu_util:.12g}",
    ]
    return "\n".join(lines) + "\n"


def main():
    args = parse_args()
    summary_path = Path(args.summary_json)
    output_dir = Path(args.output_dir)
    aggregate_dir = Path(args.aggregate_dir) if args.aggregate_dir else summary_path.parent

    if not summary_path.exists():
        raise FileNotFoundError(f"Summary JSON not found: {summary_path}")

    summary = load_json(summary_path)
    job_id = str(summary.get("job_id", "unknown"))

    job_prom_path = output_dir / f"job_{job_id}.prom"
    write_atomic(job_prom_path, format_job_metrics(summary))

    summaries = []
    for json_path in sorted(aggregate_dir.glob("gpu_summary_job_*.json")):
        try:
            summaries.append(load_json(json_path))
        except Exception:
            continue

    aggregate_prom_path = output_dir / "aggregate.prom"
    write_atomic(aggregate_prom_path, aggregate_summaries(summaries))

    print(f"Wrote {job_prom_path}")
    print(f"Wrote {aggregate_prom_path}")


if __name__ == "__main__":
    main()
