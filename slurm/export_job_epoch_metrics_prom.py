import argparse
import json
import tempfile
from pathlib import Path


def write_atomic(path: Path, content: str):
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", newline="\n", delete=False, dir=str(path.parent)) as tmp:
        tmp.write(content)
        tmp_path = Path(tmp.name)
    tmp_path.replace(path)
    path.chmod(0o644)


def prom_escape(value: str) -> str:
    return value.replace("\\", "\\\\").replace('"', '\\"')


def to_float(value, default=0.0):
    try:
        return float(value)
    except Exception:
        return default


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--summary-json", required=True)
    p.add_argument("--output-prom", required=True)
    args = p.parse_args()

    with open(args.summary_json, "r", encoding="utf-8") as f:
        data = json.load(f)

    job_id = prom_escape(str(data.get("job_id", "unknown")))
    lines = []
    for epoch_data in data.get("epochs", []):
        epoch = prom_escape(str(epoch_data.get("epoch", epoch_data.get("epoch_index", "unknown"))))
        labels = f'job_id="{job_id}",epoch="{epoch}"'
        lines.append(f"slurm_job_epoch_duration_seconds{{{labels}}} {to_float(epoch_data.get('duration_seconds')):.12g}")
        lines.append(f"slurm_job_epoch_energy_kwh{{{labels}}} {to_float(epoch_data.get('total_energy_kwh')):.12g}")
        lines.append(f"slurm_job_epoch_gpu_energy_kwh{{{labels}}} {to_float(epoch_data.get('gpu_energy_kwh')):.12g}")
        lines.append(f"slurm_job_epoch_gpu_util_avg_pct{{{labels}}} {to_float(epoch_data.get('gpu_util_avg_pct')):.12g}")
        lines.append(f"slurm_job_epoch_map50{{{labels}}} {to_float(epoch_data.get('map50')):.12g}")
        lines.append(f"slurm_job_epoch_map50_95{{{labels}}} {to_float(epoch_data.get('map50_95')):.12g}")
        lines.append(f"slurm_job_epoch_precision{{{labels}}} {to_float(epoch_data.get('precision')):.12g}")
        lines.append(f"slurm_job_epoch_recall{{{labels}}} {to_float(epoch_data.get('recall')):.12g}")
        lines.append(f"slurm_job_epoch_delta_map50{{{labels}}} {to_float(epoch_data.get('delta_map50')):.12g}")
        lines.append(f"slurm_job_epoch_mape_map50_per_wh{{{labels}}} {to_float(epoch_data.get('marginal_map50_per_wh')):.12g}")
        lines.append(f"slurm_job_epoch_cumulative_energy_kwh{{{labels}}} {to_float(epoch_data.get('cumulative_total_energy_kwh')):.12g}")
        lines.append(f"slurm_job_epoch_estimated_electricity_cost_eur{{{labels}}} {to_float(epoch_data.get('estimated_electricity_cost_eur')):.12g}")
        lines.append(f"slurm_job_epoch_estimated_co2_kg{{{labels}}} {to_float(epoch_data.get('estimated_co2_kg')):.12g}")
        lines.append(f"slurm_job_epoch_should_stop{{{labels}}} {to_float(epoch_data.get('should_stop')):.12g}")

    write_atomic(Path(args.output_prom), "\n".join(lines) + "\n")
    print(f"Wrote {args.output_prom}")


if __name__ == "__main__":
    main()
