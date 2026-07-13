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
    common_labels = (
        f'job_id="{job_id}",scenario="{prom_escape(str(data.get("scenario", "unspecified")))}",'
        f'comparison_strategy="{prom_escape(str(data.get("comparison_strategy", "unspecified")))}",'
        f'training_seed="{prom_escape(str(data.get("training_seed", 0)))}",'
        f'split_seed="{prom_escape(str(data.get("split_seed", 0)))}"'
    )
    lines = []
    phase_metrics = data.get("phase_metrics", {})
    for phase, m in phase_metrics.items():
        phase_label = prom_escape(str(phase))
        labels = f'{common_labels},phase="{phase_label}"'
        lines.append(f'slurm_job_phase_duration_seconds{{{labels}}} {to_float(m.get("duration_seconds")):.12g}')
        lines.append(f'slurm_job_phase_energy_kwh{{{labels}}} {to_float(m.get("total_energy_kwh")):.12g}')
        lines.append(f'slurm_job_phase_gpu_energy_kwh{{{labels}}} {to_float(m.get("gpu_energy_kwh")):.12g}')
        lines.append(f'slurm_job_phase_net_gpu_energy_kwh{{{labels}}} {to_float(m.get("net_gpu_energy_kwh")):.12g}')
        lines.append(f'slurm_job_phase_idle_energy_kwh{{{labels}}} {to_float(m.get("idle_energy_kwh")):.12g}')
        lines.append(f'slurm_job_phase_estimated_electricity_cost_eur{{{labels}}} {to_float(m.get("estimated_electricity_cost_eur")):.12g}')
        lines.append(f'slurm_job_phase_estimated_co2_kg{{{labels}}} {to_float(m.get("estimated_co2_kg")):.12g}')
        lines.append(f'slurm_job_phase_codecarbon_duration_seconds{{{labels}}} {to_float(m.get("codecarbon_duration_seconds", m.get("duration_seconds"))):.12g}')
        lines.append(f'slurm_job_phase_codecarbon_energy_kwh{{{labels}}} {to_float(m.get("codecarbon_energy_kwh")):.12g}')
        lines.append(f'slurm_job_phase_codecarbon_gpu_energy_kwh{{{labels}}} {to_float(m.get("codecarbon_gpu_energy_kwh")):.12g}')
        lines.append(f'slurm_job_phase_codecarbon_estimated_electricity_cost_eur{{{labels}}} {to_float(m.get("codecarbon_estimated_electricity_cost_eur")):.12g}')
        lines.append(f'slurm_job_phase_codecarbon_estimated_co2_kg{{{labels}}} {to_float(m.get("codecarbon_estimated_co2_kg")):.12g}')

    lines.append(f'slurm_job_codecarbon_total_energy_kwh{{{common_labels}}} {to_float(data.get("codecarbon_job_total_energy_kwh")):.12g}')
    lines.append(f'slurm_job_codecarbon_total_gpu_energy_kwh{{{common_labels}}} {to_float(data.get("codecarbon_job_total_gpu_energy_kwh")):.12g}')
    lines.append(f'slurm_job_codecarbon_estimated_electricity_cost_eur{{{common_labels}}} {to_float(data.get("codecarbon_estimated_electricity_cost_eur")):.12g}')
    lines.append(f'slurm_job_codecarbon_estimated_co2_kg{{{common_labels}}} {to_float(data.get("codecarbon_estimated_co2_kg")):.12g}')

    cmp = data.get("codecarbon_vs_slurm", {})
    lines.append(f'slurm_job_training_energy_compare_abs_diff_kwh{{{common_labels}}} {to_float(cmp.get("training_abs_diff_kwh")):.12g}')
    lines.append(f'slurm_job_training_energy_compare_rel_diff_pct{{{common_labels}}} {to_float(cmp.get("training_rel_diff_pct")):.12g}')
    lines.append(f'slurm_job_training_gpu_energy_compare_abs_diff_kwh{{{common_labels}}} {to_float(cmp.get("training_gpu_abs_diff_kwh")):.12g}')
    lines.append(f'slurm_job_training_gpu_energy_compare_rel_diff_pct{{{common_labels}}} {to_float(cmp.get("training_gpu_rel_diff_pct")):.12g}')
    lines.append(f'slurm_job_training_total_energy_compare_abs_diff_kwh{{{common_labels}}} {to_float(cmp.get("training_total_abs_diff_kwh")):.12g}')
    lines.append(f'slurm_job_training_total_energy_compare_rel_diff_pct{{{common_labels}}} {to_float(cmp.get("training_total_rel_diff_pct")):.12g}')

    write_atomic(Path(args.output_prom), "\n".join(lines) + "\n")
    print(f"Wrote {args.output_prom}")


if __name__ == "__main__":
    main()
