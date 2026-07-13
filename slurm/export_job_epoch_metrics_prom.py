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
    scenario = prom_escape(str(data.get("scenario", "unspecified")))
    training_seed = prom_escape(str(data.get("training_seed", "0")))
    split_seed = prom_escape(str(data.get("split_seed", "0")))
    lines = []
    for epoch_data in data.get("epochs", []):
        epoch_raw = epoch_data.get("epoch", epoch_data.get("epoch_index", "unknown"))
        epoch = prom_escape(str(epoch_raw))
        adaptive_monitor_metric = prom_escape(str(epoch_data.get("adaptive_monitor_metric", data.get("adaptive_monitor_metric", "map50"))))
        controller_mode = prom_escape(str(epoch_data.get("controller_mode", data.get("controller_mode", "none"))))
        comparison_strategy = prom_escape(str(epoch_data.get("comparison_strategy", data.get("comparison_strategy", "unspecified"))))
        try:
            epoch_label = f"E{int(float(epoch_raw)):03d}"
        except Exception:
            epoch_label = f"E{epoch}"
        labels = (
            f'job_id="{job_id}",epoch="{epoch}",epoch_label="{prom_escape(epoch_label)}",'
            f'adaptive_monitor_metric="{adaptive_monitor_metric}",controller_mode="{controller_mode}",'
            f'comparison_strategy="{comparison_strategy}",scenario="{scenario}",'
            f'training_seed="{training_seed}",split_seed="{split_seed}"'
        )
        lines.append(f"slurm_job_epoch_duration_seconds{{{labels}}} {to_float(epoch_data.get('duration_seconds')):.12g}")
        lines.append(f"slurm_job_epoch_energy_kwh{{{labels}}} {to_float(epoch_data.get('total_energy_kwh')):.12g}")
        lines.append(f"slurm_job_epoch_gpu_energy_kwh{{{labels}}} {to_float(epoch_data.get('gpu_energy_kwh')):.12g}")
        lines.append(f"slurm_job_epoch_net_gpu_energy_kwh{{{labels}}} {to_float(epoch_data.get('net_gpu_energy_kwh')):.12g}")
        lines.append(f"slurm_job_epoch_net_energy_kwh{{{labels}}} {to_float(epoch_data.get('net_total_energy_kwh')):.12g}")
        lines.append(f"slurm_job_epoch_idle_energy_kwh{{{labels}}} {to_float(epoch_data.get('idle_energy_kwh')):.12g}")
        lines.append(f"slurm_job_epoch_gpu_util_avg_pct{{{labels}}} {to_float(epoch_data.get('gpu_util_avg_pct')):.12g}")
        lines.append(f"slurm_job_epoch_map50{{{labels}}} {to_float(epoch_data.get('map50')):.12g}")
        lines.append(f"slurm_job_epoch_map50_95{{{labels}}} {to_float(epoch_data.get('map50_95')):.12g}")
        lines.append(f"slurm_job_epoch_best_map50{{{labels}}} {to_float(epoch_data.get('best_map50')):.12g}")
        lines.append(f"slurm_job_epoch_best_map50_95{{{labels}}} {to_float(epoch_data.get('best_map50_95')):.12g}")
        lines.append(f"slurm_job_epoch_precision{{{labels}}} {to_float(epoch_data.get('precision')):.12g}")
        lines.append(f"slurm_job_epoch_recall{{{labels}}} {to_float(epoch_data.get('recall')):.12g}")
        lines.append(f"slurm_job_epoch_learning_rate{{{labels}}} {to_float(epoch_data.get('learning_rate')):.12g}")
        lines.append(f"slurm_job_epoch_delta_map50{{{labels}}} {to_float(epoch_data.get('delta_map50')):.12g}")
        lines.append(f"slurm_job_epoch_delta_map50_95{{{labels}}} {to_float(epoch_data.get('delta_map50_95')):.12g}")
        lines.append(f"slurm_job_epoch_mape_map50_per_wh{{{labels}}} {to_float(epoch_data.get('marginal_map50_per_wh')):.12g}")
        lines.append(f"slurm_job_epoch_mape_map50_95_per_wh{{{labels}}} {to_float(epoch_data.get('marginal_map50_95_per_wh')):.12g}")
        lines.append(f"slurm_job_epoch_smoothed_delta_map50{{{labels}}} {to_float(epoch_data.get('smoothed_delta_map50')):.12g}")
        lines.append(f"slurm_job_epoch_smoothed_delta_map50_95{{{labels}}} {to_float(epoch_data.get('smoothed_delta_map50_95')):.12g}")
        lines.append(f"slurm_job_epoch_smoothed_mape_map50_per_wh{{{labels}}} {to_float(epoch_data.get('smoothed_mape_map50_per_wh')):.12g}")
        lines.append(f"slurm_job_epoch_smoothed_mape_map50_95_per_wh{{{labels}}} {to_float(epoch_data.get('smoothed_mape_map50_95_per_wh')):.12g}")
        lines.append(f"slurm_job_epoch_predicted_final_metric_mean{{{labels}}} {to_float(epoch_data.get('predicted_final_metric_mean')):.12g}")
        lines.append(f"slurm_job_epoch_predicted_final_metric_lower{{{labels}}} {to_float(epoch_data.get('predicted_final_metric_lower')):.12g}")
        lines.append(f"slurm_job_epoch_predicted_final_metric_upper{{{labels}}} {to_float(epoch_data.get('predicted_final_metric_upper')):.12g}")
        lines.append(f"slurm_job_epoch_predicted_remaining_gain_upper{{{labels}}} {to_float(epoch_data.get('predicted_remaining_gain_upper')):.12g}")
        lines.append(f"slurm_job_epoch_predicted_prob_gain_gt_epsilon{{{labels}}} {to_float(epoch_data.get('predicted_prob_gain_gt_epsilon')):.12g}")
        lines.append(f"slurm_job_epoch_uncertainty_interval_width{{{labels}}} {to_float(epoch_data.get('uncertainty_interval_width')):.12g}")
        lines.append(f"slurm_job_epoch_low_gain_streak{{{labels}}} {to_float(epoch_data.get('low_gain_streak')):.12g}")
        lines.append(f"slurm_job_epoch_cumulative_energy_kwh{{{labels}}} {to_float(epoch_data.get('cumulative_total_energy_kwh')):.12g}")
        lines.append(f"slurm_job_epoch_cumulative_net_energy_kwh{{{labels}}} {to_float(epoch_data.get('cumulative_net_total_energy_kwh')):.12g}")
        lines.append(f"slurm_job_epoch_estimated_electricity_cost_eur{{{labels}}} {to_float(epoch_data.get('estimated_electricity_cost_eur')):.12g}")
        lines.append(f"slurm_job_epoch_estimated_co2_kg{{{labels}}} {to_float(epoch_data.get('estimated_co2_kg')):.12g}")
        lines.append(f"slurm_job_epoch_should_stop{{{labels}}} {to_float(epoch_data.get('should_stop')):.12g}")

    summary_labels = (
        f'job_id="{job_id}",comparison_strategy="{prom_escape(str(data.get("comparison_strategy", "unspecified")))}",'
        f'scenario="{scenario}",training_seed="{training_seed}",split_seed="{split_seed}"'
    )
    for key in (
        "test_best_map50",
        "test_best_map50_95",
        "test_best_precision",
        "test_best_recall",
        "accuracy_regret_map50_95",
    ):
        if data.get(key) is not None:
            lines.append(f"slurm_job_{key}{{{summary_labels}}} {to_float(data.get(key)):.12g}")

    write_atomic(Path(args.output_prom), "\n".join(lines) + "\n")
    print(f"Wrote {args.output_prom}")


if __name__ == "__main__":
    main()
