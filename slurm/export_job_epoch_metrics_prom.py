import argparse
import json
import math
import re
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


def prometheus_metric_suffix(value: str) -> str:
    suffix = re.sub(r"[^a-zA-Z0-9_]", "_", value).strip("_").lower()
    if not suffix:
        raise ValueError("Controller diagnostic metric name cannot be empty.")
    return suffix


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
    controller_id = prom_escape(str(data.get("controller_id", "unspecified")))
    benchmark_version = prom_escape(str(data.get("benchmark_version", "unversioned")))
    benchmark_run_id = prom_escape(str(data.get("benchmark_run_id", "none")))
    benchmark_case_id = prom_escape(str(data.get("benchmark_case_id", "none")))
    benchmark_stage = prom_escape(str(data.get("benchmark_stage", "none")))
    task_type = prom_escape(str(data.get("task_type", "object_detection")))
    quality_metric = prom_escape(str(data.get("quality_metric", "map50_95")))
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
            f'training_seed="{training_seed}",split_seed="{split_seed}",controller_id="{controller_id}",'
            f'benchmark_version="{benchmark_version}",benchmark_run_id="{benchmark_run_id}",'
            f'benchmark_case_id="{benchmark_case_id}",benchmark_stage="{benchmark_stage}",'
            f'task_type="{task_type}",quality_metric="{quality_metric}"'
        )
        lines.append(f"slurm_job_epoch_duration_seconds{{{labels}}} {to_float(epoch_data.get('duration_seconds')):.12g}")
        lines.append(f"slurm_job_epoch_energy_kwh{{{labels}}} {to_float(epoch_data.get('total_energy_kwh')):.12g}")
        lines.append(f"slurm_job_epoch_gpu_energy_kwh{{{labels}}} {to_float(epoch_data.get('gpu_energy_kwh')):.12g}")
        lines.append(f"slurm_job_epoch_net_gpu_energy_kwh{{{labels}}} {to_float(epoch_data.get('net_gpu_energy_kwh')):.12g}")
        lines.append(f"slurm_job_epoch_net_energy_kwh{{{labels}}} {to_float(epoch_data.get('net_total_energy_kwh')):.12g}")
        lines.append(f"slurm_job_epoch_idle_energy_kwh{{{labels}}} {to_float(epoch_data.get('idle_energy_kwh')):.12g}")
        lines.append(f"slurm_job_epoch_gpu_util_avg_pct{{{labels}}} {to_float(epoch_data.get('gpu_util_avg_pct')):.12g}")
        lines.append(f"slurm_job_epoch_gpu_power_avg_w{{{labels}}} {to_float(epoch_data.get('gpu_power_avg_w')):.12g}")
        lines.append(f"slurm_job_epoch_gpu_power_max_w{{{labels}}} {to_float(epoch_data.get('gpu_power_max_w')):.12g}")
        lines.append(f"slurm_job_epoch_gpu_mem_used_avg_mb{{{labels}}} {to_float(epoch_data.get('gpu_mem_used_avg_mb', epoch_data.get('gpu_memory_used_mb'))):.12g}")
        lines.append(f"slurm_job_epoch_map50{{{labels}}} {to_float(epoch_data.get('map50')):.12g}")
        lines.append(f"slurm_job_epoch_map50_95{{{labels}}} {to_float(epoch_data.get('map50_95')):.12g}")
        lines.append(f"slurm_job_epoch_quality_score{{{labels}}} {to_float(epoch_data.get(quality_metric, epoch_data.get('quality_score'))):.12g}")
        lines.append(f"slurm_job_epoch_mae{{{labels}}} {to_float(epoch_data.get('mae')):.12g}")
        lines.append(f"slurm_job_epoch_rmse{{{labels}}} {to_float(epoch_data.get('rmse')):.12g}")
        lines.append(f"slurm_job_epoch_accuracy{{{labels}}} {to_float(epoch_data.get('accuracy')):.12g}")
        lines.append(f"slurm_job_epoch_macro_f1{{{labels}}} {to_float(epoch_data.get('macro_f1')):.12g}")
        lines.append(f"slurm_job_epoch_miou{{{labels}}} {to_float(epoch_data.get('miou')):.12g}")
        lines.append(f"slurm_job_epoch_dice{{{labels}}} {to_float(epoch_data.get('dice')):.12g}")
        lines.append(f"slurm_job_epoch_pixel_accuracy{{{labels}}} {to_float(epoch_data.get('pixel_accuracy')):.12g}")
        lines.append(f"slurm_job_epoch_best_map50{{{labels}}} {to_float(epoch_data.get('best_map50')):.12g}")
        lines.append(f"slurm_job_epoch_best_map50_95{{{labels}}} {to_float(epoch_data.get('best_map50_95')):.12g}")
        lines.append(f"slurm_job_epoch_precision{{{labels}}} {to_float(epoch_data.get('precision')):.12g}")
        lines.append(f"slurm_job_epoch_recall{{{labels}}} {to_float(epoch_data.get('recall')):.12g}")
        lines.append(f"slurm_job_epoch_learning_rate{{{labels}}} {to_float(epoch_data.get('learning_rate')):.12g}")
        lines.append(f"slurm_job_epoch_gradient_norm{{{labels}}} {to_float(epoch_data.get('gradient_norm')):.12g}")
        lines.append(f"slurm_job_epoch_train_loss{{{labels}}} {to_float(epoch_data.get('train_loss')):.12g}")
        lines.append(f"slurm_job_epoch_model_parameter_count{{{labels}}} {to_float(epoch_data.get('model_parameter_count', epoch_data.get('model_parameters'))):.12g}")
        lines.append(f"slurm_job_epoch_model_flops{{{labels}}} {to_float(epoch_data.get('model_flops', epoch_data.get('model_flops_estimated'))):.12g}")
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
        lines.append(f"slurm_job_epoch_conformal_correction{{{labels}}} {to_float(epoch_data.get('conformal_correction')):.12g}")
        lines.append(f"slurm_job_epoch_conformal_remaining_gain_upper{{{labels}}} {to_float(epoch_data.get('conformal_remaining_gain_upper')):.12g}")
        lines.append(f"slurm_job_epoch_predicted_remaining_energy_wh{{{labels}}} {to_float(epoch_data.get('predicted_remaining_energy_wh')):.12g}")
        lines.append(f"slurm_job_epoch_predicted_future_efficiency_per_wh{{{labels}}} {to_float(epoch_data.get('predicted_future_efficiency_per_wh')):.12g}")
        lines.append(f"slurm_job_epoch_controller_decision_state{{{labels}}} {to_float(epoch_data.get('controller_decision_state')):.12g}")
        lines.append(f"slurm_job_epoch_calibration_available{{{labels}}} {to_float(epoch_data.get('calibration_available')):.12g}")
        lines.append(f"slurm_job_epoch_plateau_slope_per_epoch{{{labels}}} {to_float(epoch_data.get('plateau_slope_per_epoch')):.12g}")
        lines.append(f"slurm_job_epoch_recent_best_gain{{{labels}}} {to_float(epoch_data.get('recent_best_gain')):.12g}")
        lines.append(f"slurm_job_epoch_observed_efficiency_per_wh{{{labels}}} {to_float(epoch_data.get('observed_efficiency_per_wh')):.12g}")
        lines.append(f"slurm_job_epoch_adaptive_efficiency_threshold{{{labels}}} {to_float(epoch_data.get('adaptive_efficiency_threshold')):.12g}")
        lines.append(f"slurm_job_epoch_pareto_continue_utility{{{labels}}} {to_float(epoch_data.get('pareto_continue_utility')):.12g}")
        lines.append(f"slurm_job_epoch_controller_evidence{{{labels}}} {to_float(epoch_data.get('controller_evidence')):.12g}")
        lines.append(f"slurm_job_epoch_dynamic_patience{{{labels}}} {to_float(epoch_data.get('dynamic_patience')):.12g}")
        lines.append(f"slurm_job_epoch_quality_target_reached{{{labels}}} {to_float(epoch_data.get('quality_target_reached')):.12g}")
        lines.append(f"slurm_job_epoch_budget_triggered{{{labels}}} {to_float(epoch_data.get('budget_triggered')):.12g}")
        lines.append(f"slurm_job_epoch_observed_job_energy_wh{{{labels}}} {to_float(epoch_data.get('observed_job_energy_wh')):.12g}")
        lines.append(f"slurm_job_epoch_projected_full_job_energy_wh{{{labels}}} {to_float(epoch_data.get('projected_full_job_energy_wh')):.12g}")
        lines.append(f"slurm_job_epoch_projected_stopped_job_energy_wh{{{labels}}} {to_float(epoch_data.get('projected_stopped_job_energy_wh')):.12g}")
        lines.append(f"slurm_job_epoch_projected_energy_saving_fraction{{{labels}}} {to_float(epoch_data.get('projected_energy_saving_fraction')):.12g}")
        lines.append(f"slurm_job_epoch_projected_next_energy_saving_fraction{{{labels}}} {to_float(epoch_data.get('projected_next_energy_saving_fraction')):.12g}")
        lines.append(f"slurm_job_epoch_energy_guard_target_saving_fraction{{{labels}}} {to_float(epoch_data.get('energy_guard_target_saving_fraction')):.12g}")
        lines.append(f"slurm_job_epoch_energy_guard_remaining_budget_wh{{{labels}}} {to_float(epoch_data.get('energy_guard_remaining_budget_wh')):.12g}")
        lines.append(f"slurm_job_epoch_energy_guard_boundary_reached{{{labels}}} {to_float(epoch_data.get('energy_guard_boundary_reached')):.12g}")
        lines.append(f"slurm_job_epoch_energy_guard_fallback_triggered{{{labels}}} {to_float(epoch_data.get('energy_guard_fallback_triggered')):.12g}")
        lines.append(f"slurm_job_epoch_energy_guard_quality_conflict{{{labels}}} {to_float(epoch_data.get('energy_guard_quality_conflict')):.12g}")
        lines.append(f"slurm_job_epoch_controller_plugin_confidence{{{labels}}} {to_float(epoch_data.get('controller_plugin_confidence')):.12g}")
        lines.append(f"slurm_job_epoch_controller_plugin_predicted_energy_saving_fraction{{{labels}}} {to_float(epoch_data.get('controller_plugin_predicted_energy_saving_fraction')):.12g}")
        lines.append(f"slurm_job_epoch_controller_plugin_predicted_quality_regret{{{labels}}} {to_float(epoch_data.get('controller_plugin_predicted_quality_regret')):.12g}")
        lines.append(f"slurm_job_epoch_controller_plugin_compute_seconds{{{labels}}} {to_float(epoch_data.get('controller_plugin_compute_seconds')):.12g}")
        fixed_plugin_metrics = {
            "controller_plugin_confidence",
            "controller_plugin_predicted_energy_saving_fraction",
            "controller_plugin_predicted_quality_regret",
            "controller_plugin_compute_seconds",
        }
        for key, raw_value in epoch_data.items():
            if not key.startswith("controller_plugin_") or key in fixed_plugin_metrics:
                continue
            value = to_float(raw_value, None)
            if value is None or not math.isfinite(value):
                continue
            metric = f"slurm_job_epoch_{prometheus_metric_suffix(key)}"
            lines.append(f"{metric}{{{labels}}} {value:.12g}")
        lines.append(f"slurm_job_epoch_low_gain_streak{{{labels}}} {to_float(epoch_data.get('low_gain_streak')):.12g}")
        lines.append(f"slurm_job_epoch_cumulative_energy_kwh{{{labels}}} {to_float(epoch_data.get('cumulative_total_energy_kwh')):.12g}")
        lines.append(f"slurm_job_epoch_cumulative_net_energy_kwh{{{labels}}} {to_float(epoch_data.get('cumulative_net_total_energy_kwh')):.12g}")
        lines.append(f"slurm_job_epoch_estimated_electricity_cost_eur{{{labels}}} {to_float(epoch_data.get('estimated_electricity_cost_eur')):.12g}")
        lines.append(f"slurm_job_epoch_estimated_co2_kg{{{labels}}} {to_float(epoch_data.get('estimated_co2_kg')):.12g}")
        lines.append(f"slurm_job_epoch_should_stop{{{labels}}} {to_float(epoch_data.get('should_stop')):.12g}")

    summary_labels = (
        f'job_id="{job_id}",comparison_strategy="{prom_escape(str(data.get("comparison_strategy", "unspecified")))}",'
        f'scenario="{scenario}",training_seed="{training_seed}",split_seed="{split_seed}"'
        f',controller_id="{controller_id}",benchmark_version="{benchmark_version}",'
        f'benchmark_run_id="{benchmark_run_id}",benchmark_case_id="{benchmark_case_id}",'
        f'benchmark_stage="{benchmark_stage}",task_type="{task_type}",quality_metric="{quality_metric}"'
    )
    for key in (
        "test_best_map50",
        "test_best_map50_95",
        "test_best_precision",
        "test_best_recall",
        "accuracy_regret_map50_95",
        "test_count_mae",
        "test_count_rmse",
        "test_count_bias",
        "test_accuracy",
        "test_macro_f1",
        "test_miou",
        "test_dice",
        "test_pixel_accuracy",
    ):
        if data.get(key) is not None:
            lines.append(f"slurm_job_{key}{{{summary_labels}}} {to_float(data.get(key)):.12g}")

    write_atomic(Path(args.output_prom), "\n".join(lines) + "\n")
    print(f"Wrote {args.output_prom}")


if __name__ == "__main__":
    main()
