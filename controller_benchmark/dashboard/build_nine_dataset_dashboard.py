from __future__ import annotations

import argparse
import json
from pathlib import Path


DS = {"type": "prometheus", "uid": "prometheus"}


def query(ref: str, expression: str, legend: str, *, table: bool = False) -> dict:
    return {
        "refId": ref,
        "expr": expression,
        "legendFormat": legend,
        "instant": table,
        "range": not table,
        "format": "table" if table else "time_series",
        "editorMode": "code",
        "datasource": DS,
    }


def filters(controller: str | None, *, epoch: bool = False) -> str:
    stage_label = "benchmark_stage" if epoch else "stage"
    case_label = "benchmark_case_id" if epoch else "case_id"
    controller_filter = f'controller_id="{controller}"' if controller else 'controller_id=~"$controllers"'
    return (
        f'benchmark_run_id="$benchmark_run_id",{controller_filter},'
        f'task_type=~"$task_types",{stage_label}=~"$stages",{case_label}=~"$cases"'
    )


def row(panel_id: int, title: str, y: int) -> dict:
    return {
        "id": panel_id,
        "title": title,
        "type": "row",
        "collapsed": False,
        "gridPos": {"x": 0, "y": y, "w": 24, "h": 1},
        "panels": [],
    }


def stat(panel_id: int, title: str, expression: str, unit: str, x: int, y: int) -> dict:
    return {
        "id": panel_id,
        "title": title,
        "type": "stat",
        "gridPos": {"x": x, "y": y, "w": 4, "h": 5},
        "datasource": DS,
        "fieldConfig": {
            "defaults": {
                "unit": unit,
                "color": {"mode": "thresholds"},
                "thresholds": {"mode": "absolute", "steps": [{"color": "green", "value": None}]},
            },
            "overrides": [],
        },
        "options": {
            "colorMode": "value",
            "graphMode": "none",
            "textMode": "value_and_name",
            "reduceOptions": {"calcs": ["lastNotNull"], "fields": "", "values": False},
        },
        "targets": [query("A", expression, "{{controller_id}}", table=True)],
    }


def bars(
    panel_id: int,
    title: str,
    metric: str,
    unit: str,
    x: int,
    y: int,
    controller: str | None,
    *,
    multiplier: str = "",
    legend: str = "{{variant}} / {{controller_id}}",
) -> dict:
    selected = filters(controller)
    value_expression = f"{metric}{{{selected}}}{multiplier}"
    value_expression = (
        f'label_join({value_expression}, "series", " / ", "controller_id", "variant")'
    )
    return {
        "id": panel_id,
        "title": title,
        "type": "barchart",
        "gridPos": {"x": x, "y": y, "w": 12, "h": 9},
        "datasource": DS,
        "fieldConfig": {
            "defaults": {
                "unit": unit,
                "color": {"mode": "palette-classic"},
                "custom": {
                    "fillOpacity": 85,
                    "lineWidth": 1,
                    "stacking": {"group": "A", "mode": "none"},
                },
            },
            "overrides": [],
        },
        "options": {
            "xField": "case_id",
            "orientation": "vertical",
            "xTickLabelRotation": 45,
            "stacking": "none",
            "groupWidth": 0.72,
            "barWidth": 0.25,
            "legend": {"showLegend": True, "displayMode": "list", "placement": "bottom"},
        },
        "targets": [query("A", value_expression, legend, table=True)],
        "transformations": [
            {"id": "sortBy", "options": {"fields": {}, "sort": [{"field": "case_id", "desc": False}]}},
            {
                "id": "groupingToMatrix",
                "options": {
                    "rowField": "case_id",
                    "columnField": "series",
                    "valueField": "Value",
                    "emptyValue": "null",
                },
            },
        ],
    }


def single_bars(
    panel_id: int,
    title: str,
    metric: str,
    unit: str,
    x: int,
    y: int,
    controller: str | None,
    *,
    multiplier: str = "",
) -> dict:
    panel = bars(panel_id, title, metric, unit, x, y, controller, multiplier=multiplier)
    selected = filters(controller)
    panel["targets"] = [
        query("A", f"{metric}{{{selected}}}{multiplier}", "{{controller_id}}", table=True)
    ]
    panel["transformations"] = panel["transformations"][:1]
    return panel


def lines(
    panel_id: int,
    title: str,
    targets: list[tuple[str, str, str]],
    unit: str,
    x: int,
    y: int,
    controller: str | None,
    *,
    task_type: str | None = None,
) -> dict:
    selected = filters(controller, epoch=True)
    if task_type:
        selected = selected.replace('task_type=~"$task_types"', f'task_type="{task_type}"')
    return {
        "id": panel_id,
        "title": title,
        "type": "timeseries",
        "gridPos": {"x": x, "y": y, "w": 12, "h": 9},
        "datasource": DS,
        "fieldConfig": {
            "defaults": {
                "unit": unit,
                "custom": {
                    "drawStyle": "line",
                    "lineWidth": 2,
                    "showPoints": "auto",
                    "spanNulls": False,
                },
            },
            "overrides": [],
        },
        "options": {
            "legend": {"showLegend": True, "displayMode": "list", "placement": "bottom"},
            "tooltip": {"mode": "multi", "sort": "desc"},
        },
        "targets": [
            query(
                ref,
                metric.replace("$filters", f"{{{selected}}}")
                if "$filters" in metric
                else f"{metric}{{{selected}}}",
                legend,
            )
            for ref, metric, legend in targets
        ],
    }


def variable(name: str, label: str, definition: str, *, multi: bool = True) -> dict:
    return {
        "name": name,
        "label": label,
        "type": "query",
        "datasource": DS,
        "definition": definition,
        "query": {"query": definition, "refId": "StandardVariableQuery"},
        "refresh": 1,
        "sort": 1,
        "multi": multi,
        "includeAll": multi,
        "allValue": ".*" if multi else None,
        "current": {"selected": multi, "text": ["All"] if multi else "", "value": ["$__all"] if multi else ""},
        "options": [],
    }


def build_dashboard(*, title: str, uid: str, controller: str | None) -> dict:
    selected = filters(controller)
    controller_match = f'controller_id="{controller}"' if controller else 'controller_id=~"$controllers"'
    panels = [
        row(1, "Campaign Summary", 0),
        stat(2, "Selected Pairs", f"count(controller_benchmark_case_info{{{selected}}})", "none", 0, 1),
        stat(
            3,
            "Energy Success",
            f"controller_benchmark_energy_success_rate{{benchmark_run_id=\"$benchmark_run_id\",{controller_match}}}",
            "percentunit",
            4,
            1,
        ),
        stat(
            4,
            "Quality Success",
            f"controller_benchmark_quality_success_rate{{benchmark_run_id=\"$benchmark_run_id\",{controller_match}}}",
            "percentunit",
            8,
            1,
        ),
        stat(
            5,
            "Joint Success",
            f"controller_benchmark_joint_success_rate{{benchmark_run_id=\"$benchmark_run_id\",{controller_match}}}",
            "percentunit",
            12,
            1,
        ),
        stat(
            6,
            "Mean Energy Saving",
            f"controller_benchmark_mean_energy_saving_fraction{{benchmark_run_id=\"$benchmark_run_id\",{controller_match}}}",
            "percentunit",
            16,
            1,
        ),
        stat(
            7,
            "Mean Validation Regret",
            f"controller_benchmark_mean_quality_regret{{benchmark_run_id=\"$benchmark_run_id\",{controller_match}}}",
            "percentunit",
            20,
            1,
        ),
        row(10, "Final Dataset-Level Comparison", 6),
        bars(11, "Full100 vs Controller GPU Energy by Dataset", "controller_benchmark_energy_wh", "watth", 0, 7, controller),
        bars(
            12,
            "Full100 vs Controller Best Validation Quality",
            "controller_benchmark_best_quality",
            "percentunit",
            12,
            7,
            controller,
        ),
        single_bars(
            13,
            "Energy Saving by Dataset",
            "controller_benchmark_energy_saving_fraction",
            "percentunit",
            0,
            16,
            controller,
        ),
        single_bars(
            14,
            "Validation Quality Regret by Dataset",
            "controller_benchmark_quality_regret",
            "percentunit",
            12,
            16,
            controller,
        ),
        bars(15, "Full100 vs Controller Epoch Count", "controller_benchmark_epochs", "none", 0, 25, controller),
        single_bars(
            16,
            "Epochs Saved by Dataset",
            "controller_benchmark_epochs_saved",
            "none",
            12,
            25,
            controller,
        ),
        bars(
            17,
            "Full100 vs Controller Duration",
            "controller_benchmark_duration_seconds",
            "s",
            0,
            34,
            controller,
        ),
        single_bars(
            18,
            "Held-Out Test Quality Regret",
            "controller_campaign_test_quality_regret",
            "percentunit",
            12,
            34,
            controller,
        ),
        row(20, "Task-Specific Quality", 43),
        lines(
            21,
            "Aerial Vehicle Detection Quality by Epoch",
            [
                ("A", "slurm_job_epoch_map50_95", "mAP50-95 / {{benchmark_case_id}} / {{controller_id}}"),
                ("B", "slurm_job_epoch_map50", "mAP50 / {{benchmark_case_id}} / {{controller_id}}"),
            ],
            "percentunit",
            0,
            44,
            controller,
            task_type="object_detection",
        ),
        lines(
            22,
            "Classification Quality by Epoch",
            [
                ("A", "slurm_job_epoch_accuracy", "Accuracy / {{benchmark_case_id}} / {{controller_id}}"),
                ("B", "slurm_job_epoch_macro_f1", "Macro-F1 / {{benchmark_case_id}} / {{controller_id}}"),
            ],
            "percentunit",
            12,
            44,
            controller,
            task_type="image_classification",
        ),
        lines(
            23,
            "Segmentation Quality by Epoch",
            [
                ("A", "slurm_job_epoch_miou", "mIoU / {{benchmark_case_id}} / {{controller_id}}"),
                ("B", "slurm_job_epoch_dice", "Dice / {{benchmark_case_id}} / {{controller_id}}"),
                ("C", "slurm_job_epoch_pixel_accuracy", "Pixel accuracy / {{benchmark_case_id}} / {{controller_id}}"),
            ],
            "percentunit",
            0,
            53,
            controller,
            task_type="semantic_segmentation",
        ),
        lines(
            24,
            "Vehicle Count Error on Held-Out Test Data",
            [
                ("A", "slurm_job_test_count_mae", "Count MAE / {{benchmark_case_id}} / {{controller_id}}"),
                ("B", "slurm_job_test_count_rmse", "Count RMSE / {{benchmark_case_id}} / {{controller_id}}"),
            ],
            "none",
            12,
            53,
            controller,
            task_type="object_detection",
        ),
        row(30, "Energy and Runtime Dynamics", 62),
        lines(
            31,
            "GPU Energy per Epoch",
            [
                (
                    "A",
                    "(slurm_job_epoch_gpu_energy_kwh$filters * 1000)",
                    "Wh / {{benchmark_case_id}} / {{controller_id}}",
                )
            ],
            "watth",
            0,
            63,
            controller,
        ),
        lines(
            32,
            "Epoch Duration",
            [("A", "slurm_job_epoch_duration_seconds", "Seconds / {{benchmark_case_id}} / {{controller_id}}")],
            "s",
            12,
            63,
            controller,
        ),
        lines(
            33,
            "GPU Utilization and Power",
            [
                ("A", "slurm_job_epoch_gpu_util_avg_pct", "GPU utilization % / {{benchmark_case_id}}"),
                ("B", "slurm_job_epoch_gpu_power_avg_w", "GPU power W / {{benchmark_case_id}}"),
            ],
            "none",
            0,
            72,
            controller,
        ),
        lines(
            34,
            "GPU Memory",
            [
                ("A", "slurm_job_epoch_gpu_mem_used_avg_mb", "GPU memory MiB / {{benchmark_case_id}} / {{controller_id}}")
            ],
            "decmbytes",
            12,
            72,
            controller,
        ),
        row(40, "Controller Inputs and Decisions", 81),
        lines(
            41,
            "Gradient Norm and Training Loss",
            [
                ("A", "slurm_job_epoch_gradient_norm", "Gradient norm / {{benchmark_case_id}}"),
                ("B", "slurm_job_epoch_train_loss", "Training loss / {{benchmark_case_id}}"),
            ],
            "none",
            0,
            82,
            controller,
        ),
        lines(
            42,
            "Learning Rate",
            [("A", "slurm_job_epoch_learning_rate", "Learning rate / {{benchmark_case_id}}")],
            "none",
            12,
            82,
            controller,
        ),
        lines(
            43,
            "RAPEC-v3 Expected Gain and Dynamic Threshold",
            [
                ("A", "slurm_job_epoch_controller_plugin_rapec_expected_quality_gain_next_horizon", "Expected gain / {{benchmark_case_id}}"),
                ("B", "slurm_job_epoch_controller_plugin_rapec_dynamic_meaningful_gain", "Meaningful gain / {{benchmark_case_id}}"),
                ("C", "slurm_job_epoch_controller_plugin_rapec_prob_gain_gt_dynamic_threshold", "P(gain > threshold) / {{benchmark_case_id}}"),
            ],
            "none",
            0,
            91,
            controller,
        ),
        lines(
            44,
            "RAPEC-v8 Bayesian Guard",
            [
                ("A", "slurm_job_epoch_controller_plugin_rapec8_posterior_probability_relevant_gain", "Posterior probability / {{benchmark_case_id}}"),
                ("B", "slurm_job_epoch_controller_plugin_rapec8_base_v3_stop", "v3 stop evidence / {{benchmark_case_id}}"),
                ("C", "slurm_job_epoch_controller_plugin_rapec8_bayesian_low_gain", "Bayesian low-gain evidence / {{benchmark_case_id}}"),
            ],
            "none",
            12,
            91,
            controller,
        ),
    ]
    variables = [
        variable(
            "benchmark_run_id",
            "Benchmark Run",
            'label_values(controller_benchmark_case_info{benchmark_version="controller-benchmark-v3-nine-dataset-fresh"}, benchmark_run_id)',
            multi=False,
        )
    ]
    if controller is None:
        variables.append(
            variable(
                "controllers",
                "Controller",
                'label_values(controller_benchmark_case_info{benchmark_run_id="$benchmark_run_id"}, controller_id)',
            )
        )
    controller_filter = (
        f'controller_id="{controller}"' if controller else 'controller_id=~"$controllers"'
    )
    variables.extend(
        [
            variable(
                "task_types",
                "Task Type",
                f'label_values(controller_benchmark_case_info{{benchmark_run_id="$benchmark_run_id",{controller_filter}}}, task_type)',
            ),
            variable(
                "stages",
                "Stage",
                f'label_values(controller_benchmark_case_info{{benchmark_run_id="$benchmark_run_id",{controller_filter},task_type=~"$task_types"}}, stage)',
            ),
            variable(
                "cases",
                "Dataset",
                f'label_values(controller_benchmark_case_info{{benchmark_run_id="$benchmark_run_id",{controller_filter},task_type=~"$task_types",stage=~"$stages"}}, case_id)',
            ),
        ]
    )
    return {
        "annotations": {"list": []},
        "editable": True,
        "graphTooltip": 1,
        "id": None,
        "panels": panels,
        "refresh": "30s",
        "schemaVersion": 39,
        "tags": ["controller-benchmark", "nine-datasets", "rapec", "energy"],
        "templating": {"list": variables},
        "time": {"from": "now-30d", "to": "now"},
        "timezone": "browser",
        "title": title,
        "uid": uid,
        "version": 1,
    }


def write_dashboard(path: Path, dashboard: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dashboard, separators=(",", ":")), encoding="utf-8", newline="\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    configurations = [
        (
            "nine-dataset-comparison.json",
            "RAPEC-v3 vs RAPEC-v8 - Fresh 9-Dataset Generalisation",
            "rapec-v3-v8-nine-dataset",
            None,
        ),
        (
            "nine-dataset-rapec-v3.json",
            "RAPEC-v3 - Fresh 9-Dataset Benchmark",
            "rapec-v3-nine-dataset",
            "rapec-v3",
        ),
        (
            "nine-dataset-rapec-v8.json",
            "RAPEC-v8 - Fresh 9-Dataset Benchmark",
            "rapec-v8-nine-dataset",
            "rapec-v8",
        ),
    ]
    for filename, title, uid, controller in configurations:
        write_dashboard(
            args.output_dir / filename,
            build_dashboard(title=title, uid=uid, controller=controller),
        )


if __name__ == "__main__":
    main()
