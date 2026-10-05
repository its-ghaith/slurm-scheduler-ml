from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any


STUDY_FILTERS = 'training_seed=~"$training_seeds",comparison_strategy=~"$strategies"'


def add_study_filters(query: str) -> str:
    def update(match: re.Match[str]) -> str:
        labels = match.group(2)
        additions = []
        if not re.search(r"(?:^|,)\s*training_seed\s*=", labels):
            additions.append('training_seed=~"$training_seeds"')
        if not re.search(r"(?:^|,)\s*comparison_strategy\s*=", labels):
            additions.append('comparison_strategy=~"$strategies"')
        merged = ",".join([labels, *additions]) if labels else ",".join(additions)
        return f"{match.group(1)}{{{merged}}}"

    return re.sub(r"(\bslurm_[A-Za-z0-9_:]+)\{([^}]*)\}", update, query)


def update_queries(node: Any) -> None:
    if isinstance(node, list):
        for item in node:
            update_queries(item)
        return
    if not isinstance(node, dict):
        return
    for key, value in list(node.items()):
        if isinstance(value, str) and key in {"expr", "definition"}:
            node[key] = add_study_filters(value.replace('comparison_set="8_run"', 'comparison_set="stage_a"'))
        elif isinstance(value, str) and key == "query" and "label_values(" in value:
            node[key] = add_study_filters(value.replace('comparison_set="8_run"', 'comparison_set="stage_a"'))
        else:
            update_queries(value)


def variable(name: str, label: str, query: str, multi: bool = True) -> dict:
    return {
        "name": name,
        "label": label,
        "type": "query",
        "hide": 0,
        "datasource": {"type": "prometheus", "uid": "prometheus"},
        "definition": query,
        "query": {"query": query, "refId": "StandardVariableQuery"},
        "refresh": 1,
        "sort": 3,
        "multi": multi,
        "includeAll": multi,
        "allValue": ".*" if multi else None,
        "current": {"selected": multi, "text": ["All"] if multi else "0", "value": ["$__all"] if multi else "0"},
        "options": [],
    }


def paired_panel(panel_id: int, title: str, expr: str, unit: str, x: int, y: int) -> dict:
    return {
        "id": panel_id,
        "title": title,
        "type": "barchart",
        "pluginVersion": "11.1.0",
        "gridPos": {"x": x, "y": y, "w": 12, "h": 9},
        "datasource": {"type": "prometheus", "uid": "prometheus"},
        "fieldConfig": {
            "defaults": {
                "unit": unit,
                "color": {"mode": "palette-classic"},
                "custom": {
                    "axisPlacement": "auto",
                    "fillOpacity": 85,
                    "lineWidth": 1,
                    "stacking": {"group": "A", "mode": "none"},
                    "hideFrom": {"legend": False, "tooltip": False, "viz": False},
                },
            },
            "overrides": [],
        },
        "options": {
            "xField": "training_seed",
            "orientation": "vertical",
            "xTickLabelRotation": 0,
            "showValue": "auto",
            "stacking": "none",
            "groupWidth": 0.7,
            "barWidth": 0.35,
            "legend": {"showLegend": True, "displayMode": "list", "placement": "bottom", "calcs": []},
            "tooltip": {"mode": "single", "sort": "none", "hideZeros": False},
        },
        "targets": [{
            "refId": "A",
            "expr": expr,
            "legendFormat": "{{comparison_strategy}}",
            "instant": True,
            "range": False,
            "format": "table",
            "editorMode": "code",
            "datasource": {"type": "prometheus", "uid": "prometheus"},
        }],
        "transformations": [
            {
                "id": "convertFieldType",
                "options": {"numericJobSort": True, "conversions": [{"targetField": "training_seed", "destinationType": "number"}]},
            },
            {"id": "sortBy", "options": {"numericJobSort": True, "fields": {}, "sort": [{"field": "training_seed", "desc": False}]}},
            {
                "id": "groupingToMatrix",
                "options": {"rowField": "training_seed", "columnField": "comparison_strategy", "valueField": "Value", "emptyValue": "null"},
            },
        ],
    }


def single_seed_panel(panel_id: int, title: str, metric: str, unit: str, x: int, y: int) -> dict:
    panel = paired_panel(
        panel_id,
        title,
        f'max by (training_seed) ({metric}{{comparison_set="stage_a",training_seed=~"$training_seeds"}})',
        unit,
        x,
        y,
    )
    panel["targets"][0]["legendFormat"] = title
    panel["transformations"] = panel["transformations"][:2]
    return panel


def study_stat(panel_id: int, title: str, metric: str, unit: str, x: int, y: int) -> dict:
    return {
        "id": panel_id,
        "title": title,
        "type": "stat",
        "pluginVersion": "11.1.0",
        "gridPos": {"x": x, "y": y, "w": 6, "h": 5},
        "datasource": {"type": "prometheus", "uid": "prometheus"},
        "fieldConfig": {"defaults": {"unit": unit, "color": {"mode": "thresholds"}, "thresholds": {"mode": "absolute", "steps": [{"color": "green", "value": None}]}}, "overrides": []},
        "options": {"colorMode": "value", "graphMode": "none", "justifyMode": "auto", "textMode": "value_and_name", "reduceOptions": {"calcs": ["lastNotNull"], "fields": "", "values": False}},
        "targets": [{"refId": "A", "expr": f'{metric}{{comparison_set="stage_a"}}', "legendFormat": title, "instant": True, "range": False, "format": "time_series", "editorMode": "code", "datasource": {"type": "prometheus", "uid": "prometheus"}}],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    dashboard = json.loads(args.source.read_text(encoding="utf-8-sig"))
    dashboard.update({
        "id": None,
        "uid": "slurm-energy-stage-a",
        "title": "Stage A - Seed Generalisation (Hybrid Min30 Eval1)",
        "version": 1,
    })
    update_queries(dashboard)

    variables = dashboard.setdefault("templating", {}).setdefault("list", [])
    variables[:] = [item for item in variables if item.get("name") not in {"training_seeds", "strategies"}]
    seed_query = 'label_values(slurm_job_info{comparison_set="stage_a"}, training_seed)'
    strategy_query = 'label_values(slurm_job_info{comparison_set="stage_a",training_seed=~"$training_seeds"}, comparison_strategy)'
    variables[0:0] = [
        variable("training_seeds", "Training Seeds", seed_query),
        variable("strategies", "Strategies", strategy_query),
    ]
    for item in variables:
        if item.get("name") in {"job_ids", "inference_job_id"}:
            item["sort"] = 3

    panels = dashboard.setdefault("panels", [])
    y = max((panel.get("gridPos", {}).get("y", 0) + panel.get("gridPos", {}).get("h", 0) for panel in panels), default=0)
    panels.append({
        "id": 200,
        "title": "Stage A: Paired Generalisation Across Training Seeds",
        "type": "row",
        "collapsed": False,
        "gridPos": {"x": 0, "y": y, "w": 24, "h": 1},
        "panels": [],
    })
    filters = 'comparison_set="stage_a",training_seed=~"$training_seeds",comparison_strategy=~"$strategies"'
    panels.extend([
        paired_panel(201, "Best Validation mAP50-95 by Seed (%)", f"max by (training_seed, comparison_strategy) (slurm_job_epoch_best_map50_95{{{filters}}}) * 100", "percent", 0, y + 1),
        paired_panel(202, "GPU Energy by Seed (Wh)", f"max by (training_seed, comparison_strategy) (slurm_job_training_energy_kwh{{{filters}}}) * 1000", "watth", 12, y + 1),
        paired_panel(203, "Completed Epochs by Seed", f"count by (training_seed, comparison_strategy) (slurm_job_epoch_duration_seconds{{{filters}}})", "none", 0, y + 10),
        paired_panel(204, "Job Duration by Seed (s)", f"max by (training_seed, comparison_strategy) (slurm_job_duration_seconds{{{filters}}})", "s", 12, y + 10),
        single_seed_panel(205, "Best Quality Regret by Seed (pp)", "slurm_stage_a_best_quality_regret_pp", "percentpoint", 0, y + 19),
        single_seed_panel(206, "GPU Energy Saving by Seed (%)", "slurm_stage_a_energy_saving_pct", "percent", 12, y + 19),
        single_seed_panel(207, "Duration Saving by Seed (%)", "slurm_stage_a_duration_saving_pct", "percent", 0, y + 28),
        single_seed_panel(208, "Epochs Saved by Seed", "slurm_stage_a_epochs_saved", "none", 12, y + 28),
        study_stat(209, "Adaptive Stop Rate (%)", "slurm_stage_a_adaptive_stop_rate_pct", "percent", 0, y + 37),
        study_stat(210, "Regret Budget Success Rate (%)", "slurm_stage_a_regret_budget_success_rate_pct", "percent", 6, y + 37),
        study_stat(211, "Mean GPU Energy Saving (%)", "slurm_stage_a_mean_energy_saving_pct", "percent", 12, y + 37),
        study_stat(212, "Mean Best Quality Regret (pp)", "slurm_stage_a_mean_best_quality_regret_pp", "percentpoint", 18, y + 37),
        study_stat(213, "Evidence-Based Stop Rate (%)", "slurm_stage_a_evidence_based_stop_rate_pct", "percent", 0, y + 42),
        study_stat(214, "Budget Fallback Rate (%)", "slurm_stage_a_budget_fallback_stop_rate_pct", "percent", 6, y + 42),
        study_stat(215, "Positive Energy Saving Rate (%)", "slurm_stage_a_positive_energy_saving_rate_pct", "percent", 12, y + 42),
        study_stat(216, "Mean Epochs Saved", "slurm_stage_a_mean_epochs_saved", "none", 18, y + 42),
    ])

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(dashboard, separators=(",", ":")), encoding="utf-8", newline="\n")


if __name__ == "__main__":
    main()
