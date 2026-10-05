#!/usr/bin/env python3
"""Create the dedicated cross-stage Generalized Energy Guard dashboard."""

from __future__ import annotations

import json
from pathlib import Path


DS = {"type": "prometheus", "uid": "prometheus"}
FILTER = 'comparison_set="energy_guard_study",stage=~"$stages",case_label=~"$cases"'


def target(ref: str, expr: str, legend: str, table: bool = True) -> dict:
    return {
        "refId": ref, "expr": expr, "legendFormat": legend, "instant": True, "range": False,
        "format": "table" if table else "time_series", "editorMode": "code", "datasource": DS,
    }


def stat(pid: int, title: str, metric: str, unit: str, x: int) -> dict:
    return {
        "id": pid, "title": title, "type": "stat", "pluginVersion": "11.1.0",
        "gridPos": {"x": x, "y": 0, "w": 3, "h": 5}, "datasource": DS,
        "fieldConfig": {"defaults": {"unit": unit, "color": {"mode": "thresholds"}, "thresholds": {"mode": "absolute", "steps": [{"color": "green", "value": None}]}}, "overrides": []},
        "options": {"colorMode": "value", "graphMode": "none", "justifyMode": "auto", "textMode": "value_and_name", "reduceOptions": {"calcs": ["lastNotNull"], "fields": "", "values": False}},
        "targets": [target("A", f'{metric}{{comparison_set="energy_guard_study"}}', title, False)],
    }


def bars(pid: int, title: str, metric: str, unit: str, x: int, y: int, w: int = 12) -> dict:
    return {
        "id": pid, "title": title, "type": "barchart", "pluginVersion": "11.1.0",
        "gridPos": {"x": x, "y": y, "w": w, "h": 9}, "datasource": DS,
        "fieldConfig": {"defaults": {"unit": unit, "color": {"mode": "palette-classic"}, "custom": {"axisPlacement": "auto", "fillOpacity": 85, "lineWidth": 1, "stacking": {"group": "A", "mode": "none"}, "hideFrom": {"legend": False, "tooltip": False, "viz": False}}}, "overrides": []},
        "options": {"xField": "case_label", "orientation": "vertical", "xTickLabelRotation": 45, "showValue": "auto", "stacking": "none", "groupWidth": 0.72, "barWidth": 0.28, "legend": {"showLegend": True, "displayMode": "list", "placement": "bottom", "calcs": []}, "tooltip": {"mode": "multi", "sort": "none", "hideZeros": False}},
        "targets": [target("A", f'{metric}{{{FILTER}}}', "{{series}}")],
        "transformations": [{"id": "sortBy", "options": {"fields": {}, "sort": [{"field": "case_label", "desc": False}]}}, {"id": "groupingToMatrix", "options": {"rowField": "case_label", "columnField": "series", "valueField": "Value", "emptyValue": "null"}}],
    }


def epoch_timeseries(pid: int, title: str, queries: list[tuple[str, str, str]], unit: str, x: int, y: int) -> dict:
    return {
        "id": pid, "title": title, "type": "timeseries", "pluginVersion": "11.1.0",
        "gridPos": {"x": x, "y": y, "w": 12, "h": 9}, "datasource": DS,
        "fieldConfig": {"defaults": {"unit": unit, "color": {"mode": "palette-classic"}, "custom": {"drawStyle": "line", "lineWidth": 2, "showPoints": "auto", "spanNulls": False}}, "overrides": []},
        "options": {"legend": {"showLegend": True, "displayMode": "list", "placement": "bottom", "calcs": []}, "tooltip": {"mode": "multi", "sort": "none"}},
        "targets": [target(ref, expr, legend, False) | {"instant": False, "range": True} for ref, expr, legend in queries],
    }


def variable(name: str, label: str, query: str, multi: bool = True) -> dict:
    return {"name": name, "label": label, "type": "query", "hide": 0, "datasource": DS, "definition": query, "query": {"query": query, "refId": "StandardVariableQuery"}, "refresh": 1, "sort": 1, "multi": multi, "includeAll": multi, "allValue": ".*" if multi else None, "current": {"selected": multi, "text": ["All"] if multi else "", "value": ["$__all"] if multi else ""}, "options": []}


def main() -> None:
    panels = [
        stat(1, "Paired Cases", "slurm_energy_guard_total_pairs", "none", 0),
        stat(2, "20% Target Success", "slurm_energy_guard_target_success_rate_pct", "percent", 3),
        stat(3, "Minimum Energy Saving", "slurm_energy_guard_min_energy_saving_pct", "percent", 6),
        stat(4, "Mean Energy Saving", "slurm_energy_guard_mean_energy_saving_pct", "percent", 9),
        stat(5, "Mean mAP Regret", "slurm_energy_guard_mean_quality_regret_pp", "percentpoint", 12),
        stat(6, "Maximum mAP Regret", "slurm_energy_guard_max_quality_regret_pp", "percentpoint", 15),
        stat(7, "Quality Conflict Rate", "slurm_energy_guard_quality_conflict_rate_pct", "percent", 18),
        stat(8, "Target", "slurm_energy_guard_target_saving_pct", "percent", 21),
        bars(10, "Full100 vs Controller GPU Energy (Wh)", "slurm_energy_guard_energy_wh", "watth", 0, 5),
        bars(11, "Energy Saving by Case (%)", "slurm_energy_guard_energy_saving_pct", "percent", 12, 5),
        bars(12, "Full100 vs Controller Best mAP50-95 (%)", "slurm_energy_guard_best_map50_95_pct", "percent", 0, 14),
        bars(13, "Best mAP50-95 Regret by Case (pp)", "slurm_energy_guard_quality_regret_pp", "percentpoint", 12, 14),
        bars(14, "Completed Epochs", "slurm_energy_guard_epochs", "none", 0, 23),
        bars(15, "Job Duration (s)", "slurm_energy_guard_duration_seconds", "s", 12, 23),
        bars(16, "Energy Target and Quality Conflict", "slurm_energy_guard_target_met", "bool", 0, 32, 12),
        bars(17, "Quality Conflict by Case", "slurm_energy_guard_quality_conflict", "bool", 12, 32, 12),
    ]
    controller_filter = 'comparison_set="energy_guard_study",job_id=~"$controller_jobs"'
    panels.extend([
        epoch_timeseries(20, "Projected Energy Saving by Epoch (%)", [("A", f'slurm_job_epoch_projected_energy_saving_fraction{{{controller_filter}}} * 100', "Job {{job_id}} projected"), ("B", f'slurm_job_epoch_energy_guard_target_saving_fraction{{{controller_filter}}} * 100', "Job {{job_id}} target")], "percent", 0, 41),
        epoch_timeseries(21, "Best mAP50-95 by Epoch (%)", [("A", f'slurm_job_epoch_best_map50_95{{{controller_filter}}} * 100', "Job {{job_id}}")], "percent", 12, 41),
        epoch_timeseries(22, "Observed and Projected Job Energy (Wh)", [("A", f'slurm_job_epoch_observed_job_energy_wh{{{controller_filter}}}', "Job {{job_id}} observed"), ("B", f'slurm_job_epoch_projected_full_job_energy_wh{{{controller_filter}}}', "Job {{job_id}} projected Full100"), ("C", f'slurm_job_epoch_projected_stopped_job_energy_wh{{{controller_filter}}}', "Job {{job_id}} projected stop")], "watth", 0, 50),
        epoch_timeseries(23, "Controller Decision and Quality Conflict", [("A", f'slurm_job_epoch_controller_decision_state{{{controller_filter}}}', "Job {{job_id}} decision"), ("B", f'slurm_job_epoch_energy_guard_quality_conflict{{{controller_filter}}}', "Job {{job_id}} conflict"), ("C", f'slurm_job_epoch_energy_guard_fallback_triggered{{{controller_filter}}}', "Job {{job_id}} fallback")], "none", 12, 50),
    ])
    dashboard = {
        "annotations": {"list": []}, "editable": True, "fiscalYearStartMonth": 0, "graphTooltip": 1,
        "id": None, "links": [], "panels": panels, "refresh": "30s", "schemaVersion": 39,
        "tags": ["slurm", "energy", "generalisation", "energy-guard"],
        "templating": {"list": [
            variable("stages", "Stages", 'label_values(slurm_energy_guard_case_info{comparison_set="energy_guard_study"}, stage)'),
            variable("cases", "Cases", 'label_values(slurm_energy_guard_case_info{comparison_set="energy_guard_study",stage=~"$stages"}, case_label)'),
            variable("controller_jobs", "Controller Jobs", 'label_values(slurm_energy_guard_case_info{comparison_set="energy_guard_study",stage=~"$stages",case_label=~"$cases"}, controller_job_id)'),
        ]},
        "time": {"from": "now-30d", "to": "now"}, "timepicker": {}, "timezone": "browser",
        "title": "Generalized Energy Guard - Cross-Stage 20% Validation", "uid": "slurm-energy-guard-study", "version": 1,
    }
    output = Path(__file__).resolve().parents[1] / "grafana" / "dashboards" / "energy-guard-study.json"
    output.write_text(json.dumps(dashboard, separators=(",", ":")), encoding="utf-8", newline="\n")
    print(output)


if __name__ == "__main__":
    main()
