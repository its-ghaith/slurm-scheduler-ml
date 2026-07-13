#!/usr/bin/env python3
"""Idempotently add study panels and synchronize the Kubernetes ConfigMap."""

from __future__ import annotations

import copy
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
DASHBOARD_PATH = ROOT / "grafana" / "dashboards" / "slurm-energy-overview.json"
MANIFEST_PATH = ROOT / "rancherConfigs" / "slurm-stack.yaml"
PANEL_IDS = set(range(116, 124))


def variable(name: str, label: str, query: str) -> dict:
    return {
        "name": name,
        "label": label,
        "type": "query",
        "hide": 0,
        "datasource": {"type": "prometheus", "uid": "prometheus"},
        "definition": query,
        "query": {"query": query, "refId": "StandardVariableQuery"},
        "refresh": 1,
        "sort": 1,
        "multi": True,
        "includeAll": True,
        "allValue": ".*",
        "current": {"selected": True, "text": ["All"], "value": ["$__all"]},
        "options": [],
    }


def study_panel(template: dict, panel_id: int, title: str, metric: str, unit: str, x: int, y: int, w: int) -> dict:
    panel = copy.deepcopy(template)
    panel.update({"id": panel_id, "title": title, "gridPos": {"x": x, "y": y, "w": w, "h": 9}})
    panel["fieldConfig"]["defaults"]["unit"] = unit
    panel["options"].update({"xField": "job", "xTickLabelRotation": 30, "showValue": "always"})
    panel["options"]["legend"]["showLegend"] = False
    factor = " * 100" if metric in {"slurm_study_test_best_map50_95"} else ""
    filters = (
        'job_id=~"$job_ids",scenario=~"$scenarios",comparison_strategy=~"$strategies",'
        'training_seed=~"$training_seeds"'
    )
    panel["targets"] = [
        {
            "expr": f"({metric}{{{filters}}}{factor})",
            "legendFormat": "Job {{job_id}} | {{comparison_strategy}} | seed {{training_seed}}",
            "refId": "A",
            "datasource": {"type": "prometheus", "uid": "prometheus"},
            "editorMode": "code",
            "exemplar": False,
            "instant": True,
            "range": False,
            "format": "time_series",
        }
    ]
    return panel


def synchronize_manifest(dashboard_text: str) -> None:
    manifest = MANIFEST_PATH.read_text(encoding="utf-8")
    marker = "  slurm-energy-overview.json: |\n"
    configmap_start = manifest.index("  name: grafana-dashboard-slurm-energy")
    data_start = manifest.index(marker, configmap_start) + len(marker)
    next_document = manifest.index("\n---\n", data_start)
    indented = "\n".join("    " + line for line in dashboard_text.rstrip().splitlines()) + "\n"
    MANIFEST_PATH.write_text(manifest[:data_start] + indented + manifest[next_document:], encoding="utf-8", newline="\n")


def main() -> None:
    dashboard = json.loads(DASHBOARD_PATH.read_text(encoding="utf-8"))
    template = next(panel for panel in dashboard["panels"] if panel.get("id") == 15)
    dashboard["panels"] = [panel for panel in dashboard["panels"] if panel.get("id") not in PANEL_IDS]

    inference = next((panel for panel in dashboard["panels"] if panel.get("id") == 90), None)
    if inference:
        inference["gridPos"]["y"] = 175

    definitions = [
        (116, "Best Checkpoint Test mAP50-95 by Job (%)", "slurm_study_test_best_map50_95", "percent", 0, 148, 8),
        (117, "Accuracy Regret vs Full100 (pp)", "slurm_study_accuracy_regret_map50_95_pp", "percent", 8, 148, 8),
        (118, "Pareto Dominated by Job (0/1)", "slurm_study_pareto_dominated", "short", 16, 148, 8),
        (119, "Gross Job GPU Energy (Wh)", "slurm_study_job_gpu_energy_wh", "watth", 0, 157, 12),
        (120, "Net Job GPU Energy above Idle (Wh)", "slurm_study_job_net_gpu_energy_wh", "watth", 12, 157, 12),
        (121, "Energy to Validation mAP50-95 50% (Wh)", "slurm_study_energy_to_map50_95_50_wh", "watth", 0, 166, 8),
        (122, "Energy to Validation mAP50-95 60% (Wh)", "slurm_study_energy_to_map50_95_60_wh", "watth", 8, 166, 8),
        (123, "Energy to Validation mAP50-95 68% (Wh)", "slurm_study_energy_to_map50_95_68_wh", "watth", 16, 166, 8),
    ]
    dashboard["panels"].extend(study_panel(template, *definition) for definition in definitions)
    dashboard["panels"].sort(key=lambda panel: (panel.get("gridPos", {}).get("y", 0), panel.get("gridPos", {}).get("x", 0)))

    variables = dashboard.setdefault("templating", {}).setdefault("list", [])
    variables[:] = [item for item in variables if item.get("name") not in {"scenarios", "strategies", "training_seeds"}]
    variables[0:0] = [
        variable("scenarios", "Scenarios", "label_values(slurm_job_info, scenario)"),
        variable("strategies", "Strategies", 'label_values(slurm_job_info{scenario=~"$scenarios"}, comparison_strategy)'),
        variable(
            "training_seeds",
            "Training Seeds",
            'label_values(slurm_job_info{scenario=~"$scenarios",comparison_strategy=~"$strategies"}, training_seed)',
        ),
    ]
    job_variable = next(item for item in variables if item.get("name") == "job_ids")
    job_query = (
        'label_values(slurm_job_info{scenario=~"$scenarios",comparison_strategy=~"$strategies",'
        'training_seed=~"$training_seeds"}, job_id)'
    )
    job_variable["definition"] = job_query
    job_variable["query"]["query"] = job_query

    dashboard["version"] = int(dashboard.get("version", 0)) + 1
    dashboard_text = json.dumps(dashboard, indent=2, ensure_ascii=False) + "\n"
    DASHBOARD_PATH.write_text(dashboard_text, encoding="utf-8", newline="\n")
    synchronize_manifest(dashboard_text)
    print(f"Updated {DASHBOARD_PATH} and {MANIFEST_PATH}")


if __name__ == "__main__":
    main()
