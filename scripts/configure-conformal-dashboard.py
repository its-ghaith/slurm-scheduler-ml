from __future__ import annotations

import copy
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
DASHBOARD = ROOT / "grafana" / "dashboards" / "slurm-energy-overview.json"
PANEL_IDS = set(range(130, 136))


def main() -> None:
    dashboard = json.loads(DASHBOARD.read_text(encoding="utf-8-sig"))
    dashboard["panels"] = [panel for panel in dashboard["panels"] if panel.get("id") not in PANEL_IDS]
    base = next(panel for panel in dashboard["panels"] if panel.get("id") == 98)
    inference = next(panel for panel in dashboard["panels"] if panel.get("id") == 90)
    inference["gridPos"]["y"] = 202

    common_filter = (
        'job_id=~"$job_ids",scenario=~"$scenarios",comparison_strategy=~"$strategies",'
        'training_seed=~"$training_seeds",controller_mode="conformal_energy_aware"'
    )
    definitions = [
        (130, "Controller Decision by Epoch (0=Continue, 1=Observe, 2=Stop)", "slurm_job_epoch_controller_decision_state", "short"),
        (131, "Raw Predicted Remaining Accuracy Gain by Epoch (pp)", "slurm_job_epoch_predicted_remaining_gain_upper", "percentpoint"),
        (132, "Conformal Upper Remaining Accuracy Gain by Epoch (pp)", "slurm_job_epoch_conformal_remaining_gain_upper", "percentpoint"),
        (133, "Predicted Remaining Training Energy by Epoch (Wh)", "slurm_job_epoch_predicted_remaining_energy_wh", "watth"),
        (134, "Upper Future Accuracy per Energy by Epoch (pp/Wh)", "slurm_job_epoch_predicted_future_efficiency_per_wh", "short"),
        (135, "Conformal Calibration Correction by Epoch (pp)", "slurm_job_epoch_conformal_correction", "percentpoint"),
    ]
    for index, (panel_id, title, metric, unit) in enumerate(definitions):
        panel = copy.deepcopy(base)
        panel["id"] = panel_id
        panel["title"] = title
        panel["gridPos"] = {"x": (index % 3) * 8, "y": 175 + (index // 3) * 9, "w": 8, "h": 9}
        panel["fieldConfig"]["defaults"]["unit"] = unit
        multiplier = " * 100" if panel_id in {131, 132, 134, 135} else ""
        panel["targets"] = [
            {
                "editorMode": "code",
                "refId": "A",
                "legendFormat": "Job {{job_id}} | {{scenario}} | seed {{training_seed}}",
                "exemplar": False,
                "instant": True,
                "format": "table",
                "datasource": {"type": "prometheus", "uid": "prometheus"},
                "range": False,
                "expr": f"({metric}{{{common_filter}}}{multiplier})",
            }
        ]
        dashboard["panels"].append(panel)

    dashboard["panels"].sort(key=lambda panel: (panel.get("gridPos", {}).get("y", 0), panel.get("gridPos", {}).get("x", 0)))
    DASHBOARD.write_text(json.dumps(dashboard, indent=2), encoding="utf-8", newline="\n")
    print(f"Configured {len(definitions)} conformal-controller panels in {DASHBOARD}")


if __name__ == "__main__":
    main()
