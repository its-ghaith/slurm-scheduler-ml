from __future__ import annotations

import argparse
import json
from pathlib import Path


def sorts_jobs_on_x_axis(panel: dict) -> bool:
    options = panel.get("options", {})
    if options.get("xField") in {"job_name", "job"}:
        return True
    return any(
        transformation.get("id") == "groupingToMatrix"
        and transformation.get("options", {}).get("rowField") == "job_name"
        for transformation in panel.get("transformations", [])
    )


def update_panel(panel: dict) -> None:
    if not sorts_jobs_on_x_axis(panel):
        return
    transformations = panel.setdefault("transformations", [])
    transformations[:] = [
        item
        for item in transformations
        if not (item.get("id") in {"convertFieldType", "sortBy"} and item.get("options", {}).get("numericJobSort"))
    ]
    x_field = panel.get("options", {}).get("xField")
    if x_field == "job":
        # seriesToRows produces the legend text (Job 1 ... Job 9). For this
        # comparison lexical and numeric order are identical.
        transformations.append(
            {
                "id": "sortBy",
                "options": {
                    "numericJobSort": True,
                    "fields": {},
                    "sort": [{"field": "job", "desc": False}],
                },
            }
        )
        return
    numeric_sort = [
        {
            "id": "convertFieldType",
            "options": {
                "numericJobSort": True,
                "conversions": [{"targetField": "job_id", "destinationType": "number"}],
            },
        },
        {
            "id": "sortBy",
            "options": {
                "numericJobSort": True,
                "fields": {},
                "sort": [{"field": "job_id", "desc": False}],
            },
        },
    ]
    grouping_index = next(
        (index for index, item in enumerate(transformations) if item.get("id") == "groupingToMatrix"),
        0,
    )
    transformations[grouping_index:grouping_index] = numeric_sort


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    dashboard = json.loads(args.source.read_text(encoding="utf-8-sig"))
    for panel in dashboard.get("panels", []):
        update_panel(panel)
    for variable in dashboard.get("templating", {}).get("list", []):
        if variable.get("name") in {"job_ids", "inference_job_id"}:
            variable["sort"] = 3  # Numerical ascending in Grafana.
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(dashboard, separators=(",", ":")), encoding="utf-8", newline="\n")


if __name__ == "__main__":
    main()
