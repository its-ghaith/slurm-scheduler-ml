from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any


ARCHIVE_RUN_ID = "20260816T142848Z-live-shadow-48-archive"


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def build(source_root: Path, report: Path, output_root: Path) -> Path:
    source_runs = sorted(
        path for path in source_root.iterdir() if path.is_dir() and (path / "status.json").is_file()
    )
    if len(source_runs) != 2:
        raise ValueError(f"Expected two archived 24-case runs, found {len(source_runs)}")

    destination = output_root / ARCHIVE_RUN_ID
    metrics = destination / "metrics"
    metrics.mkdir(parents=True, exist_ok=True)
    combined_rows: list[dict[str, Any]] = []
    source_run_ids: list[str] = []
    updated_at = ""

    for source_run in source_runs:
        status = _load(source_run / "status.json")
        source_run_ids.append(str(status["benchmark_run_id"]))
        updated_at = max(updated_at, str(status.get("updated_at") or ""))
        for row in status.get("runs") or []:
            combined = dict(row)
            combined["sequence"] = len(combined_rows) + 1
            combined_rows.append(combined)
        for pattern in ("epoch_summary_job_*.json", "shadow_summary_job_*.json"):
            for source_file in sorted((source_run / "metrics").glob(pattern)):
                shutil.copy2(source_file, metrics / source_file.name)

    if len(combined_rows) != 48:
        raise ValueError(f"Expected 48 combined status rows, found {len(combined_rows)}")
    epoch_files = list(metrics.glob("epoch_summary_job_*.json"))
    shadow_files = list(metrics.glob("shadow_summary_job_*.json"))
    if len(epoch_files) != 48 or len(shadow_files) != 48:
        raise ValueError(
            f"Expected 48 epoch and 48 shadow summaries, found {len(epoch_files)} and {len(shadow_files)}"
        )

    status = {
        "benchmark_run_id": ARCHIVE_RUN_ID,
        "controller_id": "live-shadow-suite",
        "state": "COMPLETED",
        "created_at": min(
            str(_load(path / "status.json").get("created_at") or "") for path in source_runs
        ),
        "updated_at": updated_at,
        "completed_runs": 48,
        "total_runs": 48,
        "current_job_id": None,
        "current_case_id": None,
        "error": None,
        "source_run_ids": source_run_ids,
        "runs": combined_rows,
    }
    (destination / "status.json").write_text(
        json.dumps(status, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    shutil.copy2(report, destination / "rapec-g-v5-48-offline-replay.txt")
    return destination


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    destination = build(args.source_root, args.report, args.output_root)
    print(destination)


if __name__ == "__main__":
    main()
