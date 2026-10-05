from __future__ import annotations

import argparse
import json
import time
import traceback
from pathlib import Path

from controller_benchmark.server.orchestrate import publish_partial_results


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--interval-seconds", type=int, default=15)
    args = parser.parse_args()

    run_dir = args.run_dir.resolve()
    workspace = run_dir / "workspace"
    metrics = run_dir / "metrics"
    plan = json.loads((run_dir / "run-matrix.json").read_text(encoding="utf-8"))
    last_published = -1
    while True:
        status_path = run_dir / "status.json"
        if not status_path.exists():
            time.sleep(args.interval_seconds)
            continue
        status = json.loads(status_path.read_text(encoding="utf-8"))
        completed = int(status.get("completed_runs") or 0)
        if completed > 0 and completed != last_published:
            try:
                publish_partial_results(plan, run_dir, workspace, metrics, completed)
            except Exception:
                traceback.print_exc()
            else:
                last_published = completed
        if status.get("state") in {"COMPLETED", "FAILED"}:
            return
        time.sleep(args.interval_seconds)


if __name__ == "__main__":
    main()
