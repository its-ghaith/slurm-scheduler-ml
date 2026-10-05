from __future__ import annotations

import json
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping


class ProgressReporter:
    """Write human-readable progress and a machine-readable JSONL journal."""

    def __init__(self, output_dir: Path):
        self.output_dir = output_dir
        self.path = output_dir / "progress.jsonl"
        self.started = time.monotonic()
        self._lock = threading.Lock()

    def emit(self, event: str, details: Mapping[str, Any] | None = None) -> None:
        record = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "elapsed_seconds": round(time.monotonic() - self.started, 3),
            "event": event,
            **dict(details or {}),
        }
        with self._lock:
            with self.path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
                handle.flush()
            print(self._format(record), flush=True)

    @staticmethod
    def _format(record: Mapping[str, Any]) -> str:
        elapsed = int(float(record["elapsed_seconds"]))
        hours, remainder = divmod(elapsed, 3600)
        minutes, seconds = divmod(remainder, 60)
        prefix = f"[{hours:02d}:{minutes:02d}:{seconds:02d}] {record['event']}"
        preferred = (
            "search",
            "fold",
            "holdout_task",
            "completed",
            "total",
            "trial",
            "backend",
            "quality_cvar90",
            "energy_saving_q25",
            "quality_feasible",
        )
        values = [
            f"{key}={record[key]}"
            for key in preferred
            if key in record and record[key] is not None
        ]
        return prefix + (" | " + " | ".join(values) if values else "")
