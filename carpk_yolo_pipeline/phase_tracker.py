import json
import os
import time
from contextlib import contextmanager
from pathlib import Path


class PhaseTracker:
    def __init__(self, timeline_file: str):
        self.timeline_path = Path(timeline_file)
        self.timeline_path.parent.mkdir(parents=True, exist_ok=True)

    def _append(self, payload: dict):
        with self.timeline_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(payload) + "\n")

    @contextmanager
    def phase(self, name: str, extra_end: dict | None = None):
        start = time.time()
        self._append({"phase": name, "event": "start", "ts": start})
        try:
            yield
        finally:
            end = time.time()
            payload = {
                "phase": name,
                "event": "end",
                "ts": end,
                "duration_seconds": end - start,
                # Optional CodeCarbon-compatible keys expected by Prom exporter pipeline.
                "codecarbon_energy_kwh": float(os.environ.get("CODECARBON_ENERGY_KWH", "0") or 0),
                "codecarbon_gpu_energy_kwh": float(os.environ.get("CODECARBON_GPU_ENERGY_KWH", "0") or 0),
                "codecarbon_cpu_energy_kwh": float(os.environ.get("CODECARBON_CPU_ENERGY_KWH", "0") or 0),
            }
            if extra_end:
                payload.update(extra_end)
            self._append(payload)

