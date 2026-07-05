from __future__ import annotations

import csv
from pathlib import Path


def to_float(value, default=0.0):
    try:
        return float(value)
    except Exception:
        return default


def load_gpu_rows(csv_path: Path) -> list[dict]:
    if not csv_path.exists():
        return []

    rows = []
    with csv_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(
                {
                    "ts": to_float(row.get("ts")),
                    "gpu_index": row.get("gpu_index", "0"),
                    "power_w": to_float(row.get("power_w")),
                    "util_gpu_pct": to_float(row.get("util_gpu_pct")),
                    "util_mem_pct": to_float(row.get("util_mem_pct")),
                    "mem_used_mb": to_float(row.get("mem_used_mb")),
                    "temp_c": to_float(row.get("temp_c")),
                }
            )
    rows.sort(key=lambda item: item["ts"])
    return rows


def filter_rows_by_time(rows: list[dict], start_ts: float | None = None, end_ts: float | None = None) -> list[dict]:
    return [
        row
        for row in rows
        if (start_ts is None or row["ts"] >= start_ts) and (end_ts is None or row["ts"] <= end_ts)
    ]


def summarize_rows(rows: list[dict]) -> dict:
    if not rows:
        return {
            "samples": 0,
            "gpu_power_avg_w": 0.0,
            "gpu_power_max_w": 0.0,
            "gpu_util_avg_pct": 0.0,
            "gpu_mem_used_avg_mb": 0.0,
            "gpu_temp_avg_c": 0.0,
            "gpu_energy_kwh_integrated": 0.0,
            "gpu_energy_kwh_estimated": 0.0,
            "gpu_energy_kwh": 0.0,
            "duration_seconds": 0.0,
        }

    powers = [to_float(r["power_w"]) for r in rows]
    utils = [to_float(r["util_gpu_pct"]) for r in rows]
    mems = [to_float(r["mem_used_mb"]) for r in rows]
    temps = [to_float(r["temp_c"]) for r in rows]
    ts = [to_float(r["ts"]) for r in rows]

    samples = len(rows)
    avg_power = sum(powers) / samples
    max_power = max(powers)
    avg_util = sum(utils) / samples
    avg_mem = sum(mems) / samples
    avg_temp = sum(temps) / samples

    integrated = 0.0
    for i in range(1, samples):
        dt = max(0.0, ts[i] - ts[i - 1])
        p_avg = (powers[i] + powers[i - 1]) / 2.0
        integrated += (p_avg * dt) / 3_600_000.0

    duration = max(0.0, ts[-1] - ts[0]) if samples > 1 else 0.0
    estimated = (avg_power * duration) / 3_600_000.0
    energy = max(integrated, estimated)

    return {
        "samples": samples,
        "gpu_power_avg_w": avg_power,
        "gpu_power_max_w": max_power,
        "gpu_util_avg_pct": avg_util,
        "gpu_mem_used_avg_mb": avg_mem,
        "gpu_temp_avg_c": avg_temp,
        "gpu_energy_kwh_integrated": integrated,
        "gpu_energy_kwh_estimated": estimated,
        "gpu_energy_kwh": energy,
        "duration_seconds": duration,
    }


def summarize_window(csv_path: Path, start_ts: float | None = None, end_ts: float | None = None) -> dict:
    rows = filter_rows_by_time(load_gpu_rows(csv_path), start_ts=start_ts, end_ts=end_ts)
    return summarize_rows(rows)
