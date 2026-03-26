import argparse
import csv
import json
from pathlib import Path


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--job-id", default="unknown")
    p.add_argument("--price-eur-kwh", type=float, default=0.30)
    p.add_argument("--co2-kg-kwh", type=float, default=0.4)
    p.add_argument("--pue", type=float, default=1.0)
    return p.parse_args()


def to_float(v, default=0.0):
    try:
        return float(v)
    except Exception:
        return default


def summarize(rows):
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


def main():
    args = parse_args()
    in_path = Path(args.input)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    rows = []
    if in_path.exists():
        with in_path.open("r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            rows = list(reader)

    s = summarize(rows)
    total_energy = s["gpu_energy_kwh"] * args.pue
    cost = total_energy * args.price_eur_kwh
    co2 = total_energy * args.co2_kg_kwh

    report = {
        "job_id": args.job_id,
        "samples": s["samples"],
        "duration_seconds": s["duration_seconds"],
        "gpu_power_avg_w": s["gpu_power_avg_w"],
        "gpu_power_max_w": s["gpu_power_max_w"],
        "gpu_util_avg_pct": s["gpu_util_avg_pct"],
        "gpu_mem_used_avg_mb": s["gpu_mem_used_avg_mb"],
        "gpu_temp_avg_c": s["gpu_temp_avg_c"],
        "gpu_energy_kwh_integrated": s["gpu_energy_kwh_integrated"],
        "gpu_energy_kwh_estimated": s["gpu_energy_kwh_estimated"],
        "gpu_energy_kwh": s["gpu_energy_kwh"],
        "training_energy_kwh": total_energy,
        "estimated_electricity_cost_eur": cost,
        "estimated_co2_kg": co2,
        "price_eur_kwh": args.price_eur_kwh,
        "co2_kg_kwh": args.co2_kg_kwh,
        "pue_factor": args.pue,
    }

    with out_path.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print(json.dumps(report))


if __name__ == "__main__":
    main()
