import argparse
import json
from pathlib import Path

from gpu_energy_utils import load_gpu_rows, summarize_rows


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--job-id", default="unknown")
    p.add_argument("--price-eur-kwh", type=float, default=0.30)
    p.add_argument("--co2-kg-kwh", type=float, default=0.4)
    p.add_argument("--pue", type=float, default=1.0)
    return p.parse_args()


def main():
    args = parse_args()
    in_path = Path(args.input)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    rows = load_gpu_rows(in_path) if in_path.exists() else []
    s = summarize_rows(rows)
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
