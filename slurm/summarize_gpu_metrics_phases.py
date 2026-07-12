import argparse
import json
from pathlib import Path

from gpu_energy_utils import summarize_window, to_float


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--gpu-csv", required=True)
    p.add_argument("--timeline-jsonl", required=True)
    p.add_argument("--base-summary-json", required=True)
    p.add_argument("--output-json", required=True)
    p.add_argument("--job-codecarbon-json", required=True)
    p.add_argument("--price-eur-kwh", type=float, default=0.30)
    p.add_argument("--co2-kg-kwh", type=float, default=0.4)
    p.add_argument("--pue", type=float, default=1.0)
    return p.parse_args()


def load_timeline(path: Path):
    phases = {}
    if not path.exists():
        return phases
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            event = json.loads(line)
            name = event.get("phase")
            if not name:
                continue
            rec = phases.setdefault(name, {})
            if event.get("event") == "start":
                rec["start"] = to_float(event.get("ts"))
            elif event.get("event") == "end":
                rec["end"] = to_float(event.get("ts"))
                for k in (
                    "codecarbon_energy_kwh",
                    "codecarbon_gpu_energy_kwh",
                    "codecarbon_cpu_energy_kwh",
                    "duration_seconds",
                ):
                    if k in event:
                        rec[k] = to_float(event.get(k))
    return phases


def add_phase_rollups(phase: dict, price_eur_kwh: float, co2_kg_kwh: float):
    phase["estimated_electricity_cost_eur"] = to_float(phase.get("total_energy_kwh")) * price_eur_kwh
    phase["estimated_co2_kg"] = to_float(phase.get("total_energy_kwh")) * co2_kg_kwh
    phase["codecarbon_estimated_electricity_cost_eur"] = to_float(phase.get("codecarbon_energy_kwh")) * price_eur_kwh
    phase["codecarbon_estimated_co2_kg"] = to_float(phase.get("codecarbon_energy_kwh")) * co2_kg_kwh
    phase["codecarbon_duration_seconds"] = to_float(phase.get("duration_seconds"))
    return phase


def main():
    args = parse_args()
    gpu_csv = Path(args.gpu_csv)
    timeline = load_timeline(Path(args.timeline_jsonl))
    with open(args.base_summary_json, "r", encoding="utf-8") as f:
        base = json.load(f)
    with open(args.job_codecarbon_json, "r", encoding="utf-8") as f:
        codecarbon_job = json.load(f)

    phase_metrics = {}
    for phase, rec in timeline.items():
        if "start" not in rec or "end" not in rec or rec["end"] <= rec["start"]:
            continue
        gpu_kwh = summarize_window(gpu_csv, rec["start"], rec["end"]).get("gpu_energy_kwh", 0.0)
        total_kwh = gpu_kwh * args.pue
        phase_metrics[phase] = add_phase_rollups(
            {
                "duration_seconds": rec.get("duration_seconds", rec["end"] - rec["start"]),
                "gpu_energy_kwh": gpu_kwh,
                "total_energy_kwh": total_kwh,
                "codecarbon_energy_kwh": rec.get("codecarbon_energy_kwh", 0.0),
                "codecarbon_gpu_energy_kwh": rec.get("codecarbon_gpu_energy_kwh", 0.0),
                "codecarbon_cpu_energy_kwh": rec.get("codecarbon_cpu_energy_kwh", 0.0),
            },
            args.price_eur_kwh,
            args.co2_kg_kwh,
        )

    tracked_duration = sum(to_float(m.get("duration_seconds")) for m in phase_metrics.values())
    tracked_gpu = sum(to_float(m.get("gpu_energy_kwh")) for m in phase_metrics.values())
    tracked_total = sum(to_float(m.get("total_energy_kwh")) for m in phase_metrics.values())
    tracked_cc_duration = sum(to_float(m.get("codecarbon_duration_seconds")) for m in phase_metrics.values())
    tracked_cc_total = sum(to_float(m.get("codecarbon_energy_kwh")) for m in phase_metrics.values())
    tracked_cc_gpu = sum(to_float(m.get("codecarbon_gpu_energy_kwh")) for m in phase_metrics.values())
    tracked_cc_cpu = sum(to_float(m.get("codecarbon_cpu_energy_kwh")) for m in phase_metrics.values())

    other_phase = add_phase_rollups(
        {
            "duration_seconds": max(0.0, to_float(base.get("duration_seconds")) - tracked_duration),
            "gpu_energy_kwh": max(0.0, to_float(base.get("gpu_energy_kwh")) - tracked_gpu),
            "total_energy_kwh": max(0.0, to_float(base.get("training_energy_kwh")) - tracked_total),
            "codecarbon_duration_seconds": max(
                0.0, to_float(codecarbon_job.get("duration_seconds")) - tracked_cc_duration
            ),
            "codecarbon_energy_kwh": max(0.0, to_float(codecarbon_job.get("codecarbon_energy_kwh")) - tracked_cc_total),
            "codecarbon_gpu_energy_kwh": max(
                0.0, to_float(codecarbon_job.get("codecarbon_gpu_energy_kwh")) - tracked_cc_gpu
            ),
            "codecarbon_cpu_energy_kwh": max(
                0.0, to_float(codecarbon_job.get("codecarbon_cpu_energy_kwh")) - tracked_cc_cpu
            ),
        },
        args.price_eur_kwh,
        args.co2_kg_kwh,
    )
    other_phase["codecarbon_duration_seconds"] = max(
        0.0, to_float(codecarbon_job.get("duration_seconds")) - tracked_cc_duration
    )
    if any(to_float(v) > 0.0 for v in other_phase.values() if isinstance(v, (int, float))):
        phase_metrics["other"] = other_phase

    training = phase_metrics.get("training", {})
    cc_train_gpu = to_float(training.get("codecarbon_gpu_energy_kwh"))
    slurm_train_gpu = to_float(training.get("gpu_energy_kwh"))
    cc_train_total = to_float(training.get("codecarbon_energy_kwh"))
    slurm_train_total = to_float(training.get("total_energy_kwh"))
    compare = {
        "compare_basis": "gpu_to_gpu",
        "training_abs_diff_kwh": abs(slurm_train_gpu - cc_train_gpu),
        "training_rel_diff_pct": (abs(slurm_train_gpu - cc_train_gpu) / cc_train_gpu * 100.0) if cc_train_gpu > 0 else 0.0,
        "training_gpu_abs_diff_kwh": abs(slurm_train_gpu - cc_train_gpu),
        "training_gpu_rel_diff_pct": (abs(slurm_train_gpu - cc_train_gpu) / cc_train_gpu * 100.0) if cc_train_gpu > 0 else 0.0,
        "training_total_abs_diff_kwh": abs(slurm_train_total - cc_train_total),
        "training_total_rel_diff_pct": (abs(slurm_train_total - cc_train_total) / cc_train_total * 100.0) if cc_train_total > 0 else 0.0,
    }

    out = {
        **base,
        "codecarbon_job_total_duration_seconds": to_float(codecarbon_job.get("duration_seconds")),
        "codecarbon_job_total_energy_kwh": to_float(codecarbon_job.get("codecarbon_energy_kwh")),
        "codecarbon_job_total_gpu_energy_kwh": to_float(codecarbon_job.get("codecarbon_gpu_energy_kwh")),
        "codecarbon_job_total_cpu_energy_kwh": to_float(codecarbon_job.get("codecarbon_cpu_energy_kwh")),
        "codecarbon_estimated_electricity_cost_eur": to_float(codecarbon_job.get("codecarbon_energy_kwh"))
        * args.price_eur_kwh,
        "codecarbon_estimated_co2_kg": to_float(codecarbon_job.get("codecarbon_energy_kwh")) * args.co2_kg_kwh,
        "phase_metrics": phase_metrics,
        "codecarbon_vs_slurm": compare,
    }

    Path(args.output_json).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output_json, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)

    print(json.dumps(out))


if __name__ == "__main__":
    main()
