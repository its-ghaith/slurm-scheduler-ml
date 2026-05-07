import argparse
import csv
import json
from pathlib import Path


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--gpu-csv", required=True)
    p.add_argument("--timeline-jsonl", required=True)
    p.add_argument("--base-summary-json", required=True)
    p.add_argument("--output-json", required=True)
    p.add_argument("--pue", type=float, default=1.0)
    return p.parse_args()


def to_float(v, default=0.0):
    try:
        return float(v)
    except Exception:
        return default


def load_timeline(path: Path):
    phases = {}
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


def phase_energy_from_csv(csv_path: Path, start_ts: float, end_ts: float):
    rows = []
    with csv_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for r in reader:
            ts = to_float(r.get("ts"))
            if ts >= start_ts and ts <= end_ts:
                rows.append((ts, to_float(r.get("power_w"))))
    if len(rows) < 2:
        return 0.0
    rows.sort(key=lambda x: x[0])
    e = 0.0
    for i in range(1, len(rows)):
        dt = max(0.0, rows[i][0] - rows[i - 1][0])
        p_avg = (rows[i][1] + rows[i - 1][1]) / 2.0
        e += (p_avg * dt) / 3_600_000.0
    return e


def main():
    args = parse_args()
    gpu_csv = Path(args.gpu_csv)
    timeline = load_timeline(Path(args.timeline_jsonl))
    with open(args.base_summary_json, "r", encoding="utf-8") as f:
        base = json.load(f)

    phase_metrics = {}
    for phase, rec in timeline.items():
        if "start" not in rec or "end" not in rec or rec["end"] <= rec["start"]:
            continue
        gpu_kwh = phase_energy_from_csv(gpu_csv, rec["start"], rec["end"])
        total_kwh = gpu_kwh * args.pue
        phase_metrics[phase] = {
            "duration_seconds": rec.get("duration_seconds", rec["end"] - rec["start"]),
            "gpu_energy_kwh": gpu_kwh,
            "total_energy_kwh": total_kwh,
            "codecarbon_energy_kwh": rec.get("codecarbon_energy_kwh", 0.0),
            "codecarbon_gpu_energy_kwh": rec.get("codecarbon_gpu_energy_kwh", 0.0),
            "codecarbon_cpu_energy_kwh": rec.get("codecarbon_cpu_energy_kwh", 0.0),
        }

    training = phase_metrics.get("training", {})
    cc_train = to_float(training.get("codecarbon_energy_kwh"))
    slurm_train = to_float(training.get("total_energy_kwh"))
    compare = {
        "training_abs_diff_kwh": abs(slurm_train - cc_train),
        "training_rel_diff_pct": (abs(slurm_train - cc_train) / cc_train * 100.0) if cc_train > 0 else 0.0,
    }

    out = {
        **base,
        "phase_metrics": phase_metrics,
        "codecarbon_vs_slurm": compare,
    }

    Path(args.output_json).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output_json, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)

    print(json.dumps(out))


if __name__ == "__main__":
    main()
