from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from statistics import mean, pstdev


FULL = "full100"
HYBRID = "hybrid_pareto_uncertainty_controller"


def load_job(metrics: Path, job_id: int) -> dict:
    summary = json.loads((metrics / f"epoch_summary_job_{job_id}.json").read_text(encoding="utf-8"))
    gpu = json.loads((metrics / f"gpu_summary_job_{job_id}.json").read_text(encoding="utf-8"))
    epochs = [
        json.loads(line)
        for line in (metrics / f"epoch_timeline_job_{job_id}.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    ended = [row for row in epochs if row.get("event") == "end"]
    return {
        "job_id": job_id,
        "training_seed": int(summary["training_seed"]),
        "strategy": summary["comparison_strategy"],
        "epochs": int(summary["epochs_completed"]),
        "adaptive_stopped": bool(summary.get("adaptive_stopped", False)),
        "final_map50_95": float(summary["final_map50_95"]),
        "best_map50_95": max(float(row.get("best_map50_95") or 0.0) for row in ended),
        "energy_wh": float(gpu["gpu_energy_kwh"]) * 1000.0,
        "duration_seconds": float(gpu["duration_seconds"]),
        "stop_reason": summary.get("stop_reason") or "",
    }


def safe_saving(reference: float, candidate: float) -> float:
    return (reference - candidate) / reference * 100.0 if reference else 0.0


def prometheus_line(metric: str, seed: int, value: float) -> str:
    return f'{metric}{{training_seed="{seed}"}} {value:.12g}'


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--regret-budget-pp", type=float, default=1.5)
    args = parser.parse_args()

    metrics = args.snapshot / "data" / "energy_metrics"
    manifest = json.loads((args.snapshot / "snapshot-manifest.json").read_text(encoding="utf-8-sig"))
    jobs = [load_job(metrics, job_id) for job_id in range(1, int(manifest["jobs"]) + 1)]
    grouped: dict[int, dict[str, dict]] = {}
    for job in jobs:
        grouped.setdefault(job["training_seed"], {})[job["strategy"]] = job

    rows = []
    for seed in sorted(grouped):
        pair = grouped[seed]
        if FULL not in pair or HYBRID not in pair:
            raise RuntimeError(f"Seed {seed} does not contain both required strategies")
        full, hybrid = pair[FULL], pair[HYBRID]
        best_regret_pp = (full["best_map50_95"] - hybrid["best_map50_95"]) * 100.0
        stop_trigger = "pareto_evidence" if hybrid["stop_reason"].startswith("hybrid_pareto_stop") else "budget_fallback"
        row = {
            "training_seed": seed,
            "full_job_id": full["job_id"],
            "hybrid_job_id": hybrid["job_id"],
            "full_epochs": full["epochs"],
            "hybrid_epochs": hybrid["epochs"],
            "epochs_saved": full["epochs"] - hybrid["epochs"],
            "full_best_map50_95": full["best_map50_95"],
            "hybrid_best_map50_95": hybrid["best_map50_95"],
            "best_quality_regret_pp": best_regret_pp,
            "final_quality_regret_pp": (full["final_map50_95"] - hybrid["final_map50_95"]) * 100.0,
            "full_energy_wh": full["energy_wh"],
            "hybrid_energy_wh": hybrid["energy_wh"],
            "energy_saving_pct": safe_saving(full["energy_wh"], hybrid["energy_wh"]),
            "full_duration_seconds": full["duration_seconds"],
            "hybrid_duration_seconds": hybrid["duration_seconds"],
            "duration_saving_pct": safe_saving(full["duration_seconds"], hybrid["duration_seconds"]),
            "regret_budget_satisfied": int(best_regret_pp <= args.regret_budget_pp),
            "adaptive_stopped": int(hybrid["adaptive_stopped"]),
            "evidence_based_stop": int(stop_trigger == "pareto_evidence"),
            "budget_fallback_stop": int(stop_trigger == "budget_fallback"),
            "positive_energy_saving": int(safe_saving(full["energy_wh"], hybrid["energy_wh"]) > 0.0),
            "stop_trigger": stop_trigger,
            "stop_reason": hybrid["stop_reason"],
        }
        rows.append(row)

    reports = args.snapshot / "reports"
    reports.mkdir(parents=True, exist_ok=True)
    with (reports / "stage-a-paired-analysis.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    aggregate = {
        "seeds": len(rows),
        "regret_budget_pp": args.regret_budget_pp,
        "adaptive_stop_rate_pct": mean(row["adaptive_stopped"] for row in rows) * 100.0,
        "regret_budget_success_rate_pct": mean(row["regret_budget_satisfied"] for row in rows) * 100.0,
        "evidence_based_stop_rate_pct": mean(row["evidence_based_stop"] for row in rows) * 100.0,
        "budget_fallback_stop_rate_pct": mean(row["budget_fallback_stop"] for row in rows) * 100.0,
        "positive_energy_saving_rate_pct": mean(row["positive_energy_saving"] for row in rows) * 100.0,
    }
    for field in ("epochs_saved", "best_quality_regret_pp", "final_quality_regret_pp", "energy_saving_pct", "duration_saving_pct"):
        values = [float(row[field]) for row in rows]
        aggregate[f"mean_{field}"] = mean(values)
        aggregate[f"std_{field}"] = pstdev(values)
        aggregate[f"min_{field}"] = min(values)
        aggregate[f"max_{field}"] = max(values)
    (reports / "stage-a-paired-analysis.json").write_text(
        json.dumps({"pairs": rows, "aggregate": aggregate}, indent=2), encoding="utf-8", newline="\n"
    )

    prom = []
    for row in rows:
        seed = row["training_seed"]
        for metric, field in (
            ("slurm_stage_a_best_quality_regret_pp", "best_quality_regret_pp"),
            ("slurm_stage_a_final_quality_regret_pp", "final_quality_regret_pp"),
            ("slurm_stage_a_energy_saving_pct", "energy_saving_pct"),
            ("slurm_stage_a_duration_saving_pct", "duration_saving_pct"),
            ("slurm_stage_a_epochs_saved", "epochs_saved"),
            ("slurm_stage_a_regret_budget_satisfied", "regret_budget_satisfied"),
            ("slurm_stage_a_adaptive_stopped", "adaptive_stopped"),
            ("slurm_stage_a_evidence_based_stop", "evidence_based_stop"),
            ("slurm_stage_a_budget_fallback_stop", "budget_fallback_stop"),
            ("slurm_stage_a_positive_energy_saving", "positive_energy_saving"),
        ):
            prom.append(prometheus_line(metric, seed, float(row[field])))
    for metric, field in (
        ("slurm_stage_a_mean_best_quality_regret_pp", "mean_best_quality_regret_pp"),
        ("slurm_stage_a_mean_energy_saving_pct", "mean_energy_saving_pct"),
        ("slurm_stage_a_mean_duration_saving_pct", "mean_duration_saving_pct"),
        ("slurm_stage_a_regret_budget_success_rate_pct", "regret_budget_success_rate_pct"),
        ("slurm_stage_a_adaptive_stop_rate_pct", "adaptive_stop_rate_pct"),
        ("slurm_stage_a_evidence_based_stop_rate_pct", "evidence_based_stop_rate_pct"),
        ("slurm_stage_a_budget_fallback_stop_rate_pct", "budget_fallback_stop_rate_pct"),
        ("slurm_stage_a_positive_energy_saving_rate_pct", "positive_energy_saving_rate_pct"),
        ("slurm_stage_a_mean_epochs_saved", "mean_epochs_saved"),
    ):
        prom.append(f"{metric} {aggregate[field]:.12g}")
    node_exporter = metrics / "node_exporter"
    node_exporter.mkdir(parents=True, exist_ok=True)
    (node_exporter / "stage_a_pair_metrics.prom").write_text("\n".join(prom) + "\n", encoding="utf-8", newline="\n")

    print(json.dumps(aggregate, indent=2))


if __name__ == "__main__":
    main()
