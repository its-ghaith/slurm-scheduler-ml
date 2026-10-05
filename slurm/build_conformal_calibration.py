from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

try:
    from slurm.epoch_energy_controller import _quantile, bootstrap_prediction_distribution_v2
except ImportError:  # pragma: no cover
    from epoch_energy_controller import _quantile, bootstrap_prediction_distribution_v2


def _conformal_quantile(values: list[float], alpha: float) -> float:
    if not values:
        return 0.0
    level = min(1.0, math.ceil((len(values) + 1) * (1.0 - alpha)) / len(values))
    return float(_quantile(values, level) or 0.0)


def _load_full_curves(
    input_dir: Path, *, excluded_training_seeds: set[int] | None = None
) -> list[tuple[Path, dict]]:
    excluded_training_seeds = excluded_training_seeds or set()
    curves = []
    for path in sorted(input_dir.rglob("epoch_summary_job_*.json")):
        payload = json.loads(path.read_text(encoding="utf-8-sig"))
        if payload.get("comparison_strategy") != "full100":
            continue
        training_seed = payload.get("training_seed")
        if training_seed is not None and int(training_seed) in excluded_training_seeds:
            continue
        epochs = payload.get("epochs", [])
        if len(epochs) < 100:
            continue
        curves.append((path, payload))
    return curves


def build_calibration(
    input_dir: Path,
    *,
    alpha: float,
    target_epoch: int,
    min_epoch: int,
    checkpoint_step: int,
    bin_width: int,
    bootstrap_samples: int,
    min_fit_points: int,
    regret_tolerance: float,
    excluded_training_seeds: set[int] | None = None,
) -> dict:
    excluded_training_seeds = excluded_training_seeds or set()
    curves = _load_full_curves(input_dir, excluded_training_seeds=excluded_training_seeds)
    if not curves:
        raise ValueError(f"No complete full100 curves found below {input_dir}")

    scores: dict[str, dict[int, list[float]]] = defaultdict(lambda: defaultdict(list))
    global_scores: dict[str, list[float]] = defaultdict(list)
    sources = []
    oracle_rows = []
    for path, payload in curves:
        scenario = str(payload.get("scenario", "unspecified"))
        rows = payload["epochs"][:target_epoch]
        epochs = [int(row.get("epoch_index", row.get("epoch"))) for row in rows]
        raw_values = [float(row.get("map50_95", 0.0)) for row in rows]
        best_values = []
        best = 0.0
        for value in raw_values:
            best = max(best, value)
            best_values.append(best)
        final_best = best_values[-1]
        oracle_epoch = next(
            (epoch for epoch, value in zip(epochs, best_values) if final_best - value <= regret_tolerance),
            target_epoch,
        )
        oracle_rows.append(
            {
                "scenario": scenario,
                "training_seed": payload.get("training_seed"),
                "job_id": payload.get("job_id"),
                "oracle_stop_epoch": oracle_epoch,
                "final_best_map50_95": final_best,
            }
        )
        sources.append(str(path.resolve()))

        for checkpoint in range(min_epoch, min(target_epoch, len(rows)), checkpoint_step):
            prefix_epochs = epochs[:checkpoint]
            prefix_values = best_values[:checkpoint]
            current_best = prefix_values[-1]
            predictions = bootstrap_prediction_distribution_v2(
                epochs=prefix_epochs,
                values=prefix_values,
                target_epoch=target_epoch,
                current_best=current_best,
                bootstrap_samples=bootstrap_samples,
                min_fit_points=min_fit_points,
            )
            if not predictions:
                continue
            pred_point = _quantile(predictions, 0.5)
            if pred_point is None:
                continue
            raw_gain_upper = max(0.0, pred_point - current_best)
            actual_gain = max(0.0, final_best - current_best)
            nonconformity = max(0.0, actual_gain - raw_gain_upper)
            bin_start = min_epoch + ((checkpoint - min_epoch) // bin_width) * bin_width
            scores[scenario][bin_start].append(nonconformity)
            global_scores[scenario].append(nonconformity)

    scenarios = {}
    for scenario, scenario_bins in sorted(scores.items()):
        bins = []
        for start, values in sorted(scenario_bins.items()):
            bins.append(
                {
                    "start_epoch": start,
                    "end_epoch": min(target_epoch - 1, start + bin_width - 1),
                    "correction": _conformal_quantile(values, alpha),
                    "samples": len(values),
                }
            )
        scenarios[scenario] = {
            "global_correction": _conformal_quantile(global_scores[scenario], alpha),
            "samples": len(global_scores[scenario]),
            "bins": bins,
        }

    return {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "method": "split-conformal upper remaining-gain calibration with moving-block residual bootstrap",
        "metric": "best_so_far_map50_95",
        "alpha": alpha,
        "target_epoch": target_epoch,
        "regret_tolerance": regret_tolerance,
        "calibration_curves": len(curves),
        "excluded_training_seeds": sorted(excluded_training_seeds),
        "sources": sources,
        "scenarios": scenarios,
        "oracle_definition": "first epoch t where best(T)-best(t) <= regret_tolerance",
        "oracle_rows": oracle_rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--target-epoch", type=int, default=100)
    parser.add_argument("--min-epoch", type=int, default=20)
    parser.add_argument("--checkpoint-step", type=int, default=5)
    parser.add_argument("--bin-width", type=int, default=20)
    parser.add_argument("--bootstrap-samples", type=int, default=32)
    parser.add_argument("--min-fit-points", type=int, default=12)
    parser.add_argument("--regret-tolerance", type=float, default=0.015)
    parser.add_argument("--exclude-training-seed", type=int, action="append", default=[])
    args = parser.parse_args()
    payload = build_calibration(
        args.input_dir,
        alpha=args.alpha,
        target_epoch=args.target_epoch,
        min_epoch=args.min_epoch,
        checkpoint_step=args.checkpoint_step,
        bin_width=args.bin_width,
        bootstrap_samples=args.bootstrap_samples,
        min_fit_points=args.min_fit_points,
        regret_tolerance=args.regret_tolerance,
        excluded_training_seeds=set(args.exclude_training_seed),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8", newline="\n")
    print(f"Wrote {args.output} from {payload['calibration_curves']} full100 curves")


if __name__ == "__main__":
    main()
