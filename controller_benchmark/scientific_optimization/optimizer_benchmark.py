from __future__ import annotations

import argparse
import itertools
import json
import math
import statistics
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

from .metrics import evaluate_parameters, trace_group
from .optimize import (
    _base_parameters,
    _load_sources,
    _read_json,
    _replication_audit,
    _run_search,
)
from .oracle import build_oracle_report, write_oracle_report
from .progress import ProgressReporter
from .replay_cache import PersistentReplayCache


ALLOWED_BACKENDS = ("sobol", "qnehvi", "rf-parego")


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )


def _median(values: Sequence[float]) -> float | None:
    clean = [float(value) for value in values if math.isfinite(float(value))]
    return statistics.median(clean) if clean else None


def _hypervolume(points: Sequence[tuple[float, float]]) -> float:
    """Union area of origin-anchored rectangles in normalized utility space."""
    clean = sorted(
        {(max(0.0, x), max(0.0, y)) for x, y in points if x > 0.0 and y > 0.0}
    )
    if not clean:
        return 0.0
    area = 0.0
    previous_x = 0.0
    for x in sorted({point[0] for point in clean}):
        height = max((py for px, py in clean if px >= x), default=0.0)
        area += max(0.0, x - previous_x) * height
        previous_x = x
    return area


def _anytime_hypervolume(
    records: Sequence[Mapping[str, Any]],
    *,
    quality_min: float,
    quality_max: float,
    energy_min: float,
    energy_max: float,
) -> dict[str, float]:
    quality_span = max(quality_max - quality_min, 1e-12)
    energy_span = max(energy_max - energy_min, 1e-12)
    points: list[tuple[float, float]] = []
    values = []
    for record in records:
        metrics = record["metrics"]
        if float(metrics["quality_constraint"]) <= 0.0:
            points.append(
                (
                    (quality_max - float(metrics["quality_cvar90"])) / quality_span,
                    (float(metrics["energy_saving_q25"]) - energy_min) / energy_span,
                )
            )
        values.append(_hypervolume(points))
    return {
        "final_hypervolume": values[-1] if values else 0.0,
        "anytime_hypervolume_auc": statistics.fmean(values) if values else 0.0,
    }


def _run_summary(
    *,
    backend: str,
    optimizer_seed: int,
    scope: str,
    output_dir: Path,
    elapsed_seconds: float,
    selected: Mapping[str, Any],
) -> dict[str, Any]:
    trials = json.loads((output_dir / "trials.json").read_text(encoding="utf-8"))
    feasible = [
        row for row in trials if float(row["metrics"]["quality_constraint"]) <= 0.0
    ]
    return {
        "backend": backend,
        "optimizer_seed": optimizer_seed,
        "scope": scope,
        "trials": len(trials),
        "unique_decoded_configurations": len(
            {
                json.dumps(row["parameters"], sort_keys=True, separators=(",", ":"))
                for row in trials
            }
        ),
        "feasible_trials": len(feasible),
        "feasible_fraction": len(feasible) / len(trials) if trials else 0.0,
        "best_feasible_energy_saving_q25": max(
            (float(row["metrics"]["energy_saving_q25"]) for row in feasible),
            default=None,
        ),
        "selected_trial": int(selected.get("trial_number", -1)) + 1,
        "selected_quality_feasible": bool(selected.get("quality_feasible", False)),
        "selected_quality_cvar90": float(selected["metrics"]["quality_cvar90"]),
        "selected_energy_saving_q25": float(selected["metrics"]["energy_saving_q25"]),
        "elapsed_seconds": elapsed_seconds,
        "output_dir": str(output_dir),
        "_trials": trials,
    }


def _paired_statistics(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    try:
        from scipy.stats import wilcoxon
    except ImportError:
        return [{"available": False, "reason": "scipy is not installed"}]
    final_rows = [row for row in rows if row["scope"] == "final-fit"]
    by_backend = {
        backend: {int(row["optimizer_seed"]): row for row in final_rows if row["backend"] == backend}
        for backend in sorted({row["backend"] for row in final_rows})
    }
    comparisons = []
    for left, right in itertools.combinations(sorted(by_backend), 2):
        seeds = sorted(set(by_backend[left]) & set(by_backend[right]))
        if len(seeds) < 5:
            comparisons.append(
                {
                    "left": left,
                    "right": right,
                    "paired_seeds": len(seeds),
                    "test_performed": False,
                    "reason": "at least five paired optimizer seeds are required",
                }
            )
            continue
        infeasible_seeds = [
            seed
            for seed in seeds
            if not by_backend[left][seed]["selected_quality_feasible"]
            or not by_backend[right][seed]["selected_quality_feasible"]
        ]
        if infeasible_seeds:
            comparisons.append(
                {
                    "left": left,
                    "right": right,
                    "paired_seeds": len(seeds),
                    "test_performed": False,
                    "reason": (
                        "energy is not compared inferentially because at least one paired "
                        "selection violates the quality constraint"
                    ),
                    "infeasible_optimizer_seeds": infeasible_seeds,
                }
            )
            continue
        left_energy = [by_backend[left][seed]["selected_energy_saving_q25"] for seed in seeds]
        right_energy = [by_backend[right][seed]["selected_energy_saving_q25"] for seed in seeds]
        statistic, pvalue = wilcoxon(left_energy, right_energy, zero_method="zsplit")
        comparisons.append(
            {
                "left": left,
                "right": right,
                "paired_seeds": len(seeds),
                "test_performed": True,
                "metric": "selected_energy_saving_q25",
                "wilcoxon_signed_rank_statistic": float(statistic),
                "p_value_unadjusted": float(pvalue),
                "multiple_testing_note": "Apply Holm correction before confirmatory reporting.",
            }
        )
    return comparisons


def _aggregate(
    rows: list[dict[str, Any]], oracle: Mapping[str, Any]
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    # Hypervolume is comparable only within the same dataset/task scope. Global
    # normalization would let an intrinsically difficult fold distort every other fold.
    trials_by_scope: dict[str, list[Mapping[str, Any]]] = {}
    for row in rows:
        trials = row.pop("_trials")
        trials_by_scope.setdefault(str(row["scope"]), []).extend(trials)

    ranges_by_scope: dict[str, tuple[float, float, float, float]] = {}
    for scope, trials in trials_by_scope.items():
        quality_values = [float(trial["metrics"]["quality_cvar90"]) for trial in trials]
        energy_values = [float(trial["metrics"]["energy_saving_q25"]) for trial in trials]
        ranges_by_scope[scope] = (
            min(quality_values),
            max(quality_values),
            min(energy_values),
            max(energy_values),
        )

    for row in rows:
        trials = json.loads(
            (Path(row["output_dir"]) / "trials.json").read_text(encoding="utf-8")
        )
        quality_min, quality_max, energy_min, energy_max = ranges_by_scope[
            str(row["scope"])
        ]
        row.update(
            _anytime_hypervolume(
                trials,
                quality_min=quality_min,
                quality_max=quality_max,
                energy_min=energy_min,
                energy_max=energy_max,
            )
        )

    backends = sorted({row["backend"] for row in rows})
    summaries = []
    for backend in backends:
        final = [
            row for row in rows if row["backend"] == backend and row["scope"] == "final-fit"
        ]
        outer = [
            row for row in rows if row["backend"] == backend and row["scope"] != "final-fit"
        ]
        summaries.append(
            {
                "backend": backend,
                "optimizer_seeds": len(final),
                "final_feasible_rate": statistics.fmean(
                    float(row["selected_quality_feasible"]) for row in final
                ),
                "outer_holdout_feasible_rate": statistics.fmean(
                    float(row.get("holdout_quality_feasible", False)) for row in outer
                ) if outer else None,
                "median_final_energy_saving_q25": _median(
                    [row["selected_energy_saving_q25"] for row in final]
                ),
                "median_final_quality_cvar90": _median(
                    [row["selected_quality_cvar90"] for row in final]
                ),
                "median_final_hypervolume": _median(
                    [row["final_hypervolume"] for row in final]
                ),
                "median_anytime_hypervolume_auc": _median(
                    [row["anytime_hypervolume_auc"] for row in final]
                ),
                "median_elapsed_seconds": _median([row["elapsed_seconds"] for row in final]),
            }
        )

    winner = max(
        summaries,
        key=lambda row: (
            row["outer_holdout_feasible_rate"] or 0.0,
            row["final_feasible_rate"],
            (
                row["median_final_energy_saving_q25"]
                if row["median_final_energy_saving_q25"] is not None
                else -math.inf
            ),
            -(
                row["median_final_quality_cvar90"]
                if row["median_final_quality_cvar90"] is not None
                else math.inf
            ),
        ),
    )
    candidates = [
        row
        for row in rows
        if row["backend"] == winner["backend"]
        and row["scope"] == "final-fit"
        and row["selected_quality_feasible"]
    ]
    if not candidates:
        candidates = [
            row
            for row in rows
            if row["backend"] == winner["backend"] and row["scope"] == "final-fit"
        ]
    target_energy = _median([row["selected_energy_saving_q25"] for row in candidates]) or 0.0
    representative = min(
        candidates,
        key=lambda row: (
            abs(row["selected_energy_saving_q25"] - target_energy),
            row["selected_quality_cvar90"],
        ),
    )
    selected = json.loads(
        (Path(representative["output_dir"]) / "selected-controller.json").read_text(
            encoding="utf-8"
        )
    )
    selection = {
        **selected,
        "optimizer_benchmark_selection": {
            "winning_backend": winner["backend"],
            "selection_rule": (
                "method-level outer feasibility, then final feasibility, then median robust "
                "energy saving; representative feasible median-seed controller"
            ),
            "representative_optimizer_seed": representative["optimizer_seed"],
            "outer_holdout_feasible_rate": winner["outer_holdout_feasible_rate"],
            "all_outer_holdouts_feasible": bool(
                winner["outer_holdout_feasible_rate"] is not None
                and math.isclose(winner["outer_holdout_feasible_rate"], 1.0)
            ),
            "scientific_selection_ready": bool(
                representative["selected_quality_feasible"]
                and winner["outer_holdout_feasible_rate"] is not None
                and math.isclose(winner["outer_holdout_feasible_rate"], 1.0)
            ),
            "oracle_energy_saving_q25": oracle["oracle_energy_saving_q25"],
            "selected_energy_saving_q25": representative["selected_energy_saving_q25"],
            "oracle_to_selected_energy_gap": (
                float(oracle["oracle_energy_saving_q25"])
                - float(representative["selected_energy_saving_q25"])
            ),
        },
    }
    return summaries, selection


def run_benchmark(args: argparse.Namespace) -> Path:
    output = args.output_dir.resolve()
    if output.exists():
        raise FileExistsError(f"Optimizer benchmark output is protected: {output}")
    output.mkdir(parents=True)
    progress = ProgressReporter(output)
    traces, locks = _load_sources([path.resolve() for path in args.source_dir])
    base = _base_parameters(args.base_controllers, args.base_controller_id)
    search_space = _read_json(args.search_space)
    oracle = build_oracle_report(traces)
    write_oracle_report(output, oracle)
    plan = {
        "backends": args.backends,
        "optimizer_seeds": args.optimizer_seeds,
        "trials": args.trials,
        "trials_per_outer_fold": args.trials_per_fold,
        "initial_trials": args.initial_trials,
        "outer_folds": sorted({trace.task_type for trace in traces}),
        "training_datasets_per_outer_fold": {
            task: len({trace_group(trace) for trace in traces if trace.task_type != task})
            for task in sorted({trace.task_type for trace in traces})
        },
        "holdout_datasets_per_outer_fold": {
            task: len({trace_group(trace) for trace in traces if trace.task_type == task})
            for task in sorted({trace.task_type for trace in traces})
        },
        "morris_screening": "skipped; all seven RAPEC-v9 parameters remain active",
        "rancher_jobs_submitted": 0,
    }
    _write_json(output / "plan.json", plan)
    if args.plan_only:
        progress.emit("optimizer_benchmark_planned", plan)
        return output

    cache = PersistentReplayCache(args.replay_cache_dir)
    task_types = sorted({trace.task_type for trace in traces})
    rows: list[dict[str, Any]] = []
    for backend in args.backends:
        for seed_index, optimizer_seed in enumerate(args.optimizer_seeds):
            for fold_index, holdout_task in enumerate(task_types):
                training = [trace for trace in traces if trace.task_type != holdout_task]
                holdout = [trace for trace in traces if trace.task_type == holdout_task]
                scope = f"outer-holdout-{holdout_task}"
                run_output = output / "runs" / backend / f"seed-{optimizer_seed}" / scope
                started = time.perf_counter()
                result = _run_search(
                    name=f"{backend}-seed-{optimizer_seed}-{scope}",
                    traces=training,
                    base=base,
                    search_space=search_space,
                    plugin=args.controller_plugin,
                    output_dir=run_output,
                    backend=backend,
                    trials=args.trials_per_fold,
                    initial_trials=args.initial_trials,
                    sensitivity_trajectories=0,
                    sensitivity_top_k=len(search_space["mandatory_parameters"]),
                    seed=args.seed + seed_index * 1_000 + fold_index,
                    noninferiority_probability=args.noninferiority_probability,
                    bootstrap_samples=args.bootstrap_samples,
                    progress=progress,
                    controller_id_prefix=args.controller_id_prefix,
                    selection_method="energy-under-quality-constraint",
                    run_sensitivity=False,
                    replay_cache=cache,
                    replay_workers=args.replay_workers,
                    evaluation_controller_id=args.evaluation_controller_id,
                    rf_candidate_pool=args.rf_candidate_pool,
                    rf_trees=args.rf_trees,
                    rf_min_samples_leaf=args.rf_min_samples_leaf,
                )
                selected = result["selected"]
                holdout_evaluation = evaluate_parameters(
                    holdout,
                    controller_plugin=args.controller_plugin,
                    controller_id=args.evaluation_controller_id,
                    parameters=selected["parameters"],
                    noninferiority_probability=args.noninferiority_probability,
                    bootstrap_samples=args.bootstrap_samples,
                    replay_cache=cache,
                    replay_workers=args.replay_workers,
                )
                row = _run_summary(
                    backend=backend,
                    optimizer_seed=optimizer_seed,
                    scope=scope,
                    output_dir=run_output,
                    elapsed_seconds=time.perf_counter() - started,
                    selected=selected,
                )
                row.update(
                    {
                        "holdout_quality_feasible": holdout_evaluation.metrics.quality_constraint <= 0.0,
                        "holdout_quality_cvar90": holdout_evaluation.metrics.quality_cvar90,
                        "holdout_energy_saving_q25": holdout_evaluation.metrics.energy_saving_q25,
                    }
                )
                rows.append(row)

            run_output = output / "runs" / backend / f"seed-{optimizer_seed}" / "final-fit"
            started = time.perf_counter()
            result = _run_search(
                name=f"{backend}-seed-{optimizer_seed}-final-fit",
                traces=traces,
                base=base,
                search_space=search_space,
                plugin=args.controller_plugin,
                output_dir=run_output,
                backend=backend,
                trials=args.trials,
                initial_trials=args.initial_trials,
                sensitivity_trajectories=0,
                sensitivity_top_k=len(search_space["mandatory_parameters"]),
                seed=args.seed + seed_index * 1_000 + len(task_types),
                noninferiority_probability=args.noninferiority_probability,
                bootstrap_samples=args.bootstrap_samples,
                progress=progress,
                controller_id_prefix=args.controller_id_prefix,
                selection_method="energy-under-quality-constraint",
                run_sensitivity=False,
                replay_cache=cache,
                replay_workers=args.replay_workers,
                evaluation_controller_id=args.evaluation_controller_id,
                rf_candidate_pool=args.rf_candidate_pool,
                rf_trees=args.rf_trees,
                rf_min_samples_leaf=args.rf_min_samples_leaf,
            )
            rows.append(
                _run_summary(
                    backend=backend,
                    optimizer_seed=optimizer_seed,
                    scope="final-fit",
                    output_dir=run_output,
                    elapsed_seconds=time.perf_counter() - started,
                    selected=result["selected"],
                )
            )

    summaries, selection = _aggregate(rows, oracle)
    statistics_report = _paired_statistics(rows)
    _write_json(output / "run-results.json", rows)
    _write_json(output / "method-summary.json", summaries)
    _write_json(output / "paired-statistics.json", statistics_report)
    _write_json(output / "selected-controller.json", selection)
    manifest = {
        "schema_version": 1,
        "benchmark_id": args.benchmark_id,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "method": "fair Sobol vs qLogNEHVI vs constrained RF-ParEGO comparison",
        "plan": plan,
        "source_locks": locks,
        "replication_audit": _replication_audit(traces),
        "quality_constraint": (
            "data-derived dynamic tolerance per dataset group; no fixed percentage-point margin"
        ),
        "selection": "constraint-first; quality is never traded for energy",
        "outer_holdout_usage": (
            "method comparison only; final scientific claim requires fresh Rancher seeds"
        ),
        "generalisation_gate": {
            "all_outer_holdouts_must_be_feasible": True,
            "passed": selection["optimizer_benchmark_selection"][
                "all_outer_holdouts_feasible"
            ],
        },
        "fixed_evaluation_controller_id": args.evaluation_controller_id,
        "cache": cache.statistics(),
        "old_results_overwritten": False,
        "rancher_jobs_submitted": 0,
    }
    _write_json(output / "optimizer-benchmark-manifest.json", manifest)
    progress.emit(
        "optimizer_benchmark_completed",
        {
            "backend": selection["optimizer_benchmark_selection"]["winning_backend"],
            "quality_feasible": selection.get("quality_feasible"),
            "energy_saving_q25": selection["optimizer_benchmark_selection"][
                "selected_energy_saving_q25"
            ],
            "cache_hits": cache.hits,
            "cache_writes": cache.writes,
        },
    )
    return output


def _parse_csv(value: str, cast: Any) -> list[Any]:
    return [cast(item.strip()) for item in value.split(",") if item.strip()]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-dir", type=Path, action="append", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--benchmark-id", required=True)
    parser.add_argument(
        "--backends", default="sobol,qnehvi,rf-parego",
        help="Comma-separated subset of sobol,qnehvi,rf-parego",
    )
    parser.add_argument("--optimizer-seeds", default="0,1,2")
    parser.add_argument("--trials", type=int, default=100)
    parser.add_argument("--trials-per-fold", type=int, default=100)
    parser.add_argument("--initial-trials", type=int, default=16)
    parser.add_argument("--seed", type=int, default=20260804)
    parser.add_argument("--replay-workers", type=int, default=6)
    parser.add_argument("--replay-cache-dir", type=Path, required=True)
    parser.add_argument("--noninferiority-probability", type=float, default=0.95)
    parser.add_argument("--bootstrap-samples", type=int, default=1024)
    parser.add_argument("--rf-candidate-pool", type=int, default=2048)
    parser.add_argument("--rf-trees", type=int, default=200)
    parser.add_argument("--rf-min-samples-leaf", type=int, default=2)
    parser.add_argument("--plan-only", action="store_true")
    parser.add_argument(
        "--base-controllers",
        type=Path,
        default=Path("controller_benchmark/config/rapec-v9-controller.json"),
    )
    parser.add_argument("--base-controller-id", default="rapec-v9")
    parser.add_argument(
        "--search-space",
        type=Path,
        default=Path("controller_benchmark/scientific_optimization/search-space-rapec-v9.json"),
    )
    parser.add_argument(
        "--controller-plugin",
        default=(
            "controller_benchmark.controllers.risk_constrained_bayesian_multi_horizon:"
            "RiskConstrainedBayesianMultiHorizonController"
        ),
    )
    parser.add_argument("--controller-id-prefix", default="rapec-v9-optimizer-comparison")
    parser.add_argument(
        "--evaluation-controller-id", default="rapec-v9-optimizer-comparison-fixed"
    )
    args = parser.parse_args()
    args.backends = _parse_csv(args.backends, str)
    args.optimizer_seeds = _parse_csv(args.optimizer_seeds, int)
    unknown = sorted(set(args.backends) - set(ALLOWED_BACKENDS))
    if unknown:
        parser.error(f"Unsupported backends: {unknown}")
    if not args.backends or not args.optimizer_seeds:
        parser.error("At least one backend and optimizer seed are required")
    if min(args.trials, args.trials_per_fold, args.initial_trials, args.replay_workers) < 1:
        parser.error("Trial counts and replay workers must be positive")
    print(run_benchmark(args))


if __name__ == "__main__":
    main()
