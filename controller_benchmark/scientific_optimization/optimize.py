from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

from controller_benchmark.optimization.replay import BaselineTrace, load_baseline_traces
from controller_benchmark.optimization.source_bundle import verify_source_lock

from .metrics import (
    evaluate_parameters,
    select_energy_under_quality_constraint,
    select_nash_compromise,
    trace_group,
)
from .mobo import run_qnehvi_search, run_sobol_search
from .oracle import build_oracle_report, write_oracle_report
from .parameters import ParameterCodec
from .progress import ProgressReporter
from .replay_cache import PersistentReplayCache
from .rf_parego import run_rf_parego_search
from .sensitivity import SensitivityResult, run_morris_screening


DEFAULT_PLUGIN = (
    "controller_benchmark.controllers.task_independent_rapec_v3:"
    "TaskIndependentRapecV3Controller"
)


def _read_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def _base_parameters(path: str | Path, controller_id: str = "rapec-v3") -> dict[str, Any]:
    document = _read_json(path)
    for controller in document["controllers"]:
        if controller["id"] == controller_id:
            parameters = dict(controller["controller_parameters"])
            if controller_id == "rapec-v3":
                parameters.update(
                    {
                        "warmup_signal_to_noise_ratio": 2.0,
                        "warmup_noise_stability_ratio": 2.5,
                        "warmup_required_stable_windows": 2,
                    }
                )
            return parameters
    raise ValueError(f"{controller_id} not found in {path}")


def _directed_extreme_parameters(
    base: Mapping[str, Any],
    search_space: Mapping[str, Any],
    direction: str,
) -> dict[str, Any]:
    """Build a conservative or aggressive initialization from declared ALE directions."""
    parameters = dict(base)
    for name, specification in search_space["parameters"].items():
        preferred = specification.get(direction)
        if preferred not in {"low", "high"}:
            continue
        if specification["type"] not in {"int", "float"}:
            continue
        parameters[name] = specification[preferred]

    # Preserve the historical RAPEC-v3 initialization when no directions are declared.
    if parameters == dict(base):
        legacy = {
            "safe": {
                "min_epochs_floor": "high",
                "patience": "high",
                "max_probability_gain_gt_threshold": "low",
            },
            "aggressive": {
                "min_epochs_floor": "low",
                "patience": "low",
                "max_probability_gain_gt_threshold": "high",
            },
        }
        for name, preferred in legacy[direction].items():
            specification = search_space["parameters"].get(name)
            if specification and preferred in specification:
                parameters[name] = specification[preferred]
    return parameters


def _write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        return
    keys = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def _load_sources(source_dirs: Sequence[Path]) -> tuple[list[BaselineTrace], list[dict[str, Any]]]:
    traces: list[BaselineTrace] = []
    locks = []
    seen_replicates: set[tuple[str, int, str]] = set()
    for source in source_dirs:
        lock = verify_source_lock(source)
        locks.append({"source_dir": str(source.resolve()), **lock})
        for trace in load_baseline_traces(source):
            seed = int(trace.metadata.get("training_seed", 0))
            key = (trace.case_id, seed, trace.source_job_id)
            if key not in seen_replicates:
                traces.append(trace)
                seen_replicates.add(key)
    if not traces:
        raise ValueError("No Full100 traces were loaded")
    return traces, locks


def _replication_audit(traces: Sequence[BaselineTrace]) -> dict[str, Any]:
    groups: dict[str, list[BaselineTrace]] = {}
    for trace in traces:
        groups.setdefault(trace_group(trace), []).append(trace)
    counts = {group: len(replicas) for group, replicas in sorted(groups.items())}
    return {
        "dataset_groups": len(groups),
        "traces": len(traces),
        "replicates_per_group": counts,
        "minimum_replicates": min(counts.values()),
        "recommended_minimum_replicates": 3,
        "repeated_seed_inference_available": min(counts.values()) >= 3,
    }


def _run_search(
    *,
    name: str,
    traces: Sequence[BaselineTrace],
    base: Mapping[str, Any],
    search_space: Mapping[str, Any],
    plugin: str,
    output_dir: Path,
    backend: str,
    trials: int,
    initial_trials: int,
    sensitivity_trajectories: int,
    sensitivity_top_k: int,
    seed: int,
    noninferiority_probability: float,
    bootstrap_samples: int,
    progress: ProgressReporter,
    controller_id_prefix: str,
    selection_method: str,
    run_sensitivity: bool,
    replay_cache: PersistentReplayCache,
    replay_workers: int,
    evaluation_controller_id: str,
    rf_candidate_pool: int,
    rf_trees: int,
    rf_min_samples_leaf: int,
) -> dict[str, Any]:
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(
            f"Incomplete search output is not empty and will not be overwritten: {output_dir}"
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    progress.emit(
        "search_started",
        {
            "search": name,
            "backend": backend,
            "traces": len(traces),
            "datasets": len({trace_group(trace) for trace in traces}),
            "task_types": sorted({trace.task_type for trace in traces}),
        },
    )
    full_codec = ParameterCodec.from_search_space(base, search_space)
    cache: dict[str, dict[str, Any]] = {}

    def report(event: str, details: Mapping[str, Any]) -> None:
        progress.emit(event, {"search": name, **details})

    def evaluate(parameters: Mapping[str, Any], *, include_cases: bool) -> dict[str, Any]:
        cache_key = json.dumps(parameters, sort_keys=True, separators=(",", ":"))
        if cache_key not in cache:
            result = evaluate_parameters(
                traces,
                controller_plugin=plugin,
                controller_id=evaluation_controller_id,
                parameters=parameters,
                noninferiority_probability=noninferiority_probability,
                bootstrap_samples=bootstrap_samples,
                replay_cache=replay_cache,
                replay_workers=replay_workers,
            )
            cache[cache_key] = {
                "metrics": result.metrics.to_dict(),
                "cases": list(result.cases),
            }
        value = cache[cache_key]
        return value if include_cases else value["metrics"]

    if run_sensitivity:
        sensitivity = run_morris_screening(
            full_codec,
            lambda parameters: evaluate(parameters, include_cases=False),
            trajectories=sensitivity_trajectories,
            top_k=sensitivity_top_k,
            seed=seed,
            progress=report,
        )
        mandatory = [
            parameter_name
            for parameter_name in search_space.get("mandatory_parameters", [])
            if parameter_name in full_codec.tunable_names
        ]
        active = list(mandatory)
        active.extend(
            parameter_name
            for parameter_name in sensitivity.selected_parameters
            if parameter_name not in active
        )
        active = active[: max(sensitivity_top_k, len(mandatory))]
        sensitivity = SensitivityResult(
            selected_parameters=tuple(active),
            rows=sensitivity.rows,
            evaluations=sensitivity.evaluations,
        )
    else:
        sensitivity = SensitivityResult(
            selected_parameters=full_codec.tunable_names,
            rows=(),
            evaluations=0,
        )
        report(
            "morris_screening_skipped",
            {
                "reason": "all declared controller parameters remain active",
                "active_parameters": list(full_codec.tunable_names),
            },
        )
    codec = ParameterCodec.from_search_space(
        base,
        search_space,
        active_names=sensitivity.selected_parameters,
    )
    progress.emit(
        "morris_screening_completed",
        {
            "search": name,
            "completed": sensitivity.evaluations,
            "total": sensitivity.evaluations,
            "active_parameters": list(codec.active_names),
        },
    )

    def search_evaluator(parameters: Mapping[str, Any]) -> dict[str, Any]:
        return evaluate(parameters, include_cases=True)

    safe_parameters = _directed_extreme_parameters(base, search_space, "safe")
    aggressive_parameters = _directed_extreme_parameters(base, search_space, "aggressive")
    initial_vectors = [
        codec.encode(base),
        codec.encode(safe_parameters),
        codec.encode(aggressive_parameters),
    ]

    search_arguments = {
        "dimension": codec.dimension,
        "trials": trials,
        "seed": seed + 10_000,
        "decode": codec.decode,
        "encode": codec.encode,
        "evaluator": search_evaluator,
        "initial_vectors": initial_vectors,
        "progress": report,
    }
    if backend == "qnehvi":
        records = run_qnehvi_search(
            **search_arguments,
            initial_trials=initial_trials,
        )
    elif backend == "rf-parego":
        records = run_rf_parego_search(
            **search_arguments,
            initial_trials=initial_trials,
            candidate_pool_size=rf_candidate_pool,
            trees=rf_trees,
            min_samples_leaf=rf_min_samples_leaf,
        )
    else:
        records = run_sobol_search(**search_arguments)
    if selection_method == "energy-under-quality-constraint":
        selected = select_energy_under_quality_constraint(records)
        selection_label = "Maximum energy Q25 under the hard quality constraint"
    else:
        selected = select_nash_compromise(records)
        selection_label = "Nash bargaining on the feasible Pareto front"
    selected = {
        **selected,
        "controller_plugin": plugin,
        # RAPEC-v9 includes this id in its posterior draw seed, so deployment
        # must retain the exact id used by offline parameter selection.
        "controller_id": evaluation_controller_id,
        "controller_display_id": f"{controller_id_prefix}-{name[-20:]}",
        "selection_method": selection_label,
    }
    selected_metrics = selected.get("metrics", {})
    progress.emit(
        "search_completed",
        {
            "search": name,
            "completed": len(records),
            "total": len(records),
            "trial": int(selected.get("trial_number", -1)) + 1,
            "quality_cvar90": selected_metrics.get("quality_cvar90"),
            "energy_saving_q25": selected_metrics.get("energy_saving_q25"),
            "quality_feasible": bool(selected.get("quality_feasible", False)),
        },
    )

    _write_json(output_dir / "sensitivity.json", sensitivity.to_dict())
    _write_csv(output_dir / "sensitivity.csv", sensitivity.rows)
    _write_json(output_dir / "trials.json", records)
    _write_csv(
        output_dir / "trials.csv",
        [
            {
                "trial_number": record["trial_number"],
                "backend": record["backend"],
                **record["metrics"],
                **{f"param_{key}": value for key, value in record["parameters"].items()},
            }
            for record in records
        ],
    )
    _write_json(output_dir / "selected-controller.json", selected)
    return {
        "name": name,
        "training_cases": sorted({trace.case_id for trace in traces}),
        "training_task_types": sorted({trace.task_type for trace in traces}),
        "active_parameters": list(codec.active_names),
        "selected": selected,
    }


def _load_completed_search(
    *, name: str, traces: Sequence[BaselineTrace], output_dir: Path
) -> dict[str, Any]:
    selected_path = output_dir / "selected-controller.json"
    if not selected_path.exists():
        raise FileNotFoundError(f"Completed search result not found: {selected_path}")
    sensitivity_path = output_dir / "sensitivity.json"
    sensitivity = _read_json(sensitivity_path) if sensitivity_path.exists() else {}
    return {
        "name": name,
        "training_cases": sorted({trace.case_id for trace in traces}),
        "training_task_types": sorted({trace.task_type for trace in traces}),
        "active_parameters": list(sensitivity.get("selected_parameters", [])),
        "selected": _read_json(selected_path),
    }


def optimize(args: argparse.Namespace) -> Path:
    source_dirs = [path.resolve() for path in args.source_dir]
    traces, locks = _load_sources(source_dirs)
    base = _base_parameters(args.base_controllers, args.base_controller_id)
    search_space = _read_json(args.search_space)
    output = args.output_dir.resolve()
    if output.exists() and not args.resume:
        raise FileExistsError(
            f"Scientific optimization output already exists and will not be overwritten: {output}"
        )
    output.mkdir(parents=True, exist_ok=args.resume)
    progress = ProgressReporter(output)
    replay_cache_root = (
        args.replay_cache_dir.resolve()
        if args.replay_cache_dir is not None
        else (output / "replay-cache")
    )
    replay_cache = PersistentReplayCache(replay_cache_root)
    oracle_report = build_oracle_report(traces)
    write_oracle_report(output, oracle_report)
    progress.emit(
        "optimization_resumed" if args.resume else "optimization_started",
        {
            "optimization_id": args.optimization_id,
            "backend": args.backend,
            "resume": bool(args.resume),
            "traces": len(traces),
            "datasets": len({trace_group(trace) for trace in traces}),
            "task_types": sorted({trace.task_type for trace in traces}),
        },
    )

    outer_reports: dict[str, Any] = {}
    task_types = sorted({trace.task_type for trace in traces})
    if not args.skip_outer_folds and len(task_types) > 1:
        for index, holdout_task in enumerate(task_types):
            training = [trace for trace in traces if trace.task_type != holdout_task]
            holdout = [trace for trace in traces if trace.task_type == holdout_task]
            progress.emit(
                "outer_fold_started",
                {
                    "fold": index + 1,
                    "total": len(task_types),
                    "holdout_task": holdout_task,
                },
            )
            fold_name = f"outer-holdout-{holdout_task}"
            fold_output = output / "outer-folds" / holdout_task
            if args.resume and (fold_output / "selected-controller.json").exists():
                fold = _load_completed_search(
                    name=fold_name,
                    traces=training,
                    output_dir=fold_output,
                )
                progress.emit(
                    "outer_fold_reused",
                    {
                        "fold": index + 1,
                        "completed": index + 1,
                        "total": len(task_types),
                        "holdout_task": holdout_task,
                    },
                )
            else:
                fold = _run_search(
                    name=fold_name,
                    traces=training,
                    base=base,
                    search_space=search_space,
                    plugin=args.controller_plugin,
                    output_dir=fold_output,
                    backend=args.backend,
                    trials=args.trials_per_fold,
                    initial_trials=args.initial_trials,
                    sensitivity_trajectories=args.sensitivity_trajectories,
                    sensitivity_top_k=args.sensitivity_top_k,
                    seed=args.seed + index,
                    noninferiority_probability=args.noninferiority_probability,
                    bootstrap_samples=args.bootstrap_samples,
                    progress=progress,
                    controller_id_prefix=args.controller_id_prefix,
                    selection_method=args.selection_method,
                    run_sensitivity=not args.skip_sensitivity,
                    replay_cache=replay_cache,
                    replay_workers=args.replay_workers,
                    evaluation_controller_id=args.evaluation_controller_id,
                    rf_candidate_pool=args.rf_candidate_pool,
                    rf_trees=args.rf_trees,
                    rf_min_samples_leaf=args.rf_min_samples_leaf,
                )
            selected = fold["selected"]
            holdout_evaluation = evaluate_parameters(
                holdout,
                controller_plugin=args.controller_plugin,
                controller_id=args.evaluation_controller_id,
                parameters=selected["parameters"],
                noninferiority_probability=args.noninferiority_probability,
                bootstrap_samples=args.bootstrap_samples,
                replay_cache=replay_cache,
                replay_workers=args.replay_workers,
            )
            outer_reports[holdout_task] = {
                "held_out_task": holdout_task,
                "held_out_cases": sorted({trace.case_id for trace in holdout}),
                "selected_on_other_tasks": selected,
                "holdout_metrics": holdout_evaluation.metrics.to_dict(),
                "holdout_case_results": list(holdout_evaluation.cases),
            }
            holdout_metrics = holdout_evaluation.metrics.to_dict()
            progress.emit(
                "outer_fold_completed",
                {
                    "fold": index + 1,
                    "completed": index + 1,
                    "total": len(task_types),
                    "holdout_task": holdout_task,
                    "quality_cvar90": holdout_metrics.get("quality_cvar90"),
                    "energy_saving_q25": holdout_metrics.get("energy_saving_q25"),
                    "quality_feasible": holdout_metrics.get("quality_constraint", 1.0) <= 0.0,
                },
            )
    _write_json(output / "outer-task-generalisation.json", outer_reports)

    final_output = output / "final-fit"
    if args.resume and (final_output / "selected-controller.json").exists():
        final = _load_completed_search(
            name=args.optimization_id,
            traces=traces,
            output_dir=final_output,
        )
        progress.emit(
            "final_search_reused",
            {"search": args.optimization_id},
        )
    else:
        final = _run_search(
            name=args.optimization_id,
            traces=traces,
            base=base,
            search_space=search_space,
            plugin=args.controller_plugin,
            output_dir=final_output,
            backend=args.backend,
            trials=args.trials,
            initial_trials=args.initial_trials,
            sensitivity_trajectories=args.sensitivity_trajectories,
            sensitivity_top_k=args.sensitivity_top_k,
            seed=args.seed + len(task_types),
            noninferiority_probability=args.noninferiority_probability,
            bootstrap_samples=args.bootstrap_samples,
            progress=progress,
            controller_id_prefix=args.controller_id_prefix,
            selection_method=args.selection_method,
            run_sensitivity=not args.skip_sensitivity,
            replay_cache=replay_cache,
            replay_workers=args.replay_workers,
            evaluation_controller_id=args.evaluation_controller_id,
            rf_candidate_pool=args.rf_candidate_pool,
            rf_trees=args.rf_trees,
            rf_min_samples_leaf=args.rf_min_samples_leaf,
        )
    selected = final["selected"]
    _write_json(output / "selected-controller.json", selected)
    manifest = {
        "schema_version": 2,
        "optimization_id": args.optimization_id,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "method": (
            ("Morris screening + " if not args.skip_sensitivity else "")
            + args.backend
            + " + "
            + (
                "constraint-first robust-energy selection"
                if args.selection_method == "energy-under-quality-constraint"
                else "Nash bargaining"
            )
        ),
        "backend": args.backend,
        "controller_plugin": args.controller_plugin,
        "base_controller_id": args.base_controller_id,
        "controller_id_prefix": args.controller_id_prefix,
        "evaluation_controller_id": args.evaluation_controller_id,
        "base_parameters": base,
        "search_space": search_space,
        "source_locks": locks,
        "source_data_were_modified": False,
        "replication_audit": _replication_audit(traces),
        "sensitivity_top_k": args.sensitivity_top_k,
        "sensitivity_trajectories": args.sensitivity_trajectories,
        "sensitivity_skipped": bool(args.skip_sensitivity),
        "replay_cache": replay_cache.statistics(),
        "replay_workers": args.replay_workers,
        "oracle_energy_saving_q25": oracle_report["oracle_energy_saving_q25"],
        "trials": args.trials,
        "trials_per_outer_fold": args.trials_per_fold,
        "objectives": {
            "quality": "minimize CVaR90 of dataset-normalized quality regret",
            "energy": "maximize lower-quartile relative training-energy saving",
        },
        "constraint": (
            f"bootstrap P(regret <= data-derived Full100 tolerance) >= "
            f"{args.noninferiority_probability:.3f} for every dataset group"
        ),
        "profile_selection": (
            "maximum energy-saving Q25 after hard non-inferiority filtering"
            if args.selection_method == "energy-under-quality-constraint"
            else "Nash product on the feasible Pareto front"
        ),
        "selected_quality_feasible": bool(selected["quality_feasible"]),
        "cross_validation": "outer leave-one-task-type-out; dataset groups remain intact",
        "offline_replay_limitation": (
            "Replay assumes the Full100 path up to the selected epoch. The selected controller "
            "must be confirmed on fresh Rancher seeds before scientific reporting."
        ),
        "ideal_stop_metric_used": False,
        "old_results_overwritten": False,
    }
    _write_json(output / "scientific-optimization-manifest.json", manifest)
    _write_json(output / "final-fit-summary.json", final)
    progress.emit(
        "optimization_completed",
        {
            "optimization_id": args.optimization_id,
            "quality_feasible": bool(selected["quality_feasible"]),
            "selected_trial": int(selected.get("trial_number", -1)) + 1,
            "output_dir": str(output),
            "cache_hits": replay_cache.hits,
            "cache_writes": replay_cache.writes,
        },
    )
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-dir", type=Path, action="append", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--optimization-id", required=True)
    parser.add_argument(
        "--base-controllers",
        type=Path,
        default=Path("controller_benchmark/config/rapec-v3-v8-controllers.json"),
    )
    parser.add_argument("--base-controller-id", default="rapec-v3")
    parser.add_argument("--controller-id-prefix", default="rapec-v3-ti-nash")
    parser.add_argument(
        "--search-space",
        type=Path,
        default=Path("controller_benchmark/scientific_optimization/search-space.json"),
    )
    parser.add_argument("--controller-plugin", default=DEFAULT_PLUGIN)
    parser.add_argument(
        "--selection-method",
        choices=("nash", "energy-under-quality-constraint"),
        default="nash",
    )
    parser.add_argument(
        "--backend", choices=("qnehvi", "sobol", "rf-parego"), default="qnehvi"
    )
    parser.add_argument("--trials", type=int, default=96)
    parser.add_argument("--trials-per-fold", type=int, default=64)
    parser.add_argument("--initial-trials", type=int, default=24)
    parser.add_argument("--sensitivity-trajectories", type=int, default=8)
    parser.add_argument("--sensitivity-top-k", type=int, default=8)
    parser.add_argument("--skip-sensitivity", action="store_true")
    parser.add_argument("--replay-cache-dir", type=Path)
    parser.add_argument("--replay-workers", type=int, default=1)
    parser.add_argument(
        "--evaluation-controller-id",
        default="rapec-scientific-fixed-evaluation",
        help="Fixed id used for every replay so controller-id-derived RNG seeds stay fair.",
    )
    parser.add_argument("--rf-candidate-pool", type=int, default=2048)
    parser.add_argument("--rf-trees", type=int, default=200)
    parser.add_argument("--rf-min-samples-leaf", type=int, default=2)
    parser.add_argument("--noninferiority-probability", type=float, default=0.95)
    parser.add_argument("--bootstrap-samples", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=20260801)
    parser.add_argument("--skip-outer-folds", action="store_true")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if args.trials < 2 or args.trials_per_fold < 2 or args.initial_trials < 1:
        parser.error("Trial counts must be positive and each search needs at least two trials.")
    if args.replay_workers < 1:
        parser.error("replay-workers must be at least one")
    if not 0.5 < args.noninferiority_probability < 1.0:
        parser.error("noninferiority-probability must be between 0.5 and 1.0")
    output_preexisted = args.output_dir.resolve().exists()
    try:
        result = optimize(args)
    except Exception as exc:
        output = args.output_dir.resolve()
        if (not output_preexisted or args.resume) and output.exists():
            ProgressReporter(output).emit(
                "optimization_failed",
                {
                    "optimization_id": args.optimization_id,
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                },
            )
        raise
    print(result)
    print(result / "selected-controller.json")


if __name__ == "__main__":
    main()
