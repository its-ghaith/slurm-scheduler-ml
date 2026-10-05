from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping

from .replay import BaselineTrace, ReplayResult, load_baseline_traces, replay_controller
from .source_bundle import verify_source_lock


DEFAULT_PLUGIN = (
    "controller_benchmark.controllers.risk_aware_predictive_energy:"
    "RiskAwarePredictiveEnergyController"
)


def _load_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def _base_parameters(path: str | Path, controller_id: str = "rapec-v3") -> dict[str, Any]:
    document = _load_json(path)
    for controller in document["controllers"]:
        if controller["id"] == controller_id:
            return dict(controller["controller_parameters"])
    raise ValueError(f"Controller {controller_id!r} not found in {path}.")


def _suggest_parameters(trial: Any, base: Mapping[str, Any], space: Mapping[str, Any]) -> dict[str, Any]:
    params = dict(base)
    for name, spec in space["parameters"].items():
        kind = spec["type"]
        if kind == "int":
            params[name] = trial.suggest_int(name, int(spec["low"]), int(spec["high"]), step=int(spec.get("step", 1)))
        elif kind == "float":
            params[name] = trial.suggest_float(
                name,
                float(spec["low"]),
                float(spec["high"]),
                log=bool(spec.get("log", False)),
            )
        elif kind == "categorical":
            params[name] = trial.suggest_categorical(name, spec["choices"])
        elif kind == "fixed":
            params[name] = spec["value"]
        else:
            raise ValueError(f"Unsupported search-space type for {name}: {kind}")
    return params


def _evaluate(
    traces: Iterable[BaselineTrace],
    *,
    plugin: str,
    parameters: Mapping[str, Any],
    controller_id: str,
) -> tuple[dict[str, float], list[ReplayResult]]:
    results = [
        replay_controller(
            trace,
            controller_plugin=plugin,
            controller_id=controller_id,
            parameters=parameters,
        )
        for trace in traces
    ]
    regrets = [result.quality_regret for result in results]
    normalized_regrets = [result.normalized_quality_regret for result in results]
    savings = [result.energy_saving_fraction for result in results]
    violations = [
        result.quality_regret - result.dynamic_quality_tolerance
        for result in results
    ]
    return {
        "mean_quality_regret": statistics.fmean(regrets),
        "worst_normalized_quality_regret": max(normalized_regrets),
        "median_energy_saving_fraction": statistics.median(savings),
        "quality_constraint": max(violations),
        "mean_energy_saving_fraction": statistics.fmean(savings),
        "minimum_energy_saving_fraction": min(savings),
        "maximum_quality_regret": max(regrets),
    }, results


def _objective_factory(
    traces: list[BaselineTrace],
    plugin: str,
    base: Mapping[str, Any],
    space: Mapping[str, Any],
):
    def objective(trial: Any):
        parameters = _suggest_parameters(trial, base, space)
        metrics, _ = _evaluate(
            traces,
            plugin=plugin,
            parameters=parameters,
            controller_id=f"trial-{trial.number}",
        )
        for key, value in metrics.items():
            trial.set_user_attr(key, float(value))
        return (
            metrics["mean_quality_regret"],
            metrics["worst_normalized_quality_regret"],
            metrics["median_energy_saving_fraction"],
        )
    return objective


def _normalized(value: float, values: list[float], *, maximize: bool = False) -> float:
    low, high = min(values), max(values)
    if math.isclose(low, high):
        return 0.0
    result = (value - low) / (high - low)
    return 1.0 - result if maximize else result


def _pareto_trials(study: Any) -> list[Any]:
    complete = [trial for trial in study.trials if trial.values is not None]
    feasible = [
        trial
        for trial in complete
        if float(trial.user_attrs.get("quality_constraint", 1.0)) <= 0.0
    ]
    pool = feasible or complete

    def dominates(left: Any, right: Any) -> bool:
        left_values = (left.values[0], left.values[1], -left.values[2])
        right_values = (right.values[0], right.values[1], -right.values[2])
        return all(a <= b for a, b in zip(left_values, right_values)) and any(
            a < b for a, b in zip(left_values, right_values)
        )

    return [
        candidate
        for candidate in pool
        if not any(
            other.number != candidate.number and dominates(other, candidate)
            for other in pool
        )
    ]


def _select_profiles(study: Any, base: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    complete = _pareto_trials(study)
    if not complete:
        raise RuntimeError(f"Study {study.study_name} has no completed Pareto trials.")
    feasible = [trial for trial in complete if float(trial.user_attrs.get("quality_constraint", 1.0)) <= 0.0]
    pool = feasible or complete
    conservative = min(
        pool,
        key=lambda trial: (
            float(trial.user_attrs["worst_normalized_quality_regret"]),
            float(trial.user_attrs["mean_quality_regret"]),
            -float(trial.user_attrs["median_energy_saving_fraction"]),
        ),
    )
    aggressive = max(
        pool,
        key=lambda trial: (
            float(trial.user_attrs["median_energy_saving_fraction"]),
            -float(trial.user_attrs["worst_normalized_quality_regret"]),
        ),
    )
    mean_regrets = [float(trial.user_attrs["mean_quality_regret"]) for trial in pool]
    worst_regrets = [float(trial.user_attrs["worst_normalized_quality_regret"]) for trial in pool]
    savings = [float(trial.user_attrs["median_energy_saving_fraction"]) for trial in pool]

    def distance(trial: Any) -> float:
        components = (
            _normalized(float(trial.user_attrs["mean_quality_regret"]), mean_regrets),
            _normalized(float(trial.user_attrs["worst_normalized_quality_regret"]), worst_regrets),
            _normalized(float(trial.user_attrs["median_energy_saving_fraction"]), savings, maximize=True),
        )
        return math.sqrt(sum(value * value for value in components))

    balanced = min(pool, key=distance)
    selected = {
        "conservative": conservative,
        "balanced": balanced,
        "aggressive": aggressive,
    }
    return {
        profile: {
            "trial_number": trial.number,
            "parameters": {**base, **trial.params},
            "metrics": dict(trial.user_attrs),
            "quality_feasible": float(trial.user_attrs.get("quality_constraint", 1.0)) <= 0.0,
        }
        for profile, trial in selected.items()
    }


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    keys = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def _trial_rows(study: Any) -> list[dict[str, Any]]:
    rows = []
    pareto_numbers = {trial.number for trial in _pareto_trials(study)}
    for trial in study.trials:
        if trial.values is None:
            continue
        row = {
            "study": study.study_name,
            "trial_number": trial.number,
            "pareto_optimal": int(trial.number in pareto_numbers),
            "mean_quality_regret": trial.values[0],
            "worst_normalized_quality_regret": trial.values[1],
            "median_energy_saving_fraction": trial.values[2],
            **{f"param_{key}": value for key, value in trial.params.items()},
            **{f"metric_{key}": value for key, value in trial.user_attrs.items()},
        }
        rows.append(row)
    return rows


def _run_study(
    *,
    optuna: Any,
    storage: str,
    name: str,
    traces: list[BaselineTrace],
    plugin: str,
    base: Mapping[str, Any],
    space: Mapping[str, Any],
    trials: int,
    jobs: int,
    seed: int,
) -> Any:
    sampler = optuna.samplers.NSGAIISampler(
        population_size=int(space.get("population_size", 40)),
        seed=seed,
        constraints_func=lambda trial: (
            float(trial.user_attrs.get("quality_constraint", 0.0)),
        ),
    )
    study = optuna.create_study(
        study_name=name,
        storage=storage,
        load_if_exists=True,
        directions=["minimize", "minimize", "maximize"],
        sampler=sampler,
    )
    remaining = max(0, trials - len([trial for trial in study.trials if trial.values is not None]))
    if remaining:
        study.optimize(
            _objective_factory(traces, plugin, base, space),
            n_trials=remaining,
            n_jobs=jobs,
            gc_after_trial=True,
        )
    return study


def optimize(args: argparse.Namespace) -> Path:
    try:
        import optuna
    except ImportError as exc:
        raise RuntimeError(
            "Optuna is required. Run the PowerShell wrapper or install "
            "controller_benchmark/optimization/requirements.txt."
        ) from exc

    source = args.source_dir.resolve()
    source_lock = verify_source_lock(source)
    traces = load_baseline_traces(source)
    base = _base_parameters(args.base_controllers)
    space = _load_json(args.search_space)
    output = args.output_dir.resolve()
    if output.exists() and not args.resume:
        raise FileExistsError(f"Optimization output already exists: {output}")
    output.mkdir(parents=True, exist_ok=args.resume)
    storage = f"sqlite:///{(output / 'optuna-study.db').as_posix()}"
    tasks = sorted({trace.task_type for trace in traces})
    all_trial_rows: list[dict[str, Any]] = []
    fold_reports: dict[str, Any] = {}

    for fold_index, holdout_task in enumerate(tasks):
        training = [trace for trace in traces if trace.task_type != holdout_task]
        holdout = [trace for trace in traces if trace.task_type == holdout_task]
        study = _run_study(
            optuna=optuna,
            storage=storage,
            name=f"{args.optimization_id}-loto-{holdout_task}",
            traces=training,
            plugin=args.controller_plugin,
            base=base,
            space=space,
            trials=args.trials_per_study,
            jobs=args.jobs,
            seed=args.seed + fold_index,
        )
        profiles = _select_profiles(study, base)
        profile_reports = {}
        for profile, selected in profiles.items():
            metrics, case_results = _evaluate(
                holdout,
                plugin=args.controller_plugin,
                parameters=selected["parameters"],
                controller_id=f"loto-{holdout_task}-{profile}",
            )
            profile_reports[profile] = {
                "selected_on_training_tasks": selected,
                "holdout_metrics": metrics,
                "holdout_cases": [result.to_dict() for result in case_results],
            }
        fold_reports[holdout_task] = {
            "training_tasks": sorted({trace.task_type for trace in training}),
            "holdout_task": holdout_task,
            "profiles": profile_reports,
        }
        all_trial_rows.extend(_trial_rows(study))

    final_study = _run_study(
        optuna=optuna,
        storage=storage,
        name=f"{args.optimization_id}-final-all-tasks",
        traces=traces,
        plugin=args.controller_plugin,
        base=base,
        space=space,
        trials=args.trials_per_study,
        jobs=args.jobs,
        seed=args.seed + len(tasks),
    )
    final_profiles = _select_profiles(final_study, base)
    selected_document: dict[str, Any] = {
        "schema_version": 1,
        "optimization_id": args.optimization_id,
        "controller_plugin": args.controller_plugin,
        "source_run_id": source_lock["benchmark_run_id"],
        "profiles": {},
    }
    for profile, selected in final_profiles.items():
        metrics, case_results = _evaluate(
            traces,
            plugin=args.controller_plugin,
            parameters=selected["parameters"],
            controller_id=f"po-rapec-{profile}",
        )
        selected_document["profiles"][profile] = {
            "id": f"po-rapec-{profile}-{args.optimization_id[-12:]}",
            **selected,
            "all_task_metrics": metrics,
            "cases": [result.to_dict() for result in case_results],
        }
    all_trial_rows.extend(_trial_rows(final_study))

    manifest = {
        "schema_version": 1,
        "optimization_id": args.optimization_id,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "method": "NSGA-II constrained multi-objective optimization",
        "source_dir": str(source),
        "source_file_hashes": source_lock["files"],
        "source_run_id": source_lock["benchmark_run_id"],
        "controller_plugin": args.controller_plugin,
        "trials_per_study": args.trials_per_study,
        "total_planned_trials": args.trials_per_study * (len(tasks) + 1),
        "seed": args.seed,
        "objectives": [
            "minimize mean quality regret",
            "minimize worst normalized quality regret",
            "maximize median energy saving",
        ],
        "quality_constraint": "quality_regret <= dynamic noise-derived tolerance for every optimization case",
        "evaluation": "leave-one-task-out plus final all-task fit",
        "offline_replay_limitation": (
            "Counterfactual replay assumes the Full100 trajectory up to the selected stop epoch. "
            "Selected profiles require real Rancher validation."
        ),
    }
    (output / "optimization-manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    (output / "leave-one-task-out-report.json").write_text(
        json.dumps(fold_reports, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    (output / "selected-controllers.json").write_text(
        json.dumps(selected_document, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    _write_csv(output / "all-trials.csv", all_trial_rows)
    _write_csv(
        output / "pareto-front.csv",
        [row for row in all_trial_rows if row["pareto_optimal"]],
    )
    _write_csv(
        output / "selected-case-results.csv",
        [
            {"profile": profile, **case}
            for profile, specification in selected_document["profiles"].items()
            for case in specification["cases"]
        ],
    )
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--optimization-id", required=True)
    parser.add_argument("--trials-per-study", type=int, default=500)
    parser.add_argument("--jobs", type=int, default=1)
    parser.add_argument("--seed", type=int, default=20260731)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--controller-plugin", default=DEFAULT_PLUGIN)
    parser.add_argument(
        "--base-controllers",
        type=Path,
        default=Path("controller_benchmark/config/rapec-v3-v8-controllers.json"),
    )
    parser.add_argument(
        "--search-space",
        type=Path,
        default=Path("controller_benchmark/optimization/search-space.json"),
    )
    args = parser.parse_args()
    if args.trials_per_study < 1 or args.jobs < 1:
        parser.error("trials-per-study and jobs must be positive.")
    result = optimize(args)
    print(result)
    print(result / "selected-controllers.json")


if __name__ == "__main__":
    main()
