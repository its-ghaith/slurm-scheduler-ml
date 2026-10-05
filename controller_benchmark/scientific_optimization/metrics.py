from __future__ import annotations

import hashlib
import math
import os
import random
import statistics
import threading
from atexit import register
from concurrent.futures import ProcessPoolExecutor
from concurrent.futures.process import BrokenProcessPool
from dataclasses import asdict, dataclass
from typing import TYPE_CHECKING, Any, Iterable, Mapping, Sequence

from controller_benchmark.optimization.replay import BaselineTrace, ReplayResult, replay_controller

if TYPE_CHECKING:
    from .replay_cache import PersistentReplayCache


_EXECUTOR_LOCK = threading.Lock()
_REPLAY_EXECUTORS: dict[tuple[int, int], ProcessPoolExecutor] = {}


def _replay_executor(workers: int) -> ProcessPoolExecutor:
    key = (os.getpid(), workers)
    with _EXECUTOR_LOCK:
        executor = _REPLAY_EXECUTORS.get(key)
        if executor is None:
            executor = ProcessPoolExecutor(max_workers=workers)
            _REPLAY_EXECUTORS[key] = executor
        return executor


def _drop_replay_executor(workers: int) -> None:
    with _EXECUTOR_LOCK:
        executor = _REPLAY_EXECUTORS.pop((os.getpid(), workers), None)
    if executor is not None:
        executor.shutdown(wait=True, cancel_futures=True)


def _shutdown_replay_executors() -> None:
    with _EXECUTOR_LOCK:
        executors = list(_REPLAY_EXECUTORS.values())
        _REPLAY_EXECUTORS.clear()
    for executor in executors:
        executor.shutdown(wait=True, cancel_futures=True)


register(_shutdown_replay_executors)


def _quantile(values: Sequence[float], probability: float) -> float:
    clean = sorted(float(value) for value in values if math.isfinite(float(value)))
    if not clean:
        return 0.0
    if len(clean) == 1:
        return clean[0]
    position = max(0.0, min(1.0, probability)) * (len(clean) - 1)
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return clean[lower]
    weight = position - lower
    return clean[lower] * (1.0 - weight) + clean[upper] * weight


def _mad(values: Iterable[float]) -> float:
    clean = [float(value) for value in values if math.isfinite(float(value))]
    if len(clean) < 2:
        return 0.0
    center = statistics.median(clean)
    return 1.4826 * statistics.median(abs(value - center) for value in clean)


def trace_group(trace: BaselineTrace) -> str:
    metadata = trace.metadata
    return str(
        metadata.get("dataset_fingerprint")
        or metadata.get("dataset_key")
        or metadata.get("dataset_name")
        or trace.case_id
    )


def calibrated_tolerances(traces: Sequence[BaselineTrace]) -> dict[str, float]:
    """Estimate a task-scale-free non-inferiority tolerance per dataset group."""
    from controller_benchmark.optimization.replay import dynamic_quality_tolerance

    grouped: dict[str, list[BaselineTrace]] = {}
    for trace in traces:
        grouped.setdefault(trace_group(trace), []).append(trace)
    tolerances: dict[str, float] = {}
    for group, replicas in grouped.items():
        within = statistics.median(
            dynamic_quality_tolerance(trace)["dynamic_quality_tolerance"]
            for trace in replicas
        )
        baseline_qualities = [
            max(float(row[trace.quality_metric]) for row in trace.epochs)
            for trace in replicas
        ]
        between = 1.96 * _mad(baseline_qualities)
        tolerances[group] = max(1e-6, math.sqrt(within**2 + between**2))
    return tolerances


def _bootstrap_noninferiority_probability(
    regrets: Sequence[float],
    tolerance: float,
    *,
    samples: int,
    seed: int,
) -> float:
    if not regrets:
        return 0.0
    if len(regrets) == 1:
        return float(regrets[0] <= tolerance)
    rng = random.Random(seed)
    successes = 0
    for _ in range(max(1, samples)):
        draw = [rng.choice(regrets) for _ in regrets]
        successes += statistics.fmean(draw) <= tolerance
    return successes / max(1, samples)


def _cvar(values: Sequence[float], alpha: float) -> float:
    if not values:
        return 0.0
    descending = sorted(values, reverse=True)
    count = max(1, int(math.ceil((1.0 - alpha) * len(descending))))
    return statistics.fmean(descending[:count])


@dataclass(frozen=True)
class ScientificMetrics:
    quality_cvar90: float
    energy_saving_q25: float
    quality_constraint: float
    minimum_noninferiority_probability: float
    mean_quality_regret: float
    maximum_quality_regret: float
    mean_energy_saving_fraction: float
    median_energy_saving_fraction: float
    stopped_fraction: float

    def to_dict(self) -> dict[str, float]:
        return asdict(self)


@dataclass(frozen=True)
class ScientificEvaluation:
    metrics: ScientificMetrics
    cases: tuple[Mapping[str, Any], ...]


def _replay_job(
    arguments: tuple[BaselineTrace, str, str, Mapping[str, Any]],
) -> ReplayResult:
    trace, controller_plugin, controller_id, parameters = arguments
    return replay_controller(
        trace,
        controller_plugin=controller_plugin,
        controller_id=controller_id,
        parameters=parameters,
    )


def _evaluate_replays(
    traces: Sequence[BaselineTrace],
    *,
    controller_plugin: str,
    controller_id: str,
    parameters: Mapping[str, Any],
    replay_cache: "PersistentReplayCache | None",
    replay_workers: int,
) -> list[ReplayResult]:
    results: list[ReplayResult | None] = [None] * len(traces)
    missing: list[tuple[int, BaselineTrace, str | None]] = []
    for index, trace in enumerate(traces):
        cache_key = None
        cached = None
        if replay_cache is not None:
            cache_key = replay_cache.key(
                trace,
                controller_plugin=controller_plugin,
                controller_id=controller_id,
                parameters=parameters,
            )
            cached = replay_cache.get(cache_key)
        if cached is None:
            missing.append((index, trace, cache_key))
        else:
            results[index] = cached

    jobs = [
        (trace, controller_plugin, controller_id, dict(parameters))
        for _, trace, _ in missing
    ]
    if jobs and replay_workers > 1:
        workers = min(replay_workers, len(traces))
        try:
            executor = _replay_executor(workers)
            evaluated = list(executor.map(_replay_job, jobs))
        except BrokenProcessPool:
            # A failed worker must not poison all subsequent optimizer trials.
            _drop_replay_executor(workers)
            executor = _replay_executor(workers)
            evaluated = list(executor.map(_replay_job, jobs))
    else:
        evaluated = [_replay_job(job) for job in jobs]

    for (index, _, cache_key), result in zip(missing, evaluated):
        results[index] = result
        if replay_cache is not None and cache_key is not None:
            replay_cache.put(cache_key, result)
    if any(result is None for result in results):
        raise RuntimeError("At least one controller replay did not produce a result")
    return [result for result in results if result is not None]


def evaluate_parameters(
    traces: Sequence[BaselineTrace],
    *,
    controller_plugin: str,
    controller_id: str,
    parameters: Mapping[str, Any],
    noninferiority_probability: float = 0.95,
    bootstrap_samples: int = 1024,
    replay_cache: "PersistentReplayCache | None" = None,
    replay_workers: int = 1,
) -> ScientificEvaluation:
    results = _evaluate_replays(
        traces,
        controller_plugin=controller_plugin,
        controller_id=controller_id,
        parameters=parameters,
        replay_cache=replay_cache,
        replay_workers=max(1, int(replay_workers)),
    )
    tolerances = calibrated_tolerances(traces)
    grouped: dict[str, list[tuple[BaselineTrace, ReplayResult]]] = {}
    cases: list[dict[str, Any]] = []
    normalized_regrets: list[float] = []
    for trace, result in zip(traces, results):
        group = trace_group(trace)
        grouped.setdefault(group, []).append((trace, result))
        tolerance = tolerances[group]
        normalized = result.quality_regret / tolerance
        normalized_regrets.append(normalized)
        case = result.to_dict()
        case.pop("ideal_stop_epoch", None)
        case.pop("normalized_stop_error", None)
        case.update(
            {
                "replicate_group": group,
                "training_seed": int(trace.metadata.get("training_seed", 0)),
                "calibrated_quality_tolerance": tolerance,
                "calibrated_normalized_quality_regret": normalized,
            }
        )
        cases.append(case)

    probabilities = []
    violations = []
    group_normalized_regrets = []
    group_savings = []
    for group, replicas in grouped.items():
        regrets = [result.quality_regret for _, result in replicas]
        group_normalized_regrets.append(statistics.fmean(regrets) / tolerances[group])
        group_savings.append(
            statistics.fmean(result.energy_saving_fraction for _, result in replicas)
        )
        seed = int.from_bytes(hashlib.sha256(group.encode("utf-8")).digest()[:4], "big")
        probability = _bootstrap_noninferiority_probability(
            regrets,
            tolerances[group],
            samples=bootstrap_samples,
            seed=seed,
        )
        probabilities.append(probability)
        violations.append(noninferiority_probability - probability)

    regrets = [result.quality_regret for result in results]
    savings = [result.energy_saving_fraction for result in results]
    metrics = ScientificMetrics(
        quality_cvar90=_cvar(group_normalized_regrets, 0.90),
        energy_saving_q25=_quantile(group_savings, 0.25),
        quality_constraint=max(violations, default=1.0),
        minimum_noninferiority_probability=min(probabilities, default=0.0),
        mean_quality_regret=statistics.fmean(regrets),
        maximum_quality_regret=max(regrets),
        mean_energy_saving_fraction=statistics.fmean(savings),
        median_energy_saving_fraction=statistics.median(savings),
        stopped_fraction=sum(result.stopped_early for result in results) / len(results),
    )
    return ScientificEvaluation(metrics=metrics, cases=tuple(cases))


def nondominated(records: Sequence[Mapping[str, Any]]) -> list[Mapping[str, Any]]:
    feasible = [record for record in records if record["metrics"]["quality_constraint"] <= 0.0]
    pool = feasible or list(records)

    def dominates(left: Mapping[str, Any], right: Mapping[str, Any]) -> bool:
        left_quality = left["metrics"]["quality_cvar90"]
        right_quality = right["metrics"]["quality_cvar90"]
        left_energy = left["metrics"]["energy_saving_q25"]
        right_energy = right["metrics"]["energy_saving_q25"]
        return (
            left_quality <= right_quality
            and left_energy >= right_energy
            and (left_quality < right_quality or left_energy > right_energy)
        )

    return [
        candidate
        for candidate in pool
        if not any(other is not candidate and dominates(other, candidate) for other in pool)
    ]


def select_nash_compromise(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    feasible_records = [
        record for record in records if record["metrics"]["quality_constraint"] <= 0.0
    ]
    front = nondominated(records)
    if not front:
        raise ValueError("No completed optimization records")
    qualities = [record["metrics"]["quality_cvar90"] for record in front]
    energies = [record["metrics"]["energy_saving_q25"] for record in front]
    q_ideal, q_nadir = min(qualities), max(qualities)
    e_nadir, e_ideal = min(energies), max(energies)

    def utilities(record: Mapping[str, Any]) -> tuple[float, float, float]:
        quality = record["metrics"]["quality_cvar90"]
        energy = record["metrics"]["energy_saving_q25"]
        quality_utility = 1.0 if math.isclose(q_ideal, q_nadir) else (q_nadir - quality) / (q_nadir - q_ideal)
        energy_utility = 0.0 if math.isclose(e_ideal, e_nadir) else (energy - e_nadir) / (e_ideal - e_nadir)
        return quality_utility, energy_utility, quality_utility * energy_utility

    ranked = []
    for record in front:
        quality_utility, energy_utility, product = utilities(record)
        ranked.append(
            {
                **record,
                "nash_quality_utility": quality_utility,
                "nash_energy_utility": energy_utility,
                "nash_product": product,
            }
        )
    selected = max(
        ranked,
        key=lambda record: (
            record["nash_product"],
            record["metrics"]["energy_saving_q25"],
            -record["metrics"]["quality_cvar90"],
        ),
    )
    return {
        **selected,
        "quality_feasible": bool(feasible_records),
        "selection_status": "feasible" if feasible_records else "no-feasible-trial-found",
    }


def select_energy_under_quality_constraint(
    records: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Select maximum robust energy saving after enforcing non-inferiority.

    Unlike Nash bargaining, quality is not traded against energy. It is a hard
    feasibility constraint; quality metrics only break ties between equally
    energy-efficient feasible candidates.
    """
    completed = list(records)
    if not completed:
        raise ValueError("No completed optimization records")
    feasible = [
        record
        for record in completed
        if record["metrics"]["quality_constraint"] <= 0.0
    ]
    pool = feasible or completed
    if feasible:
        selected = max(
            pool,
            key=lambda record: (
                record["metrics"]["energy_saving_q25"],
                record["metrics"]["mean_energy_saving_fraction"],
                -record["metrics"]["quality_cvar90"],
                -record["metrics"]["maximum_quality_regret"],
            ),
        )
    else:
        selected = min(
            pool,
            key=lambda record: (
                record["metrics"]["quality_constraint"],
                record["metrics"]["quality_cvar90"],
                -record["metrics"]["energy_saving_q25"],
            ),
        )
    return {
        **selected,
        "quality_feasible": bool(feasible),
        "selection_status": "feasible" if feasible else "no-feasible-trial-found",
        "constraint_first": True,
    }
