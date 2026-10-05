from __future__ import annotations

import json
import math
from typing import Any, Callable, Mapping, Sequence

import numpy as np


def _parameter_key(parameters: Mapping[str, Any]) -> str:
    return json.dumps(
        parameters,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def _normal_pdf(value: np.ndarray) -> np.ndarray:
    return np.exp(-0.5 * value**2) / math.sqrt(2.0 * math.pi)


def _normal_cdf(value: np.ndarray) -> np.ndarray:
    return np.asarray(
        [0.5 * (1.0 + math.erf(float(item) / math.sqrt(2.0))) for item in value],
        dtype=float,
    )


def _expected_improvement(
    mean: np.ndarray,
    standard_deviation: np.ndarray,
    best: float,
) -> np.ndarray:
    improvement = best - mean
    result = np.maximum(0.0, improvement)
    positive = standard_deviation > 1e-12
    if np.any(positive):
        z = improvement[positive] / standard_deviation[positive]
        result[positive] = (
            improvement[positive] * _normal_cdf(z)
            + standard_deviation[positive] * _normal_pdf(z)
        )
    return result


def _forest_distribution(model: Any, values: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    predictions = np.vstack([tree.predict(values) for tree in model.estimators_])
    return predictions.mean(axis=0), predictions.std(axis=0), predictions


def _normalize_columns(values: np.ndarray) -> np.ndarray:
    lower = values.min(axis=0)
    span = values.max(axis=0) - lower
    span = np.where(span > 1e-12, span, 1.0)
    return (values - lower) / span


def _evaluate(
    vector: Sequence[float],
    *,
    decode: Callable[[Sequence[float]], Mapping[str, Any]],
    encode: Callable[[Mapping[str, Any]], Sequence[float]],
    evaluator: Callable[[Mapping[str, Any]], Mapping[str, Any]],
    trial_number: int,
    backend: str,
) -> dict[str, Any]:
    parameters = dict(decode(vector))
    canonical_vector = [float(value) for value in encode(parameters)]
    evaluation = dict(evaluator(parameters))
    return {
        "trial_number": trial_number,
        "backend": backend,
        "normalized_parameters": canonical_vector,
        "parameters": parameters,
        **evaluation,
    }


def run_rf_parego_search(
    *,
    dimension: int,
    trials: int,
    initial_trials: int,
    seed: int,
    decode: Callable[[Sequence[float]], Mapping[str, Any]],
    encode: Callable[[Mapping[str, Any]], Sequence[float]],
    evaluator: Callable[[Mapping[str, Any]], Mapping[str, Any]],
    initial_vectors: Sequence[Sequence[float]] = (),
    candidate_pool_size: int = 2048,
    trees: int = 200,
    min_samples_leaf: int = 2,
    random_design_probability: float = 0.10,
    augmented_tchebycheff_rho: float = 0.05,
    progress: Callable[[str, Mapping[str, Any]], None] | None = None,
) -> list[dict[str, Any]]:
    """Run constrained RF-SMBO with random ParEGO scalarizations.

    Integer and categorical parameters are represented by their canonical
    decoded coordinates before fitting. A separate random forest models the
    continuous quality-constraint violation. Before the first feasible point,
    candidate selection minimizes expected violation.
    """
    try:
        import torch
        from sklearn.ensemble import RandomForestRegressor
    except ImportError as exc:
        raise RuntimeError(
            "rf-parego requires PyTorch and scikit-learn. Install the scientific "
            "optimization requirements."
        ) from exc

    if trials < 1:
        return []
    rng = np.random.default_rng(seed)
    sobol = torch.quasirandom.SobolEngine(dimension, scramble=True, seed=seed)
    records: list[dict[str, Any]] = []
    evaluated_keys: set[str] = set()

    def canonical_candidate(vector: Sequence[float]) -> tuple[list[float], dict[str, Any], str]:
        parameters = dict(decode(vector))
        canonical = [float(value) for value in encode(parameters)]
        return canonical, parameters, _parameter_key(parameters)

    def next_unique_sobol() -> tuple[list[float], dict[str, Any], str]:
        for _ in range(100_000):
            raw = sobol.draw(1).double().reshape(-1).tolist()
            candidate = canonical_candidate(raw)
            if candidate[2] not in evaluated_keys:
                return candidate
        raise RuntimeError("Could not draw a new decoded parameter configuration")

    initial_target = min(
        trials,
        max(2 * (dimension + 1), initial_trials, len(initial_vectors)),
    )
    initial_candidates: list[tuple[list[float], dict[str, Any], str]] = []
    for vector in initial_vectors:
        candidate = canonical_candidate(vector)
        if candidate[2] not in {item[2] for item in initial_candidates}:
            initial_candidates.append(candidate)
        if len(initial_candidates) >= initial_target:
            break
    while len(initial_candidates) < initial_target:
        candidate = next_unique_sobol()
        if candidate[2] not in {item[2] for item in initial_candidates}:
            initial_candidates.append(candidate)

    for vector, _, key in initial_candidates:
        evaluated_keys.add(key)
        record = _evaluate(
            vector,
            decode=decode,
            encode=encode,
            evaluator=evaluator,
            trial_number=len(records),
            backend="rf-parego-initial-sobol",
        )
        records.append(record)
        if progress:
            progress(
                "optimization_trial_completed",
                {
                    "trial": len(records),
                    "completed": len(records),
                    "total": trials,
                    "backend": record["backend"],
                    "quality_cvar90": round(record["metrics"]["quality_cvar90"], 6),
                    "energy_saving_q25": round(record["metrics"]["energy_saving_q25"], 6),
                    "quality_feasible": record["metrics"]["quality_constraint"] <= 0.0,
                },
            )

    while len(records) < trials:
        if progress:
            progress(
                "rf_surrogate_fit_started",
                {"trial": len(records) + 1, "completed": len(records), "total": trials},
            )
        train_x = np.asarray(
            [record["normalized_parameters"] for record in records], dtype=float
        )
        objectives = np.asarray(
            [
                [
                    float(record["metrics"]["quality_cvar90"]),
                    -float(record["metrics"]["energy_saving_q25"]),
                ]
                for record in records
            ],
            dtype=float,
        )
        violations = np.asarray(
            [max(0.0, float(record["metrics"]["quality_constraint"])) for record in records],
            dtype=float,
        )
        feasible = violations <= 0.0
        weights = rng.dirichlet(np.ones(2, dtype=float))
        normalized = _normalize_columns(objectives)
        weighted = normalized * weights.reshape(1, -1)
        scalarized = weighted.max(axis=1) + augmented_tchebycheff_rho * weighted.sum(axis=1)

        scalar_model = RandomForestRegressor(
            n_estimators=max(32, trees),
            min_samples_leaf=max(1, min_samples_leaf),
            max_features=1.0,
            bootstrap=True,
            random_state=seed + len(records) * 17,
            n_jobs=1,
        )
        violation_model = RandomForestRegressor(
            n_estimators=max(32, trees),
            min_samples_leaf=max(1, min_samples_leaf),
            max_features=1.0,
            bootstrap=True,
            random_state=seed + len(records) * 17 + 1,
            n_jobs=1,
        )
        scalar_model.fit(train_x, scalarized)
        violation_model.fit(train_x, violations)

        candidates: list[tuple[list[float], dict[str, Any], str]] = []
        candidate_keys: set[str] = set()
        draw_target = max(128, candidate_pool_size)
        draw_count = max(draw_target, int(draw_target * 1.25))
        for raw in sobol.draw(draw_count).double().tolist():
            candidate = canonical_candidate(raw)
            if candidate[2] in evaluated_keys or candidate[2] in candidate_keys:
                continue
            candidate_keys.add(candidate[2])
            candidates.append(candidate)
            if len(candidates) >= draw_target:
                break

        # Local perturbations improve exploitation while preserving decoded integer levels.
        ranked = np.argsort(scalarized)[: min(8, len(records))]
        for index in ranked:
            center = train_x[index]
            for scale in (0.03, 0.08, 0.16):
                for _ in range(8):
                    raw = np.clip(center + rng.normal(0.0, scale, size=dimension), 0.0, 1.0)
                    candidate = canonical_candidate(raw.tolist())
                    if candidate[2] in evaluated_keys or candidate[2] in candidate_keys:
                        continue
                    candidate_keys.add(candidate[2])
                    candidates.append(candidate)
        if not candidates:
            candidates = [next_unique_sobol()]

        candidate_x = np.asarray([candidate[0] for candidate in candidates], dtype=float)
        scalar_mean, scalar_std, _ = _forest_distribution(scalar_model, candidate_x)
        violation_mean, violation_std, violation_trees = _forest_distribution(
            violation_model, candidate_x
        )
        tree_count = violation_trees.shape[0]
        feasibility_probability = (
            (violation_trees <= 1e-12).sum(axis=0) + 1.0
        ) / (tree_count + 2.0)

        random_design = rng.random() < max(0.0, min(1.0, random_design_probability))
        if random_design:
            selected_index = int(rng.integers(0, len(candidates)))
            backend = "rf-parego-random-design"
            acquisition_value = None
        elif np.any(feasible):
            best_scalar = float(np.min(scalarized[feasible]))
            expected_improvement = _expected_improvement(
                scalar_mean, scalar_std, best_scalar
            )
            acquisition = expected_improvement * feasibility_probability
            # Retain weak exploration when EI is flat on a plateau.
            acquisition += 1e-6 * feasibility_probability * scalar_std
            selected_index = int(np.argmax(acquisition))
            backend = "rf-parego-constrained-ei"
            acquisition_value = float(acquisition[selected_index])
        else:
            scale = max(float(np.max(violation_mean)), 1e-12)
            acquisition = -violation_mean / scale + violation_std / scale
            selected_index = int(np.argmax(acquisition))
            backend = "rf-parego-feasibility-first"
            acquisition_value = float(acquisition[selected_index])

        vector, _, key = candidates[selected_index]
        evaluated_keys.add(key)
        record = _evaluate(
            vector,
            decode=decode,
            encode=encode,
            evaluator=evaluator,
            trial_number=len(records),
            backend=backend,
        )
        record["rf_parego"] = {
            "weights": weights.tolist(),
            "candidate_pool": len(candidates),
            "predicted_feasibility_probability": float(
                feasibility_probability[selected_index]
            ),
            "predicted_constraint_violation": float(violation_mean[selected_index]),
            "acquisition_value": acquisition_value,
        }
        records.append(record)
        if progress:
            progress(
                "optimization_trial_completed",
                {
                    "trial": len(records),
                    "completed": len(records),
                    "total": trials,
                    "backend": backend,
                    "quality_cvar90": round(record["metrics"]["quality_cvar90"], 6),
                    "energy_saving_q25": round(record["metrics"]["energy_saving_q25"], 6),
                    "quality_feasible": record["metrics"]["quality_constraint"] <= 0.0,
                },
            )
    return records
