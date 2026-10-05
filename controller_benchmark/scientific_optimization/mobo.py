from __future__ import annotations

import math
import json
from typing import Any, Callable, Mapping, Sequence


def _evaluate_vector(
    vector: Sequence[float],
    decode: Callable[[Sequence[float]], Mapping[str, Any]],
    evaluator: Callable[[Mapping[str, Any]], Mapping[str, Any]],
    trial_number: int,
    backend: str,
    encode: Callable[[Mapping[str, Any]], Sequence[float]] | None = None,
) -> dict[str, Any]:
    parameters = dict(decode(vector))
    evaluation = dict(evaluator(parameters))
    return {
        "trial_number": trial_number,
        "backend": backend,
        "normalized_parameters": [
            float(value) for value in (encode(parameters) if encode else vector)
        ],
        "parameters": parameters,
        **evaluation,
    }


def run_sobol_search(
    *,
    dimension: int,
    trials: int,
    seed: int,
    decode: Callable[[Sequence[float]], Mapping[str, Any]],
    evaluator: Callable[[Mapping[str, Any]], Mapping[str, Any]],
    initial_vectors: Sequence[Sequence[float]] = (),
    encode: Callable[[Mapping[str, Any]], Sequence[float]] | None = None,
    progress: Callable[[str, Mapping[str, Any]], None] | None = None,
) -> list[dict[str, Any]]:
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("The Sobol search requires PyTorch.") from exc
    engine = torch.quasirandom.SobolEngine(dimension, scramble=True, seed=seed)
    vectors = [list(vector) for vector in initial_vectors[:trials]]
    vectors.extend(engine.draw(max(0, trials - len(vectors))).double().tolist())
    records = []
    parameter_keys: set[str] = set()
    vector_index = 0
    while len(records) < trials:
        if vector_index >= len(vectors):
            vectors.extend(engine.draw(1).double().tolist())
        vector = vectors[vector_index]
        vector_index += 1
        parameters = dict(decode(vector))
        key = json.dumps(
            parameters,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        if key in parameter_keys:
            continue
        parameter_keys.add(key)
        record = _evaluate_vector(
            vector, decode, evaluator, len(records), "sobol", encode=encode
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
    return records


def run_qnehvi_search(
    *,
    dimension: int,
    trials: int,
    initial_trials: int,
    seed: int,
    decode: Callable[[Sequence[float]], Mapping[str, Any]],
    evaluator: Callable[[Mapping[str, Any]], Mapping[str, Any]],
    initial_vectors: Sequence[Sequence[float]] = (),
    encode: Callable[[Mapping[str, Any]], Sequence[float]] | None = None,
    raw_samples: int = 128,
    restarts: int = 8,
    mc_samples: int = 128,
    progress: Callable[[str, Mapping[str, Any]], None] | None = None,
) -> list[dict[str, Any]]:
    """Run constrained qNEHVI on normalized mixed-parameter projections."""
    try:
        import torch
        from botorch.acquisition.multi_objective.logei import qLogNoisyExpectedHypervolumeImprovement
        from botorch.acquisition.multi_objective.objective import IdentityMCMultiOutputObjective
        from botorch.fit import fit_gpytorch_mll
        from botorch.models import ModelListGP, SingleTaskGP
        from botorch.models.transforms.input import Normalize
        from botorch.models.transforms.outcome import Standardize
        from botorch.optim import optimize_acqf
        from botorch.sampling.normal import SobolQMCNormalSampler
        from gpytorch.mlls.sum_marginal_log_likelihood import SumMarginalLogLikelihood
    except ImportError as exc:
        raise RuntimeError(
            "qNEHVI requires torch, botorch, and gpytorch. Install scientific-optimization requirements."
        ) from exc

    dtype = torch.double
    torch.manual_seed(seed)
    initial = min(max(2 * (dimension + 1), initial_trials, len(initial_vectors)), max(1, trials))
    engine = torch.quasirandom.SobolEngine(dimension, scramble=True, seed=seed)
    raw_vectors = [list(vector) for vector in initial_vectors[:initial]]
    raw_vectors.extend(engine.draw(max(0, initial - len(raw_vectors))).double().tolist())
    records = []
    parameter_keys: set[str] = set()
    vector_index = 0
    while len(records) < initial:
        if vector_index >= len(raw_vectors):
            raw_vectors.extend(engine.draw(1).double().tolist())
        vector = raw_vectors[vector_index]
        vector_index += 1
        parameters = dict(decode(vector))
        key = json.dumps(
            parameters,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        if key in parameter_keys:
            continue
        parameter_keys.add(key)
        record = _evaluate_vector(
            vector,
            decode,
            evaluator,
            len(records),
            "qnehvi-initial-sobol",
            encode=encode,
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

    train_x = torch.tensor(
        [record["normalized_parameters"] for record in records], dtype=dtype
    )

    def output_tensor() -> Any:
        return torch.tensor(
            [
                [
                    -float(record["metrics"]["quality_cvar90"]),
                    float(record["metrics"]["energy_saving_q25"]),
                    float(record["metrics"]["quality_constraint"]),
                ]
                for record in records
            ],
            dtype=dtype,
        )

    while len(records) < trials:
        if progress:
            progress(
                "bayesian_surrogate_fit_started",
                {"trial": len(records) + 1, "completed": len(records), "total": trials},
            )
        train_y = output_tensor()
        models = [
            SingleTaskGP(
                train_x,
                train_y[:, index : index + 1],
                input_transform=Normalize(d=dimension),
                outcome_transform=Standardize(m=1),
            )
            for index in range(3)
        ]
        model = ModelListGP(*models)
        mll = SumMarginalLogLikelihood(model.likelihood, model)
        try:
            fit_gpytorch_mll(mll)
            objective = IdentityMCMultiOutputObjective(outcomes=[0, 1])
            ranges = train_y[:, :2].max(dim=0).values - train_y[:, :2].min(dim=0).values
            reference = train_y[:, :2].min(dim=0).values - torch.clamp(ranges * 0.10, min=1e-6)
            acquisition = qLogNoisyExpectedHypervolumeImprovement(
                model=model,
                ref_point=reference.tolist(),
                X_baseline=train_x,
                sampler=SobolQMCNormalSampler(sample_shape=torch.Size([mc_samples]), seed=seed + len(records)),
                objective=objective,
                constraints=[lambda samples: samples[..., 2]],
                prune_baseline=True,
            )
            candidate, _ = optimize_acqf(
                acq_function=acquisition,
                bounds=torch.stack(
                    [torch.zeros(dimension, dtype=dtype), torch.ones(dimension, dtype=dtype)]
                ),
                q=1,
                num_restarts=restarts,
                raw_samples=raw_samples,
                options={"batch_limit": 4, "maxiter": 200},
                sequential=True,
            )
            vector = candidate.detach().reshape(-1)
            backend = "qnehvi"
        except Exception as exc:  # Keep the campaign reproducible when a GP fit is singular.
            vector = engine.draw(1).to(dtype=dtype).reshape(-1)
            backend = f"sobol-fallback:{type(exc).__name__}"
        parameters = dict(decode(vector.tolist()))
        key = json.dumps(
            parameters,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        duplicate_attempts = 0
        while key in parameter_keys:
            vector = engine.draw(1).to(dtype=dtype).reshape(-1)
            parameters = dict(decode(vector.tolist()))
            key = json.dumps(
                parameters,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            )
            duplicate_attempts += 1
            backend = "sobol-decoded-duplicate-fallback"
            if duplicate_attempts > 10_000:
                raise RuntimeError("Could not draw a unique decoded parameter configuration")
        parameter_keys.add(key)
        canonical_vector = (
            [float(value) for value in encode(parameters)]
            if encode
            else vector.tolist()
        )
        record = _evaluate_vector(
            canonical_vector,
            decode,
            evaluator,
            len(records),
            backend,
            encode=encode,
        )
        records.append(record)
        train_x = torch.cat(
            [train_x, torch.tensor(canonical_vector, dtype=dtype).reshape(1, -1)],
            dim=0,
        )
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
