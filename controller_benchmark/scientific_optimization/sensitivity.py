from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Mapping, Sequence

from .parameters import ParameterCodec


@dataclass(frozen=True)
class SensitivityResult:
    selected_parameters: tuple[str, ...]
    rows: tuple[Mapping[str, Any], ...]
    evaluations: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "method": "Morris elementary-effects screening",
            "selected_parameters": list(self.selected_parameters),
            "evaluations": self.evaluations,
            "parameters": list(self.rows),
        }


def run_morris_screening(
    codec: ParameterCodec,
    evaluator: Callable[[Mapping[str, Any]], Mapping[str, float]],
    *,
    trajectories: int,
    top_k: int,
    seed: int,
    levels: int = 6,
    progress: Callable[[str, Mapping[str, Any]], None] | None = None,
) -> SensitivityResult:
    try:
        import numpy as np
        from SALib.analyze.morris import analyze
        from SALib.sample.morris import sample
    except ImportError as exc:
        raise RuntimeError(
            "Morris screening requires numpy and SALib. Install scientific-optimization requirements."
        ) from exc

    all_codec = ParameterCodec.from_search_space(codec.base, {"parameters": codec.specifications})
    names = list(all_codec.active_names)
    problem = {
        "num_vars": len(names),
        "names": names,
        "bounds": [[0.0, 1.0] for _ in names],
    }
    matrix = sample(
        problem,
        N=max(2, trajectories),
        num_levels=max(4, levels),
        optimal_trajectories=None,
        seed=seed,
    )
    quality_values = []
    energy_values = []
    for index, vector in enumerate(matrix, start=1):
        metrics = evaluator(all_codec.decode(vector.tolist()))
        quality_values.append(float(metrics["quality_cvar90"]))
        energy_values.append(float(metrics["energy_saving_q25"]))
        if progress:
            progress(
                "morris_evaluation_completed",
                {
                    "completed": index,
                    "total": len(matrix),
                    "quality_cvar90": round(float(metrics["quality_cvar90"]), 6),
                    "energy_saving_q25": round(float(metrics["energy_saving_q25"]), 6),
                },
            )

    quality = analyze(
        problem,
        matrix,
        np.asarray(quality_values),
        num_levels=max(4, levels),
        print_to_console=False,
        seed=seed,
    )
    energy = analyze(
        problem,
        matrix,
        np.asarray(energy_values),
        num_levels=max(4, levels),
        print_to_console=False,
        seed=seed + 1,
    )
    q_scale = max((float(value) for value in quality["mu_star"]), default=1.0) or 1.0
    e_scale = max((float(value) for value in energy["mu_star"]), default=1.0) or 1.0
    rows = []
    for index, name in enumerate(names):
        q_mu = float(quality["mu_star"][index])
        e_mu = float(energy["mu_star"][index])
        rows.append(
            {
                "parameter": name,
                "quality_mu_star": q_mu,
                "quality_sigma": float(quality["sigma"][index]),
                "energy_mu_star": e_mu,
                "energy_sigma": float(energy["sigma"][index]),
                "combined_score": max(q_mu / q_scale, e_mu / e_scale),
            }
        )
    rows.sort(key=lambda row: (row["combined_score"], row["quality_mu_star"], row["energy_mu_star"]), reverse=True)
    selected = tuple(row["parameter"] for row in rows[: max(1, min(top_k, len(rows)))])
    return SensitivityResult(selected, tuple(rows), len(matrix))
