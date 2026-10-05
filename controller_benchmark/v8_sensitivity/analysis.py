from __future__ import annotations

import argparse
import csv
import json
import math
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from controller_benchmark.optimization.replay import BaselineTrace, load_baseline_traces
from controller_benchmark.optimization.source_bundle import verify_source_lock
from controller_benchmark.scientific_optimization.parameters import ParameterCodec
from controller_benchmark.scientific_optimization.progress import ProgressReporter

from .evaluation import V8ReplayEvaluator


DEFAULT_PLUGIN = (
    "controller_benchmark.controllers.bayesian_guarded_rapec_v3:"
    "BayesianGuardedRapecV3Controller"
)
PRIMARY_METRICS = (
    "overall__mean_energy_saving_fraction",
    "overall__mean_quality_regret_pp",
    "overall__mean_stop_epoch",
    "overall__bayesian_veto_rate",
    "overall__controller_runtime_ms_per_epoch",
)


def _read_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    keys = list(rows[0])
    for row in rows[1:]:
        for key in row:
            if key not in keys:
                keys.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def _slug(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-")


def _base_parameters(path: str | Path) -> dict[str, Any]:
    document = _read_json(path)
    for controller in document["controllers"]:
        if controller["id"] == "rapec-v8":
            return dict(controller["controller_parameters"])
    raise ValueError(f"rapec-v8 not found in {path}")


def _load_sources(
    source_dirs: Sequence[Path],
) -> tuple[list[BaselineTrace], list[dict[str, Any]]]:
    traces: list[BaselineTrace] = []
    locks: list[dict[str, Any]] = []
    seen: set[tuple[str, int, str]] = set()
    for source in source_dirs:
        lock = verify_source_lock(source)
        locks.append({"source_dir": str(source.resolve()), **lock})
        for trace in load_baseline_traces(source):
            key = (
                trace.case_id,
                int(trace.metadata.get("training_seed", 0)),
                trace.source_job_id,
            )
            if key not in seen:
                traces.append(trace)
                seen.add(key)
    if not traces:
        raise ValueError("No Full100 traces were loaded")
    return traces, locks


def _is_power_of_two(value: int) -> bool:
    return value > 0 and value & (value - 1) == 0


def _problem(codec: ParameterCodec) -> dict[str, Any]:
    names = list(codec.active_names)
    return {
        "num_vars": len(names),
        "names": names,
        "bounds": [[0.0, 1.0] for _ in names],
    }


def _design_matrix(samples: int, dimension: int, seed: int):
    try:
        from scipy.stats import qmc
    except ImportError as exc:
        raise RuntimeError("The V8 sensitivity workflow requires scipy") from exc
    if not _is_power_of_two(samples):
        raise ValueError("DesignSamples must be a power of two")
    sampler = qmc.Sobol(d=dimension, scramble=True, seed=seed)
    return sampler.random_base2(int(math.log2(samples)))


def _sobol_matrix(problem: Mapping[str, Any], samples: int, seed: int):
    try:
        from SALib.sample.sobol import sample
    except ImportError as exc:
        raise RuntimeError("The V8 sensitivity workflow requires SALib") from exc
    if not _is_power_of_two(samples):
        raise ValueError("SobolBaseSamples must be a power of two")
    return sample(
        problem,
        samples,
        calc_second_order=True,
        scramble=True,
        seed=seed,
    )


def _load_design_records(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    records: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            records.append(json.loads(line))
    return records


def _append_design_record(path: Path, record: Mapping[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        handle.flush()


def evaluate_matrix(
    matrix,
    *,
    codec: ParameterCodec,
    evaluator: V8ReplayEvaluator,
    journal_path: Path,
    progress: ProgressReporter,
    resume: bool,
):
    import numpy as np

    records = _load_design_records(journal_path) if resume else []
    if len(records) > len(matrix):
        raise ValueError("Saved design journal is longer than the requested matrix")
    for index, saved in enumerate(records):
        expected = [float(value) for value in matrix[index]]
        actual = [float(value) for value in saved["normalized_parameters"]]
        if not np.allclose(expected, actual, atol=1e-12, rtol=0.0):
            raise ValueError(
                "Resume refused because the saved design does not match this seed and sample count"
            )

    outputs = [dict(record["outputs"]) for record in records]
    cache: dict[str, dict[str, float]] = {
        json.dumps(record["parameters"], sort_keys=True, separators=(",", ":")): dict(
            record["outputs"]
        )
        for record in records
    }
    total = len(matrix)
    report_every = max(1, total // 100)
    for index in range(len(records), total):
        vector = [float(value) for value in matrix[index]]
        parameters = codec.decode(vector)
        key = json.dumps(parameters, sort_keys=True, separators=(",", ":"))
        if key not in cache:
            cache[key] = evaluator.evaluate(parameters)
        output = cache[key]
        record = {
            "index": index,
            "normalized_parameters": vector,
            "parameters": parameters,
            "outputs": output,
        }
        _append_design_record(journal_path, record)
        records.append(record)
        outputs.append(output)
        completed = index + 1
        if completed == total or completed % report_every == 0:
            progress.emit(
                "design_evaluation_completed",
                {
                    "completed": completed,
                    "total": total,
                    "unique_parameter_vectors": len(cache),
                },
            )
    output_names = evaluator.output_names
    y = np.asarray(
        [[float(output[name]) for name in output_names] for output in outputs],
        dtype=float,
    )
    return np.asarray(matrix, dtype=float), y, records


def fit_surrogate(
    matrix,
    outputs,
    output_names: Sequence[str],
    *,
    trees: int,
    seed: int,
):
    import numpy as np

    try:
        from sklearn.ensemble import ExtraTreesRegressor
        from sklearn.metrics import mean_absolute_error, r2_score
        from sklearn.model_selection import train_test_split
    except ImportError as exc:
        raise RuntimeError("Surrogate mode requires scikit-learn") from exc

    indices = np.arange(len(matrix))
    train_indices, test_indices = train_test_split(
        indices,
        test_size=max(0.20, min(0.35, 128 / max(1, len(indices)))),
        random_state=seed,
    )
    model = ExtraTreesRegressor(
        n_estimators=max(64, trees),
        min_samples_leaf=2,
        max_features=1.0,
        random_state=seed,
        n_jobs=-1,
    )
    model.fit(matrix[train_indices], outputs[train_indices])
    predictions = model.predict(matrix[test_indices])
    validation: list[dict[str, Any]] = []
    for index, name in enumerate(output_names):
        truth = outputs[test_indices, index]
        predicted = predictions[:, index]
        span = float(np.max(truth) - np.min(truth))
        variance = float(np.var(truth))
        if variance <= 1e-15:
            r2 = None
            status = "constant-output"
        else:
            r2 = float(r2_score(truth, predicted))
            status = "reliable" if r2 >= 0.75 else "low-surrogate-fidelity"
        mae = float(mean_absolute_error(truth, predicted))
        validation.append(
            {
                "output": name,
                "r2": r2,
                "mae": mae,
                "normalized_mae": mae / span if span > 1e-15 else 0.0,
                "test_variance": variance,
                "status": status,
            }
        )
    model.fit(matrix, outputs)
    return model, validation


def calculate_sobol_indices(
    problem: Mapping[str, Any],
    matrix,
    outputs,
    output_names: Sequence[str],
    *,
    bootstrap_resamples: int,
    seed: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    import numpy as np

    try:
        from SALib.analyze.sobol import analyze
    except ImportError as exc:
        raise RuntimeError("The V8 sensitivity workflow requires SALib") from exc

    parameter_rows: list[dict[str, Any]] = []
    interaction_rows: list[dict[str, Any]] = []
    names = list(problem["names"])
    for output_index, output_name in enumerate(output_names):
        values = np.asarray(outputs[:, output_index], dtype=float)
        if float(np.var(values)) <= 1e-15:
            parameter_rows.extend(
                {
                    "output": output_name,
                    "parameter": name,
                    "s1": 0.0,
                    "s1_conf": 0.0,
                    "st": 0.0,
                    "st_conf": 0.0,
                    "interaction_gap": 0.0,
                    "status": "constant-output",
                }
                for name in names
            )
            continue
        indices = analyze(
            problem,
            values,
            calc_second_order=True,
            num_resamples=max(32, bootstrap_resamples),
            conf_level=0.95,
            print_to_console=False,
            seed=seed + output_index,
        )
        for parameter_index, name in enumerate(names):
            s1 = float(indices["S1"][parameter_index])
            st = float(indices["ST"][parameter_index])
            parameter_rows.append(
                {
                    "output": output_name,
                    "parameter": name,
                    "s1": s1,
                    "s1_conf": float(indices["S1_conf"][parameter_index]),
                    "st": st,
                    "st_conf": float(indices["ST_conf"][parameter_index]),
                    "interaction_gap": max(0.0, st - s1),
                    "status": "estimated",
                }
            )
        for left in range(len(names)):
            for right in range(left + 1, len(names)):
                interaction_rows.append(
                    {
                        "output": output_name,
                        "parameter_a": names[left],
                        "parameter_b": names[right],
                        "s2": float(indices["S2"][left, right]),
                        "s2_conf": float(indices["S2_conf"][left, right]),
                    }
                )
    return parameter_rows, interaction_rows


def build_parameter_ranking(
    rows: Sequence[Mapping[str, Any]],
    parameter_names: Sequence[str],
) -> list[dict[str, Any]]:
    lookup = {
        (str(row["output"]), str(row["parameter"])): row for row in rows
    }
    ranking: list[dict[str, Any]] = []
    for parameter in parameter_names:
        record: dict[str, Any] = {"parameter": parameter}
        total_orders: list[float] = []
        for output in PRIMARY_METRICS:
            row = lookup.get((output, parameter), {})
            key = _slug(output.replace("overall__", ""))
            value = float(row.get("st", 0.0))
            record[f"st__{key}"] = value
            record[f"st_conf__{key}"] = float(row.get("st_conf", 0.0))
            total_orders.append(max(0.0, value))
        record["combined_max_total_order"] = max(total_orders, default=0.0)
        record["combined_mean_total_order"] = (
            sum(total_orders) / len(total_orders) if total_orders else 0.0
        )
        ranking.append(record)
    ranking.sort(
        key=lambda row: (
            row["combined_max_total_order"],
            row["combined_mean_total_order"],
        ),
        reverse=True,
    )
    for index, row in enumerate(ranking, start=1):
        row["rank"] = index
    return ranking


def build_output_rankings(
    rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    grouped: dict[str, list[Mapping[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(str(row["output"]), []).append(row)
    ranked: list[dict[str, Any]] = []
    for output, values in sorted(grouped.items()):
        ordered = sorted(values, key=lambda row: float(row["st"]), reverse=True)
        for rank, row in enumerate(ordered, start=1):
            ranked.append({"rank": rank, **dict(row)})
    return ranked


def first_order_ale(
    model,
    reference,
    *,
    feature_index: int,
    output_index: int,
    bins: int,
):
    import numpy as np

    edges = np.linspace(0.0, 1.0, max(3, bins) + 1)
    bin_index = np.clip(
        np.searchsorted(edges, reference[:, feature_index], side="right") - 1,
        0,
        len(edges) - 2,
    )
    local = np.zeros(len(edges) - 1, dtype=float)
    counts = np.zeros(len(edges) - 1, dtype=int)
    for index in range(len(local)):
        selected = reference[bin_index == index]
        counts[index] = len(selected)
        if not len(selected):
            continue
        low = selected.copy()
        high = selected.copy()
        low[:, feature_index] = edges[index]
        high[:, feature_index] = edges[index + 1]
        local[index] = float(
            np.mean(model.predict(high)[:, output_index] - model.predict(low)[:, output_index])
        )
    cumulative_edges = np.concatenate(([0.0], np.cumsum(local)))
    effects = 0.5 * (cumulative_edges[:-1] + cumulative_edges[1:])
    if counts.sum():
        effects -= float(np.average(effects, weights=counts))
    centers = 0.5 * (edges[:-1] + edges[1:])
    return centers, effects, counts


def second_order_ale(
    model,
    reference,
    *,
    feature_a: int,
    feature_b: int,
    output_index: int,
    bins: int,
):
    import numpy as np

    edges = np.linspace(0.0, 1.0, max(3, bins) + 1)
    count = len(edges) - 1
    index_a = np.clip(
        np.searchsorted(edges, reference[:, feature_a], side="right") - 1,
        0,
        count - 1,
    )
    index_b = np.clip(
        np.searchsorted(edges, reference[:, feature_b], side="right") - 1,
        0,
        count - 1,
    )
    local = np.zeros((count, count), dtype=float)
    weights = np.zeros((count, count), dtype=int)
    for left in range(count):
        for right in range(count):
            selected = reference[(index_a == left) & (index_b == right)]
            weights[left, right] = len(selected)
            if not len(selected):
                continue
            ll = selected.copy()
            lh = selected.copy()
            hl = selected.copy()
            hh = selected.copy()
            for values, a_value, b_value in (
                (ll, edges[left], edges[right]),
                (lh, edges[left], edges[right + 1]),
                (hl, edges[left + 1], edges[right]),
                (hh, edges[left + 1], edges[right + 1]),
            ):
                values[:, feature_a] = a_value
                values[:, feature_b] = b_value
            local[left, right] = float(
                np.mean(
                    model.predict(hh)[:, output_index]
                    - model.predict(hl)[:, output_index]
                    - model.predict(lh)[:, output_index]
                    + model.predict(ll)[:, output_index]
                )
            )
    surface = np.cumsum(np.cumsum(local, axis=0), axis=1)
    row_mean = np.mean(surface, axis=1, keepdims=True)
    column_mean = np.mean(surface, axis=0, keepdims=True)
    surface = surface - row_mean - column_mean + float(np.mean(surface))
    centers = 0.5 * (edges[:-1] + edges[1:])
    return centers, surface, weights


def calculate_ale_reports(
    model,
    reference,
    *,
    codec: ParameterCodec,
    output_names: Sequence[str],
    parameter_rows: Sequence[Mapping[str, Any]],
    interaction_rows: Sequence[Mapping[str, Any]],
    top_parameters: int,
    top_interactions: int,
    bins: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    parameter_names = list(codec.active_names)
    output_lookup = {name: index for index, name in enumerate(output_names)}
    parameter_lookup = {name: index for index, name in enumerate(parameter_names)}
    effect_suffixes = tuple(
        metric.replace("overall__", "") for metric in PRIMARY_METRICS[:3]
    )
    effect_outputs = [
        name
        for name in output_names
        if name.endswith(effect_suffixes)
        or name in PRIMARY_METRICS[3:]
    ]
    base_vector = codec.encode(codec.base)

    def actual_value(parameter: str, normalized: float) -> Any:
        vector = list(base_vector)
        vector[parameter_lookup[parameter]] = float(normalized)
        return codec.decode(vector)[parameter]

    main_rows: list[dict[str, Any]] = []
    for output in effect_outputs:
        if output not in output_lookup:
            continue
        selected_parameters = [
            str(row["parameter"])
            for row in sorted(
                (row for row in parameter_rows if row["output"] == output),
                key=lambda row: float(row["st"]),
                reverse=True,
            )[: max(1, top_parameters)]
        ]
        for parameter in selected_parameters:
            centers, effects, counts = first_order_ale(
                model,
                reference,
                feature_index=parameter_lookup[parameter],
                output_index=output_lookup[output],
                bins=bins,
            )
            for bin_number, (center, effect, count) in enumerate(
                zip(centers, effects, counts),
                start=1,
            ):
                main_rows.append(
                    {
                        "output": output,
                        "parameter": parameter,
                        "bin": bin_number,
                        "normalized_center": float(center),
                        "parameter_value": actual_value(parameter, float(center)),
                        "ale_effect": float(effect),
                        "samples": int(count),
                    }
                )

    primary_interactions = [
        row for row in interaction_rows if row["output"] in PRIMARY_METRICS[:3]
    ]
    pair_scores: dict[tuple[str, str], float] = {}
    for row in primary_interactions:
        pair = (str(row["parameter_a"]), str(row["parameter_b"]))
        pair_scores[pair] = max(pair_scores.get(pair, 0.0), abs(float(row["s2"])))
    selected_pairs = sorted(
        pair_scores,
        key=lambda pair: pair_scores[pair],
        reverse=True,
    )[: max(0, top_interactions)]
    interaction_ale_rows: list[dict[str, Any]] = []
    for output in PRIMARY_METRICS[:3]:
        if output not in output_lookup:
            continue
        for parameter_a, parameter_b in selected_pairs:
            centers, surface, weights = second_order_ale(
                model,
                reference,
                feature_a=parameter_lookup[parameter_a],
                feature_b=parameter_lookup[parameter_b],
                output_index=output_lookup[output],
                bins=max(3, min(8, bins)),
            )
            for left, center_a in enumerate(centers):
                for right, center_b in enumerate(centers):
                    interaction_ale_rows.append(
                        {
                            "output": output,
                            "parameter_a": parameter_a,
                            "parameter_b": parameter_b,
                            "normalized_center_a": float(center_a),
                            "normalized_center_b": float(center_b),
                            "parameter_value_a": actual_value(
                                parameter_a, float(center_a)
                            ),
                            "parameter_value_b": actual_value(
                                parameter_b, float(center_b)
                            ),
                            "ale_interaction_effect": float(surface[left, right]),
                            "samples": int(weights[left, right]),
                        }
                    )
    return main_rows, interaction_ale_rows


def _plot_reports(
    output_dir: Path,
    *,
    parameter_rows: Sequence[Mapping[str, Any]],
    interaction_rows: Sequence[Mapping[str, Any]],
    ale_rows: Sequence[Mapping[str, Any]],
    ale_interaction_rows: Sequence[Mapping[str, Any]],
    ranking: Sequence[Mapping[str, Any]],
) -> list[str]:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError as exc:
        raise RuntimeError("Plot generation requires matplotlib") from exc

    plot_dir = output_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    paths: list[str] = []
    effect_outputs = sorted({str(row["output"]) for row in ale_rows})
    for output in effect_outputs:
        top_names = [
            str(row["parameter"])
            for row in sorted(
                (row for row in parameter_rows if row["output"] == output),
                key=lambda row: float(row["st"]),
                reverse=True,
            )[:10]
        ]
        selected = [
            row
            for row in parameter_rows
            if row["output"] == output and row["parameter"] in top_names
        ]
        selected.sort(key=lambda row: float(row["st"]))
        if selected:
            figure, axis = plt.subplots(figsize=(10, 6))
            axis.barh(
                [str(row["parameter"]) for row in selected],
                [max(0.0, float(row["st"])) for row in selected],
                xerr=[float(row["st_conf"]) for row in selected],
                color="#167d6d",
                alpha=0.88,
            )
            axis.set_xlabel("Sobol total-order index ST")
            axis.set_title(output.replace("overall__", "").replace("_", " ").title())
            axis.grid(axis="x", alpha=0.2)
            figure.tight_layout()
            path = plot_dir / f"sobol-total-{_slug(output)}.png"
            figure.savefig(path, dpi=160)
            plt.close(figure)
            paths.append(str(path))

        effects = [row for row in ale_rows if row["output"] == output]
        if effects:
            figure, axis = plt.subplots(figsize=(10, 6))
            for parameter in sorted({str(row["parameter"]) for row in effects}):
                values = [row for row in effects if row["parameter"] == parameter]
                values.sort(key=lambda row: int(row["bin"]))
                axis.plot(
                    [float(row["normalized_center"]) for row in values],
                    [float(row["ale_effect"]) for row in values],
                    marker="o",
                    linewidth=1.5,
                    label=parameter,
                )
            axis.axhline(0.0, color="#6f7782", linewidth=1)
            axis.set_xlabel("Normalized parameter value")
            axis.set_ylabel("Centered ALE effect")
            axis.set_title(f"ALE: {output.replace('overall__', '').replace('_', ' ').title()}")
            axis.legend(fontsize=8, ncol=2)
            axis.grid(alpha=0.2)
            figure.tight_layout()
            path = plot_dir / f"ale-{_slug(output)}.png"
            figure.savefig(path, dpi=160)
            plt.close(figure)
            paths.append(str(path))

    top_names = [str(row["parameter"]) for row in ranking[:10]]
    pair_lookup: dict[tuple[str, frozenset[str]], float] = {}
    for row in interaction_rows:
        if row["output"] in PRIMARY_METRICS[:3]:
            pair_lookup[
                (
                    str(row["output"]),
                    frozenset(
                        (str(row["parameter_a"]), str(row["parameter_b"]))
                    ),
                )
            ] = float(row["s2"])
    for output in PRIMARY_METRICS[:3]:
        matrix = np.zeros((len(top_names), len(top_names)), dtype=float)
        for left, parameter_a in enumerate(top_names):
            for right, parameter_b in enumerate(top_names):
                key = (output, frozenset((parameter_a, parameter_b)))
                matrix[left, right] = pair_lookup.get(key, 0.0)
        figure, axis = plt.subplots(figsize=(10, 8))
        image = axis.imshow(matrix, cmap="RdBu_r", vmin=-max(0.05, abs(matrix).max()), vmax=max(0.05, abs(matrix).max()))
        axis.set_xticks(range(len(top_names)), top_names, rotation=90)
        axis.set_yticks(range(len(top_names)), top_names)
        axis.set_title(f"Sobol S2: {output.replace('overall__', '').replace('_', ' ').title()}")
        figure.colorbar(image, ax=axis, label="Second-order index S2")
        figure.tight_layout()
        path = plot_dir / f"sobol-s2-{_slug(output)}.png"
        figure.savefig(path, dpi=160)
        plt.close(figure)
        paths.append(str(path))

    grouped_surfaces: dict[tuple[str, str, str], list[Mapping[str, Any]]] = {}
    for row in ale_interaction_rows:
        grouped_surfaces.setdefault(
            (
                str(row["output"]),
                str(row["parameter_a"]),
                str(row["parameter_b"]),
            ),
            [],
        ).append(row)
    for (output, parameter_a, parameter_b), rows in grouped_surfaces.items():
        a_values = sorted({float(row["normalized_center_a"]) for row in rows})
        b_values = sorted({float(row["normalized_center_b"]) for row in rows})
        surface = np.zeros((len(a_values), len(b_values)), dtype=float)
        for row in rows:
            left = a_values.index(float(row["normalized_center_a"]))
            right = b_values.index(float(row["normalized_center_b"]))
            surface[left, right] = float(row["ale_interaction_effect"])
        figure, axis = plt.subplots(figsize=(7, 6))
        image = axis.imshow(surface.T, origin="lower", aspect="auto", cmap="RdBu_r")
        axis.set_xlabel(parameter_a)
        axis.set_ylabel(parameter_b)
        axis.set_title(f"2D ALE: {output.replace('overall__', '').replace('_', ' ').title()}")
        figure.colorbar(image, ax=axis, label="Centered interaction effect")
        figure.tight_layout()
        path = plot_dir / f"ale2d-{_slug(output)}-{_slug(parameter_a)}-{_slug(parameter_b)}.png"
        figure.savefig(path, dpi=160)
        plt.close(figure)
        paths.append(str(path))
    return paths


def _summary_markdown(
    *,
    analysis_id: str,
    mode: str,
    traces: Sequence[BaselineTrace],
    ranking: Sequence[Mapping[str, Any]],
    validation: Sequence[Mapping[str, Any]],
) -> str:
    lines = [
        f"# RAPEC-v8 Sobol-ALE Sensitivity Analysis: {analysis_id}",
        "",
        f"- Mode: `{mode}`",
        f"- Full100 traces: `{len(traces)}`",
        f"- Task types: `{', '.join(sorted({trace.task_type for trace in traces}))}`",
        "- Energy scope: training/epoch GPU energy",
        "- Existing source bundles and previous results were not modified.",
        "",
        "## Parameter ranking",
        "",
        "| Rank | Parameter | Maximum ST | Mean ST |",
        "|---:|---|---:|---:|",
    ]
    for row in ranking:
        lines.append(
            f"| {row['rank']} | `{row['parameter']}` | "
            f"{float(row['combined_max_total_order']):.4f} | "
            f"{float(row['combined_mean_total_order']):.4f} |"
        )
    lines.extend(
        [
            "",
            "## Surrogate validation",
            "",
            "Sobol results from surrogate mode must be interpreted together with these holdout metrics.",
            "",
            "| Output | R2 | MAE | Status |",
            "|---|---:|---:|---|",
        ]
    )
    for row in validation:
        if row["output"] not in PRIMARY_METRICS:
            continue
        r2 = "n/a" if row["r2"] is None else f"{float(row['r2']):.4f}"
        lines.append(
            f"| `{row['output']}` | {r2} | {float(row['mae']):.6f} | {row['status']} |"
        )
    lines.extend(
        [
            "",
            "## Interpretation boundary",
            "",
            "Sobol indices quantify variance within the declared parameter ranges. They are not causal effects outside those ranges. Offline replay assumes the Full100 trajectory up to the simulated stop epoch. Fresh Rancher runs remain necessary for final confirmation.",
            "",
        ]
    )
    return "\n".join(lines)


def run_analysis(args: argparse.Namespace) -> Path:
    import numpy as np

    output_dir = args.output_dir.resolve()
    if output_dir.exists() and not args.resume:
        raise FileExistsError(
            f"Sensitivity output already exists and is protected: {output_dir}"
        )
    output_dir.mkdir(parents=True, exist_ok=args.resume)
    progress = ProgressReporter(output_dir)
    progress.emit(
        "sensitivity_started",
        {
            "analysis_id": args.analysis_id,
            "mode": args.mode,
            "design_samples": args.design_samples,
            "sobol_base_samples": args.sobol_base_samples,
        },
    )

    traces, locks = _load_sources([path.resolve() for path in args.source_dir])
    base = _base_parameters(args.base_controllers)
    search_space = _read_json(args.search_space)
    codec = ParameterCodec.from_search_space(base, search_space)
    problem = _problem(codec)
    evaluator = V8ReplayEvaluator(
        traces,
        controller_plugin=args.controller_plugin,
        controller_id=f"v8-sensitivity-{args.analysis_id}",
    )
    run_config = {
        "schema_version": 1,
        "analysis_id": args.analysis_id,
        "mode": args.mode,
        "seed": args.seed,
        "design_samples": args.design_samples,
        "sobol_base_samples": args.sobol_base_samples,
        "parameters": list(codec.active_names),
        "base_parameters": base,
        "parameter_specifications": search_space["parameters"],
        "outputs": list(evaluator.output_names),
        "source_run_ids": [lock["benchmark_run_id"] for lock in locks],
    }
    config_path = output_dir / "run-config.json"
    if args.resume and config_path.exists() and _read_json(config_path) != run_config:
        raise ValueError("Resume refused because run-config.json differs")
    _write_json(config_path, run_config)
    _write_json(output_dir / "source-audit.json", locks)

    if args.mode == "direct":
        analysis_matrix = _sobol_matrix(problem, args.sobol_base_samples, args.seed)
        design_matrix = analysis_matrix
        design_journal = output_dir / "direct-evaluations.jsonl"
    else:
        design_matrix = _design_matrix(args.design_samples, codec.dimension, args.seed)
        design_journal = output_dir / "design-evaluations.jsonl"

    design_x, design_y, design_records = evaluate_matrix(
        design_matrix,
        codec=codec,
        evaluator=evaluator,
        journal_path=design_journal,
        progress=progress,
        resume=args.resume,
    )
    flat_rows = []
    for record in design_records:
        flat_rows.append(
            {
                "index": record["index"],
                **{f"parameter__{key}": value for key, value in record["parameters"].items()},
                **record["outputs"],
            }
        )
    _write_csv(output_dir / "design-evaluations.csv", flat_rows)

    progress.emit(
        "surrogate_fit_started",
        {"completed": len(design_records), "total": len(design_records)},
    )
    model, validation = fit_surrogate(
        design_x,
        design_y,
        evaluator.output_names,
        trees=args.trees,
        seed=args.seed,
    )
    _write_json(output_dir / "surrogate-validation.json", validation)
    _write_csv(output_dir / "surrogate-validation.csv", validation)
    progress.emit(
        "surrogate_fit_completed",
        {
            "reliable_outputs": sum(row["status"] == "reliable" for row in validation),
            "total": len(validation),
        },
    )

    if args.mode == "surrogate":
        analysis_matrix = _sobol_matrix(
            problem,
            args.sobol_base_samples,
            args.seed + 1,
        )
        analysis_y = np.asarray(model.predict(analysis_matrix), dtype=float)
    else:
        analysis_y = design_y

    progress.emit(
        "sobol_analysis_started",
        {"completed": 0, "total": len(evaluator.output_names)},
    )
    parameter_rows, interaction_rows = calculate_sobol_indices(
        problem,
        analysis_matrix,
        analysis_y,
        evaluator.output_names,
        bootstrap_resamples=args.bootstrap_resamples,
        seed=args.seed,
    )
    _write_csv(output_dir / "sobol-parameters.csv", parameter_rows)
    _write_json(output_dir / "sobol-parameters.json", parameter_rows)
    _write_csv(output_dir / "sobol-interactions.csv", interaction_rows)
    _write_json(output_dir / "sobol-interactions.json", interaction_rows)
    ranking = build_parameter_ranking(
        parameter_rows,
        codec.active_names,
    )
    output_rankings = build_output_rankings(parameter_rows)
    _write_csv(output_dir / "parameter-ranking.csv", ranking)
    _write_json(output_dir / "parameter-ranking.json", ranking)
    _write_csv(output_dir / "sobol-output-rankings.csv", output_rankings)
    _write_json(output_dir / "sobol-output-rankings.json", output_rankings)
    progress.emit(
        "sobol_analysis_completed",
        {
            "completed": len(evaluator.output_names),
            "total": len(evaluator.output_names),
            "top_parameter": ranking[0]["parameter"] if ranking else None,
        },
    )

    ale_rows, ale_interaction_rows = calculate_ale_reports(
        model,
        design_x,
        codec=codec,
        output_names=evaluator.output_names,
        parameter_rows=parameter_rows,
        interaction_rows=interaction_rows,
        top_parameters=args.top_parameters,
        top_interactions=args.top_interactions,
        bins=args.ale_bins,
    )
    _write_csv(output_dir / "ale-main-effects.csv", ale_rows)
    _write_json(output_dir / "ale-main-effects.json", ale_rows)
    _write_csv(output_dir / "ale-interactions.csv", ale_interaction_rows)
    _write_json(output_dir / "ale-interactions.json", ale_interaction_rows)
    plots = _plot_reports(
        output_dir,
        parameter_rows=parameter_rows,
        interaction_rows=interaction_rows,
        ale_rows=ale_rows,
        ale_interaction_rows=ale_interaction_rows,
        ranking=ranking,
    )

    primary_validation = [
        row for row in validation if row["output"] in PRIMARY_METRICS
    ]
    reliable = all(
        row["status"] in {"reliable", "constant-output"}
        for row in primary_validation
    )
    manifest = {
        "schema_version": 1,
        "analysis_id": args.analysis_id,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "method": "Sobol S1/ST/S2 with 95% bootstrap confidence intervals and ALE",
        "execution_mode": args.mode,
        "controller_plugin": args.controller_plugin,
        "parameter_count": codec.dimension,
        "parameters": list(codec.active_names),
        "design_samples": len(design_x),
        "sobol_base_samples": args.sobol_base_samples,
        "sobol_model_evaluations": len(analysis_matrix),
        "surrogate": "ExtraTreesRegressor",
        "surrogate_primary_outputs_reliable": reliable,
        "surrogate_reliability_rule": "holdout R2 >= 0.75 or constant output",
        "source_locks": locks,
        "source_data_were_modified": False,
        "old_results_overwritten": False,
        "energy_scope": "training/epoch GPU energy",
        "task_stratification": sorted({trace.task_type for trace in traces}),
        "plot_files": plots,
        "offline_replay_limitation": (
            "Replay assumes the Full100 path up to the simulated stop epoch. "
            "Fresh Rancher confirmation is required for final claims."
        ),
    }
    _write_json(output_dir / "sensitivity-manifest.json", manifest)
    (output_dir / "SUMMARY.md").write_text(
        _summary_markdown(
            analysis_id=args.analysis_id,
            mode=args.mode,
            traces=traces,
            ranking=ranking,
            validation=validation,
        ),
        encoding="utf-8",
    )
    progress.emit(
        "sensitivity_completed",
        {
            "analysis_id": args.analysis_id,
            "top_parameter": ranking[0]["parameter"] if ranking else None,
            "surrogate_reliable": reliable,
            "output_dir": str(output_dir),
        },
    )
    return output_dir


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-dir", type=Path, action="append", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--analysis-id", required=True)
    parser.add_argument("--mode", choices=("surrogate", "direct"), default="surrogate")
    parser.add_argument(
        "--base-controllers",
        type=Path,
        default=Path("controller_benchmark/config/rapec-v3-v8-controllers.json"),
    )
    parser.add_argument(
        "--search-space",
        type=Path,
        default=Path("controller_benchmark/v8_sensitivity/search-space.json"),
    )
    parser.add_argument("--controller-plugin", default=DEFAULT_PLUGIN)
    parser.add_argument("--design-samples", type=int, default=1024)
    parser.add_argument("--sobol-base-samples", type=int, default=1024)
    parser.add_argument("--trees", type=int, default=400)
    parser.add_argument("--bootstrap-resamples", type=int, default=256)
    parser.add_argument("--top-parameters", type=int, default=8)
    parser.add_argument("--top-interactions", type=int, default=5)
    parser.add_argument("--ale-bins", type=int, default=10)
    parser.add_argument("--seed", type=int, default=20260802)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if not _is_power_of_two(args.design_samples):
        parser.error("--design-samples must be a power of two")
    if not _is_power_of_two(args.sobol_base_samples):
        parser.error("--sobol-base-samples must be a power of two")
    if args.design_samples < 32 or args.sobol_base_samples < 16:
        parser.error("Use at least 32 design samples and 16 Sobol base samples")
    try:
        result = run_analysis(args)
    except Exception as exc:
        if args.output_dir.exists():
            ProgressReporter(args.output_dir).emit(
                "sensitivity_failed",
                {
                    "analysis_id": args.analysis_id,
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                },
            )
        raise
    print(result)
    print(result / "SUMMARY.md")


if __name__ == "__main__":
    main()
