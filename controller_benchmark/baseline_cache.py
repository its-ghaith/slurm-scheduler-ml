from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .manifest import load_manifest


def _json_hash(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def enabled_cases(manifest: dict[str, Any]):
    for stage in manifest["stages"]:
        if not stage.get("enabled"):
            continue
        for case in stage["cases"]:
            yield stage, case


def required_baseline_cases(manifest: dict[str, Any]):
    seen: set[str] = set()
    for stage in manifest["stages"]:
        if not stage.get("enabled"):
            continue
        cases = {case["id"]: case for case in stage["cases"]}
        for case in stage["cases"]:
            source_id = stage.get("baseline_case_id", case["id"])
            if source_id not in cases:
                raise ValueError(f"Stage {stage['id']} references unknown baseline case {source_id}.")
            if source_id not in seen:
                seen.add(source_id)
                yield stage, cases[source_id]


def baseline_fingerprint(manifest: dict[str, Any], case: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    execution = manifest["execution"]
    config = {
        "schema": 1,
        "benchmark_version": manifest["benchmark_version"],
        "baseline_id": manifest["baselines"][0]["id"],
        "runner": case["runner"],
        "task_type": case["task_type"],
        "quality_metric": case.get("baseline_quality_metric", case["quality_metric"]),
        "scenario": case["scenario"],
        "training_seed": int(case["training_seed"]),
        "split_seed": int(execution["split_seed"]),
        "train_size": int(case["train_size"]),
        "model_version": case["model_version"],
        "pretrained": bool(case["pretrained"]),
        "max_epochs": int(execution["max_epochs"]),
        "batch_size": int(execution["batch_size"]),
        "image_size": int(execution["image_size"]),
        "cache_policy": execution["cache_policy"],
        "runtime_image_id": execution["runtime_image_id"],
        "gpu_name": execution["gpu_name"],
        "gpu_power_limit_w": str(execution["gpu_power_limit_w"]),
    }
    for key in ("measurement_protocol_version", "energy_comparison_scope"):
        if execution.get(key):
            config[key] = execution[key]
    for key in ("dataset_fingerprint", "task_definition_version", "runner_version"):
        if case.get(key):
            config[key] = case[key]
    return _json_hash(config), config


def _match_source(
    *, manifest: dict[str, Any], case: dict[str, Any], summaries: list[tuple[Path, dict[str, Any]]]
) -> tuple[Path, dict[str, Any]]:
    execution = manifest["execution"]
    def normalized_cache_policy(value: Any) -> str:
        text = str(value or "")
        return "persistent-disk" if text == "disk" else text

    matches = [
        item
        for item in summaries
        if item[1].get("comparison_strategy") == manifest["baselines"][0]["id"]
        and item[1].get("scenario") == case["scenario"]
        and int(item[1].get("training_seed", -1)) == int(case["training_seed"])
        and int(item[1].get("split_seed", -1)) == int(execution["split_seed"])
        and normalized_cache_policy(item[1].get("cache_policy")) == normalized_cache_policy(execution["cache_policy"])
        and int(item[1].get("epochs_completed", 0)) == int(execution["max_epochs"])
    ]
    if len(matches) != 1:
        raise RuntimeError(f"Expected one Full100 source for {case['id']}, found {len(matches)}.")
    return matches[0]


def import_baselines(manifest_path: str | Path, snapshot: str | Path, output_dir: str | Path) -> Path:
    manifest = load_manifest(manifest_path)
    snapshot = Path(snapshot).resolve()
    metrics = snapshot / "data" / "energy_metrics"
    output = Path(output_dir).resolve()
    catalog_path = output / "catalog.json"
    existing_catalog = None
    if catalog_path.exists():
        existing_catalog = load_baseline_catalog(catalog_path, manifest)
    elif output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Baseline output is not empty: {output}")
    summaries = []
    for path in metrics.glob("epoch_summary_job_*.json"):
        summaries.append((path, json.loads(path.read_text(encoding="utf-8-sig"))))
    output.mkdir(parents=True, exist_ok=True)
    entries: dict[str, Any] = dict((existing_catalog or {}).get("entries", {}))
    for stage, case in required_baseline_cases(manifest):
        if case["id"] in entries:
            continue
        summary_source, summary = _match_source(manifest=manifest, case=case, summaries=summaries)
        job_id = str(summary["job_id"])
        gpu_source = metrics / f"gpu_summary_job_{job_id}.json"
        if not gpu_source.exists():
            raise FileNotFoundError(gpu_source)
        gpu = json.loads(gpu_source.read_text(encoding="utf-8-sig"))
        metadata = gpu.get("run_metadata", {})
        expected = manifest["execution"]
        actual_gpu = metadata.get("gpu", {})
        checks = {
            "runtime_image_id": (metadata.get("runtime_image_id"), expected["runtime_image_id"]),
            "gpu_name": (actual_gpu.get("name"), expected["gpu_name"]),
            "gpu_power_limit_w": (str(actual_gpu.get("power_limit_w")), str(expected["gpu_power_limit_w"])),
        }
        mismatches = [f"{key}: actual={actual!r}, expected={wanted!r}" for key, (actual, wanted) in checks.items() if actual != wanted]
        if mismatches:
            raise RuntimeError(f"Incompatible baseline {case['id']}: " + "; ".join(mismatches))
        case_dir = output / "entries" / case["id"]
        case_dir.mkdir(parents=True, exist_ok=False)
        summary_target = case_dir / "epoch_summary.json"
        gpu_target = case_dir / "gpu_summary.json"
        shutil.copy2(summary_source, summary_target)
        shutil.copy2(gpu_source, gpu_target)
        fingerprint, config = baseline_fingerprint(manifest, case)
        entries[case["id"]] = {
            "stage": stage["id"],
            "fingerprint": fingerprint,
            "configuration": config,
            "source_snapshot": str(snapshot),
            "source_job_id": job_id,
            "epoch_summary": str(summary_target.relative_to(output)).replace("\\", "/"),
            "gpu_summary": str(gpu_target.relative_to(output)).replace("\\", "/"),
            "epoch_summary_sha256": _file_hash(summary_target),
            "gpu_summary_sha256": _file_hash(gpu_target),
        }
    now = datetime.now(timezone.utc).isoformat()
    catalog = dict(existing_catalog or {})
    catalog.pop("catalog_path", None)
    catalog.update({
        "schema_version": 1,
        "created_at": catalog.get("created_at", now),
        "updated_at": now,
        "benchmark_version": manifest["benchmark_version"],
        "manifest_sha256": manifest["manifest_sha256"],
        "source_snapshots": sorted(set(catalog.get("source_snapshots", [])) | {str(snapshot)}),
        "entries": entries,
    })
    temporary = catalog_path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(catalog, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    temporary.replace(catalog_path)
    return catalog_path


def load_baseline_catalog(path: str | Path, manifest: dict[str, Any]) -> dict[str, Any]:
    path = Path(path).resolve()
    catalog = json.loads(path.read_text(encoding="utf-8-sig"))
    if catalog.get("benchmark_version") != manifest["benchmark_version"]:
        raise ValueError("Baseline catalog benchmark version does not match the manifest.")
    root = path.parent
    expected_cases = {case["id"]: case for _, case in enabled_cases(manifest)}
    for case_id, case in expected_cases.items():
        entry = catalog.get("entries", {}).get(case_id)
        if entry is None:
            continue
        fingerprint, _ = baseline_fingerprint(manifest, case)
        if entry.get("fingerprint") != fingerprint:
            raise ValueError(f"Baseline fingerprint mismatch for {case_id}.")
        for field, hash_field in (("epoch_summary", "epoch_summary_sha256"), ("gpu_summary", "gpu_summary_sha256")):
            file_path = root / entry[field]
            if not file_path.exists() or _file_hash(file_path) != entry[hash_field]:
                raise ValueError(f"Baseline file validation failed for {case_id}/{field}.")
    catalog["catalog_path"] = str(path)
    return catalog


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    print(import_baselines(args.manifest, args.snapshot, args.output_dir))


if __name__ == "__main__":
    main()
