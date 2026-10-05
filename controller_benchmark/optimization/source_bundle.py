from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


LOCK_FILE = "source-lock.json"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _source_files(source_dir: Path) -> list[Path]:
    return sorted(
        path
        for path in source_dir.rglob("*")
        if path.is_file() and path.name != LOCK_FILE
    )


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def validate_source_contents(source_dir: str | Path) -> dict[str, Any]:
    root = Path(source_dir).resolve()
    manifest_path = root / "manifest.json"
    plan_path = root / "run-matrix.json"
    catalog_path = root / "baseline-cache" / "catalog.json"
    for path in (manifest_path, plan_path, catalog_path):
        if not path.is_file():
            raise FileNotFoundError(path)

    manifest = _read_json(manifest_path)
    plan = _read_json(plan_path)
    catalog = _read_json(catalog_path)
    enabled_cases = {
        case["id"]
        for stage in manifest["stages"]
        if stage.get("enabled")
        for case in stage["cases"]
    }
    entries = catalog.get("entries", {})
    missing = sorted(enabled_cases - set(entries))
    if missing:
        raise ValueError(f"Baseline catalog is missing cases: {', '.join(missing)}")

    for case_id in sorted(enabled_cases):
        entry = entries[case_id]
        summary_path = root / "baseline-cache" / entry["epoch_summary"]
        gpu_path = root / "baseline-cache" / entry["gpu_summary"]
        for path, hash_key in (
            (summary_path, "epoch_summary_sha256"),
            (gpu_path, "gpu_summary_sha256"),
        ):
            if not path.is_file():
                raise FileNotFoundError(path)
            if _sha256(path) != entry[hash_key]:
                raise ValueError(f"Checksum mismatch for {case_id}/{path.name}")
        summary = _read_json(summary_path)
        epochs = summary.get("epochs", [])
        if summary.get("comparison_strategy") != "full100":
            raise ValueError(f"{case_id} is not a Full100 baseline.")
        if len(epochs) != int(manifest["execution"]["max_epochs"]):
            raise ValueError(
                f"{case_id} has {len(epochs)} epochs, expected "
                f"{manifest['execution']['max_epochs']}."
            )

    return {
        "benchmark_run_id": plan["benchmark_run_id"],
        "benchmark_version": manifest["benchmark_version"],
        "cases": len(enabled_cases),
    }


def create_source_lock(source_dir: str | Path, run_id: str | None = None) -> Path:
    root = Path(source_dir).resolve()
    lock_path = root / LOCK_FILE
    if lock_path.exists():
        raise FileExistsError(f"Source lock already exists: {lock_path}")
    validation = validate_source_contents(root)
    if run_id and validation["benchmark_run_id"] != run_id:
        raise ValueError(
            f"Downloaded run is {validation['benchmark_run_id']}, expected {run_id}."
        )
    files = {
        path.relative_to(root).as_posix(): _sha256(path)
        for path in _source_files(root)
    }
    lock = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        **validation,
        "read_only_source": True,
        "files": files,
    }
    lock_path.write_text(
        json.dumps(lock, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return lock_path


def verify_source_lock(source_dir: str | Path) -> dict[str, Any]:
    root = Path(source_dir).resolve()
    lock_path = root / LOCK_FILE
    if not lock_path.is_file():
        raise FileNotFoundError(lock_path)
    lock = _read_json(lock_path)
    actual_files = {
        path.relative_to(root).as_posix(): _sha256(path)
        for path in _source_files(root)
    }
    expected_files = lock.get("files", {})
    if actual_files != expected_files:
        missing = sorted(set(expected_files) - set(actual_files))
        added = sorted(set(actual_files) - set(expected_files))
        changed = sorted(
            path
            for path in set(actual_files) & set(expected_files)
            if actual_files[path] != expected_files[path]
        )
        raise ValueError(
            "Optimization source was modified: "
            f"missing={missing}, added={added}, changed={changed}"
        )
    validation = validate_source_contents(root)
    if validation["benchmark_run_id"] != lock.get("benchmark_run_id"):
        raise ValueError("Source lock run id does not match the source contents.")
    return lock


def main() -> None:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    create = subparsers.add_parser("create")
    create.add_argument("--source-dir", type=Path, required=True)
    create.add_argument("--run-id")
    verify = subparsers.add_parser("verify")
    verify.add_argument("--source-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "create":
        print(create_source_lock(args.source_dir, args.run_id))
    else:
        lock = verify_source_lock(args.source_dir)
        print(
            json.dumps(
                {
                    "verified": True,
                    "benchmark_run_id": lock["benchmark_run_id"],
                    "benchmark_version": lock["benchmark_version"],
                    "cases": lock["cases"],
                    "files": len(lock["files"]),
                },
                indent=2,
            )
        )


if __name__ == "__main__":
    main()
