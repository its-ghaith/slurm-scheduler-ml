from __future__ import annotations

import argparse
import base64
import hashlib
import json
import os
import shutil
import subprocess
import time
import urllib.parse
import urllib.request
import urllib.error
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


TERMINAL_STATES = {"COMPLETED", "FAILED", "CANCELLED", "TIMEOUT", "OUT_OF_MEMORY", "NODE_FAIL"}


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def archive_completed_run(plan: dict[str, Any], run_dir: Path, status: dict[str, Any]) -> dict[str, Any]:
    execution = plan.get("execution", {})
    archive_root = Path(execution.get("long_term_archive_root", "/thesis-archive/controller-benchmarks"))
    if not archive_root.parent.exists():
        if execution.get("require_long_term_archive", False):
            raise RuntimeError(f"Long-term archive volume is unavailable: {archive_root.parent}")
        return {}

    log_destination = run_dir / "slurm-logs"
    log_destination.mkdir(parents=True, exist_ok=True)
    log_root = Path("/workspace/logs")
    job_ids = {
        str(record.get("job_id"))
        for record in status.get("runs", [])
        if record.get("job_id")
    }
    if log_root.exists():
        for job_id in sorted(job_ids):
            for source in log_root.glob(f"*{job_id}*"):
                if source.is_file():
                    shutil.copy2(source, log_destination / source.name)

    model_destination = run_dir / "model-registry"
    model_root = Path("/workspace-cache/model_registry")
    for job_id in sorted(job_ids):
        source = model_root / f"job_{job_id}"
        if source.is_dir():
            shutil.copytree(source, model_destination / source.name, dirs_exist_ok=True)

    checkpoint_roots = {
        Path(str(item["case"]["pretrained_checkpoint"])).parent
        for item in plan.get("runs", [])
        if item.get("case", {}).get("pretrained_checkpoint")
    }
    checkpoint_destination = run_dir / "pretraining-checkpoints"
    for source in sorted(checkpoint_roots):
        if source.is_dir():
            shutil.copytree(
                source,
                checkpoint_destination / source.name,
                dirs_exist_ok=True,
            )

    archive_root.mkdir(parents=True, exist_ok=True)
    run_archive_root = archive_root / plan["benchmark_run_id"]
    run_archive_root.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    destination = run_archive_root / stamp
    suffix = 1
    while destination.exists():
        destination = run_archive_root / f"{stamp}-{suffix}"
        suffix += 1
    temporary = run_archive_root / f".{destination.name}.copying"
    shutil.copytree(run_dir, temporary)

    files = []
    for path in sorted(item for item in temporary.rglob("*") if item.is_file()):
        files.append(
            {
                "path": path.relative_to(temporary).as_posix(),
                "size_bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    manifest = {
        "schema_version": 1,
        "benchmark_run_id": plan["benchmark_run_id"],
        "created_at": now(),
        "source": str(run_dir),
        "file_count": len(files),
        "files": files,
    }
    write_json(temporary / "archive-manifest.json", manifest)
    temporary.replace(destination)
    return {
        "path": str(destination),
        "manifest": str(destination / "archive-manifest.json"),
        "file_count": len(files),
    }


def run(command: list[str], *, cwd: Path | None = None, env: dict[str, str] | None = None) -> str:
    completed = subprocess.run(command, cwd=cwd, env=env, text=True, capture_output=True)
    if completed.returncode:
        raise RuntimeError(f"Command failed ({completed.returncode}): {' '.join(command)}\n{completed.stdout}\n{completed.stderr}")
    return completed.stdout.strip()


def slurm_state(job_id: str) -> str:
    completed = subprocess.run(["scontrol", "show", "job", job_id], text=True, capture_output=True)
    text = completed.stdout + completed.stderr
    marker = "JobState="
    if marker not in text:
        return "UNKNOWN"
    return text.split(marker, 1)[1].split()[0]


def wait_for_job(job_id: str, status: dict[str, Any], status_path: Path, timeout_seconds: int) -> None:
    running_since: float | None = None
    while True:
        state = slurm_state(job_id)
        if state in {"RUNNING", "COMPLETING"} and running_since is None:
            running_since = time.time()
        status["current_job_state"] = state
        status["current_job_running_seconds"] = int(time.time() - running_since) if running_since else 0
        status["updated_at"] = now()
        write_json(status_path, status)
        if state in TERMINAL_STATES:
            if state != "COMPLETED":
                raise RuntimeError(f"SLURM job {job_id} ended with {state}.")
            return
        if running_since is not None and time.time() - running_since >= timeout_seconds:
            raise TimeoutError(f"SLURM job {job_id} exceeded running timeout {timeout_seconds}s.")
        time.sleep(10)


def job_artifacts_complete(job_id: str, metrics: Path) -> bool:
    return all(
        (metrics / f"{prefix}_job_{job_id}.json").exists()
        for prefix in ("gpu_summary", "epoch_summary")
    )


def measurement_validation_errors(
    job_id: str,
    metrics: Path,
    expected_controllers: list[str],
    expected_epochs: int,
) -> list[str]:
    errors: list[str] = []
    gpu_path = metrics / f"gpu_summary_job_{job_id}.json"
    epoch_path = metrics / f"epoch_summary_job_{job_id}.json"
    shadow_path = metrics / f"shadow_summary_job_{job_id}.json"
    for path in (gpu_path, epoch_path, shadow_path):
        if not path.exists():
            errors.append(f"missing {path.name}")
    if errors:
        return errors
    gpu = json.loads(gpu_path.read_text(encoding="utf-8"))
    epoch = json.loads(epoch_path.read_text(encoding="utf-8"))
    shadow = json.loads(shadow_path.read_text(encoding="utf-8"))
    if int(gpu.get("codecarbon_measurement_available") or 0) != 1:
        errors.append("CodeCarbon job measurement unavailable")
    if int(gpu.get("lifecycle_energy_complete") or 0) != 1:
        errors.append("lifecycle energy incomplete")
    phases = gpu.get("phase_metrics") or {}
    for phase in ("preprocessing_initialization", "training", "finalization_evaluation"):
        values = phases.get(phase)
        if not isinstance(values, dict):
            errors.append(f"missing phase {phase}")
            continue
        for key in ("duration_seconds", "gpu_energy_kwh", "codecarbon_gpu_energy_kwh", "codecarbon_energy_kwh"):
            if values.get(key) is None:
                errors.append(f"missing {phase}.{key}")
    epochs = epoch.get("epochs") or []
    if len(epochs) != expected_epochs:
        errors.append(f"expected {expected_epochs} epochs, found {len(epochs)}")
    for index, row in enumerate(epochs, start=1):
        for key in ("quality_score", "duration_seconds", "gpu_energy_kwh", "gpu_util_avg_pct"):
            if row.get(key) is None:
                errors.append(f"epoch {index} missing {key}")
    controllers = {str(item.get("controller_id")): item for item in shadow.get("controllers") or []}
    for controller_id in expected_controllers:
        controller = controllers.get(controller_id)
        if controller is None:
            errors.append(f"missing shadow controller {controller_id}")
            continue
        for key in (
            "stop_epoch",
            "best_quality_at_stop",
            "training_energy_to_stop_wh",
            "controller_overhead_energy_wh",
            "controller_compute_seconds",
        ):
            if controller.get(key) is None:
                errors.append(f"{controller_id} missing {key}")
    return errors


def encode_json(value: Any) -> str:
    return base64.b64encode(json.dumps(value, separators=(",", ":")).encode()).decode()


def common_labels(item: dict[str, Any]) -> dict[str, str]:
    return {
        "MLFLOW_JOB_ENERGY_EXPERIMENT": item["execution"]["experiment_name"],
        "MLFLOW_TRACKING_URI": "http://mlflow:5000",
        "COMPARISON_STRATEGY": item["strategy"],
        "CONTROLLER_ID": item["controller_id"],
        "BENCHMARK_VERSION": item["benchmark_version"],
        "BENCHMARK_RUN_ID": item["benchmark_run_id"],
        "BENCHMARK_CASE_ID": item["benchmark_case_id"],
        "BENCHMARK_STAGE": item["benchmark_stage"],
        "BENCHMARK_TASK_TYPE": item["task_type"],
        "BENCHMARK_QUALITY_METRIC": item["quality_metric"],
        "EXPERIMENT_SCENARIO": item["scenario"],
        "TRAINING_SEED": str(item["training_seed"]),
        "SPLIT_SEED": str(item["execution"]["split_seed"]),
        "CACHE_POLICY": item["execution"]["cache_policy"],
        "RUNTIME_IMAGE_ID": item["execution"]["runtime_image_id"],
    }


def submit_run(item: dict[str, Any], workspace: Path, metrics: Path) -> str:
    environment = os.environ.copy()
    environment.update(common_labels(item))
    environment.update(
        {
            "CONTROLLER_PLUGIN": item.get("controller_plugin", ""),
            "CONTROLLER_PARAMETERS_BASE64": encode_json(item.get("controller_parameters", {})),
            "EARLY_STOP_CONTROLLER_MODE": item["controller_mode"],
            "ENERGY_ADAPTIVE_ENABLED": str(item["controller_mode"] != "none").lower(),
            "UNCERTAINTY_TARGET_EPOCH": str(item["execution"]["max_epochs"]),
            "CONTROLLER_EVALUATION_INTERVAL": "1",
            "GPU_IDLE_SAMPLE_SECONDS": "10",
            "JOB_MODEL_REGISTRY_DIR": "/workspace-cache/model_registry",
            "TORCH_HOME": "/workspace-cache/controller-pretrained-weights/torch",
            "YOLO_CONFIG_DIR": "/workspace-cache/controller-pretrained-weights/ultralytics",
        }
    )
    case = item["case"]
    if item["runner"] == "carpk_yolo":
        if environment.get("CACHE_POLICY") == "persistent-disk":
            environment["CACHE_POLICY"] = "disk"
        dataset = workspace / "rotationally-invariant-cnns" / "data" / "carpk"
        train_size = str(case["train_size"]) if int(case["train_size"]) > 0 else "False"
        overrides = [
            f"experiment={item['execution']['experiment_name']}",
            f"seeds={item['training_seed']}",
            f"dataset.path={dataset}",
            f"dataset.data_yaml={dataset / 'data.yaml'}",
            f"dataset.train_size={train_size}",
            f"model.epochs={item['execution']['max_epochs']}",
            f"model.batch_size={item['execution']['batch_size']}",
            f"model.img_size={item['execution']['image_size']}",
            f"model.patience={item['execution']['max_epochs']}",
            f"model.version={case['model_version']}",
            f"model.pretrained={str(case['pretrained']).lower()}",
        ]
        environment.update(
            {
                "REPRO_DATASET": "carpk",
                "REPRO_MODEL": "yolov8",
                "REPRO_OVERRIDES": ";".join(overrides),
                "REPRO_SCRIPT": str(workspace / "slurm" / "train_repro_phase_tracked.py"),
                "ROTA_REPO_ROOT": str(workspace / "rotationally-invariant-cnns"),
                "GPU_METRICS_DIR": str(metrics),
                "PROM_TEXTFILE_DIR": str(metrics / "node_exporter"),
                "ENERGY_ADAPTIVE_MONITOR_METRIC": "map50_95",
                "CONFORMAL_CALIBRATION_PATH": str(workspace / "slurm" / "conformal_calibration.json"),
            }
        )
        script = workspace / "slurm" / "train_repro_rotacnn_job.slurm"
    else:
        spec = dict(item)
        spec["execution"] = item["execution"]
        environment.update(
            {
                "RUN_SPEC_BASE64": encode_json(spec),
                "BENCHMARK_RUNNER": item["runner"],
                "BENCHMARK_WORKSPACE": str(workspace),
                "BENCHMARK_METRICS_DIR": str(metrics),
            }
        )
        script = workspace / "controller_benchmark" / "slurm" / "run-generic-benchmark.slurm"
    output = run(["sbatch", "--parsable", str(script)], cwd=workspace, env=environment)
    return output.split(";", 1)[0].strip()


def prometheus_label_values(label: str, benchmark_run_id: str) -> list[str]:
    matcher = f'controller_benchmark_case_info{{benchmark_run_id="{benchmark_run_id}"}}'
    query = urllib.parse.urlencode({"match[]": matcher})
    url = f"http://prometheus:9090/api/v1/label/{urllib.parse.quote(label)}/values?{query}"
    request = urllib.request.Request(url)
    with urllib.request.urlopen(request, timeout=30) as response:
        payload = json.loads(response.read())
    if payload.get("status") != "success":
        return []
    return sorted(str(value) for value in payload.get("data", []) if str(value))


def replace_dashboard_token(value: Any, token: str, replacement: str) -> Any:
    if isinstance(value, str):
        return value.replace(token, replacement)
    if isinstance(value, list):
        return [replace_dashboard_token(item, token, replacement) for item in value]
    if isinstance(value, dict):
        return {key: replace_dashboard_token(item, token, replacement) for key, item in value.items()}
    return value


def deploy_dashboard(dashboard_path: Path, benchmark_run_id: str | None = None) -> None:
    dashboard = json.loads(dashboard_path.read_text(encoding="utf-8"))
    if benchmark_run_id:
        dashboard = replace_dashboard_token(dashboard, "$benchmark_run_id", benchmark_run_id)
        values_by_variable = {
            "task_types": prometheus_label_values("task_type", benchmark_run_id),
            "stages": prometheus_label_values("stage", benchmark_run_id),
            "cases": prometheus_label_values("case_id", benchmark_run_id),
        }
        for variable in dashboard.get("templating", {}).get("list", []):
            if variable.get("name") == "benchmark_run_id":
                variable["current"] = {"selected": True, "text": benchmark_run_id, "value": benchmark_run_id}
                variable["options"] = [{"selected": True, "text": benchmark_run_id, "value": benchmark_run_id}]
            elif variable.get("name") in {"task_types", "stages", "cases"}:
                selected_values = values_by_variable.get(variable["name"], [])
                if selected_values:
                    variable["current"] = {
                        "selected": True,
                        "text": selected_values,
                        "value": selected_values,
                    }
                    variable["options"] = [
                        {"selected": True, "text": value, "value": value}
                        for value in selected_values
                    ]
                else:
                    variable["current"] = {"selected": True, "text": ["All"], "value": ["$__all"]}
    if len(str(dashboard.get("uid", ""))) > 40:
        run_id = benchmark_run_id or dashboard_path.parent.name
        dashboard["uid"] = f"cb-{run_id}".replace("_", "-")[:40]
    authorization = "Basic " + base64.b64encode(b"admin:admin").decode()
    search = urllib.request.Request(
        "http://grafana:3000/api/search?type=dash-folder",
        headers={"Authorization": authorization},
    )
    with urllib.request.urlopen(search, timeout=30) as response:
        folders = json.loads(response.read())
    folder = next((item for item in folders if item.get("title") == "SLURM Energy"), None)
    if folder is None:
        create = urllib.request.Request(
            "http://grafana:3000/api/folders",
            data=json.dumps({"title": "SLURM Energy"}).encode(),
            method="POST",
            headers={"Content-Type": "application/json", "Authorization": authorization},
        )
        with urllib.request.urlopen(create, timeout=30) as response:
            folder = json.loads(response.read())
    def post(current_dashboard: dict[str, Any]) -> None:
        payload = json.dumps({"dashboard": current_dashboard, "folderUid": folder["uid"], "overwrite": True}).encode()
        request = urllib.request.Request(
            "http://grafana:3000/api/dashboards/db", data=payload, method="POST",
            headers={"Content-Type": "application/json", "Authorization": authorization},
        )
        with urllib.request.urlopen(request, timeout=30) as response:
            if response.status >= 300:
                raise RuntimeError(f"Grafana dashboard deployment failed with HTTP {response.status}")

    try:
        post(dashboard)
    except urllib.error.HTTPError as exc:
        body = exc.read().decode(errors="replace")
        if exc.code != 400 or "provisioned dashboard" not in body:
            raise RuntimeError(f"Grafana dashboard deployment failed with HTTP {exc.code}: {body}") from exc
        run_id = benchmark_run_id or dashboard_path.parent.name
        dashboard["id"] = None
        dashboard["uid"] = f"cb-{run_id}".replace("_", "-")[:40]
        dashboard["title"] = f"{dashboard.get('title', 'Controller Benchmark')} ({run_id})"
        post(dashboard)


def prometheus_metric_types(exposition: str) -> dict[str, str]:
    """Return the metric-family types declared in Prometheus text exposition."""
    return {
        match.group(1): match.group(2)
        for match in re.finditer(
            r"^# TYPE\s+([a-zA-Z_:][a-zA-Z0-9_:]*)\s+(\S+)\s*$",
            exposition,
            flags=re.MULTILINE,
        )
    }


def normalize_prometheus_metric_types(
    exposition: str, existing_types: dict[str, str]
) -> str:
    """Make a payload type-compatible with metric families already in Pushgateway.

    Pushgateway validates metric-family types globally, including groups belonging
    to historical runs.  Old node-exporter textfiles did not always emit TYPE
    metadata, so those families are stored as ``untyped``.  Conversely, a few
    historical epoch families were explicitly gauges.  Preserve the samples and
    align only their metadata instead of deleting either old or new groups.
    """
    source_types = prometheus_metric_types(exposition)
    emitted_types: set[str] = set()
    normalized: list[str] = []
    type_pattern = re.compile(
        r"^# TYPE\s+([a-zA-Z_:][a-zA-Z0-9_:]*)\s+(\S+)\s*$"
    )
    sample_pattern = re.compile(r"^([a-zA-Z_:][a-zA-Z0-9_:]*)(?:\{|\s)")

    for line in exposition.splitlines():
        type_match = type_pattern.match(line)
        if type_match:
            metric = type_match.group(1)
            metric_type = existing_types.get(metric, type_match.group(2))
            if metric not in emitted_types:
                normalized.append(f"# TYPE {metric} {metric_type}")
                emitted_types.add(metric)
            continue

        sample_match = sample_pattern.match(line)
        if sample_match:
            metric = sample_match.group(1)
            metric_type = existing_types.get(metric)
            if metric_type and metric not in emitted_types and metric not in source_types:
                normalized.append(f"# TYPE {metric} {metric_type}")
                emitted_types.add(metric)
        normalized.append(line)

    return "\n".join(normalized) + "\n"


def publish_prometheus(prom_path: Path, benchmark_run_id: str) -> None:
    url = f"http://pushgateway:9091/metrics/job/controller_benchmark/benchmark_run_id/{benchmark_run_id}"
    with urllib.request.urlopen("http://pushgateway:9091/metrics", timeout=30) as response:
        existing_types = prometheus_metric_types(response.read().decode("utf-8", errors="replace"))
    payload = normalize_prometheus_metric_types(
        prom_path.read_text(encoding="utf-8"), existing_types
    ).encode("utf-8")
    request = urllib.request.Request(
        url, data=payload, method="PUT", headers={"Content-Type": "text/plain; version=0.0.4"}
    )
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            if response.status >= 300:
                raise RuntimeError(f"Prometheus publication failed with HTTP {response.status}")
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        first_error = next((line.strip() for line in body.splitlines() if line.strip()), "")
        raise RuntimeError(
            f"Prometheus publication failed with HTTP {exc.code}: {first_error[:1000]}"
        ) from exc


def merge_campaign_prometheus(node_exporter_dir: Path) -> Path:
    destination = node_exporter_dir / "thesis_campaign.prom"
    sources = [node_exporter_dir / "controller_benchmark.prom"] + sorted(
        path
        for path in node_exporter_dir.glob("job_*.prom")
        if path.name != destination.name
    )
    lines: list[str] = []
    metadata_seen: set[str] = set()
    samples_seen: set[str] = set()
    untyped_compatibility_metrics = {
        "slurm_job_training_energy_kwh",
        "slurm_job_estimated_electricity_cost_eur",
        "slurm_job_estimated_co2_kg",
    }
    for source in sources:
        if not source.exists():
            continue
        for raw in source.read_text(encoding="utf-8").splitlines():
            line = raw.strip()
            if not line:
                continue
            if line.startswith(("# HELP ", "# TYPE ")):
                parts = line.split()
                if (
                    line.startswith("# TYPE ")
                    and len(parts) >= 3
                    and parts[2] in untyped_compatibility_metrics
                ):
                    continue
                key = " ".join(line.split()[:3])
                if key in metadata_seen:
                    continue
                metadata_seen.add(key)
                lines.append(line)
            elif line.startswith("#"):
                lines.append(line)
            elif line not in samples_seen:
                samples_seen.add(line)
                lines.append(line)
    destination.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
    return destination


def publish_partial_results(
    plan: dict[str, Any],
    run_dir: Path,
    workspace: Path,
    metrics: Path,
    completed_count: int,
) -> Path | None:
    """Publish only validated, completed jobs while a campaign is still running."""
    completed_runs = list(plan.get("runs", []))[:completed_count]
    if not completed_runs:
        return None
    completed_job_ids: set[str] = set()
    status_path = run_dir / "status.json"
    if status_path.exists():
        status = json.loads(status_path.read_text(encoding="utf-8"))
        completed_job_ids = {
            str(record["job_id"])
            for record in status.get("runs", [])[:completed_count]
            if record.get("state") == "COMPLETED" and record.get("job_id")
        }
    if len(completed_job_ids) != len(completed_runs):
        return None

    snapshot = run_dir / "partial-snapshot"
    snapshot_metrics = snapshot / "data" / "energy_metrics"
    if snapshot_metrics.exists():
        shutil.rmtree(snapshot_metrics)
    snapshot_metrics.mkdir(parents=True)

    for source in metrics.iterdir():
        if source.name == "node_exporter":
            continue
        match = re.search(r"(?:_job_|codecarbon_job_)(\d+)", source.name)
        if not match or match.group(1) not in completed_job_ids:
            continue
        destination = snapshot_metrics / source.name
        if source.is_dir():
            shutil.copytree(source, destination)
        else:
            shutil.copy2(source, destination)

    partial_prom = snapshot_metrics / "node_exporter"
    partial_prom.mkdir()
    source_prom = metrics / "node_exporter"
    if source_prom.exists():
        for source in source_prom.glob("job_*.prom"):
            match = re.match(r"job_(\d+)(?:_(?:epochs|phases))?\.prom$", source.name)
            if match and match.group(1) in completed_job_ids:
                shutil.copy2(source, partial_prom / source.name)

    partial_plan = dict(plan)
    partial_plan["runs"] = completed_runs
    partial_plan_path = run_dir / "partial-run-matrix.json"
    write_json(partial_plan_path, partial_plan)
    analysis_module = plan.get("execution", {}).get(
        "analysis_module", "controller_benchmark.analyze"
    )
    run(
        [
            "python",
            "-m",
            analysis_module,
            "--snapshot",
            str(snapshot),
            "--plan",
            str(partial_plan_path),
        ],
        cwd=workspace,
    )
    prom = merge_campaign_prometheus(partial_prom)
    publish_prometheus(prom, plan["benchmark_run_id"])
    return prom


def prepare_workspace(plan: dict[str, Any], workspace: Path) -> None:
    data_root = workspace / "rotationally-invariant-cnns" / "data"
    data_root.mkdir(parents=True, exist_ok=True)
    carpk = data_root / "carpk"
    if carpk.is_symlink() or carpk.exists():
        if carpk.resolve() != Path("/workspace-cache/carpk").resolve():
            raise RuntimeError(f"Unexpected CARPK path in benchmark workspace: {carpk}")
    else:
        carpk.symlink_to("/workspace-cache/carpk", target_is_directory=True)
    if any(item["runner"] == "vedai_yolo" for item in plan["runs"]):
        run(
            ["python", "-m", "controller_benchmark.runners.prepare_vedai", "--source", "/workspace-cache/vedai512-source",
             "--output", "/workspace-cache/vedai512-yolo", "--seed", "0", "--download"], cwd=workspace
        )
    dataset_keys = sorted(
        {
            str(item["case"]["dataset_key"])
            for item in plan["runs"]
            if item.get("case", {}).get("dataset_key")
        }
    )
    vision_dataset_keys = {"cifar10", "cifar100", "oxford-pet", "pascal-voc2012", "tiny-imagenet", "uavid"}
    if any(item["runner"] == "visdrone_yolo" for item in plan["runs"]):
        vision_dataset_keys.add("visdrone-vehicles")
        dataset_keys.append("visdrone-vehicles")
    selected_vision_keys = sorted(set(dataset_keys) & vision_dataset_keys)
    cross_domain_keys = sorted(set(dataset_keys) - vision_dataset_keys)
    if selected_vision_keys:
        command = [
            "python",
            "-m",
            "controller_benchmark.runners.prepare_nine_datasets",
            "--root",
            "/workspace-cache/controller-datasets-v1",
        ]
        for dataset_key in selected_vision_keys:
            command.extend(["--dataset", dataset_key])
        run(command, cwd=workspace)
    if cross_domain_keys:
        asset_aliases = {
            "breast_cancer_anomaly": "breast_cancer",
            "wine_anomaly": "wine",
        }
        cross_domain_assets = sorted(
            {asset_aliases.get(dataset_key, dataset_key) for dataset_key in cross_domain_keys}
        )
        environment = os.environ.copy()
        environment["CONTROLLER_DATASET_ROOT"] = "/workspace-cache/controller-datasets"
        run(
            ["python", "-m", "controller_benchmark.runners.cross_domain_assets", *cross_domain_assets],
            cwd=workspace,
            env=environment,
        )
    label_code = (
        "from pathlib import Path; from countcv.effcv.preprocess_carpk import CarpkLabelCreator; "
        "CarpkLabelCreator(Path('/workspace-cache/carpk')).create_labels()"
    )
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(workspace / "rotationally-invariant-cnns" / "src")
    run(["python", "-c", label_code], cwd=workspace, env=environment)


def finalize(request: dict[str, Any], plan: dict[str, Any], run_dir: Path, workspace: Path, metrics: Path) -> dict[str, Any]:
    snapshot = run_dir / "snapshot"
    snapshot_metrics = snapshot / "data" / "energy_metrics"
    snapshot_metrics.parent.mkdir(parents=True, exist_ok=True)
    if snapshot_metrics.exists():
        shutil.rmtree(snapshot_metrics)
    shutil.copytree(metrics, snapshot_metrics)
    plan_path = run_dir / "run-matrix.json"
    dashboard_path = run_dir / "controller-benchmark.json"
    analysis_module = plan.get("execution", {}).get("analysis_module", "controller_benchmark.analyze")
    run(["python", "-m", analysis_module, "--snapshot", str(snapshot), "--plan", str(plan_path)], cwd=workspace)
    prom = merge_campaign_prometheus(snapshot_metrics / "node_exporter")
    publish_prometheus(prom, plan["benchmark_run_id"])
    dashboard_files = plan.get("execution", {}).get("dashboard_files", [])
    deployed_dashboards = []
    dashboard_deployment_mode = plan.get("execution", {}).get("dashboard_deployment_mode", "api")
    if dashboard_deployment_mode == "disabled":
        pass
    elif dashboard_deployment_mode == "provisioned":
        # The submission script installs the stable appendix dashboard once via
        # Grafana provisioning. Avoid creating a run-suffixed duplicate here.
        deployed_dashboards.extend(str(run_dir / dashboard_file) for dashboard_file in dashboard_files)
    elif dashboard_files:
        for dashboard_file in dashboard_files:
            current_dashboard = run_dir / dashboard_file
            deploy_dashboard(current_dashboard, plan["benchmark_run_id"])
            deployed_dashboards.append(str(current_dashboard))
    else:
        deploy_dashboard(dashboard_path, plan["benchmark_run_id"])
        deployed_dashboards.append(str(dashboard_path))
    baseline_root = run_dir / "baseline-cache"
    if int(plan["planned_baseline_runs"]) > 0:
        run(
            ["python", "-m", "controller_benchmark.baseline_cache", "--manifest", str(run_dir / "manifest.json"),
             "--snapshot", str(snapshot), "--output-dir", str(baseline_root)], cwd=workspace
        )
    if (baseline_root / "catalog.json").exists():
        canonical_parent = Path("/workspace-cache/controller-baselines")
        canonical_parent.mkdir(parents=True, exist_ok=True)
        canonical = canonical_parent / plan["benchmark_version"]
        temporary = canonical_parent / f".{plan['benchmark_version']}-{plan['benchmark_run_id']}.tmp"
        if temporary.exists():
            shutil.rmtree(temporary)
        shutil.copytree(baseline_root, temporary)
        backup = canonical_parent / f".{plan['benchmark_version']}.old"
        if backup.exists():
            shutil.rmtree(backup)
        if canonical.exists():
            canonical.replace(backup)
        temporary.replace(canonical)
        if backup.exists():
            shutil.rmtree(backup)
    return {
        "snapshot": str(snapshot),
        "baseline_catalog": str(baseline_root / "catalog.json"),
        "dashboard": deployed_dashboards[0] if deployed_dashboards else None,
        "dashboards": deployed_dashboards,
    }


def orchestrate(request_path: Path) -> None:
    request = json.loads(request_path.read_text(encoding="utf-8"))
    run_dir = Path(request["run_dir"])
    workspace = run_dir / "workspace"
    metrics = run_dir / "metrics"
    metrics.mkdir(parents=True, exist_ok=True)
    status_path = run_dir / "status.json"
    plan = json.loads((run_dir / "run-matrix.json").read_text(encoding="utf-8"))
    status = json.loads(status_path.read_text(encoding="utf-8")) if status_path.exists() else {
        "benchmark_run_id": plan["benchmark_run_id"], "controller_id": plan["controller"]["id"],
        "state": "RUNNING", "created_at": now(), "completed_runs": 0, "total_runs": len(plan["runs"]), "runs": [],
    }
    status.update({"state": "RUNNING", "updated_at": now()})
    status.pop("error", None)
    status.pop("failed_at", None)
    write_json(status_path, status)
    pretraining_job_id = str(plan.get("execution", {}).get("pretraining_job_id", "")).strip()
    if pretraining_job_id and not status.get("pretraining_completed"):
        status.update(
            {
                "state": "WAITING_FOR_PRETRAINING",
                "current_pretraining_job_id": pretraining_job_id,
                "updated_at": now(),
            }
        )
        write_json(status_path, status)
        wait_for_job(
            pretraining_job_id,
            status,
            status_path,
            int(plan["execution"].get("pretraining_timeout_seconds", 172800)),
        )
        status.update(
            {
                "state": "RUNNING",
                "pretraining_completed": True,
                "pretraining_completed_at": now(),
                "updated_at": now(),
            }
        )
        status.pop("current_pretraining_job_id", None)
        write_json(status_path, status)
    if not status.get("workspace_prepared"):
        prepare_workspace(plan, workspace)
        status["workspace_prepared"] = True
        status["updated_at"] = now()
        write_json(status_path, status)
    for index, item in enumerate(plan["runs"]):
        while len(status["runs"]) <= index:
            status["runs"].append({"sequence": item["sequence"], "case_id": item["benchmark_case_id"], "strategy": item["strategy"], "state": "PENDING"})
        record = status["runs"][index]
        if record.get("state") == "COMPLETED":
            continue
        strict_metrics = bool(plan["execution"].get("require_complete_lifecycle_metrics", False))
        max_attempts = max(1, int(plan["execution"].get("measurement_max_attempts", 3)))
        expected_controllers = [str(item["id"]) for item in plan.get("controllers", [])]
        expected_epochs = int(plan["execution"].get("max_epochs", 100))
        attempts = record.setdefault("attempts", [])
        while True:
            job_id = record.get("job_id")
            previous_state = slurm_state(str(job_id)) if job_id else "UNKNOWN"
            recovered = bool(job_id and previous_state == "UNKNOWN" and job_artifacts_complete(str(job_id), metrics))
            if job_id and previous_state in TERMINAL_STATES and previous_state != "COMPLETED":
                recovered = False
                record["last_error"] = f"SLURM job {job_id} ended with {previous_state}."
                record.pop("job_id", None)
                job_id = None
            if not job_id or (previous_state == "UNKNOWN" and not recovered):
                if len(attempts) >= max_attempts:
                    raise RuntimeError(
                        f"Case {item['benchmark_case_id']} exhausted {max_attempts} measurement attempts: "
                        f"{record.get('last_error', 'no complete artifacts')}"
                    )
                job_id = submit_run({**item, "execution": plan["execution"]}, workspace, metrics)
                attempt = {"attempt": len(attempts) + 1, "job_id": job_id, "submitted_at": now()}
                attempts.append(attempt)
                record.update({"job_id": job_id, "state": "SUBMITTED", "submitted_at": attempt["submitted_at"]})
                previous_state = "SUBMITTED"
            status.update({"current_sequence": item["sequence"], "current_job_id": job_id, "current_case_id": item["benchmark_case_id"], "updated_at": now()})
            write_json(status_path, status)
            if not recovered:
                try:
                    wait_for_job(
                        str(job_id),
                        status,
                        status_path,
                        int(plan["execution"]["timeout_seconds"]),
                    )
                except (RuntimeError, TimeoutError) as exc:
                    if attempts:
                        attempts[-1]["completed_at"] = now()
                        attempts[-1]["measurement_valid"] = False
                        attempts[-1]["validation_errors"] = [str(exc)]
                    record["last_error"] = str(exc)
                    record["state"] = "RETRYING_JOB"
                    record.pop("job_id", None)
                    status["updated_at"] = now()
                    write_json(status_path, status)
                    if len(attempts) >= max_attempts:
                        raise RuntimeError(
                            f"Case {item['benchmark_case_id']} exhausted {max_attempts} attempts: {exc}"
                        ) from exc
                    continue
            validation_errors = (
                measurement_validation_errors(
                    str(job_id), metrics, expected_controllers, expected_epochs
                )
                if strict_metrics
                else ([] if job_artifacts_complete(str(job_id), metrics) else ["incomplete artifacts"])
            )
            if not validation_errors:
                if attempts:
                    attempts[-1]["completed_at"] = now()
                    attempts[-1]["measurement_valid"] = True
                break
            if attempts:
                attempts[-1]["completed_at"] = now()
                attempts[-1]["measurement_valid"] = False
                attempts[-1]["validation_errors"] = validation_errors
            record["last_error"] = "; ".join(validation_errors)
            record["state"] = "RETRYING_MEASUREMENT"
            record.pop("job_id", None)
            status["updated_at"] = now()
            write_json(status_path, status)
        record.update({"state": "COMPLETED", "completed_at": now(), "job_id": job_id})
        status["completed_runs"] = index + 1
        status["updated_at"] = now()
        write_json(status_path, status)
        publish_partial_results(plan, run_dir, workspace, metrics, index + 1)
        time.sleep(int(plan["execution"].get("cooldown_seconds", 0)))
    result = finalize(request, plan, run_dir, workspace, metrics)
    status.update({"state": "COMPLETED", "completed_at": now(), "updated_at": now(), "result": result})
    status.pop("current_job_id", None)
    write_json(status_path, status)
    archive = archive_completed_run(plan, run_dir, status)
    if archive:
        result["long_term_archive"] = archive
        status["result"] = result
        status["updated_at"] = now()
        write_json(status_path, status)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--request", type=Path, required=True)
    args = parser.parse_args()
    try:
        orchestrate(args.request)
    except Exception as exc:
        request = json.loads(args.request.read_text(encoding="utf-8"))
        status_path = Path(request["run_dir"]) / "status.json"
        status = json.loads(status_path.read_text(encoding="utf-8")) if status_path.exists() else {}
        status.update({"state": "FAILED", "failed_at": now(), "updated_at": now(), "error": str(exc)})
        write_json(status_path, status)
        raise


if __name__ == "__main__":
    main()
