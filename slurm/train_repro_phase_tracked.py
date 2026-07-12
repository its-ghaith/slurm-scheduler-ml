import argparse
import json
import logging
import os
import shutil
import sys
import time
from contextlib import contextmanager
from pathlib import Path

import mlflow
import yaml
from codecarbon import OfflineEmissionsTracker
from ultralytics import YOLO


LOGGER = logging.getLogger("repro_rotacnn")


def _append_path(path: Path):
    s = str(path.resolve())
    if s not in sys.path:
        sys.path.insert(0, s)


def _load_yaml(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _coerce_value(raw: str):
    v = raw.strip()
    if v.lower() in {"true", "false"}:
        return v.lower() == "true"
    if v.lower() in {"none", "null"}:
        return None
    if "," in v:
        return [_coerce_value(x) for x in v.split(",")]
    try:
        if "." in v:
            return float(v)
        return int(v)
    except ValueError:
        return v


def _apply_overrides(cfg: dict, overrides: list[str]):
    for item in overrides:
        if "=" not in item:
            raise ValueError(f"Invalid override '{item}'. Use key=value.")
        key, raw = item.split("=", 1)
        value = _coerce_value(raw)
        ref = cfg
        parts = key.split(".")
        for p in parts[:-1]:
            if p not in ref or not isinstance(ref[p], dict):
                ref[p] = {}
            ref = ref[p]
        ref[parts[-1]] = value


def _write_json(path: Path, payload: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(payload) + "\n")


def _export_job_model_artifacts(
    *,
    model,
    model_name: str,
    model_version: str,
    dataset_name: str,
    job_id: str,
    run_id: str | None,
):
    trainer = getattr(model, "trainer", None)
    if trainer is None:
        LOGGER.warning("Kein Trainer am YOLO-Modell gefunden. Ueberspringe Modellexport fuer Job %s.", job_id)
        return

    best_path_raw = getattr(trainer, "best", None)
    save_dir_raw = getattr(trainer, "save_dir", None)
    if not best_path_raw:
        LOGGER.warning("Keine best.pt fuer Job %s gefunden. Ueberspringe Modellexport.", job_id)
        return

    best_path = Path(best_path_raw)
    if not best_path.exists():
        LOGGER.warning("best.pt existiert nicht fuer Job %s unter %s.", job_id, best_path)
        return

    save_dir = Path(save_dir_raw) if save_dir_raw else best_path.parent.parent
    registry_root = Path(os.environ.get("JOB_MODEL_REGISTRY_DIR", "/workspace-cache/model_registry"))
    target_dir = registry_root / f"job_{job_id}"
    target_dir.mkdir(parents=True, exist_ok=True)

    shutil.copy2(best_path, target_dir / "best.pt")

    last_path = best_path.parent / "last.pt"
    if last_path.exists():
        shutil.copy2(last_path, target_dir / "last.pt")

    results_csv = save_dir / "results.csv"
    if results_csv.exists():
        shutil.copy2(results_csv, target_dir / "results.csv")

    args_yaml = save_dir / "args.yaml"
    if args_yaml.exists():
        shutil.copy2(args_yaml, target_dir / "args.yaml")

    metadata = {
        "job_id": job_id,
        "run_id": run_id,
        "dataset": dataset_name,
        "model": model_name,
        "model_version": model_version,
        "best_weights_path": str(target_dir / "best.pt"),
        "exported_at_epoch": int(getattr(trainer, "epoch", -1)) + 1,
        "save_dir": str(save_dir),
    }
    (target_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    LOGGER.info("Job-Modell exportiert nach %s", target_dir)


@contextmanager
def tracked_phase(name: str, timeline_path: Path, metrics_path: Path, country_iso: str = "DEU"):
    start = time.time()
    _write_json(timeline_path, {"phase": name, "event": "start", "ts": start})
    tracker = OfflineEmissionsTracker(
        log_level="error",
        output_dir=str(metrics_path.parent),
        output_file=metrics_path.name,
        country_iso_code=country_iso,
    )
    tracker.start()
    try:
        yield
    finally:
        tracker.stop()
        end = time.time()
        _write_json(
            timeline_path,
            {
                "phase": name,
                "event": "end",
                "ts": end,
                "duration_seconds": round(end - start, 6),
                "codecarbon_energy_kwh": float(getattr(tracker._total_energy, "kWh", 0.0)),
                "codecarbon_gpu_energy_kwh": float(getattr(tracker._total_gpu_energy, "kWh", 0.0)),
                "codecarbon_cpu_energy_kwh": float(getattr(tracker._total_cpu_energy, "kWh", 0.0)),
            },
        )


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--repo-root", default="/workspace/rotationally-invariant-cnns")
    p.add_argument("--dataset", required=True, choices=["carpk"])
    p.add_argument("--model", required=True, choices=["yolov8", "yolov11"])
    p.add_argument("--timeline-path", required=True)
    p.add_argument("--phase-metrics-dir", required=True)
    p.add_argument("--job-metrics-path", required=True)
    p.add_argument("--gpu-csv", default=None)
    p.add_argument("--epoch-timeline-path", default=None)
    p.add_argument("--epoch-summary-path", default=None)
    p.add_argument("--price-eur-kwh", type=float, default=0.30)
    p.add_argument("--co2-kg-kwh", type=float, default=0.4)
    p.add_argument("--pue-factor", type=float, default=1.0)
    p.add_argument("--adaptive-enabled", action="store_true")
    p.add_argument("--controller-mode", choices=["none", "delta_mape", "uncertainty_aware"], default="none")
    p.add_argument("--comparison-strategy", default="unspecified")
    p.add_argument("--adaptive-monitor-metric", choices=["map50", "map50_95"], default="map50")
    p.add_argument("--adaptive-min-epochs", type=int, default=20)
    p.add_argument("--adaptive-patience", type=int, default=3)
    p.add_argument("--adaptive-smoothing-window", type=int, default=3)
    p.add_argument("--adaptive-min-delta-map50", type=float, default=0.001)
    p.add_argument("--adaptive-min-mape-map50-per-wh", type=float, default=0.0001)
    p.add_argument("--uncertainty-target-epoch", type=int, default=100)
    p.add_argument("--uncertainty-epsilon", type=float, default=0.01)
    p.add_argument("--uncertainty-alpha", type=float, default=0.05)
    p.add_argument("--uncertainty-bootstrap-samples", type=int, default=24)
    p.add_argument("--uncertainty-min-fit-points", type=int, default=8)
    p.add_argument("--override", action="append", default=[])
    return p.parse_args()


def _log_epoch_summary_metrics(epoch_summary: dict):
    mlflow.log_metric("epoch_count_completed", int(epoch_summary.get("epochs_completed", 0)))
    mlflow.log_metric("energy_adaptive_enabled", int(bool(epoch_summary.get("adaptive_enabled"))))
    mlflow.log_metric("energy_adaptive_stopped", int(bool(epoch_summary.get("adaptive_stopped"))))
    for key in (
        "final_map50",
        "final_map50_95",
        "final_precision",
        "final_recall",
        "total_energy_kwh",
        "total_gpu_energy_kwh",
        "total_duration_seconds",
        "estimated_electricity_cost_eur",
        "estimated_co2_kg",
        "best_mape_map50_per_wh",
        "best_mape_map50_95_per_wh",
        "predicted_final_metric_mean",
        "predicted_final_metric_lower",
        "predicted_final_metric_upper",
        "predicted_remaining_gain_upper",
        "predicted_prob_gain_gt_epsilon",
    ):
        value = epoch_summary.get(key)
        if value is not None:
            mlflow.log_metric(key, float(value))
    if epoch_summary.get("stop_epoch") is not None:
        mlflow.log_metric("energy_adaptive_stop_epoch", float(epoch_summary["stop_epoch"]))
    if epoch_summary.get("adaptive_monitor_metric"):
        mlflow.set_tag("energy_adaptive_monitor_metric", str(epoch_summary["adaptive_monitor_metric"]))
    if epoch_summary.get("controller_mode"):
        mlflow.set_tag("controller_mode", str(epoch_summary["controller_mode"]))
    if epoch_summary.get("comparison_strategy"):
        mlflow.set_tag("comparison_strategy", str(epoch_summary["comparison_strategy"]))
    if epoch_summary.get("stop_reason"):
        mlflow.set_tag("controller_stop_reason", str(epoch_summary["stop_reason"]))


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    args = parse_args()

    project_root = Path(__file__).resolve().parent.parent
    repo_root = Path(args.repo_root).resolve()
    src_root = repo_root / "src"
    _append_path(project_root)
    _append_path(src_root)

    from slurm.epoch_energy_controller import (
        EnergyAdaptiveConfig,
        EpochEnergyAdaptiveController,
        build_epoch_summary,
    )
    from countcv.core.experiment_utils import init_mlflow, set_seed
    from countcv.effcv.run_experiment import get_data_splitter, get_label_creator

    default_cfg = _load_yaml(src_root / "config" / "default.yaml")
    model_cfg = _load_yaml(src_root / "config" / "model" / f"{args.model}.yaml")
    dataset_cfg = _load_yaml(src_root / "config" / "dataset" / f"{args.dataset}.yaml")

    cfg = {
        "experiment": default_cfg.get("experiment", "resource-efficient-cv"),
        "remote_mlflow": bool(default_cfg.get("remote_mlflow", False)),
        "num_worker": int(default_cfg.get("num_worker", 4)),
        "seeds": list(default_cfg.get("seeds", [0])),
        "dataset": dataset_cfg,
        "model": model_cfg,
    }
    _apply_overrides(cfg, args.override)
    tracking_uri = os.environ.get("MLFLOW_TRACKING_URI", "").strip()
    use_direct_tracking_uri = tracking_uri.startswith("http://") or tracking_uri.startswith("https://")
    if not isinstance(cfg["seeds"], list):
        cfg["seeds"] = [cfg["seeds"]]

    if len(cfg["seeds"]) > 1 and args.epoch_timeline_path:
        LOGGER.warning("Epoch-level energy tracking currently assumes a single seed per job. Using shared epoch files.")

    data_dir = Path(cfg["dataset"]["path"])
    data_yaml_path = Path(cfg["dataset"]["data_yaml"])
    phase_metrics_dir = Path(args.phase_metrics_dir)
    phase_metrics_dir.mkdir(parents=True, exist_ok=True)
    timeline_path = Path(args.timeline_path)
    job_metrics_path = Path(args.job_metrics_path)
    epoch_timeline_path = Path(args.epoch_timeline_path) if args.epoch_timeline_path else None
    epoch_summary_path = Path(args.epoch_summary_path) if args.epoch_summary_path else None
    adaptive_config = EnergyAdaptiveConfig(
        enabled=args.adaptive_enabled or args.controller_mode != "none",
        controller_mode=("delta_mape" if args.adaptive_enabled and args.controller_mode == "none" else args.controller_mode),
        comparison_strategy=args.comparison_strategy,
        monitor_metric=args.adaptive_monitor_metric,
        min_epochs=args.adaptive_min_epochs,
        patience=args.adaptive_patience,
        smoothing_window=args.adaptive_smoothing_window,
        min_delta_map50=args.adaptive_min_delta_map50,
        min_mape_map50_per_wh=args.adaptive_min_mape_map50_per_wh,
        uncertainty_target_epoch=args.uncertainty_target_epoch,
        uncertainty_epsilon=args.uncertainty_epsilon,
        uncertainty_alpha=args.uncertainty_alpha,
        uncertainty_bootstrap_samples=args.uncertainty_bootstrap_samples,
        uncertainty_min_fit_points=args.uncertainty_min_fit_points,
    )

    LOGGER.info("Using config: dataset=%s model=%s experiment=%s", args.dataset, args.model, cfg["experiment"])
    LOGGER.info(
        "Epoch energy tracking: enabled=%s controller_mode=%s comparison_strategy=%s",
        bool(epoch_timeline_path and args.gpu_csv),
        adaptive_config.controller_mode,
        args.comparison_strategy,
    )

    job_start = time.time()
    job_tracker = OfflineEmissionsTracker(
        log_level="error",
        output_dir=str(phase_metrics_dir),
        output_file="codecarbon_job_total.csv",
        country_iso_code="DEU",
    )
    job_tracker.start()
    try:
        current_job_id = os.environ.get("SLURM_JOB_ID", "unknown")
        with tracked_phase("preprocessing_labels", timeline_path, phase_metrics_dir / "codecarbon_preprocessing_labels.csv"):
            creator = get_label_creator(dataset=cfg["dataset"]["dataset_name"], dataset_root=data_dir)
            creator.create_labels()

        for seed in cfg["seeds"]:
            set_seed(seed)
            flat_param_dict = {
                **dict(cfg["dataset"]),
                **dict(cfg["model"]),
                "seed": seed,
                "num_worker": cfg["num_worker"],
            }
            if use_direct_tracking_uri:
                mlflow.set_tracking_uri(tracking_uri)
                auth = None
            else:
                auth = init_mlflow(cfg["remote_mlflow"])

            with tracked_phase(
                "preprocessing_split",
                timeline_path,
                phase_metrics_dir / f"codecarbon_preprocessing_split_seed_{seed}.csv",
            ):
                splitter = get_data_splitter(dataset=cfg["dataset"]["dataset_name"], dataset_root=data_dir)
                splitter.split(train_size=0.8, seed=seed, limit_train_size=flat_param_dict.get("train_size", False))

            if cfg["model"].get("approach") == "objectdetection":
                with tracked_phase("training", timeline_path, phase_metrics_dir / f"codecarbon_training_seed_{seed}.csv"):
                    mlflow.set_experiment(cfg["experiment"])
                    base_run_name = os.environ.get("MLFLOW_RUN_NAME", f"{cfg['model']['version']}")
                    run_name = base_run_name if len(cfg["seeds"]) == 1 else f"{base_run_name}-seed-{seed}"
                    with mlflow.start_run(run_name=run_name):
                        run_id = mlflow.active_run().info.run_id
                        run_id_file = os.environ.get("MLFLOW_RUN_ID_FILE")
                        if run_id_file:
                            Path(run_id_file).write_text(run_id, encoding="utf-8")
                        for k, v in flat_param_dict.items():
                            mlflow.log_param(k, v)
                        mlflow.log_param("adaptive_enabled", args.adaptive_enabled)
                        mlflow.log_param("controller_mode", adaptive_config.controller_mode)
                        mlflow.log_param("comparison_strategy", args.comparison_strategy)
                        mlflow.log_param("adaptive_monitor_metric", args.adaptive_monitor_metric)
                        mlflow.log_param("adaptive_min_epochs", args.adaptive_min_epochs)
                        mlflow.log_param("adaptive_patience", args.adaptive_patience)
                        mlflow.log_param("adaptive_smoothing_window", args.adaptive_smoothing_window)
                        mlflow.log_param("adaptive_min_delta_map50", args.adaptive_min_delta_map50)
                        mlflow.log_param("adaptive_min_mape_map50_per_wh", args.adaptive_min_mape_map50_per_wh)
                        mlflow.log_param("uncertainty_target_epoch", args.uncertainty_target_epoch)
                        mlflow.log_param("uncertainty_epsilon", args.uncertainty_epsilon)
                        mlflow.log_param("uncertainty_alpha", args.uncertainty_alpha)
                        mlflow.log_param("uncertainty_bootstrap_samples", args.uncertainty_bootstrap_samples)
                        mlflow.log_param("uncertainty_min_fit_points", args.uncertainty_min_fit_points)

                        model = YOLO(cfg["model"]["version"])
                        if epoch_timeline_path and args.gpu_csv:
                            controller = EpochEnergyAdaptiveController(
                                gpu_csv_path=args.gpu_csv,
                                epoch_timeline_path=epoch_timeline_path,
                                pue_factor=args.pue_factor,
                                price_eur_kwh=args.price_eur_kwh,
                                co2_kg_kwh=args.co2_kg_kwh,
                                config=adaptive_config,
                            )
                            model.add_callback("on_train_epoch_start", controller.on_train_epoch_start)
                            model.add_callback("on_fit_epoch_end", controller.on_fit_epoch_end)

                        model.train(
                            data=str(data_yaml_path),
                            epochs=int(flat_param_dict.get("epochs", 100)),
                            freeze=int(flat_param_dict.get("freeze", 0)),
                            imgsz=flat_param_dict.get("img_size", 640),
                            batch=int(flat_param_dict.get("batch_size", 8)),
                            patience=int(flat_param_dict.get("patience", 8)),
                            pretrained=bool(flat_param_dict.get("pretrained", True)),
                            single_cls=True,
                            cache="ram",
                            hsv_h=float(flat_param_dict.get("hsv_h", 0.015)),
                            hsv_s=float(flat_param_dict.get("hsv_s", 0.12)),
                            hsv_v=float(flat_param_dict.get("hsv_v", 0.2)),
                            translate=float(flat_param_dict.get("translate", 0.12)),
                            scale=float(flat_param_dict.get("scale", 0.5)),
                            degrees=float(flat_param_dict.get("degrees", 0.0)),
                            shear=float(flat_param_dict.get("shear", 0.0)),
                            mosaic=float(flat_param_dict.get("mosaic", 0.0)),
                            fliplr=float(flat_param_dict.get("fliplr", 0.0)),
                            flipud=float(flat_param_dict.get("flipud", 0.0)),
                            mixup=0.0,
                            workers=0,
                        )

                        _export_job_model_artifacts(
                            model=model,
                            model_name=cfg["model"].get("model_name", args.model),
                            model_version=str(cfg["model"]["version"]),
                            dataset_name=cfg["dataset"]["dataset_name"],
                            job_id=current_job_id,
                            run_id=run_id,
                        )

                        if epoch_timeline_path and epoch_summary_path:
                            epoch_summary = build_epoch_summary(
                                epoch_timeline_path=epoch_timeline_path,
                                output_path=epoch_summary_path,
                                job_id=os.environ.get("SLURM_JOB_ID", "unknown"),
                                price_eur_kwh=args.price_eur_kwh,
                                co2_kg_kwh=args.co2_kg_kwh,
                                adaptive_enabled=adaptive_config.enabled,
                                adaptive_monitor_metric=args.adaptive_monitor_metric,
                                controller_mode=adaptive_config.controller_mode,
                                comparison_strategy=args.comparison_strategy,
                            )
                            _log_epoch_summary_metrics(epoch_summary)
                            if not auth:
                                mlflow.log_artifact(str(epoch_timeline_path), artifact_path="energy")
                                mlflow.log_artifact(str(epoch_summary_path), artifact_path="energy")
            else:
                raise ValueError("Only object detection models are supported in the CARPK-only workflow.")
    finally:
        job_tracker.stop()
        job_end = time.time()
        payload = {
            "start_ts": job_start,
            "end_ts": job_end,
            "duration_seconds": round(job_end - job_start, 6),
            "codecarbon_energy_kwh": float(getattr(job_tracker._total_energy, "kWh", 0.0)),
            "codecarbon_gpu_energy_kwh": float(getattr(job_tracker._total_gpu_energy, "kWh", 0.0)),
            "codecarbon_cpu_energy_kwh": float(getattr(job_tracker._total_cpu_energy, "kWh", 0.0)),
        }
        job_metrics_path.parent.mkdir(parents=True, exist_ok=True)
        job_metrics_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
