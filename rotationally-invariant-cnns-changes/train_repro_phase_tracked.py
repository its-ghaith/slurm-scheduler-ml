import argparse
import json
import logging
import os
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
    p.add_argument("--dataset", required=True, choices=["foci", "synthcells", "carpk"])
    p.add_argument("--model", required=True, choices=["yolov8", "yolov11", "cnn_base"])
    p.add_argument("--timeline-path", required=True)
    p.add_argument("--phase-metrics-dir", required=True)
    p.add_argument("--override", action="append", default=[])
    return p.parse_args()


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    args = parse_args()

    repo_root = Path(args.repo_root).resolve()
    src_root = repo_root / "src"
    _append_path(src_root)

    from countcv.core.experiment_utils import init_mlflow, set_seed
    from countcv.core.ultralytics_detectors import save_cropped_images
    from countcv.core.densitymap_regression import run_experiment_densitymap
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

    data_dir = Path(cfg["dataset"]["path"])
    data_yaml_path = Path(cfg["dataset"]["data_yaml"])
    phase_metrics_dir = Path(args.phase_metrics_dir)
    phase_metrics_dir.mkdir(parents=True, exist_ok=True)
    timeline_path = Path(args.timeline_path)

    LOGGER.info("Using config: dataset=%s model=%s experiment=%s", args.dataset, args.model, cfg["experiment"])

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

        run_dir = data_dir
        if cfg["model"].get("approach") == "objectdetection" and cfg["dataset"]["dataset_name"] == "foci":
            with tracked_phase(
                "preprocessing_crop",
                timeline_path,
                phase_metrics_dir / f"codecarbon_preprocessing_crop_seed_{seed}.csv",
            ):
                data_dir_cropped = data_dir / "cropped"
                save_cropped_images(data_dir=data_dir_cropped, num_worker=cfg["num_worker"])
                run_dir = data_dir_cropped

        if cfg["model"].get("approach") == "objectdetection":
            with tracked_phase("training", timeline_path, phase_metrics_dir / f"codecarbon_training_seed_{seed}.csv"):
                # Run YOLO with explicit workers to avoid shared-memory worker crashes on constrained pods.
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
                    model = YOLO(cfg["model"]["version"])
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
        elif cfg["model"].get("approach") == "densitymap":
            with tracked_phase("training", timeline_path, phase_metrics_dir / f"codecarbon_training_seed_{seed}.csv"):
                run_experiment_densitymap(
                    model_version=cfg["model"]["version"],
                    data_dir=run_dir,
                    experiment_name=cfg["experiment"],
                    model_params=flat_param_dict,
                    run_name=f"{cfg['model']['version']}",
                    auth=auth,
                    num_workers=cfg["num_worker"],
                )
        else:
            raise ValueError(f"Unknown model approach: {cfg['model'].get('approach')}")


if __name__ == "__main__":
    main()
