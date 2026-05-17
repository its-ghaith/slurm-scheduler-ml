import csv
import json
import os
from pathlib import Path

import mlflow
import yaml
from ultralytics import YOLO


def _ensure(path: Path):
    path.mkdir(parents=True, exist_ok=True)
    return path


def run_preprocessing(dataset_root: Path, artifacts_dir: Path) -> Path:
    images_train = dataset_root / "images" / "train"
    images_val = dataset_root / "images" / "val"
    labels_train = dataset_root / "labels" / "train"
    labels_val = dataset_root / "labels" / "val"
    for p in (images_train, images_val, labels_train, labels_val):
        if not p.exists():
            raise FileNotFoundError(f"Missing dataset path: {p}")

    ds_dir = _ensure(artifacts_dir / "dataset")
    yaml_path = ds_dir / "carpk_yolo.yaml"
    ds_cfg = {
        "path": str(dataset_root.resolve()),
        "train": "images/train",
        "val": "images/val",
        "names": {0: "car"},
        "nc": 1,
    }
    with yaml_path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(ds_cfg, f, sort_keys=False)
    return yaml_path


def run_hyperparameter_tuning(model_name: str, data_yaml: Path, project_dir: Path, device: str) -> dict:
    candidates = [
        {"lr0": 0.01, "imgsz": 640, "batch": 16},
        {"lr0": 0.005, "imgsz": 640, "batch": 24},
        {"lr0": 0.003, "imgsz": 960, "batch": 12},
    ]
    trials_dir = _ensure(project_dir / "tuning_trials")
    best = {"score": -1.0}
    for i, c in enumerate(candidates, start=1):
        model = YOLO(model_name)
        run_name = f"tune_{i}"
        result = model.train(
            data=str(data_yaml),
            epochs=int(os.environ.get("TUNE_EPOCHS", "5")),
            imgsz=c["imgsz"],
            batch=c["batch"],
            lr0=c["lr0"],
            project=str(trials_dir),
            name=run_name,
            device=device,
            val=True,
            save=True,
            verbose=False,
        )
        metrics = getattr(result, "results_dict", {}) or {}
        score = float(metrics.get("metrics/mAP50(B)", 0.0))
        if score > best["score"]:
            best = {"score": score, **c}

    best_path = project_dir / "best_hparams.json"
    with best_path.open("w", encoding="utf-8") as f:
        json.dump(best, f, indent=2)
    return best


def run_distributed_training(model_name: str, data_yaml: Path, project_dir: Path, device: str, hparams: dict) -> Path:
    model = YOLO(model_name)
    run_name = os.environ.get("TRAIN_RUN_NAME", "carpk_yolo_distributed")
    train_epochs = int(os.environ.get("EPOCHS", "50"))
    result = model.train(
        data=str(data_yaml),
        epochs=train_epochs,
        imgsz=int(hparams.get("imgsz", os.environ.get("IMGSZ", "960"))),
        batch=int(hparams.get("batch", os.environ.get("BATCH_SIZE", "16"))),
        lr0=float(hparams.get("lr0", os.environ.get("LR0", "0.005"))),
        optimizer=os.environ.get("OPTIMIZER", "AdamW"),
        device=device,
        project=str(project_dir),
        name=run_name,
        val=True,
        save=True,
    )
    save_dir = Path(getattr(result, "save_dir"))
    return save_dir / "weights" / "best.pt"


def run_evaluation(best_model: Path, data_yaml: Path, device: str) -> dict:
    model = YOLO(str(best_model))
    val_result = model.val(data=str(data_yaml), device=device, split="val", verbose=False)
    metrics = getattr(val_result, "results_dict", {}) or {}
    return {
        "map50": float(metrics.get("metrics/mAP50(B)", 0.0)),
        "map50_95": float(metrics.get("metrics/mAP50-95(B)", 0.0)),
        "precision": float(metrics.get("metrics/precision(B)", 0.0)),
        "recall": float(metrics.get("metrics/recall(B)", 0.0)),
    }


def run_counting_inference(best_model: Path, input_dir: Path, out_csv: Path, conf: float = 0.25):
    model = YOLO(str(best_model))
    images = sorted([p for p in input_dir.glob("*") if p.suffix.lower() in {".jpg", ".jpeg", ".png"}])
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["image", "predicted_car_count"])
        for img in images:
            results = model.predict(source=str(img), conf=conf, verbose=False)
            count = int(len(results[0].boxes)) if results else 0
            w.writerow([img.name, count])


def log_mlflow_summary(params: dict, eval_metrics: dict):
    for k, v in params.items():
        mlflow.log_param(k, v)
    for k, v in eval_metrics.items():
        mlflow.log_metric(k, float(v))

