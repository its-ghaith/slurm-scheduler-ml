from __future__ import annotations

import math
from pathlib import Path

import yaml
from ultralytics import YOLO


def _dataset_root(data_yaml: Path, configuration: dict) -> Path:
    configured = configuration.get("path")
    if configured:
        root = Path(str(configured))
        return root if root.is_absolute() else (data_yaml.parent / root).resolve()
    return data_yaml.parent.resolve()


def evaluate_count_metrics(
    weights: Path,
    data_yaml: Path,
    *,
    split: str = "test",
    image_size: int = 640,
    confidence: float = 0.25,
) -> dict[str, float]:
    configuration = yaml.safe_load(data_yaml.read_text(encoding="utf-8"))
    root = _dataset_root(data_yaml, configuration)
    split_value = configuration.get(split) or configuration.get("val")
    if not isinstance(split_value, str):
        raise ValueError(f"Count evaluation requires a directory-based {split} split.")
    image_root = root / split_value
    if not image_root.exists():
        raise FileNotFoundError(image_root)
    errors: list[float] = []
    model = YOLO(str(weights))
    predictions = model.predict(
        source=str(image_root),
        imgsz=image_size,
        conf=confidence,
        stream=True,
        verbose=False,
    )
    for result in predictions:
        image_path = Path(result.path)
        try:
            relative = image_path.relative_to(root / "images")
            label_path = root / "labels" / relative.with_suffix(".txt")
        except ValueError:
            label_path = root / "labels" / split / f"{image_path.stem}.txt"
        ground_truth = (
            sum(1 for line in label_path.read_text(encoding="utf-8").splitlines() if line.strip())
            if label_path.exists()
            else 0
        )
        predicted = len(result.boxes)
        errors.append(float(predicted - ground_truth))
    if not errors:
        raise RuntimeError(f"No images were evaluated for count metrics under {image_root}")
    absolute = [abs(value) for value in errors]
    return {
        "test_count_mae": sum(absolute) / len(absolute),
        "test_count_rmse": math.sqrt(sum(value * value for value in errors) / len(errors)),
        "test_count_bias": sum(errors) / len(errors),
        "test_count_images": float(len(errors)),
        "test_count_confidence": confidence,
    }


def evaluate_detection_quality(
    weights: Path,
    data_yaml: Path,
    *,
    image_size: int = 640,
    batch_size: int = 8,
) -> dict[str, float]:
    result = YOLO(str(weights)).val(
        data=str(data_yaml),
        split="test",
        imgsz=image_size,
        batch=batch_size,
        workers=0,
        cache=False,
        verbose=False,
    )
    box = result.box
    return {
        "test_best_map50": float(box.map50),
        "test_best_map50_95": float(box.map),
        "test_best_precision": float(box.mp),
        "test_best_recall": float(box.mr),
    }
