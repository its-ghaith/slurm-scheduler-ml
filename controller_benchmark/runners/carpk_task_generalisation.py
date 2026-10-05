from __future__ import annotations

import argparse
import json
import os
import random
from pathlib import Path

import mlflow
import numpy as np
import torch
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms

from .common import EpochTelemetry, decode_run_spec


def read_ids(path: Path) -> list[str]:
    return [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def count_vehicles(root: Path, image_id: str) -> int:
    path = root / "raw" / "labels" / f"{image_id}.txt"
    return sum(1 for line in path.read_text(encoding="utf-8").splitlines() if line.strip())


def split_ids(root: Path, split_seed: int, train_size: int) -> tuple[list[str], list[str]]:
    image_ids = read_ids(root / "raw" / "ImageSets" / "train.txt")
    random.Random(split_seed).shuffle(image_ids)
    validation_size = max(1, round(len(image_ids) * 0.2))
    validation, training = image_ids[:validation_size], image_ids[validation_size:]
    return (training[:train_size] if train_size > 0 else training), validation


IMAGE_TRANSFORM = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ]
)


class DensityClassificationDataset(Dataset):
    def __init__(self, root: Path, image_ids: list[str], thresholds: tuple[float, float]):
        self.root, self.image_ids, self.thresholds = root, image_ids, thresholds

    def __len__(self) -> int:
        return len(self.image_ids)

    def __getitem__(self, index: int):
        image_id = self.image_ids[index]
        image = Image.open(self.root / "raw" / "images" / f"{image_id}.png").convert("RGB")
        label = int(np.digitize(count_vehicles(self.root, image_id), self.thresholds, right=True))
        return IMAGE_TRANSFORM(image), torch.tensor(label, dtype=torch.long)


class BoxMaskSegmentationDataset(Dataset):
    def __init__(self, root: Path, image_ids: list[str]):
        self.root, self.image_ids = root, image_ids

    def __len__(self) -> int:
        return len(self.image_ids)

    def __getitem__(self, index: int):
        image_id = self.image_ids[index]
        image = Image.open(self.root / "raw" / "images" / f"{image_id}.png").convert("RGB")
        width, height = image.size
        mask = np.zeros((224, 224), dtype=np.float32)
        label_path = self.root / "raw" / "labels" / f"{image_id}.txt"
        for line in label_path.read_text(encoding="utf-8").splitlines():
            values = line.split()
            if len(values) < 4:
                continue
            x1, y1, x2, y2 = map(float, values[:4])
            left, right = int(224 * x1 / width), int(np.ceil(224 * x2 / width))
            top, bottom = int(224 * y1 / height), int(np.ceil(224 * y2 / height))
            mask[max(0, top):min(224, bottom), max(0, left):min(224, right)] = 1.0
        return IMAGE_TRANSFORM(image), torch.from_numpy(mask).unsqueeze(0)


class ConvBlock(nn.Sequential):
    def __init__(self, input_channels: int, output_channels: int):
        super().__init__(
            nn.Conv2d(input_channels, output_channels, 3, padding=1), nn.BatchNorm2d(output_channels), nn.ReLU(),
            nn.Conv2d(output_channels, output_channels, 3, padding=1), nn.BatchNorm2d(output_channels), nn.ReLU(),
        )


class TinyUNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder1, self.encoder2 = ConvBlock(3, 16), ConvBlock(16, 32)
        self.pool = nn.MaxPool2d(2)
        self.bottleneck = ConvBlock(32, 64)
        self.up2, self.decoder2 = nn.ConvTranspose2d(64, 32, 2, stride=2), ConvBlock(64, 32)
        self.up1, self.decoder1 = nn.ConvTranspose2d(32, 16, 2, stride=2), ConvBlock(32, 16)
        self.output = nn.Conv2d(16, 1, 1)

    def forward(self, inputs):
        enc1 = self.encoder1(inputs)
        enc2 = self.encoder2(self.pool(enc1))
        features = self.bottleneck(self.pool(enc2))
        features = self.decoder2(torch.cat((self.up2(features), enc2), dim=1))
        return self.output(self.decoder1(torch.cat((self.up1(features), enc1), dim=1)))


def gradient_norm(parameters) -> float:
    total = 0.0
    for parameter in parameters:
        if parameter.grad is None:
            continue
        norm = float(parameter.grad.detach().float().norm(2).cpu())
        total += norm * norm
    return float(total ** 0.5)


def model_parameter_count(model: nn.Module) -> int:
    return int(sum(parameter.numel() for parameter in model.parameters()))


def model_flops_estimate(task: str) -> float | None:
    # ResNet18 at 224x224 is a standard 1.8 GFLOP reference; TinyUNet is an
    # internal weak-segmentation baseline, so we keep a conservative estimate.
    if task == "image_classification":
        return 1.814e9
    if task == "semantic_segmentation":
        return 1.1e10
    return None


def classification_metrics(model, loader, device) -> tuple[float, float]:
    confusion = torch.zeros((3, 3), dtype=torch.float64)
    model.eval()
    with torch.no_grad():
        for images, targets in loader:
            predictions = model(images.to(device)).argmax(dim=1).cpu()
            for target, prediction in zip(targets, predictions):
                confusion[int(target), int(prediction)] += 1
    accuracy = float(confusion.diag().sum() / confusion.sum().clamp_min(1))
    f1_values = []
    for class_id in range(3):
        tp = confusion[class_id, class_id]
        fp = confusion[:, class_id].sum() - tp
        fn = confusion[class_id, :].sum() - tp
        f1_values.append(float(2 * tp / (2 * tp + fp + fn).clamp_min(1)))
    return accuracy, float(np.mean(f1_values))


def segmentation_metrics(model, loader, device) -> tuple[float, float]:
    intersection = union = prediction_total = target_total = 0.0
    model.eval()
    with torch.no_grad():
        for images, targets in loader:
            predictions = torch.sigmoid(model(images.to(device))) >= 0.5
            targets = targets.to(device) >= 0.5
            intersection += float((predictions & targets).sum())
            union += float((predictions | targets).sum())
            prediction_total += float(predictions.sum())
            target_total += float(targets.sum())
    return intersection / max(1.0, union), 2.0 * intersection / max(1.0, prediction_total + target_total)


def run(spec: dict, root: Path, output_dir: Path) -> None:
    telemetry = EpochTelemetry(spec, output_dir)
    telemetry.start()
    seed = int(spec["training_seed"])
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    train_ids, validation_ids = split_ids(
        root, int(spec["execution"]["split_seed"]), int(spec["case"].get("train_size", 0))
    )
    task = spec["task_type"]
    batch_size = int(spec["execution"]["batch_size"])
    metadata: dict[str, object] = {}
    if task == "image_classification":
        thresholds = tuple(float(value) for value in np.quantile([count_vehicles(root, item) for item in train_ids], [1 / 3, 2 / 3]))
        train_dataset = DensityClassificationDataset(root, train_ids, thresholds)
        validation_dataset = DensityClassificationDataset(root, validation_ids, thresholds)
        model = models.resnet18(weights=None)
        model.fc = nn.Linear(model.fc.in_features, 3)
        loss_function = nn.CrossEntropyLoss()
        metadata["density_thresholds"] = list(thresholds)
    elif task == "semantic_segmentation":
        train_dataset = BoxMaskSegmentationDataset(root, train_ids)
        validation_dataset = BoxMaskSegmentationDataset(root, validation_ids)
        model = TinyUNet()
        loss_function = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([8.0]))
        metadata["mask_provenance"] = "binary masks derived from CARPK bounding boxes"
    else:
        raise ValueError(f"Unsupported CARPK task: {task}")
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    validation_loader = DataLoader(validation_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    loss_function.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    parameters_count = model_parameter_count(model)
    flops_estimate = model_flops_estimate(task)
    model_dir = Path("/workspace/model_registry") / f"job_{telemetry.job_id}"
    model_dir.mkdir(parents=True, exist_ok=True)
    best_quality = -1.0
    mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI", "http://mlflow:5000"))
    mlflow.set_experiment(spec["execution"]["experiment_name"])
    try:
        with mlflow.start_run(run_name=f"job-energy-{telemetry.job_id}"):
            mlflow.log_params({
                "runner": spec["runner"], "task_type": task, "quality_metric": "quality_score",
                "controller_id": spec["controller_id"], "training_seed": seed,
                "benchmark_case_id": spec["benchmark_case_id"], "train_images": len(train_ids),
                "validation_images": len(validation_ids), **metadata,
            })
            for epoch in range(1, int(spec["execution"]["max_epochs"]) + 1):
                telemetry.start_epoch()
                model.train()
                losses = []
                gradient_norms = []
                for images, targets in train_loader:
                    optimizer.zero_grad(set_to_none=True)
                    output = model(images.to(device))
                    loss = loss_function(output, targets.to(device))
                    loss.backward()
                    gradient_norms.append(gradient_norm(model.parameters()))
                    optimizer.step()
                    losses.append(float(loss.detach().cpu()))
                if task == "image_classification":
                    accuracy, macro_f1 = classification_metrics(model, validation_loader, device)
                    quality, metrics = macro_f1, {"accuracy": accuracy, "macro_f1": macro_f1}
                else:
                    miou, dice = segmentation_metrics(model, validation_loader, device)
                    quality, metrics = miou, {"miou": miou, "dice": dice}
                metrics["train_loss"] = float(np.mean(losses))
                metrics["gradient_norm"] = float(np.mean(gradient_norms)) if gradient_norms else 0.0
                metrics["learning_rate"] = float(optimizer.param_groups[0]["lr"])
                metrics["model_parameter_count"] = parameters_count
                if flops_estimate is not None:
                    metrics["model_flops"] = flops_estimate
                if quality > best_quality:
                    best_quality = quality
                    torch.save(model.state_dict(), model_dir / "best.pt")
                mlflow.log_metrics({f"epoch/{key}": value for key, value in metrics.items()}, step=epoch)
                if telemetry.finish_epoch(epoch, quality, {"quality_score": quality, **metrics}):
                    break
            summary, gpu = telemetry.finalize()
            summary.update(metadata)
            output_dir.joinpath(f"epoch_summary_job_{telemetry.job_id}.json").write_text(
                json.dumps(summary, indent=2) + "\n", encoding="utf-8"
            )
            mlflow.log_metrics({
                "final_quality_score": summary["final_quality_score"],
                "best_quality_score": summary["best_quality_score"],
                "gpu_energy_wh": gpu["gpu_energy_kwh"] * 1000.0,
            })
            mlflow.log_artifact(str(telemetry.epoch_summary_path), artifact_path="energy")
            mlflow.log_artifact(str(telemetry.gpu_summary_path), artifact_path="energy")
            mlflow.log_artifact(str(model_dir / "best.pt"), artifact_path="model")
            print(json.dumps({"summary": summary, "gpu": gpu}))
    finally:
        telemetry.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-spec-base64", required=True)
    parser.add_argument("--dataset-root", type=Path, default=Path("/workspace-cache/carpk"))
    parser.add_argument("--output-dir", type=Path, default=Path("/workspace/energy_metrics"))
    args = parser.parse_args()
    run(decode_run_spec(args.run_spec_base64), args.dataset_root, args.output_dir)


if __name__ == "__main__":
    main()
