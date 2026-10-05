from __future__ import annotations

import argparse
import json
import os
import random
from pathlib import Path

import mlflow
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import models, transforms
from torchvision.datasets import CIFAR10, CIFAR100, ImageFolder

from .common import EpochTelemetry, decode_run_spec
from .classification_metrics import primary_quality


DATASET_ROOT = Path("/workspace-cache/controller-datasets-v1")


def stratified_split(targets: list[int], validation_fraction: float, seed: int) -> tuple[list[int], list[int]]:
    by_class: dict[int, list[int]] = {}
    for index, target in enumerate(targets):
        by_class.setdefault(int(target), []).append(index)
    generator = random.Random(seed)
    training: list[int] = []
    validation: list[int] = []
    for indexes in by_class.values():
        generator.shuffle(indexes)
        size = max(1, round(len(indexes) * validation_fraction))
        validation.extend(indexes[:size])
        training.extend(indexes[size:])
    generator.shuffle(training)
    generator.shuffle(validation)
    return training, validation


class TransformSubset(Dataset):
    def __init__(self, dataset: Dataset, indexes: list[int], transform):
        self.dataset = dataset
        self.indexes = indexes
        self.transform = transform

    def __len__(self) -> int:
        return len(self.indexes)

    def __getitem__(self, index: int):
        image, target = self.dataset[self.indexes[index]]
        return self.transform(image), int(target)


def dataset_bundle(dataset_key: str, split_seed: int):
    if dataset_key in {"cifar10", "cifar100"}:
        dataset_class = CIFAR10 if dataset_key == "cifar10" else CIFAR100
        root = DATASET_ROOT / dataset_key
        raw_train = dataset_class(root, train=True, download=False)
        raw_test = dataset_class(root, train=False, download=False)
        size = 32
        normalization = (
            ((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))
            if dataset_key == "cifar10"
            else ((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
        )
        train_transform = transforms.Compose(
            [
                transforms.RandomCrop(size, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(*normalization),
            ]
        )
        evaluation_transform = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize(*normalization)]
        )
        train_indexes, validation_indexes = stratified_split(raw_train.targets, 0.1, split_seed)
        training = TransformSubset(raw_train, train_indexes, train_transform)
        validation = TransformSubset(raw_train, validation_indexes, evaluation_transform)
        testing = TransformSubset(raw_test, list(range(len(raw_test))), evaluation_transform)
        return training, validation, testing
    if dataset_key == "tiny-imagenet":
        root = DATASET_ROOT / dataset_key / "tiny-imagenet-200"
        raw_train = ImageFolder(root / "train")
        raw_test = ImageFolder(root / "val-classified")
        normalization = ((0.4802, 0.4481, 0.3975), (0.2302, 0.2265, 0.2262))
        train_transform = transforms.Compose(
            [
                transforms.RandomCrop(64, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(*normalization),
            ]
        )
        evaluation_transform = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize(*normalization)]
        )
        train_indexes, validation_indexes = stratified_split(raw_train.targets, 0.1, split_seed)
        training = TransformSubset(raw_train, train_indexes, train_transform)
        validation = TransformSubset(raw_train, validation_indexes, evaluation_transform)
        testing = TransformSubset(raw_test, list(range(len(raw_test))), evaluation_transform)
        return training, validation, testing
    raise ValueError(f"Unsupported classification dataset: {dataset_key}")


def build_model(num_classes: int, *, pretrained: bool = False) -> nn.Module:
    weights = models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
    model = models.resnet18(weights=weights)
    imagenet_stem = model.conv1.weight.detach().clone() if pretrained else None
    model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    if imagenet_stem is not None:
        # Keep the established small-image architecture while transferring the
        # center of the ImageNet stem together with all residual-layer weights.
        with torch.no_grad():
            model.conv1.weight.copy_(imagenet_stem[:, :, 2:5, 2:5])
    model.maxpool = nn.Identity()
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


def confusion_metrics(model: nn.Module, loader: DataLoader, device: torch.device, classes: int) -> tuple[float, float]:
    confusion = torch.zeros((classes, classes), dtype=torch.float64)
    model.eval()
    with torch.no_grad():
        for images, targets in loader:
            predictions = model(images.to(device, non_blocking=True)).argmax(dim=1).cpu()
            encoded = targets.to(torch.int64) * classes + predictions.to(torch.int64)
            confusion += torch.bincount(encoded, minlength=classes * classes).reshape(classes, classes)
    true_positive = confusion.diag()
    accuracy = float(true_positive.sum() / confusion.sum().clamp_min(1))
    false_positive = confusion.sum(dim=0) - true_positive
    false_negative = confusion.sum(dim=1) - true_positive
    macro_f1 = float((2 * true_positive / (2 * true_positive + false_positive + false_negative).clamp_min(1)).mean())
    return accuracy, macro_f1


def gradient_norm(parameters) -> float:
    squared = 0.0
    for parameter in parameters:
        if parameter.grad is not None:
            value = float(parameter.grad.detach().float().norm(2).cpu())
            squared += value * value
    return squared**0.5


def run(spec: dict, output_dir: Path) -> None:
    telemetry = EpochTelemetry(spec, output_dir)
    telemetry.start()
    seed = int(spec["training_seed"])
    split_seed = int(spec["execution"]["split_seed"])
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    case = spec["case"]
    pretrained = bool(case.get("pretrained", False))
    primary_metric = str(case.get("primary_metric", "accuracy"))
    primary_quality(primary_metric, 0.0, 0.0)
    dataset_key = case["dataset_key"]
    classes = int(case["num_classes"])
    training, validation, testing = dataset_bundle(dataset_key, split_seed)
    batch_size = int(case.get("batch_size", spec["execution"]["batch_size"]))
    num_workers = int(case.get("workers", spec["execution"].get("classification_workers", 0)))
    loader_options = {
        "batch_size": batch_size,
        "num_workers": num_workers,
        "pin_memory": True,
        "persistent_workers": num_workers > 0,
    }
    generator = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(training, shuffle=True, generator=generator, **loader_options)
    validation_loader = DataLoader(validation, shuffle=False, **loader_options)
    test_loader = DataLoader(testing, shuffle=False, **loader_options)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(classes, pretrained=pretrained).to(device)
    loss_function = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=int(spec["execution"]["max_epochs"])
    )
    scaler = torch.cuda.amp.GradScaler(enabled=device.type == "cuda")
    model_dir = Path("/workspace-cache/model_registry") / f"job_{telemetry.job_id}"
    model_dir.mkdir(parents=True, exist_ok=True)
    best_quality = -1.0
    parameters_count = sum(parameter.numel() for parameter in model.parameters())
    flops_estimate = 1.8e9 * (int(case["input_size"]) / 224.0) ** 2
    mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI", "http://mlflow:5000"))
    mlflow.set_experiment(spec["execution"]["experiment_name"])
    try:
        with mlflow.start_run(run_name=f"job-energy-{telemetry.job_id}"):
            mlflow.log_params(
                {
                    "runner": spec["runner"],
                    "dataset": dataset_key,
                    "task_type": spec["task_type"],
                    "quality_metric": primary_metric,
                    "controller_id": spec["controller_id"],
                    "benchmark_case_id": spec["benchmark_case_id"],
                    "training_seed": seed,
                    "split_seed": split_seed,
                    "train_images": len(training),
                    "validation_images": len(validation),
                    "test_images": len(testing),
                    "classes": classes,
                    "batch_size": batch_size,
                    "model": case["model_version"],
                    "pretrained": pretrained,
                    "pretraining_source": "ImageNet-1K" if pretrained else "none",
                }
            )
            for epoch in range(1, int(spec["execution"]["max_epochs"]) + 1):
                telemetry.start_epoch()
                model.train()
                losses: list[float] = []
                norms: list[float] = []
                for images, targets in train_loader:
                    images = images.to(device, non_blocking=True)
                    targets = targets.to(device, non_blocking=True)
                    optimizer.zero_grad(set_to_none=True)
                    with torch.cuda.amp.autocast(enabled=device.type == "cuda"):
                        loss = loss_function(model(images), targets)
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                    norms.append(gradient_norm(model.parameters()))
                    scaler.step(optimizer)
                    scaler.update()
                    losses.append(float(loss.detach().cpu()))
                accuracy, macro_f1 = confusion_metrics(model, validation_loader, device, classes)
                quality = primary_quality(primary_metric, accuracy, macro_f1)
                metrics = {
                    "quality_score": quality,
                    "accuracy": accuracy,
                    "macro_f1": macro_f1,
                    "train_loss": float(np.mean(losses)),
                    "gradient_norm": float(np.mean(norms)) if norms else 0.0,
                    "learning_rate": float(optimizer.param_groups[0]["lr"]),
                    "model_parameter_count": parameters_count,
                    "model_flops": flops_estimate,
                }
                if quality > best_quality:
                    best_quality = quality
                    torch.save(model.state_dict(), model_dir / "best.pt")
                mlflow.log_metrics({f"epoch/{key}": value for key, value in metrics.items()}, step=epoch)
                should_stop = telemetry.finish_epoch(epoch, quality, metrics)
                scheduler.step()
                if should_stop:
                    break
            summary, gpu = telemetry.finalize()
            model.load_state_dict(torch.load(model_dir / "best.pt", map_location=device, weights_only=True))
            test_accuracy, test_macro_f1 = confusion_metrics(model, test_loader, device, classes)
            summary.update(
                {
                    "primary_metric": primary_metric,
                    "test_accuracy": test_accuracy,
                    "test_macro_f1": test_macro_f1,
                    "dataset_name": case["dataset_name"],
                    "dataset_key": dataset_key,
                }
            )
            telemetry.epoch_summary_path.write_text(
                json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
            )
            mlflow.log_metrics(
                {
                    f"best_validation_{primary_metric}": summary["best_quality_score"],
                    "test_accuracy": test_accuracy,
                    "test_macro_f1": test_macro_f1,
                    "gpu_energy_wh": gpu["gpu_energy_kwh"] * 1000.0,
                }
            )
            mlflow.log_artifact(str(telemetry.epoch_summary_path), artifact_path="energy")
            mlflow.log_artifact(str(telemetry.gpu_summary_path), artifact_path="energy")
            mlflow.log_artifact(str(model_dir / "best.pt"), artifact_path="model")
            print(json.dumps({"summary": summary, "gpu": gpu}))
    finally:
        telemetry.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-spec-base64", required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("/workspace/energy_metrics"))
    args = parser.parse_args()
    run(decode_run_spec(args.run_spec_base64), args.output_dir)


if __name__ == "__main__":
    main()
