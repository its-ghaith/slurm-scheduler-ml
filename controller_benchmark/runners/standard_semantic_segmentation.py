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
from torchvision import models
from torchvision.datasets import OxfordIIITPet, VOCSegmentation
from torchvision.transforms import functional as vision

from .common import EpochTelemetry, decode_run_spec


DATASET_ROOT = Path("/workspace-cache/controller-datasets-v1")
NORMALIZATION_MEAN = (0.485, 0.456, 0.406)
NORMALIZATION_STD = (0.229, 0.224, 0.225)


class IndexedSegmentationDataset(Dataset):
    def __init__(
        self,
        dataset: Dataset,
        indexes: list[int],
        *,
        input_size: int,
        train: bool,
        mask_mapping: str,
    ):
        self.dataset = dataset
        self.indexes = indexes
        self.input_size = input_size
        self.train = train
        self.mask_mapping = mask_mapping

    def __len__(self) -> int:
        return len(self.indexes)

    def __getitem__(self, index: int):
        image, mask = self.dataset[self.indexes[index]]
        return paired_transform(
            image,
            mask,
            input_size=self.input_size,
            train=self.train,
            mask_mapping=self.mask_mapping,
        )


class FolderSegmentationDataset(Dataset):
    def __init__(
        self,
        image_root: Path,
        mask_root: Path,
        image_names: list[str],
        *,
        input_size: int,
        train: bool,
    ):
        self.image_root = image_root
        self.mask_root = mask_root
        self.image_names = image_names
        self.input_size = input_size
        self.train = train

    def __len__(self) -> int:
        return len(self.image_names)

    def __getitem__(self, index: int):
        name = self.image_names[index]
        image = Image.open(self.image_root / name).convert("RGB")
        mask = Image.open(self.mask_root / name)
        return paired_transform(
            image,
            mask,
            input_size=self.input_size,
            train=self.train,
            mask_mapping="identity",
        )


def paired_transform(
    image: Image.Image,
    mask: Image.Image,
    *,
    input_size: int,
    train: bool,
    mask_mapping: str,
):
    image = image.resize((input_size, input_size), Image.Resampling.BILINEAR)
    mask = mask.resize((input_size, input_size), Image.Resampling.NEAREST)
    if train and random.random() < 0.5:
        image = vision.hflip(image)
        mask = vision.hflip(mask)
    image_tensor = vision.normalize(vision.to_tensor(image), NORMALIZATION_MEAN, NORMALIZATION_STD)
    mask_array = np.asarray(mask, dtype=np.int64)
    if mask_mapping == "oxford-binary":
        # Oxford trimaps use 1 for pet, 2 for background, and 3 for the pet border.
        mask_array = np.where(mask_array == 2, 0, 1)
    return image_tensor, torch.from_numpy(mask_array.copy()).long()


def random_split_indexes(length: int, validation_fraction: float, seed: int) -> tuple[list[int], list[int]]:
    indexes = list(range(length))
    random.Random(seed).shuffle(indexes)
    validation_size = max(1, round(length * validation_fraction))
    return indexes[validation_size:], indexes[:validation_size]


def uavid_split_names(root: Path, split_seed: int) -> tuple[list[str], list[str], list[str]]:
    train_names = sorted(path.name for path in (root / "images" / "train").glob("*.png"))
    test_names = sorted(path.name for path in (root / "images" / "val").glob("*.png"))
    sequences: dict[str, list[str]] = {}
    for name in train_names:
        sequences.setdefault(name.split("_", 1)[0], []).append(name)
    sequence_names = sorted(sequences)
    random.Random(split_seed).shuffle(sequence_names)
    validation_sequences = set(sequence_names[: max(1, round(len(sequence_names) * 0.1))])
    validation = [name for sequence in sequence_names if sequence in validation_sequences for name in sequences[sequence]]
    training = [name for sequence in sequence_names if sequence not in validation_sequences for name in sequences[sequence]]
    return training, validation, test_names


def dataset_bundle(dataset_key: str, input_size: int, split_seed: int):
    if dataset_key == "oxford-pet":
        root = DATASET_ROOT / dataset_key
        source = OxfordIIITPet(root, split="trainval", target_types="segmentation", download=False)
        test_source = OxfordIIITPet(root, split="test", target_types="segmentation", download=False)
        training, validation = random_split_indexes(len(source), 0.1, split_seed)
        return (
            IndexedSegmentationDataset(
                source, training, input_size=input_size, train=True, mask_mapping="oxford-binary"
            ),
            IndexedSegmentationDataset(
                source, validation, input_size=input_size, train=False, mask_mapping="oxford-binary"
            ),
            IndexedSegmentationDataset(
                test_source,
                list(range(len(test_source))),
                input_size=input_size,
                train=False,
                mask_mapping="oxford-binary",
            ),
        )
    if dataset_key == "pascal-voc2012":
        root = DATASET_ROOT / dataset_key
        source = VOCSegmentation(root, year="2012", image_set="train", download=False)
        test_source = VOCSegmentation(root, year="2012", image_set="val", download=False)
        training, validation = random_split_indexes(len(source), 0.1, split_seed)
        return (
            IndexedSegmentationDataset(
                source, training, input_size=input_size, train=True, mask_mapping="identity"
            ),
            IndexedSegmentationDataset(
                source, validation, input_size=input_size, train=False, mask_mapping="identity"
            ),
            IndexedSegmentationDataset(
                test_source,
                list(range(len(test_source))),
                input_size=input_size,
                train=False,
                mask_mapping="identity",
            ),
        )
    if dataset_key == "uavid":
        root = DATASET_ROOT / dataset_key
        training, validation, testing = uavid_split_names(root, split_seed)
        return (
            FolderSegmentationDataset(
                root / "images" / "train",
                root / "masks" / "train",
                training,
                input_size=input_size,
                train=True,
            ),
            FolderSegmentationDataset(
                root / "images" / "train",
                root / "masks" / "train",
                validation,
                input_size=input_size,
                train=False,
            ),
            FolderSegmentationDataset(
                root / "images" / "val",
                root / "masks" / "val",
                testing,
                input_size=input_size,
                train=False,
            ),
        )
    raise ValueError(f"Unsupported segmentation dataset: {dataset_key}")


class ConvBlock(nn.Sequential):
    def __init__(self, input_channels: int, output_channels: int):
        super().__init__(
            nn.Conv2d(input_channels, output_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(output_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(output_channels, output_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(output_channels),
            nn.ReLU(inplace=True),
        )


class TinyUNet(nn.Module):
    def __init__(self, classes: int):
        super().__init__()
        self.encoder1 = ConvBlock(3, 32)
        self.encoder2 = ConvBlock(32, 64)
        self.pool = nn.MaxPool2d(2)
        self.bottleneck = ConvBlock(64, 128)
        self.up2 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.decoder2 = ConvBlock(128, 64)
        self.up1 = nn.ConvTranspose2d(64, 32, 2, stride=2)
        self.decoder1 = ConvBlock(64, 32)
        self.output = nn.Conv2d(32, classes, 1)

    def forward(self, inputs):
        encoder1 = self.encoder1(inputs)
        encoder2 = self.encoder2(self.pool(encoder1))
        features = self.bottleneck(self.pool(encoder2))
        features = self.decoder2(torch.cat((self.up2(features), encoder2), dim=1))
        return self.output(self.decoder1(torch.cat((self.up1(features), encoder1), dim=1)))


class ResNet18UNet(nn.Module):
    """U-Net decoder with an optionally ImageNet-pretrained ResNet18 encoder."""

    def __init__(self, classes: int, *, pretrained: bool):
        super().__init__()
        weights = models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        backbone = models.resnet18(weights=weights)
        self.stem = nn.Sequential(backbone.conv1, backbone.bn1, backbone.relu)
        self.pool = backbone.maxpool
        self.encoder1 = backbone.layer1
        self.encoder2 = backbone.layer2
        self.encoder3 = backbone.layer3
        self.encoder4 = backbone.layer4
        self.up3 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.decoder3 = ConvBlock(512, 256)
        self.up2 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.decoder2 = ConvBlock(256, 128)
        self.up1 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.decoder1 = ConvBlock(128, 64)
        self.up0 = nn.ConvTranspose2d(64, 64, 2, stride=2)
        self.decoder0 = ConvBlock(128, 64)
        self.output = nn.Conv2d(64, classes, 1)

    def forward(self, inputs):
        stem = self.stem(inputs)
        encoder1 = self.encoder1(self.pool(stem))
        encoder2 = self.encoder2(encoder1)
        encoder3 = self.encoder3(encoder2)
        features = self.encoder4(encoder3)
        features = self.decoder3(torch.cat((self.up3(features), encoder3), dim=1))
        features = self.decoder2(torch.cat((self.up2(features), encoder2), dim=1))
        features = self.decoder1(torch.cat((self.up1(features), encoder1), dim=1))
        features = self.decoder0(torch.cat((self.up0(features), stem), dim=1))
        logits = self.output(features)
        return nn.functional.interpolate(
            logits, size=inputs.shape[-2:], mode="bilinear", align_corners=False
        )


def build_model(classes: int, model_version: str, *, pretrained: bool) -> nn.Module:
    if model_version == "resnet18-unet":
        return ResNet18UNet(classes, pretrained=pretrained)
    if pretrained:
        raise ValueError(
            f"Model {model_version!r} has no valid pretrained weights; use resnet18-unet."
        )
    return TinyUNet(classes)


def segmentation_metrics(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    classes: int,
) -> tuple[float, float, float]:
    confusion = torch.zeros((classes, classes), dtype=torch.float64)
    model.eval()
    with torch.no_grad():
        for images, targets in loader:
            predictions = model(images.to(device, non_blocking=True)).argmax(dim=1).cpu()
            targets = targets.to(torch.int64)
            valid = targets != 255
            encoded = targets[valid] * classes + predictions[valid]
            confusion += torch.bincount(encoded, minlength=classes * classes).reshape(classes, classes)
    true_positive = confusion.diag()
    false_positive = confusion.sum(dim=0) - true_positive
    false_negative = confusion.sum(dim=1) - true_positive
    union = true_positive + false_positive + false_negative
    valid_classes = union > 0
    miou = float((true_positive[valid_classes] / union[valid_classes]).mean())
    denominator = 2 * true_positive + false_positive + false_negative
    dice = float((2 * true_positive[valid_classes] / denominator[valid_classes].clamp_min(1)).mean())
    pixel_accuracy = float(true_positive.sum() / confusion.sum().clamp_min(1))
    return miou, dice, pixel_accuracy


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
    dataset_key = case["dataset_key"]
    classes = int(case["num_classes"])
    input_size = int(case["input_size"])
    training, validation, testing = dataset_bundle(dataset_key, input_size, split_seed)
    batch_size = int(case.get("batch_size", spec["execution"]["batch_size"]))
    num_workers = int(case.get("workers", spec["execution"].get("segmentation_workers", 0)))
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
    model = build_model(classes, str(case["model_version"]), pretrained=pretrained).to(device)
    loss_function = nn.CrossEntropyLoss(ignore_index=255)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=int(spec["execution"]["max_epochs"]), eta_min=1e-5
    )
    scaler = torch.cuda.amp.GradScaler(enabled=device.type == "cuda")
    model_dir = Path("/workspace-cache/model_registry") / f"job_{telemetry.job_id}"
    model_dir.mkdir(parents=True, exist_ok=True)
    best_quality = -1.0
    parameters_count = sum(parameter.numel() for parameter in model.parameters())
    base_flops = 30.0e9 if case["model_version"] == "resnet18-unet" else 12.0e9
    flops_estimate = base_flops * (input_size / 256.0) ** 2
    mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI", "http://mlflow:5000"))
    mlflow.set_experiment(spec["execution"]["experiment_name"])
    try:
        with mlflow.start_run(run_name=f"job-energy-{telemetry.job_id}"):
            mlflow.log_params(
                {
                    "runner": spec["runner"],
                    "dataset": dataset_key,
                    "task_type": spec["task_type"],
                    "quality_metric": "miou",
                    "controller_id": spec["controller_id"],
                    "benchmark_case_id": spec["benchmark_case_id"],
                    "training_seed": seed,
                    "split_seed": split_seed,
                    "train_images": len(training),
                    "validation_images": len(validation),
                    "test_images": len(testing),
                    "classes": classes,
                    "batch_size": batch_size,
                    "input_size": input_size,
                    "model": case["model_version"],
                    "pretrained": pretrained,
                    "pretraining_source": "ImageNet-1K encoder" if pretrained else "none",
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
                miou, dice, pixel_accuracy = segmentation_metrics(
                    model, validation_loader, device, classes
                )
                metrics = {
                    "quality_score": miou,
                    "miou": miou,
                    "dice": dice,
                    "pixel_accuracy": pixel_accuracy,
                    "train_loss": float(np.mean(losses)),
                    "gradient_norm": float(np.mean(norms)) if norms else 0.0,
                    "learning_rate": float(optimizer.param_groups[0]["lr"]),
                    "model_parameter_count": parameters_count,
                    "model_flops": flops_estimate,
                }
                if miou > best_quality:
                    best_quality = miou
                    torch.save(model.state_dict(), model_dir / "best.pt")
                mlflow.log_metrics({f"epoch/{key}": value for key, value in metrics.items()}, step=epoch)
                should_stop = telemetry.finish_epoch(epoch, miou, metrics)
                scheduler.step()
                if should_stop:
                    break
            summary, gpu = telemetry.finalize()
            model.load_state_dict(torch.load(model_dir / "best.pt", map_location=device, weights_only=True))
            test_miou, test_dice, test_pixel_accuracy = segmentation_metrics(
                model, test_loader, device, classes
            )
            summary.update(
                {
                    "primary_metric": "miou",
                    "test_miou": test_miou,
                    "test_dice": test_dice,
                    "test_pixel_accuracy": test_pixel_accuracy,
                    "dataset_name": case["dataset_name"],
                    "dataset_key": dataset_key,
                }
            )
            telemetry.epoch_summary_path.write_text(
                json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
            )
            mlflow.log_metrics(
                {
                    "best_validation_miou": summary["best_quality_score"],
                    "test_miou": test_miou,
                    "test_dice": test_dice,
                    "test_pixel_accuracy": test_pixel_accuracy,
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
