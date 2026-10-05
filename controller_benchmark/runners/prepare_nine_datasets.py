from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import time
import urllib.parse
import urllib.request
import zipfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any


DATASET_ROOT = Path("/workspace-cache/controller-datasets-v1")
TINY_IMAGENET_URL = "https://cs231n.stanford.edu/tiny-imagenet-200.zip"
TINY_IMAGENET_MD5 = "90528d7ca1a48142e341f4ef8d21d0de"
UAVID_REPOSITORY = "dronefreak/UAVid-2020"
VISDRONE_VEHICLE_CLASSES = {3, 4, 5, 8}
DOWNLOAD_RETRIES = 8
DOWNLOAD_RETRY_BASE_SECONDS = 5


def file_digest(path: Path, algorithm: str = "sha256") -> str:
    digest = hashlib.new(algorithm)
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def retry_sleep(attempt: int) -> None:
    delay = min(DOWNLOAD_RETRY_BASE_SECONDS * (2 ** attempt), 120)
    time.sleep(delay)


def download_file(url: str, target: Path, expected_size: int | None = None) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists() and (expected_size is None or target.stat().st_size == expected_size):
        return
    last_error: Exception | None = None
    for attempt in range(DOWNLOAD_RETRIES):
        temporary = target.with_suffix(target.suffix + ".part")
        start = temporary.stat().st_size if temporary.exists() else 0
        request = urllib.request.Request(url)
        if start:
            request.add_header("Range", f"bytes={start}-")
        try:
            with urllib.request.urlopen(request, timeout=180) as response:
                append = start > 0 and getattr(response, "status", None) == 206
                with temporary.open("ab" if append else "wb") as handle:
                    shutil.copyfileobj(response, handle, length=1024 * 1024)
            if expected_size is not None and temporary.stat().st_size != expected_size:
                raise RuntimeError(
                    f"Unexpected size for {target.name}: {temporary.stat().st_size} != {expected_size}"
                )
            temporary.replace(target)
            return
        except Exception as exc:
            last_error = exc
            if attempt == DOWNLOAD_RETRIES - 1:
                break
            print(
                f"Retrying download after error ({attempt + 1}/{DOWNLOAD_RETRIES}): {url}: {exc}",
                flush=True,
            )
            retry_sleep(attempt)
    raise RuntimeError(f"Could not download {url} after {DOWNLOAD_RETRIES} attempts") from last_error


def safe_extract_zip(archive: Path, destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    root = destination.resolve()
    with zipfile.ZipFile(archive) as handle:
        for member in handle.infolist():
            target = (destination / member.filename).resolve()
            if target != root and root not in target.parents:
                raise RuntimeError(f"Unsafe ZIP member: {member.filename}")
        handle.extractall(destination)


def write_marker(root: Path, payload: dict[str, Any]) -> Path:
    marker = root / "dataset-manifest.json"
    marker.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return marker


def prepare_torchvision_dataset(dataset_key: str, root: Path) -> Path:
    marker = root / "dataset-manifest.json"
    if marker.exists():
        return marker
    from torchvision.datasets import CIFAR10, CIFAR100, OxfordIIITPet, VOCSegmentation

    root.mkdir(parents=True, exist_ok=True)
    if dataset_key == "cifar10":
        CIFAR10(root, train=True, download=True)
        CIFAR10(root, train=False, download=True)
    elif dataset_key == "cifar100":
        CIFAR100(root, train=True, download=True)
        CIFAR100(root, train=False, download=True)
    elif dataset_key == "oxford-pet":
        OxfordIIITPet(root, split="trainval", target_types="segmentation", download=True)
        OxfordIIITPet(root, split="test", target_types="segmentation", download=True)
    elif dataset_key == "pascal-voc2012":
        VOCSegmentation(root, year="2012", image_set="train", download=True)
        VOCSegmentation(root, year="2012", image_set="val", download=False)
    else:
        raise ValueError(dataset_key)
    return write_marker(
        root,
        {
            "dataset_key": dataset_key,
            "source": "torchvision official dataset downloader",
            "prepared": True,
        },
    )


def prepare_tiny_imagenet(root: Path) -> Path:
    marker = root / "dataset-manifest.json"
    if marker.exists():
        return marker
    root.mkdir(parents=True, exist_ok=True)
    archive = root / "tiny-imagenet-200.zip"
    download_file(TINY_IMAGENET_URL, archive)
    actual_md5 = file_digest(archive, "md5")
    if actual_md5 != TINY_IMAGENET_MD5:
        raise RuntimeError(f"Tiny ImageNet MD5 mismatch: {actual_md5}")
    extracted = root / "tiny-imagenet-200"
    if not extracted.exists():
        safe_extract_zip(archive, root)
    validation_root = extracted / "val-classified"
    if not validation_root.exists():
        annotations = {}
        for line in (extracted / "val" / "val_annotations.txt").read_text(encoding="utf-8").splitlines():
            values = line.split("\t")
            annotations[values[0]] = values[1]
        for image_name, class_name in annotations.items():
            class_dir = validation_root / class_name
            class_dir.mkdir(parents=True, exist_ok=True)
            source = extracted / "val" / "images" / image_name
            target = class_dir / image_name
            if not target.exists():
                os.symlink(source, target)
    return write_marker(
        root,
        {
            "dataset_key": "tiny-imagenet",
            "source_url": TINY_IMAGENET_URL,
            "archive_md5": actual_md5,
            "classes": 200,
            "train_images": 100000,
            "validation_images": 10000,
        },
    )


def prepare_visdrone(root: Path) -> Path:
    marker = root / "dataset-manifest.json"
    if marker.exists():
        return marker
    from PIL import Image
    from ultralytics.utils import ASSETS_URL, TQDM
    from ultralytics.utils.downloads import download

    raw = root / "raw"
    raw.mkdir(parents=True, exist_ok=True)
    source_splits = {
        "train": "VisDrone2019-DET-train",
        "val": "VisDrone2019-DET-val",
        "test": "VisDrone2019-DET-test-dev",
    }
    urls = [f"{ASSETS_URL}/{source_name}.zip" for source_name in source_splits.values()]
    download(urls, dir=raw, threads=3)
    counts: dict[str, Any] = {}
    for split, source_name in source_splits.items():
        source_root = raw / source_name
        image_output = root / "images" / split
        label_output = root / "labels" / split
        image_output.mkdir(parents=True, exist_ok=True)
        label_output.mkdir(parents=True, exist_ok=True)
        objects = 0
        annotations = sorted((source_root / "annotations").glob("*.txt"))
        for annotation in TQDM(annotations, desc=f"Filtering VisDrone {split} vehicles"):
            image_source = source_root / "images" / annotation.with_suffix(".jpg").name
            image_target = image_output / image_source.name
            if not image_target.exists():
                os.symlink(image_source, image_target)
            width, height = Image.open(image_source).size
            labels = []
            for raw_line in annotation.read_text(encoding="utf-8").splitlines():
                values = raw_line.split(",")
                if len(values) < 6 or values[4] == "0":
                    continue
                class_id = int(values[5]) - 1
                if class_id not in VISDRONE_VEHICLE_CLASSES:
                    continue
                x, y, box_width, box_height = map(float, values[:4])
                labels.append(
                    f"0 {(x + box_width / 2) / width:.8f} {(y + box_height / 2) / height:.8f} "
                    f"{box_width / width:.8f} {box_height / height:.8f}"
                )
            (label_output / annotation.name).write_text(
                "\n".join(labels) + ("\n" if labels else ""),
                encoding="utf-8",
            )
            objects += len(labels)
        counts[split] = {"images": len(annotations), "vehicles": objects}
    (root / "data.yaml").write_text(
        f"path: {root.as_posix()}\ntrain: images/train\nval: images/val\ntest: images/test\n"
        "nc: 1\nnames:\n  0: vehicle\n",
        encoding="utf-8",
    )
    return write_marker(
        root,
        {
            "dataset_key": "visdrone-vehicles",
            "source": "VisDrone2019-DET via Ultralytics assets",
            "vehicle_source_class_ids": sorted(VISDRONE_VEHICLE_CLASSES),
            "unified_class": "vehicle",
            "counts": counts,
        },
    )


def huggingface_tree(repository: str) -> list[dict[str, Any]]:
    files = []
    url: str | None = (
        f"https://huggingface.co/api/datasets/{repository}/tree/main"
        "?recursive=true&expand=false&limit=1000"
    )
    while url:
        last_error: Exception | None = None
        for attempt in range(DOWNLOAD_RETRIES):
            try:
                with urllib.request.urlopen(url, timeout=120) as response:
                    payload = json.loads(response.read())
                    link = response.headers.get("Link", "")
                break
            except Exception as exc:
                last_error = exc
                if attempt == DOWNLOAD_RETRIES - 1:
                    raise RuntimeError(
                        f"Could not read Hugging Face tree for {repository} after {DOWNLOAD_RETRIES} attempts"
                    ) from last_error
                print(
                    f"Retrying Hugging Face tree request ({attempt + 1}/{DOWNLOAD_RETRIES}): {url}: {exc}",
                    flush=True,
                )
                retry_sleep(attempt)
        files.extend(
            {"path": str(item["path"]), "size": int(item["size"])}
            for item in payload
            if item.get("type") == "file" and item.get("size") is not None
        )
        next_link = re.search(r'<([^>]+)>;\s*rel="next"', link)
        url = next_link.group(1) if next_link else None
    return files


def prepare_uavid(root: Path) -> Path:
    marker = root / "dataset-manifest.json"
    if marker.exists():
        return marker
    root.mkdir(parents=True, exist_ok=True)
    selected = []
    for item in huggingface_tree(UAVID_REPOSITORY):
        path = str(item["path"])
        if path in {"README.md", "data.yaml"} or any(
            path.startswith(prefix)
            for prefix in ("images/train/", "images/val/", "masks/train/", "masks/val/")
        ):
            selected.append(item)

    def fetch(item: dict[str, Any]) -> None:
        relative = str(item["path"])
        encoded = urllib.parse.quote(relative, safe="/")
        url = f"https://huggingface.co/datasets/{UAVID_REPOSITORY}/resolve/main/{encoded}"
        download_file(url, root / relative, int(item["size"]))

    with ThreadPoolExecutor(max_workers=4) as executor:
        list(executor.map(fetch, selected))
    counts = {
        split: {
            "images": len(list((root / "images" / split).glob("*.png"))),
            "masks": len(list((root / "masks" / split).glob("*.png"))),
        }
        for split in ("train", "val")
    }
    for split, values in counts.items():
        if values["images"] != values["masks"] or values["images"] == 0:
            raise RuntimeError(f"Incomplete UAVid {split} split: {values}")
    return write_marker(
        root,
        {
            "dataset_key": "uavid",
            "official_url": "https://uavid.nl/",
            "download_repository": UAVID_REPOSITORY,
            "download_repository_note": "Unofficial packaging with unchanged official content",
            "license": "CC BY-NC-SA 4.0",
            "counts": counts,
        },
    )


PREPARERS = {
    "cifar10": lambda root: prepare_torchvision_dataset("cifar10", root),
    "cifar100": lambda root: prepare_torchvision_dataset("cifar100", root),
    "tiny-imagenet": prepare_tiny_imagenet,
    "oxford-pet": lambda root: prepare_torchvision_dataset("oxford-pet", root),
    "pascal-voc2012": lambda root: prepare_torchvision_dataset("pascal-voc2012", root),
    "uavid": prepare_uavid,
    "visdrone-vehicles": prepare_visdrone,
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=DATASET_ROOT)
    parser.add_argument("--dataset", action="append", choices=sorted(PREPARERS))
    args = parser.parse_args()
    selected = args.dataset or list(PREPARERS)
    for dataset_key in selected:
        marker = PREPARERS[dataset_key](args.root / dataset_key)
        print(marker)


if __name__ == "__main__":
    main()
