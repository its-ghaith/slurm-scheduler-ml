from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import shutil
import tarfile
import urllib.request
from pathlib import Path


OFFICIAL_BASE_URL = "https://downloads.greyc.fr/vedai"
SOURCE_FILES = {
    "Annotations512.tar": (1_753_088, "aa0085fbab16e61a325cd0dc18e4c4d192c4e0dcf5fc8a9a0166cfb06814725a"),
    "Vehicules512.tar.001": (699_400_192, "542aba4a228cc5a6e9018840aec60e2cb80acda662080d29139b612963d3e042"),
    "Vehicules512.tar.002": (593_733_632, "da9626f82e398b34d46e67301b4adbf0945b6b0b67571786047a82134063e2f8"),
    "TermsandConditionsofUseVeDAI2014.pdf": (53_320, "4e42ca10e169b899bf60524e092bbcf38ca91b7ced7f2bfa12d3c2ce3e099da6"),
}


def download_file(path: Path, expected_size: int, expected_sha256: str) -> None:
    current_size = path.stat().st_size if path.exists() else 0
    if current_size == expected_size and file_sha256(path) == expected_sha256:
        return
    if current_size > expected_size:
        path.unlink()
        current_size = 0
    request = urllib.request.Request(f"{OFFICIAL_BASE_URL}/{path.name}")
    if current_size:
        request.add_header("Range", f"bytes={current_size}-")
    with urllib.request.urlopen(request, timeout=120) as response:
        append = current_size > 0 and getattr(response, "status", None) == 206
        with path.open("ab" if append else "wb") as handle:
            shutil.copyfileobj(response, handle, length=1024 * 1024)
    if path.stat().st_size != expected_size:
        raise RuntimeError(f"Unexpected size for {path.name}: {path.stat().st_size} != {expected_size}")
    actual_sha256 = file_sha256(path)
    if actual_sha256 != expected_sha256:
        raise RuntimeError(f"Unexpected SHA256 for {path.name}: {actual_sha256} != {expected_sha256}")


def extract_archive(archive_path: Path, destination: Path) -> None:
    with tarfile.open(archive_path) as archive:
        root = destination.resolve()
        for member in archive.getmembers():
            target = (destination / member.name).resolve()
            if root not in target.parents and target != root:
                raise RuntimeError(f"Unsafe archive member: {member.name}")
        archive.extractall(destination)


def prepare_sources(source: Path) -> None:
    source.mkdir(parents=True, exist_ok=True)
    for name, (size, sha256) in SOURCE_FILES.items():
        download_file(source / name, size, sha256)
    annotations = source / "Annotations512"
    if not annotations.is_dir():
        extract_archive(source / "Annotations512.tar", source)
    images = source / "Vehicules512"
    if not images.is_dir():
        combined = source / "Vehicules512.tar"
        with combined.open("wb") as target:
            for name in ("Vehicules512.tar.001", "Vehicules512.tar.002"):
                with (source / name).open("rb") as part:
                    shutil.copyfileobj(part, target, length=1024 * 1024)
        extract_archive(combined, source)
        combined.unlink()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_ids(path: Path) -> list[str]:
    return [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def find_color_image(images_root: Path, image_id: str) -> Path:
    candidates = list(images_root.rglob(f"{image_id}_co.*"))
    if not candidates:
        candidates = [path for path in images_root.rglob(f"{image_id}.*") if "ir" not in path.stem.lower()]
    if len(candidates) != 1:
        raise RuntimeError(f"Expected one color image for {image_id}, found {len(candidates)}")
    return candidates[0]


def convert_annotation(source: Path, target: Path, image_size: int = 512) -> int:
    labels = []
    for line in source.read_text(encoding="utf-8").splitlines():
        values = line.split()
        if len(values) < 14:
            continue
        xs = [float(value) for value in values[6:10]]
        ys = [float(value) for value in values[10:14]]
        x1, x2 = max(0.0, min(xs)), min(float(image_size), max(xs))
        y1, y2 = max(0.0, min(ys)), min(float(image_size), max(ys))
        width, height = x2 - x1, y2 - y1
        if width <= 0 or height <= 0:
            continue
        labels.append(
            f"0 {((x1 + x2) / 2) / image_size:.8f} {((y1 + y2) / 2) / image_size:.8f} "
            f"{width / image_size:.8f} {height / image_size:.8f}"
        )
    target.write_text("\n".join(labels) + ("\n" if labels else ""), encoding="utf-8")
    return len(labels)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--download", action="store_true")
    args = parser.parse_args()
    source, output = args.source.resolve(), args.output.resolve()
    if args.download:
        prepare_sources(source)
    annotations = source / "Annotations512"
    marker = output / "dataset-manifest.json"
    if marker.exists():
        print(marker)
        return
    train_ids = read_ids(annotations / "fold01.txt")
    test_ids = read_ids(annotations / "fold01test.txt")
    random.Random(args.seed).shuffle(train_ids)
    val_size = max(1, round(len(train_ids) * 0.1))
    splits = {"train": train_ids[val_size:], "val": train_ids[:val_size], "test": test_ids}
    images_root = source / "Vehicules512"
    counts = {}
    for split, image_ids in splits.items():
        image_dir, label_dir = output / "images" / split, output / "labels" / split
        image_dir.mkdir(parents=True, exist_ok=True)
        label_dir.mkdir(parents=True, exist_ok=True)
        objects = 0
        for image_id in image_ids:
            image = find_color_image(images_root, image_id)
            link = image_dir / image.name
            if not link.exists():
                os.symlink(image, link)
            objects += convert_annotation(annotations / f"{image_id}.txt", label_dir / f"{image.stem}.txt")
        counts[split] = {"images": len(image_ids), "objects": objects}
    yaml = output / "data.yaml"
    yaml.write_text(
        f"path: {output.as_posix()}\ntrain: images/train\nval: images/val\ntest: images/test\n"
        "nc: 1\nnames:\n  0: vehicle\n",
        encoding="utf-8",
    )
    manifest = {
        "dataset": "VEDAI 512",
        "official_url": "https://downloads.greyc.fr/vedai/",
        "fold": 1,
        "validation_fraction": 0.1,
        "split_seed": args.seed,
        "counts": counts,
        "source_files": {
            path.name: file_sha256(path)
            for path in source.iterdir()
            if path.is_file() and path.name.startswith(("Annotations512", "Vehicules512"))
        },
    }
    marker.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    shutil.copy2(source / "TermsandConditionsofUseVeDAI2014.pdf", output)
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
