from __future__ import annotations

import argparse
import json
import os
import shutil
import tarfile
import urllib.request
import zipfile
from pathlib import Path
from typing import Callable


ASSET_ROOT = Path(
    os.environ.get(
        "CONTROLLER_DATASET_ROOT", "/workspace-cache/controller-datasets"
    )
)

DOWNLOADS: dict[str, tuple[str, str]] = {
    "adult": (
        "https://archive.ics.uci.edu/static/public/2/adult.zip",
        "adult.zip",
    ),
    "imdb": (
        "https://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz",
        "aclImdb_v1.tar.gz",
    ),
    "bank_marketing": (
        "https://archive.ics.uci.edu/static/public/222/bank+marketing.zip",
        "bank-marketing.zip",
    ),
    "bike_sharing": (
        "https://archive.ics.uci.edu/static/public/275/bike+sharing+dataset.zip",
        "bike-sharing.zip",
    ),
    "ag_news": (
        "https://s3.amazonaws.com/fast-ai-nlp/ag_news_csv.tgz",
        "ag_news_csv.tgz",
    ),
    "dbpedia": (
        "https://s3.amazonaws.com/fast-ai-nlp/dbpedia_csv.tgz",
        "dbpedia_csv.tgz",
    ),
    "yelp_review_full": (
        "https://s3.amazonaws.com/fast-ai-nlp/yelp_review_full_csv.tgz",
        "yelp_review_full_csv.tgz",
    ),
    "letter_recognition": (
        "https://archive.ics.uci.edu/static/public/59/letter%2Brecognition.zip",
        "letter-recognition.zip",
    ),
    "sensorless_drive": (
        "https://archive.ics.uci.edu/static/public/325/dataset%2Bfor%2Bsensorless%2Bdrive%2Bdiagnosis.zip",
        "sensorless-drive.zip",
    ),
    "online_news_popularity": (
        "https://archive.ics.uci.edu/static/public/332/online%2Bnews%2Bpopularity.zip",
        "online-news-popularity.zip",
    ),
    "superconductivity": (
        "https://archive.ics.uci.edu/static/public/464/superconductivty%2Bdata.zip",
        "superconductivity.zip",
    ),
    "year_prediction_msd": (
        "https://archive.ics.uci.edu/ml/machine-learning-databases/00203/YearPredictionMSD.txt.zip",
        "YearPredictionMSD.txt.zip",
    ),
    "wikitext103": (
        "https://huggingface.co/datasets/mattdangerw/wikitext-103-raw/resolve/main/wikitext-103-raw-v1.zip?download=true",
        "wikitext-103-raw-v1.zip",
    ),
    "etth1": (
        "https://raw.githubusercontent.com/zhouhaoyi/ETDataset/main/ETT-small/ETTh1.csv",
        "ETTh1.csv",
    ),
    "etth2": (
        "https://raw.githubusercontent.com/zhouhaoyi/ETDataset/main/ETT-small/ETTh2.csv",
        "ETTh2.csv",
    ),
    "ettm1": (
        "https://raw.githubusercontent.com/zhouhaoyi/ETDataset/main/ETT-small/ETTm1.csv",
        "ETTm1.csv",
    ),
    "ettm2": (
        "https://raw.githubusercontent.com/zhouhaoyi/ETDataset/main/ETT-small/ETTm2.csv",
        "ETTm2.csv",
    ),
    "movielens100k": (
        "https://files.grouplens.org/datasets/movielens/ml-100k.zip",
        "ml-100k.zip",
    ),
    "movielens1m": (
        "https://files.grouplens.org/datasets/movielens/ml-1m.zip",
        "ml-1m.zip",
    ),
    "movielens10m": (
        "https://files.grouplens.org/datasets/movielens/ml-10m.zip",
        "ml-10m.zip",
    ),
    "movielens20m": (
        "https://files.grouplens.org/datasets/movielens/ml-20m.zip",
        "ml-20m.zip",
    ),
    "movielens25m": (
        "https://files.grouplens.org/datasets/movielens/ml-25m.zip",
        "ml-25m.zip",
    ),
    "monash_traffic_hourly": (
        "https://zenodo.org/records/4656132/files/traffic_hourly_dataset.zip?download=1",
        "traffic_hourly_dataset.zip",
    ),
    "monash_electricity_hourly": (
        "https://zenodo.org/records/4656140/files/electricity_hourly_dataset.zip?download=1",
        "electricity_hourly_dataset.zip",
    ),
    "monash_solar_10_minutes": (
        "https://zenodo.org/records/4656144/files/solar_10_minutes_dataset.zip?download=1",
        "solar_10_minutes_dataset.zip",
    ),
}


FILE_COLLECTIONS: dict[str, dict[str, str]] = {
    "go_emotions": {
        "train.parquet": "https://huggingface.co/datasets/google-research-datasets/go_emotions/resolve/refs%2Fconvert%2Fparquet/simplified/train/0000.parquet",
        "validation.parquet": "https://huggingface.co/datasets/google-research-datasets/go_emotions/resolve/refs%2Fconvert%2Fparquet/simplified/validation/0000.parquet",
        "test.parquet": "https://huggingface.co/datasets/google-research-datasets/go_emotions/resolve/refs%2Fconvert%2Fparquet/simplified/test/0000.parquet",
    },
    "multi_eurlex": {
        "train.parquet": "https://huggingface.co/datasets/coastalcph/multi_eurlex/resolve/refs%2Fconvert%2Fparquet/en/train/0000.parquet",
        "validation.parquet": "https://huggingface.co/datasets/coastalcph/multi_eurlex/resolve/refs%2Fconvert%2Fparquet/en/validation/0000.parquet",
        "test.parquet": "https://huggingface.co/datasets/coastalcph/multi_eurlex/resolve/refs%2Fconvert%2Fparquet/en/test/0000.parquet",
    },
    "marc_multilingual": {
        "train.parquet": "https://huggingface.co/datasets/goosmanlei/amazon_reviews_multi/resolve/refs%2Fconvert%2Fparquet/all_languages/train/0000.parquet",
        "validation.parquet": "https://huggingface.co/datasets/goosmanlei/amazon_reviews_multi/resolve/refs%2Fconvert%2Fparquet/all_languages/validation/0000.parquet",
        "test.parquet": "https://huggingface.co/datasets/goosmanlei/amazon_reviews_multi/resolve/refs%2Fconvert%2Fparquet/all_languages/test/0000.parquet",
    },
    "c4_en_shard": {
        "train.parquet": "https://huggingface.co/datasets/allenai/c4/resolve/refs%2Fconvert%2Fparquet/en/partial-train/0000.parquet",
        "validation.parquet": "https://huggingface.co/datasets/allenai/c4/resolve/refs%2Fconvert%2Fparquet/en/partial-validation/0000.parquet",
    },
    "lm1b_shard": {
        "train.parquet": "https://huggingface.co/datasets/lm1b/resolve/refs%2Fconvert%2Fparquet/plain_text/train/0000.parquet",
        "test.parquet": "https://huggingface.co/datasets/lm1b/resolve/refs%2Fconvert%2Fparquet/plain_text/test/0000.parquet",
    },
    "amazon_books_5core": {
        "train.csv": "https://huggingface.co/datasets/McAuley-Lab/Amazon-Reviews-2023/resolve/main/benchmark/5core/last_out/Books.train.csv",
        "validation.csv": "https://huggingface.co/datasets/McAuley-Lab/Amazon-Reviews-2023/resolve/main/benchmark/5core/last_out/Books.valid.csv",
        "test.csv": "https://huggingface.co/datasets/McAuley-Lab/Amazon-Reviews-2023/resolve/main/benchmark/5core/last_out/Books.test.csv",
    },
    "amazon_electronics_5core": {
        "train.csv": "https://huggingface.co/datasets/McAuley-Lab/Amazon-Reviews-2023/resolve/main/benchmark/5core/last_out/Electronics.train.csv",
        "validation.csv": "https://huggingface.co/datasets/McAuley-Lab/Amazon-Reviews-2023/resolve/main/benchmark/5core/last_out/Electronics.valid.csv",
        "test.csv": "https://huggingface.co/datasets/McAuley-Lab/Amazon-Reviews-2023/resolve/main/benchmark/5core/last_out/Electronics.test.csv",
    },
}

WIKITEXT2_FILES = {
    "wiki.train.raw": "https://raw.githubusercontent.com/pytorch/examples/main/word_language_model/data/wikitext-2/train.txt",
    "wiki.valid.raw": "https://raw.githubusercontent.com/pytorch/examples/main/word_language_model/data/wikitext-2/valid.txt",
    "wiki.test.raw": "https://raw.githubusercontent.com/pytorch/examples/main/word_language_model/data/wikitext-2/test.txt",
}

PENN_TREEBANK_FILES = {
    "ptb.train.txt": "https://raw.githubusercontent.com/wojzaremba/lstm/master/data/ptb.train.txt",
    "ptb.valid.txt": "https://raw.githubusercontent.com/wojzaremba/lstm/master/data/ptb.valid.txt",
    "ptb.test.txt": "https://raw.githubusercontent.com/wojzaremba/lstm/master/data/ptb.test.txt",
}

TINY_SHAKESPEARE_FILES = {
    "input.txt": "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt",
}


def _download(url: str, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".part")
    request = urllib.request.Request(url, headers={"User-Agent": "RAPEC-benchmark/1.0"})
    with urllib.request.urlopen(request, timeout=120) as response, temporary.open("wb") as handle:
        shutil.copyfileobj(response, handle)
    temporary.replace(destination)


def _safe_target(root: Path, member_name: str) -> Path:
    target = (root / member_name).resolve()
    root = root.resolve()
    if target != root and root not in target.parents:
        raise ValueError(f"Archive member escapes target directory: {member_name}")
    return target


def _extract_zip(archive: Path, target: Path) -> None:
    with zipfile.ZipFile(archive) as handle:
        for member in handle.infolist():
            _safe_target(target, member.filename)
        handle.extractall(target)


def _extract_tar(archive: Path, target: Path) -> None:
    with tarfile.open(archive) as handle:
        for member in handle.getmembers():
            _safe_target(target, member.name)
        handle.extractall(target, filter="data")


def _prepare_downloaded(key: str, extractor: Callable[[Path, Path], None] | None) -> Path:
    url, filename = DOWNLOADS[key]
    root = ASSET_ROOT / key
    marker = root / ".ready.json"
    if marker.exists():
        return root
    root.mkdir(parents=True, exist_ok=True)
    archive = root / filename
    if not archive.exists():
        _download(url, archive)
    if extractor is not None:
        extractor(archive, root)
    marker.write_text(
        json.dumps({"dataset": key, "source": url, "archive": filename}, indent=2)
        + "\n",
        encoding="utf-8",
    )
    return root


def _prepare_wikitext2() -> Path:
    root = ASSET_ROOT / "wikitext2"
    target = root / "wikitext-2-raw"
    marker = root / ".ready.json"
    if marker.exists() and all((target / name).exists() for name in WIKITEXT2_FILES):
        return root
    target.mkdir(parents=True, exist_ok=True)
    for filename, url in WIKITEXT2_FILES.items():
        destination = target / filename
        if not destination.exists():
            _download(url, destination)
    marker.write_text(
        json.dumps(
            {
                "dataset": "wikitext2",
                "sources": WIKITEXT2_FILES,
                "layout": "wikitext-2-raw",
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return root


def _prepare_file_collection(key: str, files: dict[str, str]) -> Path:
    root = ASSET_ROOT / key
    marker = root / ".ready.json"
    if marker.exists() and all((root / name).exists() for name in files):
        return root
    root.mkdir(parents=True, exist_ok=True)
    for filename, url in files.items():
        destination = root / filename
        if not destination.exists():
            _download(url, destination)
    marker.write_text(
        json.dumps({"dataset": key, "sources": files}, indent=2) + "\n",
        encoding="utf-8",
    )
    return root


def prepare_asset(key: str) -> Path:
    if key in FILE_COLLECTIONS:
        return _prepare_file_collection(key, FILE_COLLECTIONS[key])
    if key == "adult":
        return _prepare_downloaded(key, _extract_zip)
    if key == "imdb":
        return _prepare_downloaded(key, _extract_tar)
    if key in {"ag_news", "dbpedia", "yelp_review_full"}:
        return _prepare_downloaded(key, _extract_tar)
    if key in {
        "letter_recognition",
        "sensorless_drive",
        "online_news_popularity",
        "superconductivity",
        "year_prediction_msd",
        "wikitext103",
    }:
        return _prepare_downloaded(key, _extract_zip)
    if key in {"bank_marketing", "bike_sharing"}:
        root = _prepare_downloaded(key, _extract_zip)
        if key == "bank_marketing" and not any(root.rglob("bank-full.csv")):
            nested = next(root.rglob("bank.zip"), None)
            if nested is None:
                raise FileNotFoundError("bank.zip was not found in the UCI archive.")
            _extract_zip(nested, root / "bank")
        return root
    if key == "wikitext2":
        return _prepare_wikitext2()
    if key == "penn_treebank":
        return _prepare_file_collection(key, PENN_TREEBANK_FILES)
    if key == "tiny_shakespeare":
        return _prepare_file_collection(key, TINY_SHAKESPEARE_FILES)
    if key in {"etth1", "etth2", "ettm1", "ettm2"}:
        return _prepare_downloaded(key, None)
    if key in {
        "movielens100k",
        "movielens1m",
        "movielens10m",
        "movielens20m",
        "movielens25m",
        "monash_traffic_hourly",
        "monash_electricity_hourly",
        "monash_solar_10_minutes",
    }:
        return _prepare_downloaded(key, _extract_zip)
    if key == "covertype_anomaly":
        return prepare_asset("covertype")
    if key == "sensorless_drive_anomaly":
        return prepare_asset("sensorless_drive")
    if key in {"california_housing", "kddcup99", "covertype"}:
        data_home = ASSET_ROOT / "scikit-learn"
        data_home.mkdir(parents=True, exist_ok=True)
        if key == "california_housing":
            from sklearn.datasets import fetch_california_housing

            fetch_california_housing(data_home=data_home, download_if_missing=True)
        elif key == "kddcup99":
            from sklearn.datasets import fetch_kddcup99

            fetch_kddcup99(
                data_home=data_home,
                subset="SA",
                percent10=True,
                download_if_missing=True,
            )
        else:
            from sklearn.datasets import fetch_covtype

            fetch_covtype(data_home=data_home, download_if_missing=True)
        return data_home
    if key in {"diabetes", "breast_cancer", "wine"}:
        root = ASSET_ROOT / key
        root.mkdir(parents=True, exist_ok=True)
        return root
    if key in {
        "cartpole",
        "cartpole_heavy",
        "cartpole_noisy",
        "mountaincar",
        "minatar_breakout",
        "minatar_seaquest",
        "minatar_asterix",
    }:
        # The deterministic classic-control environments are generated locally.
        root = ASSET_ROOT / key
        root.mkdir(parents=True, exist_ok=True)
        return root
    raise ValueError(f"Unsupported cross-domain asset: {key}")


def main() -> None:
    parser = argparse.ArgumentParser()
    valid_assets = sorted(
        [
            *DOWNLOADS,
            *FILE_COLLECTIONS,
            "wikitext2",
            "penn_treebank",
            "tiny_shakespeare",
            "california_housing",
            "kddcup99",
            "covertype",
            "diabetes",
            "breast_cancer",
            "wine",
            "cartpole",
            "cartpole_heavy",
            "mountaincar",
            "covertype_anomaly",
            "sensorless_drive_anomaly",
            "cartpole_noisy",
            "minatar_breakout",
            "minatar_seaquest",
            "minatar_asterix",
        ]
    )
    parser.add_argument("assets", nargs="*")
    parser.add_argument("--all", action="store_true")
    args = parser.parse_args()
    invalid = sorted(set(args.assets) - set(valid_assets))
    if invalid:
        parser.error(f"unsupported assets: {', '.join(invalid)}")
    selected = (
        valid_assets
        if args.all
        else args.assets
    )
    if not selected:
        parser.error("Select at least one asset or use --all.")
    for key in selected:
        print(f"{key}: {prepare_asset(key)}")


if __name__ == "__main__":
    main()
