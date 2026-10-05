from __future__ import annotations

import argparse
import copy
import json
import math
import random
import re
import sys
import zlib
from collections import Counter, deque
from pathlib import Path
from typing import Any

import mlflow
import numpy as np
import pandas as pd
import torch
from sklearn.compose import ColumnTransformer
from sklearn.datasets import (
    fetch_california_housing,
    fetch_covtype,
    fetch_kddcup99,
    load_breast_cancer,
    load_diabetes,
    load_wine,
)
from sklearn.metrics import f1_score, mean_squared_error, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from controller_benchmark.task_adapter import QualityDefinition, quality_definition

from .common import EpochTelemetry, decode_run_spec
from .cross_domain_assets import ASSET_ROOT, prepare_asset


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _gradient_norm(parameters) -> float:
    squared = 0.0
    for parameter in parameters:
        if parameter.grad is not None:
            value = float(parameter.grad.detach().float().norm(2).cpu())
            squared += value * value
    return math.sqrt(squared)


def _parameter_count(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())


def assert_cuda_model(model: nn.Module, device: torch.device, case_id: str) -> None:
    """Fail closed when a CUDA-required benchmark is not entirely on one GPU."""
    if device.type != "cuda":
        raise RuntimeError(f"Case {case_id} requires CUDA, but selected device is {device}.")
    expected_index = device.index if device.index is not None else torch.cuda.current_device()
    misplaced = []
    for name, tensor in [*model.named_parameters(), *model.named_buffers()]:
        if tensor.device.type != "cuda" or tensor.device.index != expected_index:
            misplaced.append(f"{name}={tensor.device}")
    if misplaced:
        preview = ", ".join(misplaced[:8])
        raise RuntimeError(
            f"Case {case_id} has model tensors outside cuda:{expected_index}: {preview}"
        )


def _pad_features(values: np.ndarray, width: int = 128) -> np.ndarray:
    if values.shape[1] > width:
        raise ValueError(f"Tabular feature width {values.shape[1]} exceeds {width}.")
    result = np.zeros((len(values), width), dtype=np.float32)
    result[:, : values.shape[1]] = values.astype(np.float32)
    return result


TOKEN_PATTERN = re.compile(r"[A-Za-z0-9']+|[^\w\s]")


def _hashed_token_ids(
    text: str,
    vocabulary_size: int,
    *,
    limit: int | None = None,
) -> list[int]:
    values: list[int] = []
    for match in TOKEN_PATTERN.finditer(text):
        token = match.group(0).lower()
        values.append(
            2 + zlib.crc32(token.encode("utf-8")) % max(1, vocabulary_size - 2)
        )
        if limit is not None and len(values) >= limit:
            break
    return values


def _loader(*arrays: np.ndarray, batch_size: int, shuffle: bool, seed: int) -> DataLoader:
    tensors = [torch.as_tensor(array) for array in arrays]
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(
        TensorDataset(*tensors),
        batch_size=batch_size,
        shuffle=shuffle,
        generator=generator if shuffle else None,
        num_workers=0,
        pin_memory=True,
    )


class MLP(nn.Module):
    def __init__(self, inputs: int, outputs: int):
        super().__init__()
        width = min(512, max(64, 2 ** math.ceil(math.log2(max(16, inputs)))))
        self.network = nn.Sequential(
            nn.Linear(inputs, width),
            nn.LayerNorm(width),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(width, width // 2),
            nn.GELU(),
            nn.Linear(width // 2, outputs),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.network(inputs)


class Autoencoder(nn.Module):
    def __init__(self, inputs: int):
        super().__init__()
        hidden = min(256, max(32, inputs * 2))
        latent = max(8, hidden // 4)
        self.network = nn.Sequential(
            nn.Linear(inputs, hidden),
            nn.ReLU(),
            nn.Linear(hidden, latent),
            nn.ReLU(),
            nn.Linear(latent, hidden),
            nn.ReLU(),
            nn.Linear(hidden, inputs),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.network(inputs)


class TextGRU(nn.Module):
    def __init__(self, vocabulary: int, classes: int = 2):
        super().__init__()
        self.embedding = nn.Embedding(vocabulary, 128, padding_idx=0)
        self.gru = nn.GRU(128, 192, batch_first=True, bidirectional=True)
        self.output = nn.Linear(384, classes)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        embedded = self.embedding(tokens)
        _, hidden = self.gru(embedded)
        return self.output(torch.cat((hidden[-2], hidden[-1]), dim=1))


class TransformerTextClassifier(nn.Module):
    def __init__(
        self,
        vocabulary: int,
        classes: int,
        *,
        sequence_length: int,
        width: int = 256,
        layers: int = 4,
        heads: int = 8,
    ):
        super().__init__()
        self.embedding = nn.Embedding(vocabulary, width, padding_idx=0)
        self.position = nn.Embedding(sequence_length, width)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=width,
            nhead=heads,
            dim_feedforward=width * 4,
            dropout=0.1,
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=layers)
        self.norm = nn.LayerNorm(width)
        self.output = nn.Linear(width, classes)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        positions = torch.arange(tokens.shape[1], device=tokens.device)
        mask = tokens == 0
        features = self.embedding(tokens) + self.position(positions)[None, :, :]
        encoded = self.encoder(features, src_key_padding_mask=mask)
        valid = (~mask).unsqueeze(-1).float()
        pooled = (encoded * valid).sum(dim=1) / valid.sum(dim=1).clamp_min(1.0)
        return self.output(self.norm(pooled))


class TransformerLanguageModel(nn.Module):
    def __init__(
        self,
        vocabulary: int,
        sequence_length: int,
        *,
        width: int = 192,
        layers: int = 3,
        heads: int = 6,
        feedforward: int | None = None,
    ):
        super().__init__()
        self.embedding = nn.Embedding(vocabulary, width)
        self.position = nn.Embedding(sequence_length, width)
        layer = nn.TransformerEncoderLayer(
            d_model=width,
            nhead=heads,
            dim_feedforward=feedforward or width * 4,
            dropout=0.1,
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=layers)
        self.output = nn.Linear(width, vocabulary)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        positions = torch.arange(tokens.shape[1], device=tokens.device)
        features = self.embedding(tokens) + self.position(positions)[None, :, :]
        return self.output(self.encoder(features))


class ForecastLSTM(nn.Module):
    def __init__(self, features: int, horizon: int = 1):
        super().__init__()
        self.lstm = nn.LSTM(features, 128, num_layers=2, dropout=0.1, batch_first=True)
        self.output = nn.Linear(128, horizon)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        values, _ = self.lstm(inputs)
        prediction = self.output(values[:, -1])
        return prediction.squeeze(-1) if prediction.shape[-1] == 1 else prediction


class TimeSeriesTransformer(nn.Module):
    def __init__(
        self,
        features: int,
        horizon: int,
        *,
        sequence_length: int,
        width: int = 256,
        layers: int = 4,
        heads: int = 8,
    ):
        super().__init__()
        self.input_projection = nn.Linear(features, width)
        self.position = nn.Embedding(sequence_length, width)
        layer = nn.TransformerEncoderLayer(
            d_model=width,
            nhead=heads,
            dim_feedforward=width * 4,
            dropout=0.1,
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=layers)
        self.output = nn.Sequential(nn.LayerNorm(width), nn.Linear(width, horizon))

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        positions = torch.arange(inputs.shape[1], device=inputs.device)
        features = self.input_projection(inputs) + self.position(positions)[None, :, :]
        prediction = self.output(self.encoder(features)[:, -1])
        return prediction.squeeze(-1) if prediction.shape[-1] == 1 else prediction


class MatrixFactorization(nn.Module):
    def __init__(self, users: int, items: int):
        super().__init__()
        self.user = nn.Embedding(users, 64)
        self.item = nn.Embedding(items, 64)
        self.user_bias = nn.Embedding(users, 1)
        self.item_bias = nn.Embedding(items, 1)
        self.global_bias = nn.Parameter(torch.zeros(()))

    def forward(self, pairs: torch.Tensor) -> torch.Tensor:
        users, items = pairs[:, 0], pairs[:, 1]
        score = (self.user(users) * self.item(items)).sum(dim=1)
        return score + self.user_bias(users).squeeze(1) + self.item_bias(items).squeeze(1) + self.global_bias


class NeuralCollaborativeFiltering(nn.Module):
    def __init__(self, users: int, items: int, *, embedding_dim: int = 128, width: int = 512):
        super().__init__()
        self.user = nn.Embedding(users, embedding_dim)
        self.item = nn.Embedding(items, embedding_dim)
        self.network = nn.Sequential(
            nn.Linear(embedding_dim * 2, width),
            nn.LayerNorm(width),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(width, width // 2),
            nn.GELU(),
            nn.Linear(width // 2, 1),
        )

    def forward(self, pairs: torch.Tensor) -> torch.Tensor:
        users, items = pairs[:, 0], pairs[:, 1]
        features = torch.cat((self.user(users), self.item(items)), dim=1)
        return self.network(features).squeeze(1)


class TrainingTask:
    def __init__(self, spec: dict[str, Any], device: torch.device):
        self.spec = spec
        self.case = spec["case"]
        self.device = device
        self.seed = int(spec["training_seed"])
        self.quality = quality_definition(self.case)
        self.model: nn.Module
        self.optimizer: torch.optim.Optimizer
        self.scheduler: Any
        self.primary_metric = self.quality.metric
        self.pretrained_loaded = False
        self.pretrained_source_case_id: str | None = None

    def apply_pretrained_if_requested(self) -> None:
        if not bool(self.case.get("pretrained", False)):
            return
        source = str(self.case.get("pretrained_source_case_id", "")).strip()
        if not source:
            raise ValueError("Pretrained case requires pretrained_source_case_id.")
        checkpoint = Path(
            self.case.get(
                "pretrained_checkpoint",
                f"/workspace-cache/controller-pretrained-cross-domain/{source}.pt",
            )
        )
        if not checkpoint.exists():
            raise FileNotFoundError(
                f"Pretrained source checkpoint is missing: {checkpoint}. "
                "Run prepare-cross-domain-pretrained-models-rancher.ps1 first."
            )
        document = torch.load(checkpoint, map_location="cpu", weights_only=True)
        source_state = document.get("model_state", document)
        target_state = self.model.state_dict()
        compatible = {
            key: value
            for key, value in source_state.items()
            if key in target_state and target_state[key].shape == value.shape
        }
        if not compatible:
            raise RuntimeError(f"No compatible weights in pretrained checkpoint {checkpoint}.")
        self.model.load_state_dict(compatible, strict=False)
        self.pretrained_loaded = True
        self.pretrained_source_case_id = source

    def train_epoch(self, checkpoint: int) -> dict[str, float]:
        raise NotImplementedError

    def validate(self) -> dict[str, float]:
        raise NotImplementedError

    def test(self) -> dict[str, float]:
        raise NotImplementedError

    def epoch(self, checkpoint: int) -> tuple[float, dict[str, float]]:
        train_metrics = self.train_epoch(checkpoint)
        validation = self.validate()
        raw_quality = float(validation[self.primary_metric])
        quality = self.quality.normalize(raw_quality)
        metrics = {
            **train_metrics,
            **validation,
            **self.quality.metadata(),
            "raw_quality_value": raw_quality,
            "quality_score": quality,
            "learning_rate": float(self.optimizer.param_groups[0]["lr"]),
            "model_parameter_count": _parameter_count(self.model),
            "model_flops": float(self.case.get("model_flops_estimate", 0.0)),
            "pretrained_weights_loaded": int(self.pretrained_loaded),
        }
        if self.scheduler is not None:
            self.scheduler.step()
        return quality, metrics


class TabularTask(TrainingTask):
    def __init__(self, spec: dict[str, Any], device: torch.device):
        super().__init__(spec, device)
        key = self.case["dataset_key"]
        if key in {"adult", "covertype", "bank_marketing", "sensorless_drive", "letter_recognition"}:
            loader = {
                "adult": self._adult,
                "covertype": self._covertype,
                "bank_marketing": self._bank_marketing,
                "sensorless_drive": self._sensorless_drive,
                "letter_recognition": self._letter_recognition,
            }[key]
            train_x, val_x, test_x, train_y, val_y, test_y = loader()
            self.mode = "classification"
            classes = int(max(train_y.max(), val_y.max(), test_y.max())) + 1
            self.model = MLP(128, classes).to(device)
            self.loss = nn.CrossEntropyLoss()
            y_dtype = np.int64
        elif key in {
            "california_housing",
            "diabetes",
            "bike_sharing",
            "year_prediction_msd",
            "online_news_popularity",
            "superconductivity",
        }:
            loader = {
                "california_housing": self._california,
                "diabetes": self._diabetes,
                "bike_sharing": self._bike_sharing,
                "year_prediction_msd": self._year_prediction_msd,
                "online_news_popularity": self._online_news_popularity,
                "superconductivity": self._superconductivity,
            }[key]
            train_x, val_x, test_x, train_y, val_y, test_y = loader()
            self.mode = "regression"
            self.model = MLP(128, 1).to(device)
            self.loss = nn.MSELoss()
            y_dtype = np.float32
        elif key in {
            "kddcup99",
            "breast_cancer_anomaly",
            "wine_anomaly",
            "covertype_anomaly",
            "sensorless_drive_anomaly",
        }:
            loader = {
                "kddcup99": self._kdd,
                "breast_cancer_anomaly": self._breast_cancer_anomaly,
                "wine_anomaly": self._wine_anomaly,
                "covertype_anomaly": self._covertype_anomaly,
                "sensorless_drive_anomaly": self._sensorless_drive_anomaly,
            }[key]
            train_x, val_x, test_x, train_y, val_y, test_y = loader()
            self.mode = "anomaly"
            self.model = Autoencoder(128).to(device)
            self.loss = nn.MSELoss()
            y_dtype = np.float32
        else:
            raise ValueError(f"Unsupported tabular dataset: {key}")
        train_x, val_x, test_x = (
            _pad_features(train_x),
            _pad_features(val_x),
            _pad_features(test_x),
        )
        self.apply_pretrained_if_requested()
        batch = int(self.case.get("batch_size", 256))
        if self.mode == "anomaly":
            normal = train_x[train_y == 0]
            self.train_loader = _loader(normal.astype(np.float32), batch_size=batch, shuffle=True, seed=self.seed)
        else:
            self.train_loader = _loader(train_x.astype(np.float32), train_y.astype(y_dtype), batch_size=batch, shuffle=True, seed=self.seed)
        self.validation = (torch.tensor(val_x, dtype=torch.float32), torch.tensor(val_y))
        self.testing = (torch.tensor(test_x, dtype=torch.float32), torch.tensor(test_y))
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=1e-3, weight_decay=1e-4)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=int(spec["execution"]["max_epochs"]), eta_min=1e-5
        )

    def _adult(self):
        root = prepare_asset("adult")
        names = [
            "age", "workclass", "fnlwgt", "education", "education_num", "marital_status",
            "occupation", "relationship", "race", "sex", "capital_gain", "capital_loss",
            "hours_per_week", "native_country", "income",
        ]
        train = pd.read_csv(root / "adult.data", names=names, na_values="?", skipinitialspace=True).dropna()
        test = pd.read_csv(root / "adult.test", names=names, na_values="?", skipinitialspace=True, skiprows=1).dropna()

        def separate(frame):
            frame = frame.copy()
            labels = frame.pop("income").astype(str).str.replace(".", "", regex=False).str.contains(">50K").astype(np.int64)
            return frame, labels.to_numpy()

        train_x, train_y = separate(train)
        test_x, test_y = separate(test)
        train_x, val_x, train_y, val_y = train_test_split(
            train_x,
            train_y,
            test_size=0.15,
            random_state=self.seed,
            stratify=train_y,
        )
        categorical = list(train_x.select_dtypes(include=["object"]).columns)
        numeric = [column for column in train_x.columns if column not in categorical]
        transformer = ColumnTransformer(
            [("num", StandardScaler(), numeric), ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical)]
        )
        return (
            transformer.fit_transform(train_x).astype(np.float32),
            transformer.transform(val_x).astype(np.float32),
            transformer.transform(test_x).astype(np.float32),
            train_y,
            val_y,
            test_y,
        )

    def _california(self):
        root = prepare_asset("california_housing")
        dataset = fetch_california_housing(data_home=root)
        train_x, remainder_x, train_y, remainder_y = train_test_split(
            dataset.data,
            dataset.target,
            test_size=0.30,
            random_state=self.seed,
        )
        val_x, test_x, val_y, test_y = train_test_split(
            remainder_x,
            remainder_y,
            test_size=0.50,
            random_state=self.seed,
        )
        scaler = StandardScaler().fit(train_x)
        return (
            scaler.transform(train_x).astype(np.float32),
            scaler.transform(val_x).astype(np.float32),
            scaler.transform(test_x).astype(np.float32),
            train_y.astype(np.float32),
            val_y.astype(np.float32),
            test_y.astype(np.float32),
        )

    def _covertype(self):
        root = prepare_asset("covertype")
        dataset = fetch_covtype(data_home=root)
        labels = dataset.target.astype(np.int64) - 1
        return self._numeric_split(dataset.data, labels, stratify=labels)

    def _bank_marketing(self):
        root = prepare_asset("bank_marketing")
        path = next(root.rglob("bank-full.csv"))
        frame = pd.read_csv(path, sep=";")
        labels = (frame.pop("y") == "yes").astype(np.int64).to_numpy()
        return self._frame_split(frame, labels, stratify=labels)

    def _sensorless_drive(self):
        root = prepare_asset("sensorless_drive")
        path = next(root.rglob("Sensorless_drive_diagnosis.txt"))
        frame = pd.read_csv(path, sep=r"\s+", header=None)
        labels = frame.iloc[:, -1].to_numpy(dtype=np.int64) - 1
        return self._numeric_split(frame.iloc[:, :-1].to_numpy(dtype=np.float32), labels, stratify=labels)

    def _letter_recognition(self):
        root = prepare_asset("letter_recognition")
        path = next(root.rglob("letter-recognition.data"))
        frame = pd.read_csv(path, header=None)
        labels = frame.iloc[:, 0].astype(str).map(lambda value: ord(value.upper()) - ord("A")).to_numpy(dtype=np.int64)
        return self._numeric_split(frame.iloc[:, 1:].to_numpy(dtype=np.float32), labels, stratify=labels)

    def _diabetes(self):
        prepare_asset("diabetes")
        dataset = load_diabetes()
        return self._numeric_split(dataset.data, dataset.target.astype(np.float32))

    def _bike_sharing(self):
        root = prepare_asset("bike_sharing")
        path = next(root.rglob("hour.csv"))
        frame = pd.read_csv(path)
        labels = frame.pop("cnt").to_numpy(dtype=np.float32)
        frame = frame.drop(columns=["dteday", "casual", "registered"], errors="ignore")
        return self._numeric_split(frame.to_numpy(dtype=np.float32), labels)

    def _year_prediction_msd(self):
        root = prepare_asset("year_prediction_msd")
        path = next(root.rglob("YearPredictionMSD.txt"))
        frame = pd.read_csv(path, header=None, dtype=np.float32)
        labels = frame.iloc[:, 0].to_numpy(dtype=np.float32)
        return self._numeric_split(frame.iloc[:, 1:].to_numpy(dtype=np.float32), labels)

    def _online_news_popularity(self):
        root = prepare_asset("online_news_popularity")
        path = next(root.rglob("OnlineNewsPopularity.csv"))
        frame = pd.read_csv(path)
        frame.columns = frame.columns.astype(str).str.strip()
        labels = np.log1p(frame.pop("shares").to_numpy(dtype=np.float32))
        frame = frame.drop(columns=["url", "timedelta"], errors="ignore")
        return self._numeric_split(frame.to_numpy(dtype=np.float32), labels)

    def _superconductivity(self):
        root = prepare_asset("superconductivity")
        path = next(path for path in root.rglob("train.csv") if path.is_file())
        frame = pd.read_csv(path)
        labels = frame.pop("critical_temp").to_numpy(dtype=np.float32)
        return self._numeric_split(frame.to_numpy(dtype=np.float32), labels)

    def _breast_cancer_anomaly(self):
        prepare_asset("breast_cancer")
        dataset = load_breast_cancer()
        # Malignant samples are the positive anomaly class.
        labels = (dataset.target == 0).astype(np.int64)
        return self._numeric_split(dataset.data, labels, stratify=labels)

    def _wine_anomaly(self):
        prepare_asset("wine")
        dataset = load_wine()
        counts = Counter(dataset.target.tolist())
        anomaly_class = min(counts, key=counts.get)
        labels = (dataset.target == anomaly_class).astype(np.int64)
        return self._numeric_split(dataset.data, labels, stratify=labels)

    def _covertype_anomaly(self):
        root = prepare_asset("covertype_anomaly")
        dataset = fetch_covtype(data_home=root)
        counts = Counter(dataset.target.tolist())
        anomaly_class = min(counts, key=counts.get)
        labels = (dataset.target == anomaly_class).astype(np.int64)
        return self._numeric_split(dataset.data, labels, stratify=labels)

    def _sensorless_drive_anomaly(self):
        root = prepare_asset("sensorless_drive_anomaly")
        path = next(root.rglob("Sensorless_drive_diagnosis.txt"))
        frame = pd.read_csv(path, sep=r"\s+", header=None)
        raw_labels = frame.iloc[:, -1].to_numpy(dtype=np.int64)
        counts = Counter(raw_labels.tolist())
        anomaly_class = min(counts, key=counts.get)
        labels = (raw_labels == anomaly_class).astype(np.int64)
        return self._numeric_split(frame.iloc[:, :-1].to_numpy(dtype=np.float32), labels, stratify=labels)

    def _limit_rows(self, values, labels):
        maximum = int(self.case.get("max_tabular_samples", 0))
        if maximum <= 0 or len(labels) <= maximum:
            return values, labels
        indices = np.random.default_rng(self.seed).choice(len(labels), size=maximum, replace=False)
        if isinstance(values, pd.DataFrame):
            values = values.iloc[indices].reset_index(drop=True)
        else:
            values = np.asarray(values)[indices]
        return values, np.asarray(labels)[indices]

    def _numeric_split(self, values, labels, stratify=None):
        values, labels = self._limit_rows(values, labels)
        stratify = labels if stratify is not None else None
        train_x, val_x, test_x, train_y, val_y, test_y = self._split(
            np.asarray(values), np.asarray(labels), stratify=stratify
        )
        if bool(self.case.get("standardize_regression_target", False)):
            target_mean = float(np.mean(train_y))
            target_std = max(float(np.std(train_y)), 1e-9)
            train_y = (train_y - target_mean) / target_std
            val_y = (val_y - target_mean) / target_std
            test_y = (test_y - target_mean) / target_std
        scaler = StandardScaler().fit(train_x)
        return (
            scaler.transform(train_x).astype(np.float32),
            scaler.transform(val_x).astype(np.float32),
            scaler.transform(test_x).astype(np.float32),
            train_y,
            val_y,
            test_y,
        )

    def _frame_split(self, frame: pd.DataFrame, labels, stratify=None):
        frame, labels = self._limit_rows(frame, labels)
        stratify = labels if stratify is not None else None
        train_x, remainder_x, train_y, remainder_y = train_test_split(
            frame,
            labels,
            test_size=0.30,
            random_state=self.seed,
            stratify=stratify,
        )
        remainder_stratify = remainder_y if stratify is not None else None
        val_x, test_x, val_y, test_y = train_test_split(
            remainder_x,
            remainder_y,
            test_size=0.50,
            random_state=self.seed,
            stratify=remainder_stratify,
        )
        categorical = list(train_x.select_dtypes(include=["object"]).columns)
        numeric = [column for column in train_x.columns if column not in categorical]
        transformer = ColumnTransformer(
            [
                ("num", StandardScaler(), numeric),
                (
                    "cat",
                    OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                    categorical,
                ),
            ]
        )
        return (
            transformer.fit_transform(train_x).astype(np.float32),
            transformer.transform(val_x).astype(np.float32),
            transformer.transform(test_x).astype(np.float32),
            np.asarray(train_y),
            np.asarray(val_y),
            np.asarray(test_y),
        )

    def _kdd(self):
        root = prepare_asset("kddcup99")
        subset = None if self.case.get("kdd_subset") == "full" else "SA"
        dataset = fetch_kddcup99(data_home=root, subset=subset, percent10=True)
        frame = pd.DataFrame(dataset.data)
        for column in frame.columns:
            if frame[column].dtype == object:
                frame[column] = frame[column].map(lambda value: value.decode("utf-8", errors="ignore") if isinstance(value, bytes) else value)
        labels = np.array([0 if value == b"normal." else 1 for value in dataset.target], dtype=np.int64)
        frame, labels = self._limit_rows(frame, labels)
        train_x, remainder_x, train_y, remainder_y = train_test_split(
            frame,
            labels,
            test_size=0.30,
            random_state=self.seed,
            stratify=labels,
        )
        val_x, test_x, val_y, test_y = train_test_split(
            remainder_x,
            remainder_y,
            test_size=0.50,
            random_state=self.seed,
            stratify=remainder_y,
        )
        categorical = list(train_x.select_dtypes(include=["object"]).columns)
        numeric = [column for column in train_x.columns if column not in categorical]
        transformer = ColumnTransformer(
            [("num", StandardScaler(), numeric), ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical)]
        )
        return (
            transformer.fit_transform(train_x).astype(np.float32),
            transformer.transform(val_x).astype(np.float32),
            transformer.transform(test_x).astype(np.float32),
            train_y,
            val_y,
            test_y,
        )

    def _split(self, x, y, stratify=None):
        train_x, remainder_x, train_y, remainder_y = train_test_split(
            x, y, test_size=0.30, random_state=self.seed, stratify=stratify
        )
        rest_stratify = remainder_y if stratify is not None else None
        val_x, test_x, val_y, test_y = train_test_split(
            remainder_x, remainder_y, test_size=0.50, random_state=self.seed, stratify=rest_stratify
        )
        return train_x, val_x, test_x, train_y, val_y, test_y

    def train_epoch(self, checkpoint: int) -> dict[str, float]:
        self.model.train()
        losses, norms = [], []
        for batch in self.train_loader:
            inputs = batch[0].to(self.device, non_blocking=True)
            self.optimizer.zero_grad(set_to_none=True)
            if self.mode == "anomaly":
                loss = self.loss(self.model(inputs), inputs)
            else:
                targets = batch[1].to(self.device, non_blocking=True)
                outputs = self.model(inputs)
                if self.mode == "regression":
                    outputs = outputs.squeeze(1)
                loss = self.loss(outputs, targets)
            loss.backward()
            norms.append(_gradient_norm(self.model.parameters()))
            self.optimizer.step()
            losses.append(float(loss.detach().cpu()))
        return {"train_loss": float(np.mean(losses)), "gradient_norm": float(np.mean(norms))}

    def _metrics(self, data) -> dict[str, float]:
        inputs, targets = data
        self.model.eval()
        with torch.no_grad():
            inputs_device = inputs.to(self.device)
            if self.mode == "classification":
                prediction = self.model(inputs_device).argmax(dim=1).cpu().numpy()
                return {"macro_f1": float(f1_score(targets.numpy(), prediction, average="macro"))}
            if self.mode == "regression":
                prediction = self.model(inputs_device).squeeze(1).cpu().numpy()
                rmse = math.sqrt(mean_squared_error(targets.numpy(), prediction))
                scale = max(float(np.std(targets.numpy())), 1e-9)
                return {"nrmse": rmse / scale, "rmse": rmse}
            errors = ((self.model(inputs_device) - inputs_device) ** 2).mean(dim=1).cpu().numpy()
            return {"auroc": float(roc_auc_score(targets.numpy(), errors))}

    def validate(self) -> dict[str, float]:
        return self._metrics(self.validation)

    def test(self) -> dict[str, float]:
        return {f"test_{key}": value for key, value in self._metrics(self.testing).items()}


class TextClassificationTask(TrainingTask):
    def __init__(self, spec: dict[str, Any], device: torch.device):
        super().__init__(spec, device)
        key = self.case["dataset_key"]
        predefined_validation = False
        if key == "imdb":
            root = prepare_asset("imdb") / "aclImdb"
            train_rows = self._read_imdb(root / "train")
            test_rows = self._read_imdb(root / "test")
        elif key in {"ag_news", "dbpedia", "yelp_review_full"}:
            train_rows, test_rows = self._read_csv_text(key)
        elif key in {"go_emotions", "multi_eurlex", "marc_multilingual"}:
            train_rows, validation_rows, test_rows = self._read_parquet_text(key)
            predefined_validation = True
        else:
            raise ValueError(f"Unsupported text classification dataset: {key}")
        random.Random(self.seed).shuffle(train_rows)
        if not predefined_validation:
            validation_count = int(self.case.get("validation_samples", 2500))
            validation_rows = train_rows[:validation_count]
            train_rows = train_rows[validation_count:]
        train_rows = train_rows[: int(self.case.get("max_train_samples", 10000))]
        validation_rows = validation_rows[: int(self.case.get("validation_samples", 2500))]
        test_rows = test_rows[: int(self.case.get("max_test_samples", 5000))]
        vocabulary_size = int(self.case.get("vocabulary_size", 20000))
        sequence_length = int(self.case.get("sequence_length", 256))
        self.multilabel = bool(self.case.get("multilabel", False))
        configured_classes = int(self.case.get("num_classes", 0))

        def encode(rows):
            encoded = []
            for text, _ in rows:
                values = _hashed_token_ids(
                    text,
                    vocabulary_size,
                    limit=sequence_length,
                )
                encoded.append(values + [0] * (sequence_length - len(values)))
            if self.multilabel:
                if configured_classes <= 0:
                    raise ValueError("Multi-label text cases require num_classes.")
                targets = np.zeros((len(rows), configured_classes), dtype=np.float32)
                for row_index, (_, labels) in enumerate(rows):
                    targets[row_index, np.asarray(labels, dtype=np.int64)] = 1.0
            else:
                targets = np.asarray([label for _, label in rows], dtype=np.int64)
            return np.asarray(encoded, dtype=np.int64), targets

        train_x, train_y = encode(train_rows)
        val_x, val_y = encode(validation_rows)
        test_x, test_y = encode(test_rows)
        self.train_loader = _loader(train_x, train_y, batch_size=int(self.case.get("batch_size", 64)), shuffle=True, seed=self.seed)
        self.validation = (torch.tensor(val_x), torch.tensor(val_y))
        self.testing = (torch.tensor(test_x), torch.tensor(test_y))
        classes = (
            configured_classes
            if self.multilabel
            else int(max(train_y.max(), val_y.max(), test_y.max())) + 1
        )
        if str(self.case.get("model_version", "")).startswith("transformer-text"):
            self.model = TransformerTextClassifier(
                vocabulary_size,
                classes=classes,
                sequence_length=sequence_length,
                width=int(self.case.get("model_width", 256)),
                layers=int(self.case.get("model_layers", 4)),
                heads=int(self.case.get("attention_heads", 8)),
            ).to(device)
        else:
            self.model = TextGRU(vocabulary_size, classes=classes).to(device)
        self.apply_pretrained_if_requested()
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=float(self.case.get("learning_rate", 2e-3)),
            weight_decay=float(self.case.get("weight_decay", 1e-4)),
        )
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=int(spec["execution"]["max_epochs"]), eta_min=1e-5)
        self.loss = nn.BCEWithLogitsLoss() if self.multilabel else nn.CrossEntropyLoss()

    def _read_imdb(self, root: Path):
        rows = []
        for label, name in enumerate(("neg", "pos")):
            for path in sorted((root / name).glob("*.txt")):
                rows.append((path.read_text(encoding="utf-8", errors="ignore"), label))
        return rows

    def _read_csv_text(self, key: str):
        root = prepare_asset(key)
        folder_name = {
            "ag_news": "ag_news_csv",
            "dbpedia": "dbpedia_csv",
            "yelp_review_full": "yelp_review_full_csv",
        }[key]
        folder = root / folder_name
        train_limit = int(self.case.get("max_train_samples", 20000)) + int(
            self.case.get("validation_samples", 5000)
        )
        test_limit = int(self.case.get("max_test_samples", 5000))

        default_classes = {"ag_news": 4, "dbpedia": 14, "yelp_review_full": 5}[key]
        classes = int(self.case.get("num_classes", default_classes))

        def read(path: Path, limit: int):
            per_class = int(math.ceil(limit / classes))
            selected: dict[int, list[tuple[str, int]]] = {
                label: [] for label in range(classes)
            }
            for frame in pd.read_csv(path, header=None, chunksize=50000):
                labels = frame.iloc[:, 0].astype(np.int64) - 1
                text = frame.iloc[:, 1:].fillna("").astype(str).agg(" ".join, axis=1)
                for label in range(classes):
                    remaining = per_class - len(selected[label])
                    if remaining <= 0:
                        continue
                    positions = np.flatnonzero(labels.to_numpy() == label)[:remaining]
                    selected[label].extend(
                        (text.iloc[position], label) for position in positions
                    )
                if all(len(rows) >= per_class for rows in selected.values()):
                    break
            rows = [row for label in range(classes) for row in selected[label]]
            random.Random(self.seed).shuffle(rows)
            return rows[:limit]

        return read(folder / "train.csv", train_limit), read(
            folder / "test.csv", test_limit
        )

    def _read_parquet_text(self, key: str):
        root = prepare_asset(key)

        def read_split(split: str):
            frame = pd.read_parquet(root / f"{split}.parquet")
            if key == "marc_multilingual":
                text = (
                    frame["review_title"].fillna("").astype(str)
                    + " "
                    + frame["review_body"].fillna("").astype(str)
                )
                labels = frame["stars"].astype(np.int64) - 1
                rows = list(zip(text.tolist(), labels.tolist()))
            else:
                rows = [
                    (str(text), [int(label) for label in labels])
                    for text, labels in zip(frame["text"], frame["labels"])
                ]
            random.Random(self.seed + {"train": 1, "validation": 2, "test": 3}[split]).shuffle(rows)
            return rows

        return read_split("train"), read_split("validation"), read_split("test")

    def train_epoch(self, checkpoint: int) -> dict[str, float]:
        self.model.train()
        losses, norms = [], []
        for inputs, targets in self.train_loader:
            self.optimizer.zero_grad(set_to_none=True)
            loss = self.loss(self.model(inputs.to(self.device)), targets.to(self.device))
            loss.backward()
            norms.append(_gradient_norm(self.model.parameters()))
            nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()
            losses.append(float(loss.detach().cpu()))
        return {"train_loss": float(np.mean(losses)), "gradient_norm": float(np.mean(norms))}

    def _metrics(self, data):
        inputs, targets = data
        predictions = []
        self.model.eval()
        with torch.no_grad():
            for start in range(0, len(inputs), 256):
                logits = self.model(inputs[start : start + 256].to(self.device))
                if self.multilabel:
                    predictions.extend((torch.sigmoid(logits) >= 0.5).int().cpu().tolist())
                else:
                    predictions.extend(logits.argmax(1).cpu().tolist())
        return {
            "macro_f1": float(
                f1_score(
                    targets.numpy(),
                    np.asarray(predictions),
                    average="macro",
                    zero_division=0,
                )
            )
        }

    def validate(self):
        return self._metrics(self.validation)

    def test(self):
        return {f"test_{key}": value for key, value in self._metrics(self.testing).items()}


class LanguageModelTask(TrainingTask):
    def __init__(self, spec: dict[str, Any], device: torch.device):
        super().__init__(spec, device)
        key = self.case["dataset_key"]
        vocabulary_size = int(self.case.get("vocabulary_size", 12000))
        sequence_length = int(self.case.get("sequence_length", 64))
        max_tokens = int(self.case.get("max_train_tokens", 300000))
        train_text, valid_text, test_text = self._corpus(
            key,
            train_character_limit=max_tokens * 32,
        )
        def encode(text, limit):
            values = np.asarray(
                _hashed_token_ids(text, vocabulary_size, limit=limit), dtype=np.int64
            )
            count = (len(values) - 1) // sequence_length
            x = np.stack([values[i * sequence_length : (i + 1) * sequence_length] for i in range(count)])
            y = np.stack([values[i * sequence_length + 1 : (i + 1) * sequence_length + 1] for i in range(count)])
            return x, y
        train_x, train_y = encode(train_text, max_tokens)
        val_x, val_y = encode(valid_text, 60000)
        test_x, test_y = encode(test_text, 60000)
        self.train_loader = _loader(train_x, train_y, batch_size=int(self.case.get("batch_size", 32)), shuffle=True, seed=self.seed)
        self.validation_loader = _loader(val_x, val_y, batch_size=64, shuffle=False, seed=self.seed)
        self.test_loader = _loader(test_x, test_y, batch_size=64, shuffle=False, seed=self.seed)
        self.model = TransformerLanguageModel(
            vocabulary_size,
            sequence_length,
            width=int(self.case.get("model_width", 192)),
            layers=int(self.case.get("model_layers", 3)),
            heads=int(self.case.get("attention_heads", 6)),
        ).to(device)
        self.apply_pretrained_if_requested()
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=float(self.case.get("learning_rate", 3e-4)),
            weight_decay=float(self.case.get("weight_decay", 0.01)),
        )
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=int(spec["execution"]["max_epochs"]), eta_min=1e-5)
        self.loss = nn.CrossEntropyLoss()

    @staticmethod
    def _read_text(path: Path, character_limit: int | None = None) -> str:
        with path.open("r", encoding="utf-8", errors="ignore") as handle:
            return handle.read(character_limit)

    def _corpus(
        self,
        key: str,
        *,
        train_character_limit: int | None = None,
    ) -> tuple[str, str, str]:
        if key == "wikitext103":
            root = prepare_asset(key)
            corpus = next(root.rglob("wikitext-103-raw"))
            return (
                self._read_text(corpus / "wiki.train.raw", train_character_limit),
                self._read_text(corpus / "wiki.valid.raw"),
                self._read_text(corpus / "wiki.test.raw"),
            )
        if key == "wikitext2":
            root = prepare_asset(key) / "wikitext-2-raw"
            return (
                self._read_text(root / "wiki.train.raw", train_character_limit),
                self._read_text(root / "wiki.valid.raw"),
                self._read_text(root / "wiki.test.raw"),
            )
        if key == "penn_treebank":
            root = prepare_asset(key)
            return (
                self._read_text(root / "ptb.train.txt", train_character_limit),
                self._read_text(root / "ptb.valid.txt"),
                self._read_text(root / "ptb.test.txt"),
            )
        if key == "tiny_shakespeare":
            text = (prepare_asset(key) / "input.txt").read_text(encoding="utf-8")
            first, second = int(len(text) * 0.80), int(len(text) * 0.90)
            return text[:first], text[first:second], text[second:]
        if key in {"c4_en_shard", "lm1b_shard"}:
            root = prepare_asset(key)

            def parquet_text(path: Path, limit: int | None = None) -> str:
                frame = pd.read_parquet(path, columns=["text"])
                parts: list[str] = []
                characters = 0
                for value in frame["text"]:
                    piece = str(value)
                    if limit is not None and characters >= limit:
                        break
                    parts.append(piece)
                    characters += len(piece) + 1
                result = "\n".join(parts)
                return result if limit is None else result[:limit]

            train = parquet_text(root / "train.parquet", train_character_limit)
            evaluation_path = root / (
                "validation.parquet" if key == "c4_en_shard" else "test.parquet"
            )
            evaluation = parquet_text(evaluation_path, 4_000_000)
            midpoint = len(evaluation) // 2
            return train, evaluation[:midpoint], evaluation[midpoint:]
        raise ValueError(f"Unsupported language modeling dataset: {key}")

    def train_epoch(self, checkpoint: int):
        self.model.train()
        losses, norms = [], []
        for inputs, targets in self.train_loader:
            self.optimizer.zero_grad(set_to_none=True)
            logits = self.model(inputs.to(self.device))
            loss = self.loss(logits.flatten(0, 1), targets.to(self.device).flatten())
            loss.backward()
            norms.append(_gradient_norm(self.model.parameters()))
            nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()
            losses.append(float(loss.detach().cpu()))
        return {"train_loss": float(np.mean(losses)), "gradient_norm": float(np.mean(norms))}

    def _metrics(self, loader):
        losses = []
        self.model.eval()
        with torch.no_grad():
            for inputs, targets in loader:
                logits = self.model(inputs.to(self.device))
                losses.append(float(self.loss(logits.flatten(0, 1), targets.to(self.device).flatten()).cpu()))
        loss = float(np.mean(losses))
        return {"cross_entropy": loss, "perplexity": float(math.exp(min(20.0, loss)))}

    def validate(self):
        return self._metrics(self.validation_loader)

    def test(self):
        return {f"test_{key}": value for key, value in self._metrics(self.test_loader).items()}


class TimeSeriesTask(TrainingTask):
    def __init__(self, spec: dict[str, Any], device: torch.device):
        super().__init__(spec, device)
        key = self.case["dataset_key"]
        filenames = {
            "etth1": "ETTh1.csv",
            "etth2": "ETTh2.csv",
            "ettm1": "ETTm1.csv",
            "ettm2": "ETTm2.csv",
        }
        length = int(self.case.get("sequence_length", 96))
        horizon = int(self.case.get("forecast_horizon", 1))
        if horizon < 1:
            raise ValueError("forecast_horizon must be positive")
        if key in filenames:
            path = prepare_asset(key) / filenames[key]
            frame = pd.read_csv(path).drop(columns=["date"], errors="ignore")
            values = frame.to_numpy(dtype=np.float32)
            train_end, validation_end = int(len(values) * 0.70), int(len(values) * 0.85)
            mean, std = values[:train_end].mean(0), values[:train_end].std(0).clip(1e-6)
            values = (values - mean) / std
            target_column = frame.columns.get_loc("OT") if "OT" in frame.columns else values.shape[1] - 1

            def windows(start, end):
                x, y = [], []
                for index in range(max(start, length), end - horizon + 1):
                    x.append(values[index - length : index])
                    y.append(values[index : index + horizon, target_column])
                targets = np.asarray(y, dtype=np.float32)
                if horizon == 1:
                    targets = targets[:, 0]
                return np.asarray(x, dtype=np.float32), targets

            train_x, train_y = windows(0, train_end)
            val_x, val_y = windows(train_end - length, validation_end)
            test_x, test_y = windows(validation_end - length, len(values))
            feature_count = values.shape[1]
        elif key.startswith("monash_"):
            series = self._read_monash_series(prepare_asset(key))
            train_x, train_y, val_x, val_y, test_x, test_y = self._monash_windows(
                series,
                length=length,
                horizon=horizon,
                max_train=int(self.case.get("max_train_windows", 30000)),
                max_validation=int(self.case.get("max_validation_windows", 5000)),
                max_test=int(self.case.get("max_test_windows", 5000)),
            )
            feature_count = 1
        else:
            raise ValueError(f"Unsupported time-series dataset: {key}")
        self.train_loader = _loader(train_x, train_y, batch_size=int(self.case.get("batch_size", 128)), shuffle=True, seed=self.seed)
        self.validation = (torch.tensor(val_x), torch.tensor(val_y))
        self.testing = (torch.tensor(test_x), torch.tensor(test_y))
        if str(self.case.get("model_version", "")).startswith("timeseries-transformer"):
            self.model = TimeSeriesTransformer(
                feature_count,
                horizon=horizon,
                sequence_length=length,
                width=int(self.case.get("model_width", 256)),
                layers=int(self.case.get("model_layers", 4)),
                heads=int(self.case.get("attention_heads", 8)),
            ).to(device)
        else:
            self.model = ForecastLSTM(feature_count, horizon=horizon).to(device)
        self.apply_pretrained_if_requested()
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=float(self.case.get("learning_rate", 1e-3)),
            weight_decay=float(self.case.get("weight_decay", 1e-4)),
        )
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=int(spec["execution"]["max_epochs"]), eta_min=1e-5)
        self.loss = nn.MSELoss()

    @staticmethod
    def _read_monash_series(root: Path) -> list[np.ndarray]:
        path = next(root.rglob("*.tsf"), None)
        if path is None:
            raise FileNotFoundError(f"No Monash .tsf file found below {root}.")
        values: list[np.ndarray] = []
        in_data = False
        with path.open("r", encoding="utf-8", errors="ignore") as handle:
            for raw_line in handle:
                line = raw_line.strip()
                if not line or line.startswith("#"):
                    continue
                if line.lower() == "@data":
                    in_data = True
                    continue
                if not in_data or line.startswith("@"):
                    continue
                encoded = line.rsplit(":", 1)[-1]
                array = np.asarray(
                    [float(item) if item != "?" else np.nan for item in encoded.split(",")],
                    dtype=np.float32,
                )
                if np.isnan(array).all():
                    continue
                if np.isnan(array).any():
                    fill = float(np.nanmean(array))
                    array = np.nan_to_num(array, nan=fill)
                values.append(array)
        if not values:
            raise ValueError(f"No usable series found in {path}.")
        return values

    def _monash_windows(
        self,
        series: list[np.ndarray],
        *,
        length: int,
        horizon: int,
        max_train: int,
        max_validation: int,
        max_test: int,
    ):
        normalized: list[np.ndarray] = []
        for values in series:
            train_end = int(len(values) * 0.70)
            mean = float(values[:train_end].mean())
            std = max(float(values[:train_end].std()), 1e-6)
            normalized.append((values - mean) / std)

        def sample(split: str, count: int, seed_offset: int):
            rng = np.random.default_rng(self.seed + seed_offset)
            candidates: list[tuple[int, int, int]] = []
            for series_index, values in enumerate(normalized):
                train_end = int(len(values) * 0.70)
                validation_end = int(len(values) * 0.85)
                if split == "train":
                    low, high = length, train_end - horizon + 1
                elif split == "validation":
                    low, high = max(length, train_end), validation_end - horizon + 1
                else:
                    low, high = max(length, validation_end), len(values) - horizon + 1
                if high > low:
                    candidates.append((series_index, low, high))
            if not candidates:
                raise ValueError(f"No {split} windows fit length={length}, horizon={horizon}.")
            x = np.empty((count, length, 1), dtype=np.float32)
            y = np.empty((count, horizon), dtype=np.float32)
            for row in range(count):
                series_index, low, high = candidates[int(rng.integers(0, len(candidates)))]
                end = int(rng.integers(low, high))
                values = normalized[series_index]
                x[row, :, 0] = values[end - length : end]
                y[row] = values[end : end + horizon]
            return x, y[:, 0] if horizon == 1 else y

        train_x, train_y = sample("train", max_train, 11)
        val_x, val_y = sample("validation", max_validation, 12)
        test_x, test_y = sample("test", max_test, 13)
        return train_x, train_y, val_x, val_y, test_x, test_y

    def train_epoch(self, checkpoint: int):
        self.model.train()
        losses, norms = [], []
        for inputs, targets in self.train_loader:
            self.optimizer.zero_grad(set_to_none=True)
            loss = self.loss(self.model(inputs.to(self.device)), targets.to(self.device))
            loss.backward()
            norms.append(_gradient_norm(self.model.parameters()))
            self.optimizer.step()
            losses.append(float(loss.detach().cpu()))
        return {"train_loss": float(np.mean(losses)), "gradient_norm": float(np.mean(norms))}

    def _metrics(self, data):
        inputs, targets = data
        predictions = []
        self.model.eval()
        with torch.no_grad():
            for start in range(0, len(inputs), 512):
                predictions.extend(self.model(inputs[start : start + 512].to(self.device)).cpu().tolist())
        rmse = math.sqrt(mean_squared_error(targets.numpy(), predictions))
        return {"nrmse": rmse / max(float(targets.std()), 1e-9), "rmse": rmse}

    def validate(self):
        return self._metrics(self.validation)

    def test(self):
        return {f"test_{key}": value for key, value in self._metrics(self.testing).items()}


class RecommendationTask(TrainingTask):
    def __init__(self, spec: dict[str, Any], device: torch.device):
        super().__init__(spec, device)
        key = self.case["dataset_key"]
        root = prepare_asset(key)
        predefined_splits = None
        if key == "movielens100k":
            path = root / "ml-100k" / "u.data"
            frame = pd.read_csv(path, sep="\t", names=["user", "item", "rating", "timestamp"])
        elif key in {"movielens1m", "movielens10m"}:
            folder = "ml-1m" if key == "movielens1m" else "ml-10M100K"
            path = root / folder / "ratings.dat"
            frame = pd.read_csv(
                path,
                sep="::",
                engine="python",
                names=["user", "item", "rating", "timestamp"],
            )
        elif key in {"movielens20m", "movielens25m"}:
            folder = "ml-20m" if key == "movielens20m" else "ml-25m"
            path = root / folder / "ratings.csv"
            frame = pd.read_csv(path).rename(columns={"userId": "user", "movieId": "item"})
        elif key in {"amazon_books_5core", "amazon_electronics_5core"}:
            def amazon_split(name: str):
                return pd.read_csv(root / f"{name}.csv").rename(
                    columns={"user_id": "user", "parent_asin": "item"}
                )

            predefined_splits = (
                amazon_split("train"),
                amazon_split("validation"),
                amazon_split("test"),
            )
            frame = predefined_splits[0]
        else:
            raise ValueError(f"Unsupported recommendation dataset: {key}")
        max_samples = int(self.case.get("max_samples", 0))
        user_buckets = int(self.case.get("user_buckets", 8192))
        item_buckets = int(self.case.get("item_buckets", 16384))
        self.ranking = self.primary_metric == "ndcg_at_10"

        def prepare_frame(data: pd.DataFrame, limit: int = 0):
            if limit > 0 and len(data) > limit:
                data = data.sample(n=limit, random_state=self.seed).reset_index(drop=True)
            data = data.copy()
            data["user"] = (
                pd.util.hash_pandas_object(data["user"].astype(str), index=False).to_numpy()
                % user_buckets
            )
            data["item"] = (
                pd.util.hash_pandas_object(data["item"].astype(str), index=False).to_numpy()
                % item_buckets
            )
            return data

        if predefined_splits is None:
            frame = prepare_frame(frame, max_samples)
            train, remainder = train_test_split(frame, test_size=0.30, random_state=self.seed)
            validation, testing = train_test_split(remainder, test_size=0.50, random_state=self.seed)
        else:
            train = prepare_frame(predefined_splits[0], max_samples)
            evaluation_limit = int(self.case.get("max_evaluation_interactions", 5000))
            validation = prepare_frame(predefined_splits[1], evaluation_limit)
            testing = prepare_frame(predefined_splits[2], evaluation_limit)
        pair = lambda data: data[["user", "item"]].to_numpy(dtype=np.int64)
        target = lambda data: data["rating"].to_numpy(dtype=np.float32)
        if self.ranking:
            train = train[train["rating"] >= 4.0].copy()
            validation = validation[validation["rating"] >= 4.0].copy()
            testing = testing[testing["rating"] >= 4.0].copy()
            positive_pairs = pair(train)
            rng = np.random.default_rng(self.seed + 101)
            negative_pairs = positive_pairs.copy()
            negative_pairs[:, 1] = rng.integers(0, item_buckets, size=len(negative_pairs))
            training_pairs = np.concatenate((positive_pairs, negative_pairs), axis=0)
            training_targets = np.concatenate(
                (np.ones(len(positive_pairs)), np.zeros(len(negative_pairs)))
            ).astype(np.float32)
            permutation = rng.permutation(len(training_pairs))
            self.train_loader = _loader(
                training_pairs[permutation],
                training_targets[permutation],
                batch_size=int(self.case.get("batch_size", 1024)),
                shuffle=True,
                seed=self.seed,
            )
            evaluation_limit = int(self.case.get("max_evaluation_interactions", 2000))
            self.validation = torch.tensor(pair(validation)[:evaluation_limit])
            self.testing = torch.tensor(pair(testing)[:evaluation_limit])
        else:
            self.train_loader = _loader(pair(train), target(train), batch_size=int(self.case.get("batch_size", 1024)), shuffle=True, seed=self.seed)
            self.validation = (torch.tensor(pair(validation)), torch.tensor(target(validation)))
            self.testing = (torch.tensor(pair(testing)), torch.tensor(target(testing)))
        if str(self.case.get("model_version", "")).startswith("neural-cf"):
            self.model = NeuralCollaborativeFiltering(
                user_buckets,
                item_buckets,
                embedding_dim=int(self.case.get("embedding_dim", 128)),
                width=int(self.case.get("model_width", 512)),
            ).to(device)
        else:
            self.model = MatrixFactorization(user_buckets, item_buckets).to(device)
        self.apply_pretrained_if_requested()
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=float(self.case.get("learning_rate", 3e-3)),
            weight_decay=float(self.case.get("weight_decay", 1e-5)),
        )
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=int(spec["execution"]["max_epochs"]), eta_min=1e-5)
        self.loss = nn.BCEWithLogitsLoss() if self.ranking else nn.MSELoss()
        self.item_buckets = item_buckets

    def train_epoch(self, checkpoint: int):
        self.model.train()
        losses, norms = [], []
        for pairs, targets in self.train_loader:
            self.optimizer.zero_grad(set_to_none=True)
            loss = self.loss(self.model(pairs.to(self.device)), targets.to(self.device))
            loss.backward()
            norms.append(_gradient_norm(self.model.parameters()))
            self.optimizer.step()
            losses.append(float(loss.detach().cpu()))
        return {"train_loss": float(np.mean(losses)), "gradient_norm": float(np.mean(norms))}

    def _metrics(self, data):
        if self.ranking:
            return self._ranking_metrics(data)
        pairs, targets = data
        predictions = []
        self.model.eval()
        with torch.no_grad():
            for start in range(0, len(pairs), 4096):
                predictions.extend(self.model(pairs[start : start + 4096].to(self.device)).cpu().tolist())
        rmse = math.sqrt(mean_squared_error(targets.numpy(), predictions))
        return {"rmse": rmse}

    def _ranking_metrics(self, positive_pairs: torch.Tensor):
        rng = np.random.default_rng(self.seed + 202)
        ndcg, hits = [], []
        self.model.eval()
        with torch.no_grad():
            for user, positive_item in positive_pairs.numpy():
                candidates = rng.integers(0, self.item_buckets, size=100, dtype=np.int64)
                candidates[0] = positive_item
                pairs = np.column_stack(
                    (np.full(100, user, dtype=np.int64), candidates)
                )
                scores = self.model(torch.tensor(pairs, device=self.device)).cpu().numpy()
                rank = int(np.count_nonzero(scores[1:] > scores[0]))
                hits.append(float(rank < 10))
                ndcg.append(1.0 / math.log2(rank + 2.0) if rank < 10 else 0.0)
        return {"ndcg_at_10": float(np.mean(ndcg)), "hit_rate_at_10": float(np.mean(hits))}

    def validate(self):
        return self._metrics(self.validation)

    def test(self):
        return {f"test_{key}": value for key, value in self._metrics(self.testing).items()}


class CartPoleEnvironment:
    """Small deterministic CartPole implementation for dependency-free RL runs."""

    def __init__(
        self,
        seed: int,
        *,
        gravity: float = 9.8,
        mass_pole: float = 0.1,
        observation_noise: float = 0.0,
        gravity_jitter: float = 0.0,
    ):
        self.random = np.random.default_rng(seed)
        self.state = np.zeros(4, dtype=np.float32)
        self.steps = 0
        self.gravity = gravity
        self.episode_gravity = gravity
        self.mass_pole = mass_pole
        self.observation_noise = observation_noise
        self.gravity_jitter = gravity_jitter

    def _observation(self) -> np.ndarray:
        if self.observation_noise <= 0:
            return self.state.copy()
        noise = self.random.normal(0.0, self.observation_noise, size=4).astype(np.float32)
        return self.state.copy() + noise

    def reset(self, seed: int | None = None) -> np.ndarray:
        if seed is not None:
            self.random = np.random.default_rng(seed)
        jitter = self.random.uniform(-self.gravity_jitter, self.gravity_jitter)
        self.episode_gravity = self.gravity * (1.0 + jitter)
        self.state = self.random.uniform(-0.05, 0.05, size=4).astype(np.float32)
        self.steps = 0
        return self._observation()

    def step(self, action: int) -> tuple[np.ndarray, float, bool]:
        gravity, mass_cart, mass_pole, half_length, force, tau = self.episode_gravity, 1.0, self.mass_pole, 0.5, 10.0, 0.02
        total_mass = mass_cart + mass_pole
        pole_mass_length = mass_pole * half_length
        x, x_dot, theta, theta_dot = map(float, self.state)
        applied = force if action == 1 else -force
        cosine, sine = math.cos(theta), math.sin(theta)
        temporary = (applied + pole_mass_length * theta_dot**2 * sine) / total_mass
        theta_acceleration = (gravity * sine - cosine * temporary) / (
            half_length * (4.0 / 3.0 - mass_pole * cosine**2 / total_mass)
        )
        x_acceleration = temporary - pole_mass_length * theta_acceleration * cosine / total_mass
        x += tau * x_dot
        x_dot += tau * x_acceleration
        theta += tau * theta_dot
        theta_dot += tau * theta_acceleration
        self.state = np.array([x, x_dot, theta, theta_dot], dtype=np.float32)
        self.steps += 1
        done = bool(abs(x) > 2.4 or abs(theta) > 12 * math.pi / 180 or self.steps >= 500)
        return self._observation(), 1.0, done


class MountainCarEnvironment:
    def __init__(self, seed: int):
        self.random = np.random.default_rng(seed)
        self.state = np.zeros(2, dtype=np.float32)
        self.steps = 0

    def reset(self, seed: int | None = None) -> np.ndarray:
        if seed is not None:
            self.random = np.random.default_rng(seed)
        self.state = np.array([self.random.uniform(-0.6, -0.4), 0.0], dtype=np.float32)
        self.steps = 0
        return self.state.copy()

    def step(self, action: int) -> tuple[np.ndarray, float, bool]:
        position, velocity = map(float, self.state)
        velocity += (action - 1) * 0.001 + math.cos(3.0 * position) * -0.0025
        velocity = max(-0.07, min(0.07, velocity))
        position = max(-1.2, min(0.6, position + velocity))
        if position <= -1.2 and velocity < 0:
            velocity = 0.0
        self.state = np.array([position, velocity], dtype=np.float32)
        self.steps += 1
        done = bool(position >= 0.5 or self.steps >= 200)
        return self.state.copy(), -1.0, done


class DqnNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(6, 128), nn.ReLU(), nn.Linear(128, 128), nn.ReLU(), nn.Linear(128, 3)
        )

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.network(state)


class MinAtarDqnNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(
            nn.Conv2d(10, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(64 * 10 * 10, 256),
            nn.ReLU(),
            nn.Linear(256, 6),
        )

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.network(state)


class MinAtarEnvironment:
    def __init__(self, game: str, seed: int, *, maximum_steps: int = 5000):
        dependency_root = Path("/workspace-cache/controller-python-deps/hardest-v1")
        if dependency_root.exists() and str(dependency_root) not in sys.path:
            sys.path.insert(0, str(dependency_root))
        from minatar import Environment

        self.environment = Environment(game, sticky_action_prob=0.1, difficulty_ramping=True)
        self.maximum_steps = maximum_steps
        self.steps = 0
        self.environment.seed(seed)

    def reset(self, seed: int | None = None) -> np.ndarray:
        if seed is not None:
            self.environment.seed(seed)
        self.environment.reset()
        self.steps = 0
        return self.environment.state().astype(np.float32)

    def step(self, action: int) -> tuple[np.ndarray, float, bool]:
        reward, terminal = self.environment.act(int(action))
        self.steps += 1
        done = bool(terminal or self.steps >= self.maximum_steps)
        return self.environment.state().astype(np.float32), float(reward), done


class ReinforcementLearningTask(TrainingTask):
    def __init__(self, spec: dict[str, Any], device: torch.device):
        super().__init__(spec, device)
        key = self.case["dataset_key"]
        prepare_asset(key)
        self.is_minatar = False
        if key == "cartpole":
            self.environment = CartPoleEnvironment(self.seed)
            self.action_count = 2
            self.maximum_reward = 500.0
        elif key == "cartpole_heavy":
            self.environment = CartPoleEnvironment(self.seed, gravity=12.0, mass_pole=0.5)
            self.action_count = 2
            self.maximum_reward = 500.0
        elif key == "cartpole_noisy":
            self.environment = CartPoleEnvironment(
                self.seed,
                mass_pole=0.25,
                observation_noise=0.02,
                gravity_jitter=0.25,
            )
            self.action_count = 2
            self.maximum_reward = 500.0
        elif key == "mountaincar":
            self.environment = MountainCarEnvironment(self.seed)
            self.action_count = 3
            self.maximum_reward = 200.0
        elif key in {"minatar_breakout", "minatar_seaquest", "minatar_asterix"}:
            game = key.removeprefix("minatar_")
            self.environment = MinAtarEnvironment(
                game,
                self.seed,
                maximum_steps=int(self.case.get("maximum_episode_steps", 5000)),
            )
            self.action_count = 6
            self.maximum_reward = float(self.case.get("reward_scale", 20.0))
            self.is_minatar = True
        else:
            raise ValueError(f"Unsupported reinforcement-learning environment: {key}")
        network = MinAtarDqnNetwork if self.is_minatar else DqnNetwork
        self.model = network().to(device)
        self.target = network().to(device)
        self.target.load_state_dict(self.model.state_dict())
        self.apply_pretrained_if_requested()
        self.target.load_state_dict(self.model.state_dict())
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=1e-3)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=int(spec["execution"]["max_epochs"]), eta_min=1e-5
        )
        self.loss = nn.SmoothL1Loss()
        self.replay: deque[tuple[np.ndarray, int, float, np.ndarray, float]] = deque(maxlen=50000)
        self.episodes_per_checkpoint = int(self.case.get("episodes_per_checkpoint", 5))
        self.evaluation_episodes = int(self.case.get("evaluation_episodes", 10))
        self.training_episode = 0

    def _action(self, state: np.ndarray, epsilon: float) -> int:
        if random.random() < epsilon:
            return random.randrange(self.action_count)
        with torch.no_grad():
            value = self.model(self._state_tensor(state).unsqueeze(0))
        return int(value[:, : self.action_count].argmax(dim=1).item())

    def _state_tensor(self, state: np.ndarray) -> torch.Tensor:
        if self.is_minatar:
            padded = np.zeros((10, 10, 10), dtype=np.float32)
            padded[:, :, : state.shape[2]] = state
            return torch.tensor(np.moveaxis(padded, -1, 0), device=self.device)
        padded = np.zeros(6, dtype=np.float32)
        padded[: len(state)] = state
        return torch.tensor(padded, device=self.device)

    def _optimize(self) -> tuple[float | None, float | None]:
        if len(self.replay) < 256:
            return None, None
        batch = random.sample(self.replay, 128)
        states = torch.stack([self._state_tensor(row[0]) for row in batch])
        actions = torch.tensor([row[1] for row in batch], device=self.device).long()
        rewards = torch.tensor([row[2] for row in batch], device=self.device)
        next_states = torch.stack([self._state_tensor(row[3]) for row in batch])
        active = torch.tensor([row[4] for row in batch], device=self.device)
        estimate = self.model(states).gather(1, actions[:, None]).squeeze(1)
        with torch.no_grad():
            target = rewards + 0.99 * active * self.target(next_states)[:, : self.action_count].max(dim=1).values
        self.optimizer.zero_grad(set_to_none=True)
        loss = self.loss(estimate, target)
        loss.backward()
        norm = _gradient_norm(self.model.parameters())
        nn.utils.clip_grad_norm_(self.model.parameters(), 10.0)
        self.optimizer.step()
        return float(loss.detach().cpu()), norm

    def train_epoch(self, checkpoint: int):
        losses, norms = [], []
        epsilon = max(0.02, 1.0 - checkpoint / max(1, self.spec["execution"]["max_epochs"] * 0.60))
        self.model.train()
        for _ in range(self.episodes_per_checkpoint):
            state = self.environment.reset(self.seed + self.training_episode)
            self.training_episode += 1
            done = False
            while not done:
                action = self._action(state, epsilon)
                next_state, reward, done = self.environment.step(action)
                self.replay.append((state, action, reward, next_state, 0.0 if done else 1.0))
                state = next_state
                loss, norm = self._optimize()
                if loss is not None:
                    losses.append(loss)
                    norms.append(float(norm or 0.0))
        if checkpoint % 2 == 0:
            self.target.load_state_dict(self.model.state_dict())
        return {
            "train_loss": float(np.mean(losses)) if losses else 0.0,
            "gradient_norm": float(np.mean(norms)) if norms else 0.0,
            "exploration_epsilon": epsilon,
        }

    def _evaluation(self, seed_offset: int) -> dict[str, float]:
        rewards = []
        self.model.eval()
        for episode in range(self.evaluation_episodes):
            state = self.environment.reset(seed_offset + episode)
            total = 0.0
            done = False
            while not done:
                action = self._action(state, 0.0)
                state, reward, done = self.environment.step(action)
                total += reward
            rewards.append(total)
        return {
            "normalized_eval_reward": (
                float(1.0 - math.exp(-max(0.0, float(np.mean(rewards))) / self.maximum_reward))
                if self.is_minatar
                else float(np.mean(rewards) / self.maximum_reward)
                if np.mean(rewards) >= 0
                else float(1.0 + np.mean(rewards) / self.maximum_reward)
            ),
            "mean_eval_reward": float(np.mean(rewards)),
            "std_eval_reward": float(np.std(rewards)),
        }

    def validate(self):
        return self._evaluation(100000 + self.seed * 100)

    def test(self):
        return {f"test_{key}": value for key, value in self._evaluation(200000 + self.seed * 100).items()}


def build_task(spec: dict[str, Any], device: torch.device) -> TrainingTask:
    family = spec["case"]["task_family"]
    if family == "tabular":
        return TabularTask(spec, device)
    if family == "text_classification":
        return TextClassificationTask(spec, device)
    if family == "language_modeling":
        return LanguageModelTask(spec, device)
    if family == "time_series_forecasting":
        return TimeSeriesTask(spec, device)
    if family == "recommendation":
        return RecommendationTask(spec, device)
    if family == "reinforcement_learning":
        return ReinforcementLearningTask(spec, device)
    raise ValueError(f"Unsupported cross-domain task family: {family}")


def run(spec: dict[str, Any], output_dir: Path) -> None:
    telemetry = EpochTelemetry(spec, output_dir)
    telemetry.start()
    seed = int(spec["training_seed"])
    _seed_everything(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if bool(spec["case"].get("require_cuda", False)) and device.type != "cuda":
        raise RuntimeError(
            f"Case {spec['case']['id']} requires CUDA, but no GPU is available."
        )
    task = build_task(spec, device)
    if bool(spec["case"].get("require_cuda", False)):
        assert_cuda_model(task.model, device, str(spec["case"]["id"]))
    model_dir = Path("/workspace-cache/model_registry") / f"job_{telemetry.job_id}"
    model_dir.mkdir(parents=True, exist_ok=True)
    best_quality = -1.0
    best_state: dict[str, torch.Tensor] | None = None
    mlflow.set_tracking_uri(str(spec["execution"].get("mlflow_tracking_uri", "http://mlflow:5000")))
    mlflow.set_experiment(spec["execution"]["experiment_name"])
    try:
        with mlflow.start_run(run_name=f"job-energy-{telemetry.job_id}"):
            mlflow.log_params(
                {
                    "runner": spec["runner"],
                    "dataset": spec["case"]["dataset_key"],
                    "task_type": spec["task_type"],
                    "quality_metric": task.primary_metric,
                    "quality_transform": task.quality.transform,
                    "controller_id": spec["controller_id"],
                    "benchmark_case_id": spec["benchmark_case_id"],
                    "training_seed": seed,
                    "model": spec["case"]["model_version"],
                    "pretrained": bool(spec["case"].get("pretrained", False)),
                    "device": device.type,
                    "cuda_required": bool(spec["case"].get("require_cuda", False)),
                    "progress_unit": spec["case"].get("progress_unit", "epoch"),
                }
            )
            for checkpoint in range(1, int(spec["execution"]["max_epochs"]) + 1):
                telemetry.start_epoch()
                cuda_start = None
                cuda_end = None
                if device.type == "cuda":
                    torch.cuda.synchronize(device)
                    torch.cuda.reset_peak_memory_stats(device)
                    cuda_start = torch.cuda.Event(enable_timing=True)
                    cuda_end = torch.cuda.Event(enable_timing=True)
                    cuda_start.record()
                quality, metrics = task.epoch(checkpoint)
                if device.type == "cuda":
                    cuda_end.record()
                    torch.cuda.synchronize(device)
                    metrics.update(
                        {
                            "cuda_training_elapsed_ms": float(cuda_start.elapsed_time(cuda_end)),
                            "cuda_memory_allocated_mb": float(
                                torch.cuda.memory_allocated(device) / (1024.0**2)
                            ),
                            "cuda_peak_memory_allocated_mb": float(
                                torch.cuda.max_memory_allocated(device) / (1024.0**2)
                            ),
                            "cuda_device_index": float(torch.cuda.current_device()),
                            "cuda_execution_verified": 1.0,
                        }
                    )
                if quality > best_quality:
                    best_quality = quality
                    best_state = copy.deepcopy(task.model.state_dict())
                mlflow.log_metrics(
                    {
                        f"checkpoint/{key}": float(value)
                        for key, value in metrics.items()
                        if isinstance(value, (int, float)) and math.isfinite(float(value))
                    },
                    step=checkpoint,
                )
                if telemetry.finish_epoch(checkpoint, quality, metrics):
                    break
            summary, gpu = telemetry.finalize()
            if best_state is not None:
                torch.save(best_state, model_dir / "best.pt")
                task.model.load_state_dict(best_state)
            test_metrics = task.test()
            summary.update(
                {
                    "primary_metric": task.primary_metric,
                    "dataset_name": spec["case"]["dataset_name"],
                    "dataset_key": spec["case"]["dataset_key"],
                    "quality_definition": task.quality.metadata(),
                    **test_metrics,
                }
            )
            telemetry.epoch_summary_path.write_text(
                json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
                encoding="utf-8",
            )
            mlflow.log_metrics(
                {
                    **{key: value for key, value in test_metrics.items()},
                    "best_validation_quality_score": summary["best_quality_score"],
                    "gpu_energy_wh": gpu["gpu_energy_kwh"] * 1000.0,
                }
            )
            mlflow.log_artifact(str(telemetry.epoch_summary_path), artifact_path="energy")
            mlflow.log_artifact(str(telemetry.gpu_summary_path), artifact_path="energy")
            if (model_dir / "best.pt").exists():
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
