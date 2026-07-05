import json
import logging
import shutil
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any, Literal, get_args

import albumentations as A
import cv2
import mlflow
import numpy as np
import omegaconf
import torch
import yaml
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import v2
from tqdm import tqdm

logger = logging.getLogger(__name__)


class LabelCreator(ABC):
	"""
	Create labels in dot / xyxy / xywh format for a dataset.
	Returns a list of object counts per image. Use dataset specific class"""

	def __init__(self, dataset_root: Path):
		self.dst_root = dataset_root / "labels"
		self.raw_path = dataset_root / "raw"
		self.dots_dir = self.dst_root / "dots"
		self.xyxy_dir = self.dst_root / "xyxy"
		self.xywh_dir = self.dst_root / "xywh"
		for d in (self.dots_dir, self.xyxy_dir, self.xywh_dir):
			d.mkdir(parents=True, exist_ok=True)

	@abstractmethod
	def create_labels(self):
		"""Create labels in dot / xyxy / xywh format for a dataset."""
		pass


SetType = Literal["train", "val", "test"]


class DatasetSplitter(ABC):
	"""Base class for dataset splitting into train/val/test sets."""

	def __init__(self, dataset_root: Path):
		self.img_dest = dataset_root / "images"
		self.label_root = dataset_root / "labels"
		self.label_xyxy = self.label_root / "xyxy"
		self.label_xywh = self.label_root / "xywh"
		self.raw_dir = dataset_root / "raw"
		self.raw_images: list[Path] = []
		self.label_files: list[Path] = []

		self._reset_split_dirs()

	@abstractmethod
	def split(
		self,
		train_size: float | int = 0.8,
		seed: int = 1,
		limit_train_size: int | bool = False,
	):
		"""
		Splits synthetic cell data into train/val/test folders for Ultralytics YOLO compatibility.
		- Test set: Always the same 20% test images
		- Train and val sets: Randomly split from the remaining data based on and `seed` parameter, stratified by count.
		Args:
			train_size: Float between 0-1 to indicate how much of the available data is used for training vs. validation
			limit_train_size: limit the number of train images for train size experiments. Does not influence val size
		"""
		pass

	def _stratified_train_val_split(
		self,
		counts: np.ndarray,
		train_size: float = 0.8,
		limit_train_size: int | bool = False,
		fixed_test_indices: bool = False,
		seed: int = 1,
	) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
		"""
		Creates stratified train/val/test splits based on object counts.
		- If fixed_test_indices is given, only train/val are stratified and test is skipped;
			counts is then expected to contain only counts for train/val data.
		- Otherwise a 20% test split is created automatically.
		Returns train_indices, val_indices, test_indices.
		"""
		if isinstance(train_size, int) & (train_size > 1):
			train_size = train_size / len(self.raw_images)
			if train_size > 1:
				ValueError(f"Fraction for training is larger than 1. train_size = {train_size}")

		indices = np.arange(len(counts))
		num_bins = 10
		quantiles = np.linspace(0, 1, num_bins + 1)[:-1]
		bin_edges = np.unique(np.quantile(counts, quantiles))
		bins = np.digitize(counts, bin_edges, right=False)

		def validate_bin_count(bins: np.ndarray) -> np.ndarray | None:
			_, bin_count = np.unique(bins, return_counts=True)
			if np.any(bin_count < 2):  # ensure each bin contains more than one value, else set stratify to False
				logger.info("train data is not stratified -> not enough data")
				return None
			return bins

		if not fixed_test_indices:  # default 20% test split with fixed seed
			# split into train_val and test
			bins = validate_bin_count(bins)
			try:
				train_val_idx, test_idx = train_test_split(indices, test_size=0.2, random_state=1, stratify=bins)
				bins_train_val = bins[train_val_idx] if bins is not None else None
			except ValueError:
				logger.warning("Could not stratify train/test split. Falling back to unstratified sampling.")
				train_val_idx, test_idx = train_test_split(indices, test_size=0.2, random_state=1, stratify=None)
				bins_train_val = None
		else:  # skip test split
			test_idx = None
			train_val_idx = indices
			bins_train_val = None

		# stratify train/val from remaining data -> 2 step split ensures same val data for different train_limit
		try:
			train_idx, val_idx = train_test_split(
				train_val_idx, train_size=train_size, random_state=seed, stratify=bins_train_val
			)
		except ValueError:
			train_idx, val_idx = train_test_split(
				train_val_idx, train_size=train_size, random_state=seed, stratify=None
			)
			logger.warning(f"Could not stratify train val data. Not enough data samples: {len(train_val_idx)}")

		if limit_train_size:
			if bins is not None:
				bins = validate_bin_count(bins[train_idx])
			try:
				train_idx, _ = train_test_split(  # discard remaining train data
					train_idx,
					train_size=limit_train_size,
					random_state=seed,
					stratify=bins,
				)
			except ValueError:
				logger.warning("Train size limit too small for stratification. Falling back to unstratified sampling.")
				train_idx, _ = train_test_split(
					train_idx,
					train_size=limit_train_size,
					random_state=seed,
					stratify=None,
				)

		return train_idx, val_idx, test_idx

	def _reset_split_dirs(self):
		# remove only splits (train, val, test), not all labels
		for split in get_args(SetType):
			set_path_img = self.img_dest / split
			set_path_label = self.label_root / split

			shutil.rmtree(set_path_img, ignore_errors=True)
			shutil.rmtree(set_path_label, ignore_errors=True)
			# create dirs
			set_path_img.mkdir(parents=True, exist_ok=True)
			set_path_label.mkdir(parents=True, exist_ok=True)

	def _load_counts_from_labels(self, file_paths: None | list[Path] = None) -> np.ndarray:
		"""Reads object counts per image from existing label JSONs.
		If file_paths is set their names are used as labelnames (param file_paths required for carpk where only count
		for trainval is required)
		"""
		if file_paths:
			label_files = [self.label_xyxy / file.with_suffix(".json").name for file in file_paths]
		else:
			label_files = sorted(self.label_xyxy.glob("*.json"))
		counts = []
		for file in label_files:
			try:
				with file.open() as f:
					data = json.load(f)
				counts.append(int(data.get("count", 0)))
			except Exception:
				logger.error(f"Error reading Label {file} to receive count. Stratify split failed.")
		if not counts:
			raise ValueError(f"No JSON label files found in {self.label_xyxy}")
		return np.array(counts)

	def _copy_and_log(self, img_list, label_list, subset_name: str):
		for img, lab in zip(img_list, label_list, strict=False):
			dest_img = self.img_dest / subset_name / img.name
			src_xywh_label = self.label_xywh / lab.with_suffix(".txt").name
			dest_xywh_label = (self.label_root / subset_name / lab.name).with_suffix(".txt")
			try:
				shutil.copy2(img, dest_img)
				shutil.copy2(src_xywh_label, dest_xywh_label)

			except FileNotFoundError:  # Not all images need to be present
				continue


@dataclass
class SyncedTransformConfig:
	dataset_name: str = "carpk"
	img_size: int | tuple[int, int] = 256
	tilesize: int | None = 224
	sigma: float = 2.0
	flip_rotate: bool = False
	infer_normalisation: bool = False
	degree: int = 0
	mean: tuple[float, float, float] = (0.485, 0.456, 0.406)
	std: tuple[float, float, float] = (0.229, 0.224, 0.225)
	alpha: float = field(default=100)  # Not part of __init__, computed only
	centercrop: int | None = field(default=None)
	keypoint_adjustment: int = field(default=0)

	def __post_init__(self):
		"""Compute derived values and validate"""
		# Always compute alpha from sigma
		self.alpha = 25 * self.sigma**2

		if isinstance(self.img_size, tuple) and len(self.img_size) != 2:
			raise ValueError(f"size tuple must have 2 elements, got {len(self.img_size)}")
		if self.degree < 0 or self.degree > 180:
			raise ValueError(f"degree must be in [0, 180], got {self.degree}")

	@classmethod
	def validate_dict(cls, param_dict: dict) -> dict:
		valid_fields = {f.name for f in fields(cls)}
		config_dict = {k: v for k, v in param_dict.items() if k in valid_fields}
		# convert lists to tuples
		for key in ["img_size", "mean", "std"]:
			if key in config_dict and (
				isinstance(config_dict[key], list) or omegaconf.OmegaConf.is_list(config_dict[key])
			):
				config_dict[key] = tuple(config_dict[key])
		return config_dict

	@staticmethod
	def tuples_to_lists(config_dict: dict) -> dict:
		for key in ["img_size", "mean", "std"]:
			if key in config_dict and isinstance(config_dict[key], tuple):
				config_dict[key] = list(config_dict[key])
		return config_dict

	@classmethod
	def from_yaml(cls, *paths: str | Path):
		"""Load config from YAMLs"""
		merged_config = {}

		# Load and merge all config files
		for path in paths:
			if isinstance(path, str):
				path = Path(path)
			assert path.exists()
			assert path.suffix == ".yaml"

			with open(path) as f:
				config = yaml.safe_load(f) or {}
			merged_config.update(config)

		config_dict = cls.validate_dict(merged_config)
		return cls(**config_dict)

	@classmethod
	def from_dict(cls, param_dict: dict):
		"""Load directly from dict"""
		config_dict = cls.validate_dict(param_dict)
		return cls(**config_dict)

	def log_locally(self, checkpoint: Path):
		try:
			config_dict = asdict(self)
			config_dict = self.tuples_to_lists(config_dict)
			with open(checkpoint, "w") as f:
				yaml.dump(config_dict, f)
		except Exception as e:
			logger.warning(f"Unable to save transform config locally. /n {e}")

	def log_to_mlflow(self):
		"""Log config to MLflow"""
		config_dict = asdict(self)
		config_dict = self.tuples_to_lists(config_dict)
		# Log as params (good for filtering/comparing runs)
		mlflow.log_params({f"transform_{k}": v for k, v in config_dict.items()})

		# Also log as a dict artifact (preserves structure)
		mlflow.log_dict(config_dict, "configs/transform_config.yaml")


class SyncedTransform(torch.nn.Module):
	"""Transforms images to tensors with synchronised data augmentation in training data between
	images and dot annotations. Applies gaussian blur to dot annotations

	Args:
			config: Custom Dataclass that collects the transform configuration
	"""

	def __init__(self, config: SyncedTransformConfig):
		super().__init__()
		self.config = config
		if config.flip_rotate:
			if config.tilesize:  # Square symmtery only works for squares
				symmetries = A.SquareSymmetry(p=0.5)
			else:  # Else we do flipping which covers 180 degree rotations but not change the dimensions of the image
				symmetries = A.Compose(
					[  # 50% no augment, equal chance for horizontal, vertical or both flips
						A.VerticalFlip(p=0.5),
						A.HorizontalFlip(p=0.5),
					],
					p=0.4,
				)
		else:
			symmetries = A.NoOp()
		self.train_transform = A.Compose(  # TODO: Add parameter options for adjusting the augmentations
			[
				A.CenterCrop(height=config.centercrop, width=config.centercrop) if config.centercrop else A.NoOp(p=1.0),
				(
					A.Resize(config.img_size[1], config.img_size[0], interpolation=cv2.INTER_AREA)
					if isinstance(config.img_size, tuple)
					else A.SmallestMaxSize(config.img_size, interpolation=cv2.INTER_AREA)
				),
				(  # Cropping to square if using tiles
					A.RandomCrop(
						height=config.tilesize, width=config.tilesize, p=1.0, pad_if_needed=True, pad_position="center"
					)
					if config.tilesize
					else A.NoOp(p=1.0)
				),
				symmetries,
				A.Rotate(limit=config.degree, p=0.5),
				A.Normalize(mean=config.mean, std=config.std),
				A.ToTensorV2(),
			],
			keypoint_params=A.KeypointParams("xy"),
		)
		self.val_transform = A.Compose(
			[
				A.CenterCrop(height=config.centercrop, width=config.centercrop) if config.centercrop else A.NoOp(p=1),
				(
					A.Resize(config.img_size[1], config.img_size[0], interpolation=cv2.INTER_AREA)
					if isinstance(config.img_size, tuple)
					else A.SmallestMaxSize(config.img_size, interpolation=cv2.INTER_AREA)
				),
				# Pad validation images to at least tile_size for inference
				(
					A.PadIfNeeded(
						min_height=config.tilesize, min_width=config.tilesize, border_mode=cv2.BORDER_CONSTANT, fill=0
					)
					if config.tilesize
					else A.NoOp(p=1)
				),
				A.Normalize(mean=config.mean, std=config.std),
				A.ToTensorV2(),
			],
			keypoint_params=A.KeypointParams("xy"),
		)

		if config.sigma > 0:
			kernel_size = 2 * round(4 * config.sigma) + 1
			self.gaussian_kernel = v2.GaussianBlur(kernel_size=kernel_size, sigma=config.sigma)
		if config.sigma == 0:
			self.gaussian_kernel = v2.Lambda(lambda x: x)

	def create_density_map(self, dot_annotations: np.ndarray, img_size: tuple[int, int, int]):
		h, w = img_size[1:]
		density_map = torch.zeros((1, h, w))
		for y, x in dot_annotations:
			(x, y) = min(max(x, 0), h - 1), min(max(y, 0), w - 1)  # Fix Albumentations edge cases
			density_map[:, x, y] += self.config.alpha
		return density_map

	def pad_target_if_needed(self, target: torch.Tensor, padded_shape: tuple[int, int]):
		"""Pad target tensor to match padded image dimensions"""
		if target is None:
			return None

		_, h, w = target.shape
		target_h, target_w = padded_shape

		if h < target_h or w < target_w:
			pad_h = max(0, target_h - h)
			pad_w = max(0, target_w - w)
			target = torch.nn.functional.pad(target, (0, pad_w, 0, pad_h))

		return target

	def forward(self, image: np.ndarray, dot_annotations: dict[str, np.ndarray] | None, mode="train"):
		transform = self.train_transform if mode == "train" else self.val_transform
		if dot_annotations is None:
			transformed_items = transform(image=np.array(image, dtype=np.uint8))
			image_transformed = transformed_items["image"]
			return image_transformed, None
		else:
			# Increase every label point if the labels were based on cropped images instead of full ones
			dot_label = np.array(dot_annotations["center"]) + self.config.keypoint_adjustment

			transformed_items = transform(image=np.array(image, dtype=np.uint8), keypoints=dot_label)
			image_transformed = transformed_items["image"]
			dot_annotations_transformed = np.array(transformed_items["keypoints"]).round().astype(int)
			density_map = self.create_density_map(dot_annotations_transformed, img_size=image_transformed.shape)
			density_map = self.gaussian_kernel(density_map)

			if mode != "train" and self.config.tilesize:
				density_map = self.pad_target_if_needed(density_map, image_transformed.shape[1:])

			return image_transformed, density_map


# Usecase: Zellschädigung. Read dataset, make dataloaders and if necessary, produce labels
ReturnFormat = Literal["image_label", "image_only", "path_label", "path_only", "label_only"]


def collect_paths(image_path: Path) -> list[Path]:
	"Finds all files ending with .png or .RB.TIF (usecase)"
	files = sorted([p for p in image_path.rglob("*") if any(p.match(pat) for pat in ("*.png", "*.RB.TIF"))])
	return files


class CountingDataset(Dataset):
	def __init__(
		self,
		data_dir: Path,
		set_type: SetType,
		transform: SyncedTransform | v2.Transform | None = None,
		return_format: ReturnFormat = "image_label",
		prefetch_labels: bool = False,
		fetch_dot_labels: bool = False,
	):
		assert set_type in SetType.__args__
		assert return_format in ReturnFormat.__args__

		image_path = data_dir / "images" / set_type
		self.data_dir = data_dir
		self.return_format: ReturnFormat = return_format
		self.set_type = set_type
		self.prefetch_labels = prefetch_labels

		self.fetch_dot_labels = fetch_dot_labels

		# Collect all image file paths ending in .RB.TIF or png recursively
		self.paths = collect_paths(image_path)
		# set transformation that converts to tensor to ensure correct type
		if self.fetch_dot_labels:
			self.transform = transform if transform is not None else (lambda img, label, **kwargs: (img, label))
		else:
			self.transform = transform if transform is not None else v2.Compose([v2.ToImage()])
		self.labels = None
		if self.prefetch_labels:
			self.labels = [self._load_label_for_path(p) for p in self.paths]

	def __len__(self):
		return len(self.paths)

	def __getitem__(self, idx: int):  # ty:ignore[invalid-method-override]
		img_path = self.paths[idx]

		if self.return_format in ("path_only", "label_only", "path_label"):
			label = self.get_label(idx)
			if self.return_format == "path_only":
				return img_path
			if self.return_format == "label_only":  # NOTE: For dot annotations this does not perform a transform
				return label
			return img_path, label  # "path_label"

		image = cv2.imread(str(img_path))
		# Handle grayscale and color images
		if len(image.shape) == 2:  # Grayscale
			image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
		elif image.shape[2] == 4:  # RGBA
			image = cv2.cvtColor(image, cv2.COLOR_BGRA2RGB)
		else:  # BGR
			image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

		label = self._load_label_for_path(img_path)
		if self.transform:
			if self.fetch_dot_labels:
				image, label = self.transform(image, label, mode=self.set_type)
			else:
				image = self.transform(image)

		if self.return_format == "image_only":
			return image
		# Default
		return image, label

	def _load_label_for_path(self, img_path: Path) -> dict[str, Any]:
		"""Read label JSON if present."""
		img_name_crp = img_path.with_suffix(".json")
		mode = "dots" if self.fetch_dot_labels else "xyxy"
		label_file = self.data_dir / "labels" / mode / img_name_crp.name
		if label_file.exists():
			with open(label_file) as f:
				return json.load(f)
		else:
			raise FileNotFoundError(f"label {label_file} for image {img_path} is missing.")

	def get_label(self, idx: int) -> dict[str, Any]:
		"""Return GT label dict for item idx (does not load the image)."""
		if self.labels is not None:
			return self.labels[idx]
		return self._load_label_for_path(self.paths[idx])


class CachedDataset(Dataset):
	"""
	Wraps another dataset and caches all items in memory after the first load.
	Memory considerations:
	- All dataset items are loaded into RAM during initialization
	- For a dataset with N samples of HxW images: ~N*H*W*C*4 bytes (float32)
	- Example: 1000 images of 512x512 RGB = ~3GB of RAM
	"""

	def __init__(self, base_dataset, runtime_transform: None | A.BasicTransform = None):
		self.base_dataset = base_dataset
		self.runtime_transform = runtime_transform

		# Extract properties from base dataset if available
		self.fetch_dot_labels = getattr(base_dataset, "fetch_dot_labels", False)
		self.set_type = getattr(base_dataset, "set_type", None)
		self.samples: list[tuple[Any, Any]] = []
		self._cache_data()

	def _cache_data(self):
		# [Cache] Loading and caching {len(self.base_dataset)} samples
		for i in range(len(self.base_dataset)):
			img, target = self.base_dataset[i]
			self.samples.append((img, target))
		logger.info(f"[Cache] Cached {len(self.samples)} samples ({len(self.samples) * 2} tensors).")

	def __len__(self):
		return len(self.samples)

	def __getitem__(self, idx: int):  # ty:ignore[invalid-method-override]
		img, target = self.samples[idx]

		# Apply runtime transform if provided
		if self.runtime_transform:
			# No need to clone - Albumentations creates new arrays/tensors
			if self.fetch_dot_labels:
				img_tensor, target_tensor = self.runtime_transform(img, target, mode=self.set_type)
			else:
				img_tensor = self.runtime_transform(img)
				target_tensor = target.clone()
		else:  # Clone if no transform to avoid keeping computation graphs
			img_tensor = img.clone()
			target_tensor = target.clone()

		return img_tensor, target_tensor


def infer_normalisation_constants(
	data_dir: Path, transform_config: SyncedTransformConfig
) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
	"""Takes data dir and infers channel norm and std from the training data.
	This is used for normalisation of the images during transform."""
	logger.info("Inferring normalisation constants...")
	transform_config.mean = (0, 0, 0)
	transform_config.std = (1, 1, 1)
	transform = SyncedTransform(transform_config)
	train_dataset = CountingDataset(
		data_dir, set_type="train", transform=transform, return_format="image_only", fetch_dot_labels=True
	)
	train_dataset.set_type = "val"
	train_loader = DataLoader(train_dataset, batch_size=1)

	mean = torch.zeros(3)
	std = torch.zeros(3)
	total_pixels = 0

	for img in tqdm(train_loader):  # img shape: (batch, 3, H, W)
		batch_size = img.shape[0]
		img_reshaped = img.reshape(batch_size, 3, -1)  # (batch, 3, H*W)
		mean += img_reshaped.mean(dim=(0, 2)).sum(dim=0)
		total_pixels += batch_size

	mean = mean / (total_pixels * img.shape[2] * img.shape[3])

	# For std, you need to use the global mean
	# This requires a second pass or storing all data
	variance = torch.zeros(3)
	for img in train_loader:
		img_reshaped = img.reshape(img.shape[0], 3, -1)
		variance += ((img_reshaped - mean.unsqueeze(-1)) ** 2).sum(dim=(0, 2))

	std = torch.sqrt(variance / (total_pixels * img.shape[2] * img.shape[3]))

	return tuple(mean.tolist()), tuple(std.tolist())


