from __future__ import annotations

import logging
import os
import random
import tempfile
from collections.abc import Callable, Iterable, Sequence
from dataclasses import astuple, dataclass
from pathlib import Path

import cv2
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from torchvision.transforms import v2 as T
from torchvision.utils import save_image
from tqdm import tqdm
from ultralytics import YOLO

from countcv.core.data import SyncedTransform
from countcv.core.experiment_utils import calc_metrics
from countcv.core.fcrn_model import BaseModel

Box = tuple[float, float, float, float]

logger = logging.getLogger(__name__)


@dataclass
class YoloDetection:
	"""Single YOLO-predicted bounding box with its associated confidence."""

	xmin: float
	ymin: float
	xmax: float
	ymax: float
	confidence: float

	@property
	def xyxy(self) -> Box:
		"""Return the (xmin, ymin, xmax, ymax) tuple consumed by plotting helpers."""

		return (self.xmin, self.ymin, self.xmax, self.ymax)


@dataclass
class YoloCountStats:
	"""Deterministic prediction interval driven by confidence thresholds."""

	pred_count: int
	interval_min: int
	interval_max: int
	low_conf_threshold: float
	base_conf_threshold: float
	high_conf_threshold: float

	@property
	def interval(self) -> tuple[int, int]:
		"""Return (pred_min, pred_max), matching the plotting notation."""

		return (self.interval_min, self.interval_max)


@dataclass
class YoloPredictionResult:
	"""Bundle filtered YOLO detections with the statistics derived from them."""

	detections: list[YoloDetection]
	stats: YoloCountStats

	@property
	def boxes(self) -> list[Box]:
		return [det.xyxy for det in self.detections]

	@property
	def pred_count(self) -> int:
		return self.stats.pred_count


@dataclass
class DensityMapResults:
	"Divide quantile prediction into lower and upper bound + median"

	density_map: np.ndarray
	density_count: float
	lower_bound: float | None = None
	upper_bound: float | None = None


@dataclass
class ImageResult:
	"""Container for visual artifacts plus YOLO detection-level uncertainty information."""

	path: Path
	image: np.ndarray
	densitymap_prediction: DensityMapResults | None
	gt_boxes: list[Box]
	gt_count: int | float
	yolo_prediction: YoloPredictionResult | None

	def __post_init__(self):
		if self.densitymap_prediction is None and self.yolo_prediction is None:
			raise ValueError("At least one of densitymap_prediction or yolo_prediction must be provided")


@dataclass
class PredictionResults:
	ground_truth: np.ndarray
	yolo: np.ndarray
	density: np.ndarray
	ensemble: np.ndarray
	yolo_interval_min: np.ndarray
	yolo_interval_max: np.ndarray
	density_interval_min: np.ndarray
	density_interval_max: np.ndarray

	def __post_init__(self):
		arrays = astuple(self)
		first_len = len(arrays[0])
		if not all(len(arr) == first_len for arr in arrays[1:]):
			raise ValueError(f"All arrays must have the same length as ground_truth (length {first_len})")


def _select_image_subset(image_paths: Sequence[Path], num_images: int, seed: int | None) -> list[Path]:
	"""Return a deterministic subset of images."""

	if num_images <= 0:
		raise ValueError("num_images must be greater than zero.")
	images = list(image_paths)
	if seed is not None:
		rng = random.Random(seed)
		rng.shuffle(images)
	return images[: min(num_images, len(images))]


def _read_gt_boxes(label_dir: Path | None, img_path: Path, img_size: int) -> list[Box]:
	"""Parse YOLO-format labels and convert them to pixel coordinates."""

	boxes: list[Box] = []
	if not label_dir:
		return boxes

	label_file = label_dir / img_path.with_suffix(".txt").name
	if not label_file.exists():
		return boxes

	with label_file.open("r") as f:
		for line in f:
			parts = line.strip().split()
			if len(parts) != 5:
				continue
			_, xc, yc, w, h = map(float, parts)
			xmin = (xc - w / 2) * img_size
			ymin = (yc - h / 2) * img_size
			xmax = (xc + w / 2) * img_size
			ymax = (yc + h / 2) * img_size
			boxes.append((xmin, ymin, xmax, ymax))
	return boxes


def _confidence_thresholds(base_conf: float, interval_delta: float) -> tuple[float, float, float]:
	"""Return the (low, base, high) confidence triplet constrained to [0, 1]."""

	low_conf = max(base_conf - interval_delta, 0.0)
	high_conf = min(base_conf + interval_delta, 1.0)
	return (low_conf, base_conf, high_conf)


def _build_threshold_interval(
	detections: Sequence[YoloDetection], thresholds: tuple[float, float, float]
) -> YoloCountStats:
	"""
	Derive deterministic bounds by counting detections above three confidence cutoffs:
	- base_conf - delta → upper bound (more boxes)
	- base_conf → predicted value
	- base_conf + delta → lower bound (fewer boxes)
	"""

	low_conf, base_conf, high_conf = thresholds

	def count(threshold: float) -> int:
		return sum(1 for det in detections if det.confidence >= threshold)

	lower = count(high_conf)
	predicted = count(base_conf)
	upper = count(low_conf)

	return YoloCountStats(
		pred_count=predicted,
		interval_min=lower,
		interval_max=upper,
		low_conf_threshold=float(low_conf),
		base_conf_threshold=float(base_conf),
		high_conf_threshold=float(high_conf),
	)


def _predict_yolo_result(
	model: YOLO, image_tensor: torch.Tensor, conf: float, overlap_thresh: float, interval_delta: float = 0.15
) -> YoloPredictionResult:
	"""
	Run YOLO inference once using the relaxed (low) threshold and split the detections
	into three confidence bins to form the deterministic interval.
	"""

	thresholds = _confidence_thresholds(base_conf=conf, interval_delta=interval_delta)
	low_conf, base_conf, _ = thresholds
	detections: list[YoloDetection] = []
	try:
		with tempfile.NamedTemporaryFile(suffix=".TIF", delete=False) as tmp:
			tmp_name = tmp.name
		save_image(image_tensor, tmp_name)
		results = model.predict(source=tmp_name, conf=low_conf, iou=overlap_thresh, verbose=False)
	finally:
		os.unlink(tmp_name)
	for res in results:
		if res.boxes is None or res.boxes.xyxy is None:
			continue
		xyxy = res.boxes.xyxy.detach().cpu().numpy()
		if res.boxes.conf is not None:
			confidences = res.boxes.conf.detach().cpu().numpy()
		else:
			confidences = np.ones((xyxy.shape[0],), dtype=np.float32)
		for coords, conf_value in zip(xyxy, confidences, strict=True):
			xmin, ymin, xmax, ymax = map(float, coords.tolist())
			detections.append(
				YoloDetection(
					xmin=xmin,
					ymin=ymin,
					xmax=xmax,
					ymax=ymax,
					confidence=float(conf_value),
				)
			)

	stats = _build_threshold_interval(detections, thresholds=thresholds)
	filtered_detections = [det for det in detections if det.confidence >= base_conf]
	return YoloPredictionResult(detections=filtered_detections, stats=stats)


def _ensure_rgb(image: np.ndarray) -> np.ndarray:
	"""Return an RGB image regardless of the file's original channel layout."""

	if image.ndim == 2:
		return cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
	if image.shape[2] == 4:
		return cv2.cvtColor(image, cv2.COLOR_BGRA2RGB)
	return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def _normalize_density_map(density_map: torch.Tensor) -> np.ndarray:
	"""Convert a density prediction into a normalized NumPy array."""

	density_2d = density_map.squeeze().detach().cpu().to(torch.float32)
	min_val = float(density_2d.min())
	max_val = float(density_2d.max())
	scale = max(max_val - min_val, 1e-8)
	normalized = (density_2d - min_val) / scale
	return normalized.numpy()


def predict_density(
	img_path: Path,
	dens_model: BaseModel,
	device: torch.device,
	transform: SyncedTransform,
) -> DensityMapResults:
	"""
	Load an image, run the density model, and return the normalized map and count.

	Optionally accepts a transform override to facilitate reuse outside this module.
	"""
	dens_model.to(device)
	density_scale = transform.config.alpha
	raw_image = cv2.imread(str(img_path), cv2.IMREAD_UNCHANGED)
	if raw_image is None:
		raise FileNotFoundError(f"Unable to read image at {img_path}")
	rgb_image = _ensure_rgb(raw_image)
	density_tensor, _ = transform(rgb_image, None, mode="test")

	density_input = density_tensor.unsqueeze(0).to(device=device, dtype=torch.float32)
	density_output = dens_model.predict(density_input, device=device)

	mean_idx = density_output.shape[1] // 2
	mean_output = density_output[:, mean_idx, :, :]
	density_map = _normalize_density_map(mean_output)

	density_count = density_output.sum(dim=(0, 2, 3)) / density_scale
	if density_output.shape[1] == 1:
		count_mean = density_count.item()
		count_lower, count_upper = None, None
	else:
		count_lower, count_mean, count_upper = density_count.tolist()
	return DensityMapResults(density_map, count_mean, count_lower, count_upper)


def _predict_image(
	img_path: Path,
	label_dir: Path | None,
	img_size: int,
	yolo_model: YOLO | None,
	yolo_transform: T.Compose,
	display_transform: T.Compose,
	dens_model: BaseModel | None,
	dens_transform: SyncedTransform | None,
	device: str,
	conf: float,
	bbox_overlap_thresh: float,
) -> ImageResult:
	"""Prepare predictions, annotations, and tensors for a single image."""

	assert (yolo_model is not None) | (dens_model is not None)
	with Image.open(img_path) as pil_image:
		img_tensor = yolo_transform(pil_image)
		display_tensor = display_transform(pil_image)
	img_np = display_tensor.permute(1, 2, 0).cpu().numpy()

	if dens_model:
		assert dens_transform is not None
		densitymap_prediction = predict_density(
			img_path=img_path,
			dens_model=dens_model,
			device=device,
			transform=dens_transform,
		)
	else:
		densitymap_prediction = None

	if yolo_model:
		yolo_prediction = _predict_yolo_result(yolo_model, img_tensor, conf=conf, overlap_thresh=bbox_overlap_thresh)
	else:
		yolo_prediction = None
	gt_boxes = _read_gt_boxes(label_dir, img_path, img_size)
	gt_count = len(gt_boxes) if label_dir else np.nan

	return ImageResult(
		path=img_path,
		image=img_np,
		densitymap_prediction=densitymap_prediction,
		gt_boxes=gt_boxes,
		gt_count=gt_count,
		yolo_prediction=yolo_prediction,
	)


def build_image_transform(img_size: int, keep_rgb: bool = False) -> T.Compose:
	"""Return the tensor transform used for YOLO inference and visualization."""

	transforms = [
		T.ToImage(),
		T.CenterCrop((img_size, img_size)),
		T.ToDtype(torch.float32, scale=True),
	]
	if not keep_rgb:
		transforms.append(T.Lambda(lambda t: t[0:1, ...]))  # keep single channel
	return T.Compose(transforms)


def _draw_boxes(ax: plt.Axes, boxes: Iterable[Box], color: str, linewidth: float) -> None:
	"""Overlay rectangular boxes on the given axis."""

	for xmin, ymin, xmax, ymax in boxes:
		rect = mpatches.Rectangle(
			(xmin, ymin),
			xmax - xmin,
			ymax - ymin,
			edgecolor=color,
			fill=False,
			linewidth=linewidth,
			linestyle="--",
		)
		ax.add_patch(rect)


def _plot_row(ax_gt: plt.Axes, ax_det: plt.Axes, ax_density: plt.Axes, example: ImageResult) -> None:
	"""Render GT, detector predictions, and density map for a single example."""

	h, w = example.image.shape[:2]
	extent = (0, w, h, 0)

	ax_gt.imshow(example.image, origin="upper", extent=extent)
	_draw_boxes(ax_gt, example.gt_boxes, color="green", linewidth=1.0)
	ax_gt.text(
		0.02,
		0.98,
		f"Ground Truth: {'unlabeled' if example.gt_count is np.nan else example.gt_count}\n{example.path.name}",
		transform=ax_gt.transAxes,
		fontsize=14,
		verticalalignment="top",
		bbox=dict(facecolor="white", alpha=0.6, edgecolor="none"),
	)
	ax_gt.axis("off")

	if example.yolo_prediction is not None:
		ax_det.imshow(example.image, origin="upper", extent=extent)
		stats_yolo = example.yolo_prediction.stats
		lo, hi = stats_yolo.interval
		_draw_boxes(ax_det, example.yolo_prediction.boxes, color="red", linewidth=1.0)
		ax_det.text(
			0.02,
			0.98,
			f"YOLO: {stats_yolo.pred_count} [{lo}, {hi}]",
			transform=ax_det.transAxes,
			fontsize=14,
			verticalalignment="top",
			bbox=dict(facecolor="white", alpha=0.6, edgecolor="none"),
		)
	ax_det.axis("off")

	if example.densitymap_prediction is not None:
		ax_density.imshow(example.image, origin="upper", extent=extent)
		lo_dmap, hi_dmap = example.densitymap_prediction.lower_bound, example.densitymap_prediction.upper_bound
		ax_density.imshow(example.densitymap_prediction.density_map, cmap="gray")
		if lo_dmap and hi_dmap:
			txt = f"Density Map: {example.densitymap_prediction.density_count:.2f} [{lo_dmap:.2f}, {hi_dmap:.2f}]"
		else:
			txt = f"Density Map: {example.densitymap_prediction.density_count:.2f}"
		ax_density.text(
			0.02,
			0.98,
			txt,
			transform=ax_density.transAxes,
			fontsize=14,
			verticalalignment="top",
			bbox=dict(facecolor="white", alpha=0.6, edgecolor="none"),
		)
	ax_density.axis("off")


def _collect_counts(results: Sequence[ImageResult]) -> PredictionResults:
	"""Aggregate GT and prediction counts into aligned arrays."""
	gt_counts = np.array([res.gt_count for res in results])

	# Collect YOLO predictions if available
	yolo_counts = np.array(
		[res.yolo_prediction.stats.pred_count if res.yolo_prediction is not None else np.nan for res in results]
	)
	yolo_interval_min = np.array(
		[res.yolo_prediction.stats.interval_min if res.yolo_prediction is not None else np.nan for res in results]
	)
	yolo_interval_max = np.array(
		[res.yolo_prediction.stats.interval_max if res.yolo_prediction is not None else np.nan for res in results]
	)
	density_counts = np.array(
		[
			res.densitymap_prediction.density_count if res.densitymap_prediction is not None else np.nan
			for res in results
		]
	)
	density_interval_min = np.array(
		[res.densitymap_prediction.lower_bound if res.densitymap_prediction is not None else np.nan for res in results]
	)
	density_interval_max = np.array(
		[res.densitymap_prediction.upper_bound if res.densitymap_prediction is not None else np.nan for res in results]
	)

	ensemble_counts = np.rint((yolo_counts + density_counts) / 2.0)
	return PredictionResults(
		**{
			"ground_truth": gt_counts,
			"yolo": yolo_counts,
			"density": density_counts,
			"ensemble": ensemble_counts,
			"yolo_interval_min": yolo_interval_min,
			"yolo_interval_max": yolo_interval_max,
			"density_interval_min": density_interval_min,
			"density_interval_max": density_interval_max,
		}
	)


def _evaluate_prediction_metrics(counts: PredictionResults) -> dict[str, dict[str, float]]:
	"""Compute metrics for detector, density, and ensemble predictions."""

	gt_counts = counts.ground_truth
	return {
		"yolo": calc_metrics(p=counts.yolo, t=gt_counts),
		"density": calc_metrics(p=counts.density, t=gt_counts),
		"ensemble": calc_metrics(p=counts.ensemble, t=gt_counts),
	}


def make_results_from_paths(
	image_paths: Sequence[Path],
	label_path: Path | None,
	yolo_model_path: Path | None,
	dens_model: BaseModel | None,
	dens_transform: SyncedTransform | None,
	img_size: int = 220,
	confidence: float = 0.25,
	seed: int | None = None,
	device: str = "cuda",
	bbox_overlap_thresh: float = 0.05,
	out_path: Path | None = None,
	progress_callback: Callable[[int, int], None] | None = None,
) -> Sequence[PredictionResults]:
	if not image_paths:
		raise ValueError("image_paths must not be empty.")

	yolo_transform = build_image_transform(img_size)
	display_transform = build_image_transform(img_size, keep_rgb=True)
	yolo_model = YOLO(yolo_model_path) if yolo_model_path else None
	counts = []
	for idx, img_path in tqdm(enumerate(image_paths)):
		if progress_callback:
			progress_callback(idx, len(image_paths))
		try:
			img_result = _predict_image(
				img_path=img_path,
				label_dir=label_path,
				dens_transform=dens_transform,
				img_size=img_size,
				yolo_model=yolo_model,
				yolo_transform=yolo_transform,
				display_transform=display_transform,
				dens_model=dens_model,
				device=device,
				conf=confidence,
				bbox_overlap_thresh=bbox_overlap_thresh,
			)
			counts.append(_collect_counts([img_result]))
			if out_path:
				save_prediction_image(img_result, out_path)
		except Exception as e:
			logger.error(f"Unable to process image {img_path}, due to \n{e}", exc_info=True)
	return counts


def plot_results_from_paths(
	image_paths: Sequence[Path],
	label_path: Path | None,
	yolo_model_path: Path,
	dens_model: BaseModel,
	dens_transform: SyncedTransform,
	num_images: int = 50,
	img_size: int = 220,
	confidence: float = 0.25,
	seed: int | None = None,
	device: str = "cuda",
	bbox_overlap_thresh: float = 0.05,
) -> dict[str, object]:
	"""Visualise YOLO detections, ground-truth boxes, and density maps."""

	if not image_paths:
		raise ValueError("image_paths must not be empty.")

	selected_paths = _select_image_subset(image_paths, num_images, seed)

	yolo_transform = build_image_transform(img_size)
	display_transform = build_image_transform(img_size, keep_rgb=True)
	yolo_model = YOLO(yolo_model_path)

	nrows = len(selected_paths)
	row_height = 7.0
	fig_height = max(row_height, row_height * nrows)
	fig, axes = plt.subplots(nrows=nrows, ncols=3, figsize=(20, fig_height), dpi=120)
	axes = np.atleast_2d(axes)

	results: list[ImageResult] = []
	for row_axes, img_path in zip(axes, selected_paths, strict=True):
		ax_gt, ax_det, ax_density = row_axes
		img_result = _predict_image(
			img_path=img_path,
			label_dir=label_path,
			dens_transform=dens_transform,
			img_size=img_size,
			yolo_model=yolo_model,
			yolo_transform=yolo_transform,
			display_transform=display_transform,
			dens_model=dens_model,
			device=device,
			conf=confidence,
			bbox_overlap_thresh=bbox_overlap_thresh,
		)
		_plot_row(ax_gt, ax_det, ax_density, img_result)
		results.append(img_result)

	counts = _collect_counts(results)
	metrics = _evaluate_prediction_metrics(counts)

	fig.subplots_adjust(wspace=0.02, hspace=0.02)
	plt.show()

	return {
		"figure": fig,
		"axes": axes,
		"results": results,
		"counts": counts,
		"metrics": metrics,
	}


def save_prediction_image(img_result: ImageResult, dst_path: Path):
	"""Persist a single prediction triptych using a tighter layout and higher DPI."""

	fig, axes = plt.subplots(
		nrows=1,
		ncols=3,
		figsize=(12, 4),
		dpi=80,
		constrained_layout=True,
	)
	ax_gt, ax_det, ax_density = axes
	_plot_row(ax_gt, ax_det, ax_density, img_result)

	img_name = img_result.path.name
	fig.savefig(dst_path / f"result_{img_name}", bbox_inches="tight", pad_inches=0.15)
	plt.close(fig)
