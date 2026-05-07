import json
import logging
import os
from pathlib import Path

import cv2
import numpy as np

from countcv.core.data import DatasetSplitter, LabelCreator


class SynthCellsSplitter(DatasetSplitter):
	def __init__(self, dataset_root: Path):
		super().__init__(dataset_root)
		self.raw_images = sorted(list(self.raw_dir.glob("*cell.png")))
		self.label_files = [self.label_xywh / file_path.with_suffix(".txt").name for file_path in self.raw_images]
		self.counts = self._load_counts_from_labels(file_paths=self.label_files)

	def split(self, train_size: float | int = 0.8, seed: int = 1, limit_train_size: int | bool = False):
		train_idx, val_idx, test_idx = self._stratified_train_val_split(
			train_size=train_size, counts=self.counts, limit_train_size=limit_train_size, seed=seed
		)
		assert isinstance(test_idx, np.ndarray)

		images = np.array(self.raw_images)
		labels = np.array(self.label_files)
		for indices, settype in zip([train_idx, val_idx, test_idx], ["train", "val", "test"], strict=True):
			self._copy_and_log(images[indices], labels[indices], settype)
		logging.info(
			f"Dataset split complete (train: {len(train_idx)}, val: {len(val_idx)}, test: {len(test_idx)}). \
			Images in {self.img_dest}."
		)


class SynthCellsLabelCreator(LabelCreator):
	def __init__(self, dataset_root: Path, bbox_width=11, bbox_height=11):
		super().__init__(dataset_root)
		self.bbox_width = bbox_width
		self.bbox_height = bbox_height

	def create_labels(self):
		pairs = self._get_and_validate_file_pairs_synthcells()
		for img_name, label_name in pairs:
			xyxy_dest = self.xyxy_dir / Path(img_name).with_suffix(".json").name
			xywh_dest = self.xywh_dir / Path(img_name).with_suffix(".txt").name
			dot_dest = self.dots_dir / Path(img_name).with_suffix(".json").name
			self._write_json_label_from_png(
				self.raw_path / label_name,
				dot_dest,
				xyxy_dest,
				xywh_dest,
			)

	def _get_and_validate_file_pairs_synthcells(self) -> list[tuple[str, str]]:
		"""Get image–label file pairs by matching filenames: e.g. '001cell.png' -> '001dots.png'."""
		image_files = sorted(f for f in os.listdir(self.raw_path) if f.endswith("cell.png"))
		label_files = sorted(f for f in os.listdir(self.raw_path) if f.endswith("dots.png"))

		pairs = []
		for img_name in image_files:
			label_name = img_name.replace("cell", "dots", 1)  # Replace only first occurrence
			if label_name not in label_files:
				raise FileNotFoundError(f"Label file '{label_name}' not found for image '{img_name}'.")
			pairs.append((img_name, label_name))

		if len(pairs) == 0:
			raise ValueError("No image-label-pairs found")

		return pairs

	def _write_json_label_from_png(self, label_path: Path, dot_dest: Path, xyxy_dest: Path, xywh_dest):
		arr = cv2.imread(str(label_path), cv2.IMREAD_GRAYSCALE)
		# --- Find bright pixels (nonzero) ---
		ys, xs = np.nonzero(arr)  # row (y), col (x) indices of non-zero pixels
		points = list(zip(xs.tolist(), ys.tolist(), strict=True))  # convert to (x, y)
		data = {"count": len(points), "center": points}
		with dot_dest.open("w") as f:
			json.dump(data, f, indent=4)
		self._create_bboxes(arr.shape, points, xyxy_path=xyxy_dest, xywh_path=xywh_dest)

	def _create_bboxes(self, img_size, points, xyxy_path, xywh_path):
		bboxes_xyxy = []
		bboxes_xywh = []
		img_width, img_height = img_size

		for x, y in points:
			xmin = max(x - self.bbox_width / 2, 0)
			ymin = max(y - self.bbox_height / 2, 0)
			xmax = min(x + self.bbox_width / 2, img_width)
			ymax = min(y + self.bbox_height / 2, img_height)
			bboxes_xyxy.append([xmin, ymin, xmax, ymax])

			xcenter = ((xmin + xmax) / 2) / img_width  # use xyxy to ensure bbox does not extend beyond img borders
			ycenter = ((ymin + ymax) / 2) / img_height
			xwidth = (xmax - xmin) / img_width
			ywidth = (ymax - ymin) / img_height

			bboxes_xywh.append(f"{0} {xcenter:.6f} {ycenter:.6f} {xwidth:.6f} {ywidth:.6f}")

		xyxy_out_data = {"count": len(bboxes_xyxy), "bboxes": bboxes_xyxy}
		with xyxy_path.open("w") as f:
			json.dump(xyxy_out_data, f, indent=4)
		xywh_path.write_text("\n".join(bboxes_xywh) + "\n", encoding="utf-8")
