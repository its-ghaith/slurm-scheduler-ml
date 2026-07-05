import json
import logging
from pathlib import Path

import numpy as np
import PIL
from PIL import Image

from countcv.core.data import DatasetSplitter, LabelCreator, SetType


class CarpkSplitter(DatasetSplitter):
	def __init__(self, dataset_root: Path):
		super().__init__(dataset_root)
		self.split_dir = self.raw_dir / "ImageSets"
		self.raw_images = sorted(list(self.raw_dir.glob("images/*.png")))
		self.label_files = sorted(list(self.label_xywh.glob("*.txt")))

	def split(self, train_size: float | int = 0.8, seed: int = 1, limit_train_size: int | bool = False):
		logging.info("Splitting CarPK dataset...")

		train_val_imgs, train_val_labels = self._read_imageSets("train")

		test_imgs, test_label = self._read_imageSets("test")

		train_val_count = self._load_counts_from_labels(train_val_labels)

		train_idx, val_idx, _ = self._stratified_train_val_split(
			train_size=train_size,
			counts=train_val_count,
			limit_train_size=limit_train_size,
			seed=seed,
			fixed_test_indices=True,
		)

		train_val_labels = np.array(train_val_labels)
		train_val_imgs = np.array(train_val_imgs)
		self._copy_and_log(train_val_imgs[train_idx], train_val_labels[train_idx], "train")
		self._copy_and_log(train_val_imgs[val_idx], train_val_labels[val_idx], "val")
		self._copy_and_log(test_imgs, test_label, "test")
		logging.info(
			f"Dataset split complete (train: {len(train_idx)}, val: {len(val_idx)}, test: {len(test_imgs)}). \
			Images in {self.img_dest}."
		)

	def _read_imageSets(self, subset: SetType) -> tuple[list[Path], list[Path]]:
		"""txt files for train and test set from original carpk dataset. Returns image_paths, label_paths"""
		filename = subset + ".txt"
		path = self.split_dir / filename
		existing_label_names = {p.stem for p in self.label_files}
		# read and normalize lines once
		lines = [line.strip() for line in path.read_text().splitlines() if line.strip()]
		# filter lists for robustness in ci with missing raw
		img_paths = [self.raw_dir / "images" / (name + ".png") for name in lines]
		img_paths = [
			img_path for img_path in img_paths if img_path in self.raw_images and img_path.stem in existing_label_names
		]  # check if img exists in raw and associated label exists
		label_paths = [self.label_xywh / (name + ".txt") for name in lines]
		label_paths = [label_path for label_path in label_paths if label_path in self.label_files]

		return img_paths, label_paths


class CarpkLabelCreator(LabelCreator):
	def __init__(self, dataset_root: Path):
		super().__init__(dataset_root)

	def create_labels(self):
		label_paths = list((self.raw_path / "labels").glob("*.txt"))
		for lab in label_paths:
			img = (self.raw_path / "images" / lab.stem).with_suffix(".png")
			try:
				with Image.open(img) as im:
					w, h = im.size
			except (FileNotFoundError, PIL.UnidentifiedImageError):
				logging.error(f"Skipping label creation {lab} as no associated image exists.")
				continue
			json_path = self.xyxy_dir / lab.with_suffix(".json").name
			yolo_path = self.xywh_dir / lab.with_suffix(".txt").name
			centers = self.write_bbox_labels(lab, json_path, yolo_path, w, h)
			if centers is not None:
				dots_data = {"count": len(centers), "center": centers}
				savepath = (self.dots_dir / lab.name).with_suffix(".json")
				with open(savepath, "w") as f:
					json.dump(dots_data, f, indent=4)

	def write_bbox_labels(self, txt_path: Path, json_path: Path, yolo_label_path: Path, img_w, img_h) -> list:
		"""Convert bounding boxes from txt (xyxy + label) to JSON xyxy and to normalized xywh txt format.
		Returns center of dot labels"""
		bboxes = []
		yolo_bboxes = []

		def xyxy_to_xywh(xyxy: list[int], w, h):
			x1, y1, x2, y2 = xyxy
			assert x2 > x1 and y2 > y1
			w = (x2 - x1) / img_w
			h = (y2 - y1) / img_h
			x_c = (x1 + x2) / 2.0 / img_w
			y_c = (y1 + y2) / 2.0 / img_h
			return f"{0} {x_c:.6f} {y_c:.6f} {w:.6f} {h:.6f}"

		with txt_path.open("r") as f:
			for line in f:
				parts = line.strip().split()
				if len(parts) < 4:
					continue  # skip malformed lines
				# take only the first 4 values (x1, y1, x2, y2)
				coords = list(map(int, parts[:4]))
				bboxes.append(coords)
				yolo_bboxes.append(xyxy_to_xywh(coords, img_w, img_h))

		data = {
			"count": len(bboxes),
			"bboxes": bboxes,
		}
		with json_path.open("w") as f:
			json.dump(data, f, indent=4)

		yolo_label_path.write_text("\n".join(yolo_bboxes) + "\n", encoding="utf-8")
		# create data for dot labels
		bboxes = np.array(bboxes)
		centers = np.column_stack(((bboxes[:, 0] + bboxes[:, 2]) // 2, (bboxes[:, 1] + bboxes[:, 3]) // 2))
		return centers.tolist()
