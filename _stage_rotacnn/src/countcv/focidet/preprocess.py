import json
import logging
import shutil
from pathlib import Path

import numpy as np
import PIL
from PIL import Image

from countcv.core.data import DatasetSplitter, LabelCreator


class FociDataSplitter(DatasetSplitter):
	def __init__(self, dataset_root: Path):
		super().__init__(dataset_root)
		self.label_files = list(self.label_xywh.glob("*.txt"))
		self.raw_images = self._find_labelled_images()
		self.counts = self._load_counts_from_labels(file_paths=self.label_files)

	def _find_labelled_images(self):
		raw_images = [f for f in self.raw_dir.rglob("*.TIF") if "Rejects" not in f.parts]
		image_dict = {img.stem: img for img in raw_images}
		img_sorted = []
		for lab in self.label_files:
			img_path = image_dict.get(Path(lab).with_suffix(".TIF").stem)
			img_sorted.append(img_path)
		return img_sorted

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


class FociDataLabelCreator(LabelCreator):
	def __init__(self, dataset_root: Path):
		super().__init__(dataset_root)
		self.img_w = 220
		self.img_h = 220

	def create_labels(self):
		image_paths = [f for f in self.raw_path.glob("**/*.TIF") if "Rejects" not in f.parts]
		image_dict = {img.stem: img for img in image_paths}
		label_paths = list((self.raw_path / "labels_xywh").glob("*.txt"))
		for lab in label_paths:
			img = image_dict[(Path(lab).with_suffix(".TIF").stem)]
			try:
				with Image.open(img):
					pass
			except (FileNotFoundError, PIL.UnidentifiedImageError):
				logging.error(f"Skipping label creation {lab} as no associated image exists.")
				continue
			json_path = self.xyxy_dir / lab.with_suffix(".json").name
			yolo_path = self.xywh_dir / lab.name
			centers = self.write_bbox_labels(lab, json_path, yolo_path, self.img_w, self.img_h)
			if centers is not None:
				dots_data = {"count": len(centers), "center": centers}
				savepath = (self.dots_dir / lab.name).with_suffix(".json")
				with open(savepath, "w") as f:
					json.dump(dots_data, f, indent=4)

	def write_bbox_labels(self, txt_path: Path, json_path: Path, yolo_label_path: Path, img_w, img_h) -> list:
		"""Labels already in correct xywh format. Create xyxy format
		Returns center of dot labels"""
		bboxes = []

		def xywh_to_xyxy(xywh: list[int], img_w: int, img_h: int):
			x, y, w, h = xywh
			assert x > 0 and y > 0

			x1 = (x - w / 2) * img_w
			x2 = (x + w / 2) * img_w
			y1 = (y - h / 2) * img_h
			y2 = (y + h / 2) * img_h
			return list(map(int, [x1, y1, x2, y2]))

		with txt_path.open("r") as f:
			for line in f:
				parts = list(map(float, line.strip().split()))
				if len(parts) < 4:
					continue  # skip malformed lines
				# take only the first 4 values (x1, y1, x2, y2)
				relative_coords = list(parts[1:])
				bboxes.append(xywh_to_xyxy(relative_coords, img_w, img_h))

		data = {
			"count": len(bboxes),
			"bboxes": bboxes,
		}
		with json_path.open("w") as f:
			json.dump(data, f, indent=4)

		shutil.copy2(txt_path, yolo_label_path)
		# create data for dot labels
		bboxes = np.array(bboxes)
		centers = (
			np.column_stack(((bboxes[:, 0] + bboxes[:, 2]) // 2, (bboxes[:, 1] + bboxes[:, 3]) // 2))
			if bboxes.size != 0
			else bboxes
		)
		return centers.tolist()
