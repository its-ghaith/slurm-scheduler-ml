"""
Script to generate YOLO-format labels for fluorescent foci detection using GroundingDINO,
and optionally save images with drawn nucleus (blue) and foci (red) bounding boxes.
"""

import json
import warnings
from pathlib import Path

import cv2
import groundingdino.datasets.transforms as T
import numpy as np
import torch
from groundingdino.util.inference import load_image, load_model, predict
from PIL import Image
from torchvision.ops import nms
from torchvision.transforms import v2

from countcv.core.ultralytics_detectors import save_cropped_images

# comes from inside dependencies
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", message="torch.meshgrid")
warnings.filterwarnings("ignore", message=".*use_reentrant.*")


class BboxLabelGenerator:
	def __init__(
		self,
		label_dir: Path,
		config_path: Path,
		weights_path: Path,
		prompt: str,
		box_threshold: float,
		text_threshold: float,
		max_frac: float = 0.5,
		iou_threshold: float = 0.3,
	):
		self.label_dir = label_dir
		self.prompt = prompt
		self.box_threshold = box_threshold
		self.text_threshold = text_threshold
		self.max_frac = max_frac
		self.iou_threshold = iou_threshold

		# Load GroundingDINO model
		self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
		self.dino = load_model(str(config_path), str(weights_path))
		self.dino.to(self.device)

		# Ensure output directories exist
		self.label_dir.mkdir(parents=True, exist_ok=True)
		self.xyxy_dir = self.label_dir / "xyxy"
		self.xyxy_dir.mkdir(parents=True, exist_ok=True)

	@staticmethod
	def _detect_nucleus_box(img_np: np.ndarray):
		"""Detects the largest blue-region (nucleus) by thresholding the blue channel."""
		blue = img_np[..., 2]
		_, mask = cv2.threshold(blue, 30, 255, cv2.THRESH_BINARY)
		kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
		mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
		contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
		if not contours:
			h, w = img_np.shape[:2]
			return (0, 0, w, h)
		cnt = max(contours, key=cv2.contourArea)
		x, y, w, h = cv2.boundingRect(cnt)
		return (x, y, x + w, y + h)

	def _filter_by_size(self, boxes, roi_box):
		"""Removes overlapping and overly large boxes relative to the nucleus."""
		x0, y0, x1, y1 = roi_box
		RoiW, RoiH = x1 - x0, y1 - y0
		if not boxes:
			return []
		boxes_tensor = torch.tensor([[x, y, x + w, y + h] for x, y, w, h, _ in boxes], dtype=torch.float32)
		scores_tensor = torch.tensor([s for *_, s in boxes], dtype=torch.float32)
		keep_idx = nms(boxes_tensor, scores_tensor, iou_threshold=self.iou_threshold)
		filtered = [boxes[i] for i in keep_idx]
		final = []
		for x, y, w, h, score in filtered:
			if w / RoiW < self.max_frac and h / RoiH < self.max_frac:
				final.append((x, y, w, h, score))
		return final

	def _predict_on_resized_crop(self, crop_tensor, crop_box):
		"""Runs DINO on the crop and maps predicted boxes back to full-image coordinates."""
		x0_img, y0_img, x1_img, y1_img = crop_box
		Wc, Hc = x1_img - x0_img, y1_img - y0_img
		boxes, scores, _ = predict(
			self.dino,
			crop_tensor,
			self.prompt,
			box_threshold=self.box_threshold,
			text_threshold=self.text_threshold,
			device=self.device,
		)
		full_boxes = []
		for (cx, cy, bw, bh), score in zip(boxes, scores, strict=False):
			x0n, y0n = cx - bw / 2, cy - bh / 2
			x1n, y1n = cx + bw / 2, cy + bh / 2
			x0c, y0c = x0n * Wc, y0n * Hc
			x1c, y1c = x1n * Wc, y1n * Hc
			wf, hf = x1c - x0c, y1c - y0c
			x0f, y0f = x0_img + x0c, y0_img + y0c
			full_boxes.append((x0f, y0f, wf, hf, score))
		return self._filter_by_size(full_boxes, roi_box=crop_box)

	def _get_foci_boxes(self, path_str: str):
		"""Loads an image, detects the nucleus, crops, binarizes, and predicts foci boxes."""
		img_np, _ = load_image(path_str)
		nuc_box = self._detect_nucleus_box(img_np)
		x0, y0, x1, y1 = nuc_box
		crop_np = img_np[y0:y1, x0:x1]
		transform = T.Compose(  # same transformation as in load_image from grounding dino
			[
				T.RandomResize([800], max_size=1333),
				T.ToTensor(),
				T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
			]
		)
		crop_tensor, _ = transform(Image.fromarray(crop_np), None)
		# Binarize red channel
		red = crop_tensor[0]
		crop_tensor[0] = (red > -0.1).float()
		foci = self._predict_on_resized_crop(crop_tensor, crop_box=nuc_box)
		return img_np, nuc_box, foci

	def _process_image(self, img_path: Path, save_images_dir: Path = None):
		"""Processes a single image: writes YOLO label and optionally saves a boxed image."""
		img_np, nuc_box, foci_boxes = self._get_foci_boxes(str(img_path))
		h, w = img_np.shape[:2]
		# Write YOLO label file
		subdir = img_path.parent.name
		label_file = self.label_dir / subdir / f"{img_path.stem}.txt"
		json_file = self.xyxy_dir / f"{img_path.stem}.json"
		xyxy_list = []
		with open(label_file, "w") as f:
			for x0, y0, bw, bh, _ in foci_boxes:
				xc = x0 + bw / 2
				yc = y0 + bh / 2
				f.write(f"0 {xc / w:.6f} {yc / h:.6f} {bw / w:.6f} {bh / h:.6f}\n")

				# collect XYXY coords
				xyxy_list.append([float(x0), float(y0), float(x0 + bw), float(y0 + bh)])

		# dump JSON once
		json_data = {"count": len(xyxy_list), "bboxes": xyxy_list}
		with open(json_file, "w") as json_file:
			json.dump(json_data, json_file, indent=4)

		# Optionally save image with boxes
		if save_images_dir:
			img = img_np.copy()
			x0, y0, x1, y1 = nuc_box
			# draw nucleus bbox
			cv2.rectangle(img, (int(x0), int(y0)), (int(x1), int(y1)), (0, 0, 255), 1)
			for x0f, y0f, bw, bh, _ in foci_boxes:
				cv2.rectangle(img, (int(x0f), int(y0f)), (int(x0f + bw), int(y0f + bh)), (255, 0, 0), 1)
			img_pil = Image.fromarray(img)
			count = len(foci_boxes)
			out_name = f"{img_path.stem}_{count}{img_path.suffix}"
			save_path = save_images_dir / out_name
			img_pil.save(save_path, format="TIFF", compression="none")

	def create_labels(self, image_dir: Path, save_images_dir: Path = None):
		"""Iterates over all images in the directory and processes them.

		Args:
			image_dir (Path): where original images are stored
			save_images_dir (Path): optional. where result images with bboxes should be stored. Defaults to None.
		"""
		if save_images_dir:
			save_images_dir.mkdir(parents=True, exist_ok=True)

		for subdir in image_dir.iterdir():
			(self.label_dir / subdir.name).mkdir(
				exist_ok=True, parents=True
			)  # ensure subdir for set exists in label_dir
			for img_path in subdir.glob("*.TIF"):
				self._process_image(img_path, save_images_dir)


def create_yolo_labels(image_dir: Path, label_dir: Path, save_images_dir: Path = None):
	config_path = Path("./config/GroundingDINO_SwinT_OGC.py")
	weights_path = Path("./models/groundingdino_swint_ogc.pth")
	prompt = "many separate bright dots in the nucleus of different size"
	box_threshold = 0.2
	text_threshold = 0.3
	max_frac = 0.5
	iou_threshold = 0.3

	label_generator = BboxLabelGenerator(
		label_dir=label_dir,
		config_path=config_path,
		weights_path=weights_path,
		prompt=prompt,
		box_threshold=box_threshold,
		text_threshold=text_threshold,
		max_frac=max_frac,
		iou_threshold=iou_threshold,
	)
	label_generator.create_labels(image_dir=image_dir, save_images_dir=save_images_dir)


def create_cropped(cropped_path: Path):
	transform = v2.Compose(
		[
			v2.ToImage(),
			v2.CenterCrop((220, 220)),
			v2.ToDtype(torch.float32, scale=True),
		]
	)
	save_cropped_images(data_dir=cropped_path, transform=transform)


if __name__ == "__main__":
	image_dir = Path("./data/uc_cells/cropped/images")
	label_dir = Path("./data/uc_cells/labels")
	save_images_dir = Path("./data/uc_cells/boxed_images")  # set to None to skip saving images

	label_dir.mkdir(exist_ok=True, parents=True)
	create_cropped(image_dir.parent)
	create_yolo_labels(image_dir, label_dir=label_dir, save_images_dir=None)
