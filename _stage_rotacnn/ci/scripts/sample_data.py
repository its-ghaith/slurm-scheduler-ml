"""
Sample 30 images from each dataset directory into data_subset/.

Rules:
- For synthetic_cells/raw: sample 30 *ID groups* (e.g. '001dots.png' and '001cell.png')
  together, so labels/images stay aligned.
- For uc_cells/images/{train,val,test}: sample 30 random images individually.
"""

import logging
import random
import shutil
from pathlib import Path

N_SAMPLES = 10

# source -> destination mapping (uc_cells are sampled individually)
UC_DIRS = {"data/uc_cells/raw": "data_subset/data/uc_cells/raw"}

SYNTHETIC_SRC = Path("data/synthetic_cells/raw")
SYNTHETIC_DST = Path("data_subset/data/synthetic_cells/raw")

CARPK_DIRS = {"data/carpk/raw/images": "data_subset/data/carpk/raw/images"}


def copy_files(files: list[Path], dst: Path):
	dst.mkdir(parents=True, exist_ok=True)
	for f in files:
		shutil.copy2(f, dst)


def sample_data(src: Path, dst: Path, n: int = N_SAMPLES):
	files = [p for p in src.rglob("*") if p.is_file()]
	chosen = random.sample(files, min(n, len(files)))
	copy_files(chosen, dst)
	logging.info(f"Copied {len(chosen)} UC cells images for {src} -> {dst}")


def sample_uc_data(src: Path, dst: Path, n: int = N_SAMPLES):
	label_xywh = src / "labels_xywh"

	labels = list(label_xywh.glob("*.txt"))
	# Sample the labels instead of the images and retrieve images from label names
	chosen_labels = random.sample(labels, min(n, len(labels)))
	img_files = [p for p in src.rglob("*.TIF") if p.is_file()]
	image_dict = {img.stem: img for img in img_files}
	img_chosen = []
	for lab in chosen_labels:
		img_path = image_dict.get(Path(lab).with_suffix(".TIF").stem)
		img_chosen.append(img_path)
	copy_files(img_chosen, dst)
	copy_files(chosen_labels, dst / "labels_xywh")
	logging.info(f"Copied {len(img_chosen)} UC cells images for {src} -> {dst}")


def sample_synthetic_cells(src: Path, dst: Path, n: int = N_SAMPLES):
	"""Sample n IDs, copying both dots+cell images per ID."""
	# Extract IDs by stripping suffix (e.g. '001' from '001dots.png')
	ids = sorted({p.stem[:-4] for p in src.iterdir() if p.is_file()})
	chosen_ids = random.sample(ids, min(n, len(ids)))
	all_files = []
	for id_ in chosen_ids:
		for suffix in ("dots.png", "cell.png"):
			f = src / f"{id_}{suffix}"
			if f.exists():
				all_files.append(f)
			else:
				logging.warning(f"[WARN] Expected file {f} not found, skipping")

	copy_files(all_files, dst)
	logging.info(f"Copied {len(all_files)} synthetic cell files for {len(chosen_ids)} IDs")


if __name__ == "__main__":
	logging.basicConfig(level=logging.INFO)
	# synthetic cells (paired sampling)
	sample_synthetic_cells(SYNTHETIC_SRC, SYNTHETIC_DST)

	# uc_cells and carpk splits (independent sampling)
	for src_str, dst_str in UC_DIRS.items():
		src, dst = Path(src_str), Path(dst_str)
		sample_uc_data(src, dst)

	for src_str, dst_str in CARPK_DIRS.items():
		src, dst = Path(src_str), Path(dst_str)
		sample_data(src, dst, n=3 * N_SAMPLES)
