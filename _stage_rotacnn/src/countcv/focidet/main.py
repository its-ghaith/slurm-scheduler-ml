import argparse
import dataclasses
import logging
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Literal

import pandas as pd
import torch

from countcv.core.data import collect_paths
from countcv.core.experiment_utils import get_models
from countcv.focidet.predict import PredictionResults, make_results_from_paths

logger = logging.getLogger(__name__)


def parse_args():
	parser = argparse.ArgumentParser(description="Run inference with YOLO and/or density models")
	# Add your other arguments here
	parser.add_argument("imgdir", help="Directory path to the location of the images for inference")
	parser.add_argument(
		"output_dir", metavar="output-dir", help="location where the resulting images and files are stored."
	)
	parser.add_argument(
		"-m",
		"--modeldir",
		help="Path to the directory of the ML models. Directory needs to contain 'cnn_base' and 'yolo' subdirectories",
		default="checkpoints",
	)
	parser.add_argument(
		"--models",
		type=str,
		choices=["both", "yolo", "density"],
		default="both",
		help="Which model(s) to use for inference (default: both)",
	)
	parser.add_argument(
		"--images",
		help="Boolean action. Use '--no-images' to avoid writing images and only return results.csv",
		default=True,
		action=argparse.BooleanOptionalAction,
	)
	return parser.parse_args()


def save_results(
	counts: Sequence[PredictionResults], outdir: Path, image_paths: list[Path], format: Literal["german", "english"]
):
	outdir.mkdir(parents=True, exist_ok=True)
	res_df = pd.DataFrame(
		{k: v[i] for k, v in dataclasses.asdict(item).items()}
		for item in counts
		for i in range(len(next(iter(dataclasses.asdict(item).values()))))
	)
	res_df["filename"] = image_paths
	res_df.set_index("filename")
	# drop density intervals from df (currently not used)
	res_df = res_df.drop(columns=["density_interval_min", "density_interval_max"])
	if format == "german":
		res_df.to_csv(outdir / "foci_counts.csv", float_format="%.2f", index=False, decimal=",", sep=";")
	elif format == "english":
		res_df.to_csv(outdir / "foci_counts.csv", float_format="%.2f", index=False, decimal=".", sep=",")


def main(
	output_dir: Path,
	img_dir: str,
	model_dir: str,
	write_images: bool,
	models: Literal["yolo", "density", "both"],
	progress_callback: Callable[[int, int], None] | None = None,
	format: Literal["german", "english"] = "german",
) -> Sequence[PredictionResults]:
	# Convert argparse choice to boolean flags
	load_yolo = models in ["both", "yolo"]
	load_density = models in ["both", "density"]
	imgdir = Path(img_dir)
	modeldir = Path(model_dir)

	logger.info("Start script")

	# Create list of image paths from the image directory
	device = "cuda" if torch.cuda.is_available() else "cpu"
	image_paths = collect_paths(imgdir)
	yolo_model_path, dens_model, dens_transform = get_models(modeldir, device, load_density, load_yolo)
	logger.info("Loaded models")
	logger.info("Start inference")
	label_path = Path(
		imgdir / ".." / ".." / "labels" / "xywh",
	)

	counts = make_results_from_paths(
		image_paths=image_paths,
		label_path=label_path if label_path.exists() else None,
		yolo_model_path=yolo_model_path,
		dens_model=dens_model,
		dens_transform=dens_transform,
		device=device,
		out_path=output_dir if write_images else None,
		progress_callback=progress_callback,
	)
	assert len(counts) > 0, f"No images processed, but {len(image_paths)} images in directory"
	save_results(counts, output_dir, image_paths, format=format)
	logger.info("All images processed")
	return counts


if __name__ == "__main__":
	logging.basicConfig(
		level=logging.INFO,
		format="[%(asctime)s] %(levelname)s - %(message)s",
		datefmt="%H:%M:%S",
		handlers=[logging.FileHandler("results_logs.log"), logging.StreamHandler()],
	)
	args = parse_args()
	main(
		output_dir=args.output_dir,
		img_dir=args.imgdir,
		model_dir=args.modeldir,
		write_images=args.images,
		models=args.models,
		progress_callback=None,
	)
