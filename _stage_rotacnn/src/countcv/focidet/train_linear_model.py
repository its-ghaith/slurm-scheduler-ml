import logging
from pathlib import Path

import albumentations as A
import mlflow
import numpy as np
import pandas as pd
import torch
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader

from countcv.core.data import CountingDataset
from countcv.core.experiment_utils import experiment_run


class LinearDataset(CountingDataset):
	"""Wrapper around the CountingDataset which basically does nothing but retrieve the
	images and labels. No transform, as the result is independent due to global avg pooling."""

	def __init__(self, data_dir, set_type):
		super().__init__(
			data_dir,
			set_type,
			transform=None,
			return_format="image_label",
			prefetch_labels=False,
			fetch_dot_labels=True,
		)
		self.img_transform = A.Compose([A.CenterCrop(220, 220), A.ToTensorV2()])
		self.scaler = StandardScaler()

	def __getitem__(self, idx: int):  # type: ignore[override]
		image, label = super().__getitem__(idx)
		label = label["count"]
		image = self.img_transform(image=image)["image"].to(dtype=torch.float)
		image = image.mean(dim=(1, 2))
		return image, label


def gather_images(loader: DataLoader, scaler: StandardScaler | None, only_red_channel: bool):
	intensities = torch.empty(size=(len(loader.dataset), 3))
	targets = torch.empty(size=(len(loader.dataset),))
	offset = 0
	for inputs, labels in loader:
		bsz = inputs.size(0)
		intensities[offset : (offset + bsz)] = inputs
		targets[offset : (offset + bsz)] = labels
		offset += bsz

	intensities = intensities.cpu().numpy()
	targets = targets.cpu().numpy()
	if scaler:
		intensities, targets = scaler.fit_transform(intensities, targets)
	if only_red_channel:
		intensities = intensities[:, 0:1]
	return intensities, targets, scaler


def run_linear_experiment(
	model_version: str,
	data_dir: Path,
	experiment_name: str,
	run_name: str,
	num_workers: int = 4,
	linear: bool = True,
	only_red_channel: bool = False,
	standardize: bool = True,
):
	logging.info("Start linear experiment")
	torch.cuda.empty_cache()

	with experiment_run(
		experiment_name=experiment_name,
		run_name=run_name,
		log_params={},
	):
		checkpoint_path = f"checkpoints/{model_version}_{experiment_name}.pt"
		Path(checkpoint_path).parent.mkdir(parents=True, exist_ok=True)

		train_dataset = LinearDataset(data_dir=data_dir, set_type="train")
		train_loader = DataLoader(train_dataset, batch_size=4, num_workers=num_workers)

		val_dataset = LinearDataset(data_dir=data_dir, set_type="val")
		val_loader = DataLoader(val_dataset, batch_size=4, num_workers=num_workers)

		scaler = StandardScaler() if standardize else None
		X_train, y_train, scaler = gather_images(train_loader, scaler, only_red_channel)
		X_val, y_val, scaler = gather_images(val_loader, scaler, only_red_channel)

		model = LinearRegression(fit_intercept=not only_red_channel) if linear else DummyRegressor(strategy="mean")
		model.fit(X_train, y_train)
		pred = model.predict(X_val)

		names = ["intercept", "red", "green", "blue"]
		if linear:
			if only_red_channel:
				names = names[0:2]
			coefs_df = pd.DataFrame({"name": names, "coef": np.insert(model.coef_, 0, model.intercept_)})
		else:
			coefs_df = pd.DataFrame({"name": ["mean"], "coef": model.constant_[0]})

		logging.info(coefs_df)
		if standardize:
			y_val = scaler.inverse_transform(y_val)
			pred = scaler.inverse_transform(pred)
		mae = np.abs(y_val - pred).mean()
		logging.info(f"MAE: {mae}")

		mlflow.sklearn.log_model(
			model,
			"model",
			signature=mlflow.models.infer_signature(
				model_input=np.zeros_like(X_train), model_output=np.zeros_like(y_train)
			),
		)


if __name__ == "__main__":
	logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
	model_version = "linear"
	data_dir = Path("data/uc_cells")
	experiment_name = "linear_model"

	run_linear_experiment(
		model_version=model_version,
		data_dir=data_dir,
		experiment_name=experiment_name,
		run_name=f"{model_version}",
		linear=True,
		only_red_channel=False,
		standardize=True,
	)
