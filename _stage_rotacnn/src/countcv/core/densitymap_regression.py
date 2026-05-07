import logging
from collections.abc import Sized
from pathlib import Path
from typing import cast

import mlflow
import optuna
import torch
import yaml
from torch import nn, optim
from torch.utils.data import DataLoader

from config.auth_mgr import AuthMgr
from countcv.core.data import SyncedTransformConfig, get_densitymap_loaders, infer_normalisation_constants
from countcv.core.experiment_utils import calc_energy_efficiency, calc_metrics, experiment_run
from countcv.core.fcrn_model import (
	BaseModel,
	DeeplabV3,
	FCRNBase,
	FCRNSkip,
	ModelFactory,
	QuantileDensityMapLoss,
	SAUnet,
	SOTAnet4,
)

logger = logging.getLogger(__name__)

MODEL_REGISTRY = {
	"fcrn-base": (FCRNBase, {}),
	"fcrn-skip": (FCRNSkip, {}),
	"SAUnet": (SAUnet, {"self_attention": True}),
	"Unet": (SAUnet, {"self_attention": False}),
	"sota-net": (SOTAnet4, {}),
	"deeplabv3": (DeeplabV3, {"freeze_backbone": True, "unfreeze_layer4": False}),
}


def train_one_epoch(
	model: nn.Module,
	loader: DataLoader,
	criterion: nn.Module,
	optimizer: optim.Optimizer,
	device: torch.device,
):
	model.train()
	running_loss = 0.0
	total = 0
	for inputs, targets in loader:
		inputs, targets = inputs.to(device), targets.to(device)
		optimizer.zero_grad()
		outputs = model(inputs)
		loss = criterion(outputs, targets)
		loss.backward()

		torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
		optimizer.step()

		running_loss += loss.item() * inputs.size(0)
		total += targets.size(0)
	epoch_loss = running_loss / total
	return epoch_loss


def evaluate(
	model: BaseModel,
	loader: DataLoader,
	criterion: nn.Module,
	device: torch.device,
	quantiles: bool,
	alpha: int,
	best_mae: float,
) -> tuple[float, float, dict[str, float]]:
	model.eval()
	running_loss = 0.0

	assert hasattr(loader.dataset, "__len__")
	out_dims = 3 if quantiles else 1
	out_counts = torch.empty(size=(len(cast(Sized, loader.dataset)), out_dims), device=device)
	true_counts = torch.empty(size=(len(cast(Sized, loader.dataset)),), device=device)
	total = 0
	offset = 0
	with torch.no_grad():
		for inputs, targets in loader:
			bsz = inputs.size(0)
			inputs, targets = inputs.to(device), targets.to(device)
			outputs = model.predict(inputs, device=device)  # B x Q x H x W
			loss = criterion(outputs, targets)
			running_loss += loss.item() * bsz
			total += bsz
			out_counts[offset : (offset + bsz), :] = outputs.sum(dim=(2, 3)) / alpha  # B x Q
			true_counts[offset : (offset + bsz)] = targets.sum(dim=(1, 2, 3)) / alpha  # B
			offset += bsz

		epoch_loss = running_loss / total
		middle_q = out_dims // 2  # If Q=3 this is 1 if Q=1 this is 0
		metrics_dict = calc_metrics(out_counts[:, middle_q].cpu().numpy(), true_counts.cpu().numpy())
		epoch_mae = metrics_dict.pop("mae")
		if quantiles:
			above_lower = true_counts >= out_counts[:, 0]
			below_upper = true_counts <= out_counts[:, 2]
			coverage = (above_lower & below_upper).to(dtype=torch.float).mean()
			metrics_dict["coverage"] = coverage.cpu().numpy().item()

	if epoch_mae < best_mae:
		logger.info(f"Validation MAE improved: {best_mae:.4f} -> {epoch_mae:.4f}.")
		best_mae = epoch_mae
		mlflow.log_metric("mae", best_mae)
	return epoch_loss, best_mae, metrics_dict


def run_experiment_densitymap(
	model_version: str,
	data_dir: Path,
	experiment_name: str,
	run_name: str,
	model_params: dict,
	auth: AuthMgr | None = None,
	num_workers: int = 4,
	pruning_callback=None,
) -> dict[str, float]:
	logger.info("Start density map experiment")
	torch.cuda.empty_cache()
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	model_class, preset_arguments = MODEL_REGISTRY[model_version]
	quantiles = model_params.pop("quantiles", False)
	transform_config = SyncedTransformConfig.from_dict(model_params)
	best_mae = torch.inf
	mae_patience = 0

	with experiment_run(
		experiment_name=experiment_name,
		run_name=run_name,
		auth=auth,
		log_params=model_params,
	) as tracker:
		checkpoint_path = Path(f"checkpoints/{model_version}_{experiment_name}/")
		checkpoint_path.mkdir(parents=True, exist_ok=True)
		if model_version in MODEL_REGISTRY:
			model_class = ModelFactory(model_class, tile_size=transform_config.tilesize, **preset_arguments)
			model = model_class(input_channels=3, output_channels=3 if quantiles else 1)
		else:
			ValueError(f"{model_version} not in MODEL_REGISTRY")

		num_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
		mlflow.log_param("trainable_parameter", num_trainable_params)
		model.to(device)
		# model.compile(mode="reduce-overhead")
		criterion = QuantileDensityMapLoss(
			alpha=transform_config.alpha,
			tau=[0.25, 0.5, 0.75] if quantiles else 0.5,
			lambda_count=model_params["lambda_count"],
		)

		optimizer = optim.AdamW(
			model.parameters(), lr=3e-3, betas=(model_params.get("beta1", 0.9), model_params.get("beta2", 0.999))
		)  # optim.SGD(model.parameters(), lr=0.01, weight_decay=0.0005, momentum=0.9)
		scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.3, patience=5)

		if transform_config.infer_normalisation:
			mean, std = infer_normalisation_constants(
				data_dir,
				transform_config,
			)
			transform_config.mean = mean
			transform_config.std = std

		train_loader, val_loader, transform = get_densitymap_loaders(
			transform_config=transform_config,
			batch_size=model_params["batch_size"],
			data_dir=data_dir,
			num_workers=num_workers,
		)
		patience = 0
		best_val_loss = torch.inf
		logger.info("Starting training...")
		for epoch in range(model_params["epochs"]):
			train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
			if epoch % model_params["eval_every_n"] == 0:
				val_loss, best_mae, metrics = evaluate(
					model,
					val_loader,
					criterion,
					device,
					quantiles,
					transform_config.alpha,
					best_mae=best_mae,
				)
				# Metric tracking
				metrics = metrics | {
					"lr": optimizer.param_groups[0]["lr"],
					"train_loss": train_loss,
					"val_loss": val_loss,
					"energy_efficiency": calc_energy_efficiency(
						entropy=metrics["entropy_gain"], energy=tracker._total_energy.kWh
					),
					"best_mae": best_mae,
				}
				logger.info({"epoch": epoch} | metrics)
				if auth:
					mlflow.environment_variables.MLFLOW_TRACKING_TOKEN.set(auth.get_token())  # Update authentication
				for name, value in metrics.items():
					mlflow.log_metric(name, value, step=epoch)
				scheduler.step(val_loss)

				# Early stopping criterion
				if val_loss <= best_val_loss:
					best_val_loss = val_loss
					patience = 0
				else:
					patience += 1
				if patience >= model_params["max_patience"]:
					break

				if model_params["stop_at_mae"] is not None and best_mae <= model_params["stop_at_mae"]:
					logger.info(f"Stopping at epoch {epoch} with MAE {best_mae:.4f}")
					if mae_patience == 1:
						break
					else:
						mae_patience = 1
				else:
					mae_patience = 0

				if pruning_callback is not None:
					should_prune = pruning_callback(epoch, best_mae)
					if should_prune:
						logger.info(f"Trial pruned at epoch {epoch} with MAE {best_mae:.4f}")
						raise optuna.TrialPruned()

		logger.info("Saving model and transform at checkpoint and MLFlow.")
		# log model at checkpoint
		torch.save(model.state_dict(), checkpoint_path / "model.pt")
		# Log model in mlflow
		inputs, outputs = next(iter(train_loader))
		mlflow.log_metric("early_stopped_at_epoch", epoch)
		transform_config.log_locally(checkpoint_path / "transform_config.yaml")
		mlflow.pytorch.log_model(
			model,
			name="model",
			registered_model_name="model",
			signature=mlflow.models.infer_signature(
				model_input=inputs.cpu().numpy(), model_output=outputs.cpu().numpy()
			),
			extra_files=[str(checkpoint_path / "transform_config.yaml")],
		)

		del model_params, transform_config, train_loader, val_loader, model
		return {"mae": best_mae}


if __name__ == "__main__":
	logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
	dataset = "foci"
	experiment_name = "test_foci_model_artifacts"
	auth = None
	with open("src/config/model/cnn_base.yaml") as f:
		flat_param_dict = yaml.load(f, Loader=yaml.SafeLoader)

	with open(f"src/config/dataset/{dataset}.yaml") as f:
		flat_param_dict = flat_param_dict | yaml.load(f, Loader=yaml.SafeLoader)

	run_experiment_densitymap(
		model_version=flat_param_dict["version"],
		data_dir=Path(flat_param_dict["path"]),
		experiment_name=experiment_name,
		run_name=f"{flat_param_dict['version']}",
		auth=auth,
		model_params=flat_param_dict,
	)
