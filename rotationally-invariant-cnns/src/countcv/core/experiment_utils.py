import logging
import os
import random
import sys
import tempfile
import warnings
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Any

import mlflow
import numpy as np
import torch
import yaml
from codecarbon import OfflineEmissionsTracker

from config.auth_mgr import AuthMgr
from countcv.core.data import SyncedTransform, SyncedTransformConfig


@contextmanager
def experiment_run(experiment_name: str, run_name: str, log_params: dict, auth: AuthMgr | None = None, seed: int = 42):
	"""Call init_mlflow() before using this context_manager!
	Set up of ml flow and carbon tracker.
	You must implement model initialisation and training when using this contextmanager function.

	Args:
		experiment_name (str): mlflow experiment name
		run_name (str): mlflow runname
		log_params (dict): addition params to log in mlflow
		auth (AuthMgr | None, optional): Uses remote mlflow via auth if set. Defaults to None.

	Returns:
		str: mlflow runid

	Yields:
		Iterator[str]: model initialisation and training
	"""
	# mlflow
	set_seed(seed)
	mlflow.set_experiment(experiment_name)
	with mlflow.start_run(run_name=run_name):
		mlflow.log_param("Cuda version", torch.version.cuda)
		mlflow.log_param("python_version", ".".join(map(str, sys.version_info[:3])))
		for k, v in log_params.items():
			mlflow.log_param(k, v)

		# CodeCarbon
		energy_logs_dir = Path(os.environ.get("CODECARBON_OUTPUT_DIR", "energy_logs"))
		energy_logs_dir.mkdir(exist_ok=True)
		tracker = OfflineEmissionsTracker(
			log_level="error",
			output_dir=str(energy_logs_dir),
			output_file=f"{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}_emissions.csv",
			country_iso_code="DEU",
		)
		tracker.start()

		try:
			yield tracker  # specific training logic
		finally:
			tracker.stop()
			mlflow.log_metrics(
				{
					"energy_joule": round(tracker._total_energy.kWh * 3.6e6, 5),
					"energy_gpu": round(tracker._total_gpu_energy.kWh * 3.6e6, 5),
					"energy_cpu": round(tracker._total_cpu_energy.kWh * 3.6e6, 5),
					"duration_sec": round(tracker.final_emissions_data.duration, 5),
				}
			)


def init_mlflow(remote_mlflow: bool = False) -> AuthMgr | None:
	if remote_mlflow:
		config_file = Path(__file__).resolve().parent.parent / "config" / "config.yaml"
		if not config_file.exists():
			raise FileNotFoundError(f"Could not find {str(config_file)} to init mlflow.")
		credentials = yaml.safe_load(config_file.read_text())
		cache_file = Path.home() / ".cache" / "mlflow_oidc" / "refresh.json"
		auth = AuthMgr(credentials["oidc_url"], credentials["client_id"], cache_file=cache_file)
		try:
			auth.read_cache()
			mlflow.set_tracking_uri("https://mlflow.ai-env.de")
			mlflow.environment_variables.MLFLOW_TRACKING_TOKEN.set(auth.get_token())
		except Exception:
			auth.login()
			auth.write_cache()
			mlflow.set_tracking_uri("https://mlflow.ai-env.de")
			mlflow.environment_variables.MLFLOW_TRACKING_TOKEN.set(auth.get_token())

		return auth
	else:
		mlflow_uri = os.getenv("MLFLOW_TRACKING_URI", default="mlruns")
		Path(mlflow_uri).mkdir(exist_ok=True)  # Create folder mlruns to locally track runs
		mlflow.set_tracking_uri(mlflow_uri)
		return None


def set_seed(seed: int = 42):
	torch.manual_seed(seed)
	torch.cuda.manual_seed_all(seed)
	np.random.seed(seed)
	random.seed(seed)
	torch.backends.cudnn.deterministic = True
	torch.backends.cudnn.benchmark = False


def calc_metrics(p: torch.Tensor | np.ndarray, t: torch.Tensor | np.ndarray) -> dict[str, float]:
	t: np.ndarray = np.asarray(t, dtype=float).ravel()
	p: np.ndarray = np.asarray(p, dtype=float).ravel()
	if t.shape != p.shape:
		raise ValueError("Ground truth `t` and predictions `p` must have the same shape.")
	mean_absolute_error = float(np.mean(np.abs(t - p)))
	mean_error = float(np.mean(t - p))
	error_variance = float(np.mean((t - p) ** 2))
	mape = float(np.mean(np.abs(t - p) / (t + 1)))
	entropy_gain = calc_entropy_gain(t, p)
	mean_count = float(np.mean(t))
	metrics_dict = {
		"mae": mean_absolute_error,
		"mean_error": mean_error,
		"error_variance": error_variance,
		"entropy_gain": entropy_gain,
		"mean_count": mean_count,
		"mape": mape,
	}
	return metrics_dict


def calc_energy_efficiency(entropy: float, energy: float, eps: float = 1) -> float:
	"""
	Calculates energy efficiency according to paper from entropy gain (in bits) and energy (in kwh).
	If entropy gain is negative, use 0 instead.
	Returns:
		efficiency in 1/T or bits/J
	"""
	efficiency = max(entropy, 0) / (energy * 3.6e6 + eps)
	return efficiency


def calc_entropy_gain(t: np.ndarray, p: np.ndarray) -> float:
	"""
	Compute the information entropy gain (in bits) between ground truth `t`
	and predictions `p` under a Gaussian approximation.

	Formula
	-------
	info_gain = 0.5 * log2(Var[t] / MSE)

	where:
	- Var[t] = variance of the ground truth targets t.
	- MSE = mean((p - t)^2), the mean squared error of predictions p.

	Parameters
	----------
	t : np.ndarray
		Ground truth values (1D array).
	p : np.ndarray
		Predicted values (same shape as `t`).

	Returns
	-------
	float
		Information entropy gain in bits.
		Returns 0.0 and raises a warning if variance of `t` is zero.

	Notes
	-----
	- A small epsilon is added to denominators to ensure numerical stability.
	- Negative values can occur if MSE >= Var[t].
	"""
	# --- Input validation ---
	t = np.asarray(t, dtype=float).ravel()
	p = np.asarray(p, dtype=float).ravel()
	if t.shape != p.shape:
		raise ValueError("Ground truth `t` and predictions `p` must have the same shape.")

	# --- Core calculation ---
	eps = 1e-12
	var_prior = np.var(t)
	if var_prior <= 0:
		warnings.warn("Variance of ground truth `t` is zero; info gain is undefined. Returning 0.0.", stacklevel=2)
		return 0.0

	mse = np.mean((p - t) ** 2)
	info_gain = 0.5 * np.log2((var_prior + eps) / (mse + eps))
	return info_gain


def load_mlflow_model(tracking_uri: str, run_id: str) -> tuple[Any, SyncedTransform] | Any:
	mlflow.set_tracking_uri(tracking_uri)
	model = mlflow.pyfunc.load_model(f"runs:/{run_id}/model").get_raw_model()
	try:
		with tempfile.TemporaryDirectory() as tmpdir:
			filename = mlflow.artifacts.download_artifacts(
				artifact_uri=f"runs:/{run_id}/configs/transform_config.yaml",
				dst_path=tmpdir,
			)
			with open(filename) as f:
				config_dict = yaml.safe_load(f)
				config = SyncedTransformConfig(config_dict)
		transform = SyncedTransform(config)
	except Exception as e:
		logging.warning(e, exc_info=True)
		return model
	return model, transform


def get_models(
	model_directory: Path, device: str, load_density: bool = True, load_yolo: bool = True
) -> tuple[Path | None, Any | None, SyncedTransform | None]:
	yolo_model_path = model_directory / "yolo" / "best.pt" if load_yolo else None
	dens_model = None
	dens_transform = None
	return yolo_model_path, dens_model, dens_transform
