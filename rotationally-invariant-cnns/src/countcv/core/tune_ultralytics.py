from copy import deepcopy
from pathlib import Path

import optuna
from ultralytics_detectors import run_experiment_ultralytics


def tune_ultralytics_augmentations(
	model_version: str,
	data_dir: Path,
	data_yaml_path: Path,
	experiment_name: str,
	run_name: str,
	mlflow_log_params: dict,
	extra_params: dict,
	n_trials: int = 10,
	auth=None,
):
	"""Optuna-based hyperparameter tuning wrapper for Ultralytics YOLO using count-based evaluation."""

	# search ranges; for carpk and cell?
	search_space_carpk = {
		"fliplr": [0.0, 0.5],
		"flipud": [0.0, 0.5],
	}

	search_space_synth = {
		"hsv_h": (0.02, 0.03),
		"hsv_s": (0.1, 0.5),
		"hsv_v": (0.2, 0.5),
		"degrees": (0, 20),
		"translate": (0.05, 0.15),
		"scale": (0.15, 0.4),
		"mosaic": (0.25, 0.7),
		"fliplr": [0.0, 0.5],
		"flipud": [0.0, 0.5],
	}

	def objective(trial):
		# copy base params and inject trial augmentations
		params = deepcopy(extra_params)
		search_space = search_space_carpk if params["dataset_name"] == "carpk" else search_space_synth

		params.update(
			{
				# "hsv_h": trial.suggest_float("hsv_h", *search_space["hsv_h"]),
				# "hsv_s": trial.suggest_float("hsv_s", *search_space["hsv_s"]),
				# "hsv_v": trial.suggest_float("hsv_v", *search_space["hsv_v"]),
				# "degrees": trial.suggest_float("degrees", *search_space["degrees"]),
				# "translate": trial.suggest_float("translate", *search_space["translate"]),
				# "scale": trial.suggest_float("scale", *search_space["scale"]),
				# "mosaic": trial.suggest_float("mosaic", *search_space["mosaic"]),
				"fliplr": trial.suggest_float("fliplr", *search_space["fliplr"]),
				"flipud": trial.suggest_float("flipud", *search_space["flipud"]),
			}
		)

		trial_name = f"{run_name}_trial{trial.number}"
		# train one short run
		metrics = run_experiment_ultralytics(
			model_version=model_version,
			data_dir=data_dir,
			data_yaml_path=data_yaml_path,
			experiment_name=experiment_name,
			run_name=trial_name,
			mlflow_log_params={**mlflow_log_params, **params},
			extra_params=params,
			auth=auth,
		)

		mae = float(metrics.get("mae", 9999))
		trial.report(mae, step=1)
		return mae

	study = optuna.create_study(direction="minimize")
	study.optimize(objective, n_trials=n_trials)
	print("Best trial:", study.best_trial.number)
	print("Best params:", study.best_trial.params)
	print("Best MAE:", study.best_value)
	return study.best_trial.params
