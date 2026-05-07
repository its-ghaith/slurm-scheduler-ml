import logging
from copy import deepcopy
from pathlib import Path

import optuna
import yaml
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler


def create_pruning_callback(trial: optuna.Trial):
	"""
	Factory function that creates a callback for reporting intermediate values to Optuna.

	This callback should be called during training at each evaluation step.
	Returns a callable that takes (epoch, mae) and reports to the trial.
	"""

	def report_and_check_pruning(epoch: int, mae: float) -> bool:
		"""
		Report intermediate MAE to Optuna and check if trial should be pruned.

		Args:
			epoch: Current epoch number
			mae: Current validation MAE

		Returns:
			True if trial should be pruned (stop training), False otherwise
		"""
		trial.report(mae, step=epoch)
		return trial.should_prune()

	return report_and_check_pruning


def tune_dmap_hyperparams(
	model_version: str,
	data_dir: Path,
	experiment_name: str,
	run_name: str,
	extra_params: dict,
	n_trials: int | None = 20,
	timeout: float | None = None,
	auth=None,
	storage: str | None = None,
	n_startup_trials: int = 5,
	n_warmup_steps: int = 10,
	n_jobs: int = 1,
):
	"""
	Optuna-based hyperparameter tuning with early stopping of unpromising trials.

	Args:
		model_version: Model architecture version string
		data_dir: Path to dataset directory
		experiment_name: MLflow experiment name
		run_name: Base name for runs
		extra_params: Base training parameters
		n_trials: Number of optimization trials
		timeout: Run study for the specified number of seconds instead of a fixed number of runs
		auth: Authentication manager (optional)
		storage: Optuna storage URL for persistence (e.g., "sqlite:///optuna_study.db")
		n_startup_trials: Number of trials before pruner starts (allows baseline establishment)
		n_warmup_steps: Number of epochs before pruning can occur within a trial
		pruner_type: "median" or "hyperband" - pruning strategy
		n_jobs: Number of parallel trials (use with storage for proper synchronization)

	Returns:
		Dictionary of best hyperparameters
	"""
	# Import here to avoid circular dependency issues
	from countcv.core.densitymap_regression import run_experiment_densitymap

	# Define search space
	search_space = {
		"sigma": (1.5, 3),
		"lambda_count": (1e-4, 1e-2),
		"flip_rotate": [True, False],
		"infer_normalisation": [True, False],
		"degree": (0.0, 90.0),
		"beta1": (0.8, 0.9999),
		"beta2": (0.8, 0.9999),
	}

	pruner = MedianPruner(
		n_startup_trials=n_startup_trials,  # Don't prune first N trials (need baseline)
		n_warmup_steps=n_warmup_steps,  # Don't prune before N steps within a trial
		interval_steps=1,  # Check pruning at every report
	)

	# TPE sampler with multivariate option for correlated hyperparameters
	sampler = TPESampler(
		n_startup_trials=n_startup_trials,
		multivariate=True,  # Model parameter correlations
		seed=42,
	)

	def objective(trial: optuna.Trial) -> float:
		"""Objective function with intermediate value reporting for pruning."""

		# Sample hyperparameters
		params = deepcopy(extra_params)
		params.update(
			{
				"sigma": trial.suggest_float("sigma", *search_space["sigma"]),
				"lambda_count": trial.suggest_float("lambda_count", *search_space["lambda_count"], log=True),
				"flip_rotate": trial.suggest_categorical("flip_rotate", search_space["flip_rotate"]),
				"infer_normalisation": trial.suggest_categorical(
					"infer_normalisation", search_space["infer_normalisation"]
				),
				"degree": trial.suggest_float("degree", *search_space["degree"]),
				"beta1": trial.suggest_float("beta1", *search_space["beta1"]),
				"beta2": trial.suggest_float("beta2", *search_space["beta2"]),
			}
		)

		logging.info(f"Trial {trial.number} params: {params}")
		trial_name = f"{run_name}_trial{trial.number}"

		# Create pruning callback
		pruning_callback = create_pruning_callback(trial)

		try:
			# Run experiment with pruning callback
			metrics = run_experiment_densitymap(
				model_version=model_version,
				data_dir=data_dir,
				experiment_name=experiment_name,
				run_name=trial_name,
				model_params=params,
				auth=auth,
				pruning_callback=pruning_callback,  # Pass callback for intermediate reporting
			)

			mae = float(metrics.get("mae", float("inf")))

		except optuna.TrialPruned:
			# Re-raise to let Optuna handle it
			logging.info(f"Trial {trial.number} was pruned.")
			raise

		logging.info(f"Trial {trial.number} completed with MAE: {mae}")
		return mae

	# Create or load study
	study = optuna.create_study(
		study_name=f"{experiment_name}_{run_name}",
		direction="minimize",
		pruner=pruner,
		sampler=sampler,
		storage=storage,
		load_if_exists=True,  # Resume if study exists in storage
	)

	# Add callbacks for logging
	def log_trial_callback(study: optuna.Study, trial: optuna.Trial):
		"""Log trial results including pruned status."""
		assert hasattr(trial, "state")
		assert hasattr(trial, "last_step")
		assert hasattr(trial, "value")

		if trial.state == optuna.trial.TrialState.PRUNED:
			logging.info(f"Trial {trial.number} pruned at step {trial.last_step}")
		elif trial.state == optuna.trial.TrialState.COMPLETE:
			logging.info(
				f"Trial {trial.number} finished with MAE: {trial.value:.4f}. Best so far: {study.best_value:.4f}"
			)

	if n_trials and timeout:
		ValueError("Only set 'n_trials' or 'timeout', not both.")
	if n_trials is None and timeout is None:
		ValueError("Set either 'n_trials' or 'timeout'")

	# Run optimization
	study.optimize(
		objective,
		n_trials=n_trials,
		timeout=timeout,
		n_jobs=n_jobs,
		callbacks=[log_trial_callback],
		catch=(Exception,),  # Catch exceptions to continue optimization
		gc_after_trial=True,  # Help with GPU memory
	)

	# Report results
	pruned_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED]
	complete_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]

	logging.info("=" * 60)
	logging.info("Study Statistics:")
	logging.info(f"  Completed trials: {len(complete_trials)}")
	logging.info(f"  Pruned trials: {len(pruned_trials)}")
	logging.info(f"  Total trials: {len(study.trials)}")

	if complete_trials:
		logging.info(f"\nBest trial: {study.best_trial.number}")
		logging.info(f"Best MAE: {study.best_value:.4f}")
		logging.info("Best hyperparameters:")
		for key, value in study.best_trial.params.items():
			logging.info(f"  {key}: {value}")

	# Calculate energy savings from pruning
	if pruned_trials:
		avg_pruned_step = sum(t.last_step or 0 for t in pruned_trials) / len(pruned_trials)
		max_steps = extra_params.get("epochs", 100) // extra_params.get("eval_every_n", 1)
		savings_pct = (1 - avg_pruned_step / max_steps) * 100 * len(pruned_trials) / len(study.trials)
		logging.info(f"\nEstimated compute savings from pruning: ~{savings_pct:.1f}%")

	return study.best_trial.params if complete_trials else {}


if __name__ == "__main__":
	logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

	# Optuna logging configuration
	optuna.logging.set_verbosity(optuna.logging.INFO)

	dataset = "foci"
	experiment_name = "tune_foci_hyperparam"
	auth = None

	with open("config/model/cnn_base.yaml") as f:
		flat_param_dict = yaml.load(f, Loader=yaml.SafeLoader)

	with open(f"config/dataset/{dataset}.yaml") as f:
		flat_param_dict = flat_param_dict | yaml.load(f, Loader=yaml.SafeLoader)

	best_params = tune_dmap_hyperparams(
		model_version=flat_param_dict["version"],
		data_dir=Path(flat_param_dict["path"]),
		experiment_name=experiment_name,
		run_name=f"{flat_param_dict['version']}",
		auth=auth,
		extra_params=flat_param_dict,
		n_trials=20,
		n_startup_trials=5,  # Complete at least 5 trials before pruning
		n_warmup_steps=20,  # Don't prune before epoch 20
	)

	# Optionally save best params
	if best_params:
		with open(f"best_params_{dataset}.yaml", "w") as f:
			yaml.dump(best_params, f)
		logging.info(f"Best parameters saved to best_params_{dataset}.yaml")
