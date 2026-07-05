import logging
from pathlib import Path

import hydra

from countcv.core.experiment_utils import init_mlflow, set_seed
from countcv.core.ultralytics_detectors import run_experiment_ultralytics
from countcv.effcv.preprocess_carpk import CarpkLabelCreator, CarpkSplitter

logger = logging.getLogger(__name__)


@hydra.main(version_base="1.3", config_path="../../config", config_name="default")
def train(cfg):
	# load cfg params
	remote_mlflow = cfg.remote_mlflow
	num_worker = cfg.num_worker
	model_version = cfg.model.version
	dataset: str = cfg.dataset.dataset_name
	data_dir = Path(cfg.dataset.path)
	data_yaml_path = cfg.dataset.data_yaml
	experiment_name = cfg.experiment

	if dataset != "carpk":
		raise ValueError(f"Only 'carpk' is supported in this repository. Received: {dataset}")

	creator = get_label_creator(dataset=dataset, dataset_root=data_dir)
	creator.create_labels()

	for seed in cfg.seeds:
		set_seed(seed)
		flat_param_dict = {
			**dict(cfg.dataset),
			**dict(cfg.model),
			"seed": seed,
			"num_worker": num_worker,
		}  # for mlflow log
		auth = init_mlflow(remote_mlflow)

		train_size = flat_param_dict.get("train_size", False)
		splitter = get_data_splitter(dataset=dataset, dataset_root=data_dir)
		splitter.split(train_size=0.8, seed=seed, limit_train_size=train_size)

		logger.info(f"Start training of {model_version}.")
		run_dir = data_dir

		if cfg.model.approach == "objectdetection":
			run_experiment_ultralytics(
				data_dir=run_dir,
				data_yaml_path=data_yaml_path,
				model_version=model_version,
				auth=auth,
				experiment_name=experiment_name,
				run_name=f"{model_version}",
				mlflow_log_params=flat_param_dict,
				extra_params=flat_param_dict,
			)
		else:
			raise ValueError(f"Only object detection is supported for CARPK. Received approach: {cfg.model.approach}")


def get_label_creator(dataset: str, dataset_root: Path):
	if dataset != "carpk":
		raise ValueError(f"Unknown dataset: {dataset}")
	return CarpkLabelCreator(dataset_root=dataset_root)


def get_data_splitter(dataset: str, dataset_root: Path):
	if dataset != "carpk":
		raise ValueError(f"Unknown dataset: {dataset}")
	return CarpkSplitter(dataset_root)


if __name__ == "__main__":
	logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
	train()
