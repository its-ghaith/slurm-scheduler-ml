import logging
from pathlib import Path

import hydra

from countcv.core.densitymap_regression import run_experiment_densitymap
from countcv.core.experiment_utils import init_mlflow, set_seed
from countcv.core.ultralytics_detectors import run_experiment_ultralytics, save_cropped_images
from countcv.effcv.preprocess_carpk import CarpkLabelCreator, CarpkSplitter
from countcv.effcv.preprocess_synthcell import SynthCellsLabelCreator, SynthCellsSplitter
from countcv.focidet.preprocess import FociDataLabelCreator, FociDataSplitter

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
		data_dir_cropped = data_dir / "cropped"
		if cfg.dataset.dataset_name == "foci":
			save_cropped_images(data_dir=data_dir_cropped, num_worker=num_worker)
		# Need to define new data_dir, otherwise resplitting does not work as datasplitter chooses cropped/ as origin
		run_dir = (
			data_dir_cropped
			if cfg.model.approach == "objectdetection" and cfg.dataset.dataset_name == "foci"
			else data_dir
		)

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

		elif cfg.model.approach == "densitymap":
			run_experiment_densitymap(
				model_version=model_version,
				data_dir=run_dir,
				experiment_name=experiment_name,
				model_params=flat_param_dict,
				run_name=f"{model_version}",
				auth=auth,
				num_workers=num_worker,
			)


def get_label_creator(dataset: str, dataset_root: Path):
	if dataset == "carpk":
		return CarpkLabelCreator(dataset_root=dataset_root)
	elif dataset == "synthcells":
		return SynthCellsLabelCreator(dataset_root=dataset_root)
	elif dataset == "foci":
		return FociDataLabelCreator(dataset_root=dataset_root)
	else:
		raise ValueError(f"Unknown dataset: {dataset}")


def get_data_splitter(dataset: str, dataset_root: Path):
	if dataset == "carpk":
		return CarpkSplitter(dataset_root)
	elif dataset == "synthcells":
		return SynthCellsSplitter(dataset_root)
	elif dataset == "foci":
		return FociDataSplitter(dataset_root)
	else:
		raise ValueError(f"Unknown dataset: {dataset}")


if __name__ == "__main__":
	logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
	train()
