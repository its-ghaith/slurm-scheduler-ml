import logging
import re
from pathlib import Path
from typing import cast, no_type_check

import mlflow
import pandas as pd
import torch
from omegaconf import ListConfig
from torchmetrics.detection import IntersectionOverUnion
from torchmetrics.detection.mean_ap import MeanAveragePrecision
from tqdm import tqdm
from ultralytics import YOLO, settings
from ultralytics.utils.plotting import plot_results

from config.auth_mgr import AuthMgr
from countcv.core.data import CountingDataset
from countcv.core.experiment_utils import calc_energy_efficiency, calc_metrics, experiment_run


def run_experiment_ultralytics(
	model_version: str,
	data_dir: Path,
	data_yaml_path: Path,
	experiment_name: str,
	run_name: str,
	mlflow_log_params: dict,
	extra_params: dict,
	auth: AuthMgr | None = None,
) -> dict:
	# get train params from extra_params
	epochs = int(extra_params.get("epochs", 10))
	freeze = int(extra_params.get("freeze", 0))
	patience = int(extra_params.get("patience", 100))
	batch = int(extra_params.get("batch_size", 16))
	num_worker = int(extra_params.get("num_worker", 12))
	imgsz = extra_params.get("img_size", 640)
	pretrained = bool(extra_params.get("pretrained", True))
	if isinstance(imgsz, ListConfig):  # hydra creates ListConfig; predict can only handle list
		imgsz = list(imgsz)

	torch.cuda.empty_cache()
	settings.update({"mlflow": False})
	settings.update({"weights_dir": "/models"})
	with experiment_run(
		experiment_name=experiment_name,
		run_name=run_name,
		log_params=mlflow_log_params,
		auth=auth,
	) as tracker:
		model = YOLO(model_version)
		num_trainable_params = sum(p.numel() for p in model.model.parameters() if p.requires_grad)
		mlflow.log_param("trainable_parameter", num_trainable_params)
		results = model.train(
			data=data_yaml_path,
			epochs=epochs,
			freeze=freeze,
			imgsz=imgsz,
			batch=batch,
			patience=patience,
			pretrained=pretrained,
			single_cls=True,
			cache="ram",
			hsv_h=float(extra_params.get("hsv_h", 0.015)),
			hsv_s=float(extra_params.get("hsv_s", 0.12)),
			hsv_v=float(extra_params.get("hsv_v", 0.2)),
			translate=float(extra_params.get("translate", 0.12)),
			scale=float(extra_params.get("scale", 0.5)),
			degrees=float(extra_params.get("degrees", 0.0)),
			shear=float(extra_params.get("shear", 0.0)),
			mosaic=float(extra_params.get("mosaic", 0.0)),
			fliplr=float(extra_params.get("fliplr", 0.0)),
			flipud=float(extra_params.get("flipud", 0.0)),
			mixup=0.0,
		)
		save_yolo_training_plots(results)

		best_weights_path = model.trainer.best
		logging.info("Weights saved in: %s", best_weights_path)

		# log yolo artefacts and metrics to existing mlflow run:
		if not auth:  # log_artifacts does not work on remote mlflow (permission denied)
			mlflow.log_artifacts(str(results.save_dir), artifact_path="yolo_outputs")
		epoch_metrics = pd.read_csv(results.save_dir / "results.csv")
		for i, row in epoch_metrics.iterrows():
			epoch = int(row["epoch"]) if "epoch" in row else cast(int, i)
			for col in epoch_metrics.columns:
				if col != "epoch":
					safe_col = re.sub(r"[^\w\-/.: ]", "_", col.strip())
					if auth:
						mlflow.environment_variables.MLFLOW_TRACKING_TOKEN.set(auth.get_token())
					mlflow.log_metric(safe_col, row[col], step=epoch)
		mlflow.log_metric("early_stopped_at_epoch", len(epoch_metrics["epoch"]))

		# count and mean iou evaluation on val data
		# TODO move this out of code carbon code for final energy experiments
		return evaluate_yolo(
			model_path=best_weights_path,
			data_dir=data_dir,
			batch=batch,
			workers=num_worker,
			imgsz=imgsz,
			tracker=tracker,
		)


def save_yolo_training_plots(results):
	"""
	Save all main Ultralytics YOLO training plots to disk.
	This replicates what YOLO logs to MLflow automatically.
	"""
	save_dir = Path(results.save_dir)
	metrics = results

	logging.info(f"Saving training plots to: {save_dir}")

	results_csv = save_dir / "results.csv"
	if results_csv.exists():
		plot_results(file=str(results_csv), dir=str(save_dir))

	if hasattr(metrics, "confusion_matrix") and metrics.confusion_matrix is not None:
		metrics.confusion_matrix.plot(save_dir=save_dir)

	if hasattr(metrics, "pr_curve") and metrics.pr_curve is not None:
		metrics.pr_curve.plot(save_dir=save_dir)

	if hasattr(metrics, "f1_curve") and metrics.f1_curve is not None:
		metrics.f1_curve.plot(save_dir=save_dir)


@no_type_check
def evaluate_yolo(
	model_path: str,
	data_dir: Path,
	conf_threshold: float = 0.3,
	iou_threshold: float = 0.5,
	workers: int = 12,
	batch: int = 16,
	imgsz: int | list = 640,
	tracker=None,
):
	"""Evaluation specific to cell uc object detection results.

	Args:
		model_path (str): trained object detection model
		data_dir (Path): dataset directory
		conf_threshold (float, optional): bbox confidence threshold. Defaults to 0.3.
		iou_threshold (float, optional): intersection over union threshold (gt/pred) of bboxes. Defaults to 0.5.
	"""
	if isinstance(imgsz, ListConfig):  # hydra creates ListConfig; predict can only handle list
		imgsz = list(imgsz)
	model = YOLO(model_path)
	logging.info(f"loaded model from {model_path}.")
	val_dataset = CountingDataset(
		data_dir=data_dir,
		set_type="val",
		transform=None,
		return_format="path_label",
		prefetch_labels=True,
	)
	image_paths = val_dataset.paths

	map_metric = MeanAveragePrecision(box_format="xyxy", iou_type="bbox", iou_thresholds=[iou_threshold])
	iou_metric = IntersectionOverUnion(box_format="xyxy", iou_threshold=iou_threshold)
	gts = val_dataset.labels
	assert gts is not None
	gt_counts, pred_counts = [], []

	# workaround to allow fast processing on small laptop gpu and huge hpc gpu
	n = len(image_paths)
	for start in range(0, n, batch):
		end = min(start + batch, n)
		batch_paths = image_paths[start:end]
		batch_gts = gts[start:end]
		results = model.predict(
			source=batch_paths,
			imgsz=imgsz,
			conf=conf_threshold,
			batch=min(batch, end - start),
			workers=workers,
			half=True,
			verbose=False,
			stream=False,
			save=False,
			max_det=300,
			retina_masks=False,
			iou=0.7,  # default parameter -> configure dependent on dataset?
		)

		for r, gt in zip(results, batch_gts, strict=False):
			#  Predictions -> tensors on CPU
			if getattr(r, "boxes", None) is not None and r.boxes.xyxy is not None and r.boxes.xyxy.numel() > 0:
				boxes_tensor = r.boxes.xyxy.detach().to("cpu", dtype=torch.float32)
				scores_tensor = r.boxes.conf.detach().to("cpu", dtype=torch.float32)
				n_pred = boxes_tensor.shape[0]
			else:
				boxes_tensor = torch.zeros((0, 4), dtype=torch.float32)
				scores_tensor = torch.zeros((0,), dtype=torch.float32)
				n_pred = 0

			pred_dict: dict[str, torch.Tensor] = {
				"boxes": boxes_tensor,
				"scores": scores_tensor,
				"labels": torch.zeros(n_pred, dtype=torch.int32),  # single class
			}

			# ---- Ground truth dict[str, Any] → tensors ----
			gt_bboxes = gt.get("bboxes", [])
			if gt_bboxes:
				gt_boxes = torch.tensor(gt_bboxes, dtype=torch.float32)
				n_gt = int(gt_boxes.shape[0])
			else:
				gt_boxes = torch.zeros((0, 4), dtype=torch.float32)
				n_gt = 0
			target_dict = {
				"boxes": gt_boxes,
				"labels": torch.zeros(n_gt, dtype=torch.int64),
			}

			map_metric.update([pred_dict], [target_dict])
			iou_metric.update([pred_dict], [target_dict])

			gt_counts.append(int(gt.get("count", n_gt)))
			pred_counts.append(n_pred)
			del r, boxes_tensor, scores_tensor, pred_dict, target_dict, gt_boxes

	pred_counts, gt_counts = torch.tensor(pred_counts), torch.tensor(gt_counts)
	metrics = calc_metrics(pred_counts, gt_counts)
	metrics["energy_efficiency"] = calc_energy_efficiency(metrics["entropy_gain"], tracker._total_energy.kWh)

	map_results = map_metric.compute()
	iou_results = iou_metric.compute()
	logging.info(f"Count MAE: {metrics.get('mae'):.4f}")
	logging.info(f"mAP (0.50): {map_results['map']:.4f}")
	# logging.info(f"mAR (0.50): {map_results['mar_100']:.4f}")
	logging.info(f"Mean IoU (threshold={iou_threshold}): {iou_results['iou']:.4f}")

	mlflow.log_metric("val_map", float(map_results["map"]))
	# mlflow.log_metric("val_mar", float(map_results["mar_100"]))
	mlflow.log_metric("val_mean_iou", float(iou_results["iou"]))
	for name, value in metrics.items():
		mlflow.log_metric(name, value)
	return metrics
