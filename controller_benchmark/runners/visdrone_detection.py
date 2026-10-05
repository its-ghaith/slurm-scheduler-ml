from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import mlflow
from ultralytics import YOLO

from .common import EpochTelemetry, decode_run_spec, number
from .detection_count_metrics import evaluate_count_metrics, evaluate_detection_quality


def metric(metrics: dict, *keys: str) -> float:
    for key in keys:
        value = number(metrics.get(key))
        if value is not None:
            return value
    return 0.0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-spec-base64", required=True)
    parser.add_argument(
        "--data-yaml",
        type=Path,
        default=Path("/workspace-cache/controller-datasets-v1/visdrone-vehicles/data.yaml"),
    )
    parser.add_argument("--output-dir", type=Path, default=Path("/workspace/energy_metrics"))
    args = parser.parse_args()
    spec = decode_run_spec(args.run_spec_base64)
    pretrained = bool(spec["case"].get("pretrained", False))
    mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI", "http://mlflow:5000"))
    mlflow.set_experiment(spec["execution"]["experiment_name"])
    telemetry = EpochTelemetry(spec, args.output_dir)
    telemetry.start()
    try:
        with mlflow.start_run(run_name=f"job-energy-{telemetry.job_id}"):
            mlflow.log_params(
                {
                    "runner": spec["runner"],
                    "scenario": spec["scenario"],
                    "controller_id": spec["controller_id"],
                    "training_seed": spec["training_seed"],
                    "benchmark_case_id": spec["benchmark_case_id"],
                    "dataset": "VisDrone2019-DET vehicle subset",
                    "vehicle_classes": "car,van,truck,bus",
                    "unified_class": "vehicle",
                    "model": spec["case"]["model_version"],
                    "pretrained": pretrained,
                    "pretraining_source": "COCO" if pretrained else "none",
                }
            )
            resume_from_job_id = str(spec["case"].get("resume_from_job_id", "") or "")
            resume_checkpoint = (
                Path("/workspace/runs/controller-benchmark")
                / f"job-{resume_from_job_id}"
                / "weights"
                / "last.pt"
            )
            if resume_from_job_id:
                if not resume_checkpoint.exists():
                    raise FileNotFoundError(f"Resume checkpoint not found: {resume_checkpoint}")
                model = YOLO(str(resume_checkpoint))
                mlflow.log_param("resume_from_job_id", resume_from_job_id)
                mlflow.log_param("resume_checkpoint", str(resume_checkpoint))
            else:
                model = YOLO(spec["case"]["model_version"])

            def on_start(trainer):
                telemetry.start_epoch()

            def on_end(trainer):
                values = getattr(trainer, "metrics", {}) or {}
                map50_95 = metric(values, "metrics/mAP50-95(B)", "metrics/mAP50-95")
                extra = {
                    "map50_95": map50_95,
                    "quality_score": map50_95,
                    "map50": metric(values, "metrics/mAP50(B)", "metrics/mAP50"),
                    "precision": metric(values, "metrics/precision(B)", "metrics/precision"),
                    "recall": metric(values, "metrics/recall(B)", "metrics/recall"),
                }
                if telemetry.finish_epoch(int(trainer.epoch) + 1, map50_95, extra):
                    trainer.stop = True

            model.add_callback("on_train_epoch_start", on_start)
            model.add_callback("on_fit_epoch_end", on_end)
            train_kwargs = {
                "data": str(args.data_yaml),
                "epochs": int(spec["execution"]["max_epochs"]),
                "imgsz": int(spec["execution"]["image_size"]),
                "batch": int(spec["execution"]["batch_size"]),
                "patience": int(spec["execution"]["max_epochs"]),
                "pretrained": pretrained,
                "single_cls": True,
                "cache": False,
                "workers": int(spec["case"].get("workers", 0)),
                "seed": int(spec["training_seed"]),
                "deterministic": True,
                "project": "/workspace/runs/controller-benchmark",
                "name": f"job-{telemetry.job_id}",
            }
            if resume_from_job_id:
                train_kwargs["resume"] = True
            model.train(**train_kwargs)
            trainer = model.trainer
            best_path = Path(trainer.best)
            summary, gpu = telemetry.finalize()
            test_metrics = evaluate_detection_quality(
                best_path,
                args.data_yaml,
                image_size=int(spec["execution"]["image_size"]),
                batch_size=int(spec["execution"]["batch_size"]),
            )
            count_metrics = evaluate_count_metrics(
                best_path,
                args.data_yaml,
                image_size=int(spec["execution"]["image_size"]),
            )
            summary.update(
                {
                    "dataset_name": spec["case"]["dataset_name"],
                    "primary_metric": "map50_95",
                    **test_metrics,
                    **count_metrics,
                }
            )
            telemetry.epoch_summary_path.write_text(
                json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
            )
            mlflow.log_metrics(
                {
                    f"final_{spec['quality_metric']}": summary[f"final_{spec['quality_metric']}"],
                    f"best_{spec['quality_metric']}": summary[f"best_{spec['quality_metric']}"],
                    "gpu_energy_wh": gpu["gpu_energy_kwh"] * 1000.0,
                    **test_metrics,
                    **count_metrics,
                }
            )
            mlflow.log_artifact(str(telemetry.epoch_summary_path), artifact_path="energy")
            mlflow.log_artifact(str(telemetry.gpu_summary_path), artifact_path="energy")
            print(json.dumps({"summary": summary, "gpu": gpu}))
    finally:
        telemetry.close()


if __name__ == "__main__":
    main()
