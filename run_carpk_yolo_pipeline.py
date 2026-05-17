import json
import os
from pathlib import Path

import mlflow

from carpk_yolo_pipeline.phase_tracker import PhaseTracker
from carpk_yolo_pipeline.phases import (
    log_mlflow_summary,
    run_counting_inference,
    run_distributed_training,
    run_evaluation,
    run_hyperparameter_tuning,
    run_preprocessing,
)


def main():
    project_root = Path(os.environ.get("PROJECT_ROOT", ".")).resolve()
    artifacts_dir = project_root / os.environ.get("PIPELINE_ARTIFACTS_DIR", "pipeline_artifacts")
    dataset_root = Path(os.environ.get("CARPK_DATASET_DIR", str(project_root / "data" / "carpk"))).resolve()
    inference_dir = Path(os.environ.get("CARPK_INFER_DIR", str(dataset_root / "images" / "test"))).resolve()
    timeline_file = str(artifacts_dir / "phase_timeline.jsonl")
    tracker = PhaseTracker(timeline_file)

    mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000"))
    mlflow.set_experiment(os.environ.get("MLFLOW_EXPERIMENT_NAME", "carpk-yolo-pipeline"))

    model_name = os.environ.get("YOLO_MODEL", "yolov8n.pt")
    train_device = os.environ.get("TRAIN_DEVICE", "0")
    do_tuning = os.environ.get("RUN_TUNING", "1") == "1"

    with mlflow.start_run(run_name=os.environ.get("MLFLOW_RUN_NAME", "carpk-yolo-e2e")):
        with tracker.phase("preprocessing"):
            data_yaml = run_preprocessing(dataset_root=dataset_root, artifacts_dir=artifacts_dir)

        if do_tuning:
            with tracker.phase("hyperparameter_tuning"):
                best_hparams = run_hyperparameter_tuning(
                    model_name=model_name,
                    data_yaml=data_yaml,
                    project_dir=artifacts_dir,
                    device=train_device,
                )
        else:
            best_hparams = {
                "lr0": float(os.environ.get("LR0", "0.005")),
                "imgsz": int(os.environ.get("IMGSZ", "960")),
                "batch": int(os.environ.get("BATCH_SIZE", "16")),
            }

        with tracker.phase("training"):
            best_model = run_distributed_training(
                model_name=model_name,
                data_yaml=data_yaml,
                project_dir=artifacts_dir,
                device=train_device,
                hparams=best_hparams,
            )

        with tracker.phase("evaluation"):
            eval_metrics = run_evaluation(best_model=best_model, data_yaml=data_yaml, device=train_device)

        count_csv = artifacts_dir / "predictions" / "car_counts.csv"
        with tracker.phase("counting_inference"):
            run_counting_inference(best_model=best_model, input_dir=inference_dir, out_csv=count_csv)

        log_mlflow_summary(
            params={
                "dataset_root": str(dataset_root),
                "model_name": model_name,
                "train_device": train_device,
                "best_model_path": str(best_model),
                **best_hparams,
            },
            eval_metrics=eval_metrics,
        )
        mlflow.log_artifact(str(data_yaml))
        if count_csv.exists():
            mlflow.log_artifact(str(count_csv))
        mlflow.log_artifact(timeline_file)

        summary = {
            "best_model": str(best_model),
            "phase_timeline": timeline_file,
            "evaluation": eval_metrics,
            "best_hparams": best_hparams,
            "count_csv": str(count_csv),
        }
        summary_path = artifacts_dir / "pipeline_summary.json"
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        mlflow.log_artifact(str(summary_path))
        print(json.dumps(summary))


if __name__ == "__main__":
    main()

