import argparse
import json
import mlflow


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--summary-json", required=True)
    p.add_argument("--tracking-uri", required=True)
    p.add_argument("--experiment", default="slurm-job-energy")
    p.add_argument("--run-name", default=None)
    p.add_argument("--run-id", default=None)
    args = p.parse_args()

    with open(args.summary_json, "r", encoding="utf-8") as f:
        data = json.load(f)

    mlflow.set_tracking_uri(args.tracking_uri)
    if args.run_id:
        run_ctx = mlflow.start_run(run_id=args.run_id)
    else:
        mlflow.set_experiment(args.experiment)
        run_name = args.run_name or f"slurm-job-{data.get('job_id', 'unknown')}"
        run_ctx = mlflow.start_run(run_name=run_name)

    with run_ctx:
        for k, v in data.items():
            if isinstance(v, (int, float)):
                mlflow.log_metric(k, float(v))
            else:
                mlflow.log_param(k, str(v))
        mlflow.log_artifact(args.summary_json, artifact_path="energy")


if __name__ == "__main__":
    main()
