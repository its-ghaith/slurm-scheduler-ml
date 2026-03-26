#!/bin/bash
set -euo pipefail

BATCH_SIZES=(16 32 64 128)
JOB_FILE="slurm/train_mlflow_job.slurm"

if [ ! -f "$JOB_FILE" ]; then
  echo "Missing $JOB_FILE"
  exit 1
fi

export MLFLOW_TRACKING_URI="${MLFLOW_TRACKING_URI:-http://127.0.0.1:5000}"

echo "Submitting jobs with MLFLOW_TRACKING_URI=${MLFLOW_TRACKING_URI}"

for bs in "${BATCH_SIZES[@]}"; do
  echo "Submit batch size ${bs}"
  sbatch --export=ALL,BATCH_SIZE=${bs},MLFLOW_TRACKING_URI=${MLFLOW_TRACKING_URI} "$JOB_FILE"
done
