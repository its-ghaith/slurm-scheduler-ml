#!/bin/bash
#SBATCH --job-name=mlflow
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --time=08:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail

cd "${SLURM_SUBMIT_DIR}"
mkdir -p logs mlflow_data mlruns

module purge
# Optional: an deinen Cluster anpassen
# module load python/3.10

if [ -f ".venv/bin/activate" ]; then
  source .venv/bin/activate
fi

PORT="${MLFLOW_PORT:-5000}"
HOST="${MLFLOW_HOST:-0.0.0.0}"

echo "Starting MLflow on $(hostname):${PORT}"

mlflow server \
  --host "${HOST}" \
  --port "${PORT}" \
  --backend-store-uri "sqlite:///$(pwd)/mlflow_data/mlflow.db" \
  --default-artifact-root "$(pwd)/mlruns"
