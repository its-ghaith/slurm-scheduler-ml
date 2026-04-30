#!/bin/bash
#SBATCH --job-name=ml-energy-local
#SBATCH --partition=debug
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --gres=gpu:1
#SBATCH --time=00:20:00
#SBATCH --chdir=/workspace
#SBATCH --output=/workspace/logs/%x-%j.out
#SBATCH --error=/workspace/logs/%x-%j.err

set -euo pipefail

cd "${SLURM_SUBMIT_DIR}"

export BATCH_SIZE="${BATCH_SIZE:-32}"
export MLFLOW_TRACKING_URI="${MLFLOW_TRACKING_URI:-http://mlflow:5000}"
export GPU_SAMPLE_INTERVAL_SEC="${GPU_SAMPLE_INTERVAL_SEC:-0.2}"
export ELECTRICITY_PRICE_EUR_PER_KWH="${ELECTRICITY_PRICE_EUR_PER_KWH:-0.30}"
export GRID_CO2_KG_PER_KWH="${GRID_CO2_KG_PER_KWH:-0.4}"
export PUE_FACTOR="${PUE_FACTOR:-1.0}"
export GPU_METRICS_DIR="${GPU_METRICS_DIR:-/workspace/energy_metrics}"
export PROM_TEXTFILE_DIR="${PROM_TEXTFILE_DIR:-${GPU_METRICS_DIR}/node_exporter}"
export MLFLOW_LOG_JOB_ENERGY="${MLFLOW_LOG_JOB_ENERGY:-1}"
export MLFLOW_JOB_ENERGY_EXPERIMENT="${MLFLOW_JOB_ENERGY_EXPERIMENT:-ml-energy-poc}"
export MLFLOW_EXPERIMENT_NAME="${MLFLOW_EXPERIMENT_NAME:-ml-energy-poc}"
export TRAIN_CMD="${TRAIN_CMD:-python train_with_energy_tracking_mlflow.py}"

echo "Running local SLURM simulation on: $(hostname)"
echo "BATCH_SIZE=${BATCH_SIZE}"
echo "MLFLOW_TRACKING_URI=${MLFLOW_TRACKING_URI}"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
echo "GPU_SAMPLE_INTERVAL_SEC=${GPU_SAMPLE_INTERVAL_SEC}"
echo "TRAIN_CMD=${TRAIN_CMD}"

python -c "import torch,sys; print('torch.cuda.is_available()=', torch.cuda.is_available()); sys.exit(0 if torch.cuda.is_available() else 1)"

mkdir -p "${GPU_METRICS_DIR}"
mkdir -p "${PROM_TEXTFILE_DIR}"
JOB_ID="${SLURM_JOB_ID:-local}"
GPU_CSV="${GPU_METRICS_DIR}/gpu_metrics_job_${JOB_ID}.csv"
GPU_SUMMARY_JSON="${GPU_METRICS_DIR}/gpu_summary_job_${JOB_ID}.json"
MLFLOW_RUN_ID_FILE="${GPU_METRICS_DIR}/mlflow_run_id_job_${JOB_ID}.txt"
export MLFLOW_RUN_NAME="job-energy-${JOB_ID}"
export MLFLOW_RUN_ID_FILE

bash slurm/gpu_monitor.sh "${GPU_CSV}" "${GPU_SAMPLE_INTERVAL_SEC}" &
MONITOR_PID=$!

set +e
bash -lc "${TRAIN_CMD}"
TRAIN_EXIT=$?
set -e

kill "${MONITOR_PID}" 2>/dev/null || true
wait "${MONITOR_PID}" 2>/dev/null || true

python slurm/summarize_gpu_metrics.py \
  --input "${GPU_CSV}" \
  --output "${GPU_SUMMARY_JSON}" \
  --job-id "${JOB_ID}" \
  --price-eur-kwh "${ELECTRICITY_PRICE_EUR_PER_KWH}" \
  --co2-kg-kwh "${GRID_CO2_KG_PER_KWH}" \
  --pue "${PUE_FACTOR}"

if [ "${MLFLOW_LOG_JOB_ENERGY}" = "1" ] || [ "${MLFLOW_LOG_JOB_ENERGY,,}" = "true" ] || [ "${MLFLOW_LOG_JOB_ENERGY,,}" = "yes" ]; then
  RUN_ID=""
  if [ -f "${MLFLOW_RUN_ID_FILE}" ]; then
    RUN_ID="$(cat "${MLFLOW_RUN_ID_FILE}" 2>/dev/null || true)"
  fi
  python slurm/log_job_energy_mlflow.py \
    --summary-json "${GPU_SUMMARY_JSON}" \
    --tracking-uri "${MLFLOW_TRACKING_URI}" \
    --experiment "${MLFLOW_JOB_ENERGY_EXPERIMENT}" \
    --run-name "job-energy-${JOB_ID}" \
    ${RUN_ID:+--run-id "${RUN_ID}"} || true
fi

python slurm/export_job_metrics_prom.py \
  --summary-json "${GPU_SUMMARY_JSON}" \
  --output-dir "${PROM_TEXTFILE_DIR}" \
  --aggregate-dir "${GPU_METRICS_DIR}"

echo "GPU summary written to: ${GPU_SUMMARY_JSON}"
exit "${TRAIN_EXIT}"
