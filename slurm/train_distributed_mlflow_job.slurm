#!/bin/bash
#SBATCH --job-name=ml-energy-ddp
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=2
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail

cd "${SLURM_SUBMIT_DIR}"
mkdir -p logs

if [ -f ".venv/bin/activate" ]; then
  source .venv/bin/activate
fi

export BATCH_SIZE="${BATCH_SIZE:-64}"
export EPOCHS="${EPOCHS:-3}"
export LR="${LR:-0.001}"
export MLFLOW_TRACKING_URI="${MLFLOW_TRACKING_URI:-http://127.0.0.1:5000}"
export MLFLOW_EXPERIMENT_NAME="${MLFLOW_EXPERIMENT_NAME:-carpk-yolo-distributed}"
export MLFLOW_RUN_NAME="${MLFLOW_RUN_NAME:-ddp-job-${SLURM_JOB_ID:-local}}"
export DIST_BACKEND="${DIST_BACKEND:-nccl}"
export YOLO_MODEL="${YOLO_MODEL:-yolov8n.pt}"
export CARPK_DATASET_DIR="${CARPK_DATASET_DIR:-${SLURM_SUBMIT_DIR}/data/carpk}"
export CARPK_INFER_DIR="${CARPK_INFER_DIR:-${CARPK_DATASET_DIR}/images/test}"
export RUN_TUNING="${RUN_TUNING:-1}"
export TRAIN_DEVICE="${TRAIN_DEVICE:-0,1}"

export GPU_SAMPLE_INTERVAL_SEC="${GPU_SAMPLE_INTERVAL_SEC:-1.0}"
export ELECTRICITY_PRICE_EUR_PER_KWH="${ELECTRICITY_PRICE_EUR_PER_KWH:-0.30}"
export GRID_CO2_KG_PER_KWH="${GRID_CO2_KG_PER_KWH:-0.4}"
export PUE_FACTOR="${PUE_FACTOR:-1.0}"
export GPU_METRICS_DIR="${GPU_METRICS_DIR:-${SLURM_SUBMIT_DIR}/energy_metrics}"
export PROM_TEXTFILE_DIR="${PROM_TEXTFILE_DIR:-${GPU_METRICS_DIR}/node_exporter}"
export MLFLOW_LOG_JOB_ENERGY="${MLFLOW_LOG_JOB_ENERGY:-1}"
export MLFLOW_JOB_ENERGY_EXPERIMENT="${MLFLOW_JOB_ENERGY_EXPERIMENT:-slurm-job-energy}"

mkdir -p "${GPU_METRICS_DIR}" "${PROM_TEXTFILE_DIR}"
JOB_ID="${SLURM_JOB_ID:-local}"
GPU_CSV="${GPU_METRICS_DIR}/gpu_metrics_job_${JOB_ID}.csv"
GPU_SUMMARY_JSON="${GPU_METRICS_DIR}/gpu_summary_job_${JOB_ID}.json"
PHASES_JSON="${GPU_METRICS_DIR}/gpu_phases_job_${JOB_ID}.json"
PHASE_SUMMARY_JSON="${GPU_METRICS_DIR}/gpu_summary_phases_job_${JOB_ID}.json"
export PIPELINE_ARTIFACTS_DIR="${GPU_METRICS_DIR}/pipeline_job_${JOB_ID}"
TIMELINE_JSONL="${PIPELINE_ARTIFACTS_DIR}/phase_timeline.jsonl"

echo "Starting distributed run on host $(hostname)"
echo "SLURM_NTASKS=${SLURM_NTASKS:-2} BATCH_SIZE=${BATCH_SIZE} EPOCHS=${EPOCHS}"

bash slurm/gpu_monitor.sh "${GPU_CSV}" "${GPU_SAMPLE_INTERVAL_SEC}" &
MONITOR_PID=$!

set +e
python train_distributed_with_energy_tracking_mlflow.py
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

python slurm/summarize_gpu_metrics_phases.py \
  --gpu-csv "${GPU_CSV}" \
  --timeline-jsonl "${TIMELINE_JSONL}" \
  --base-summary-json "${GPU_SUMMARY_JSON}" \
  --output-json "${PHASE_SUMMARY_JSON}" \
  --pue "${PUE_FACTOR}" || true

if [ "${MLFLOW_LOG_JOB_ENERGY}" = "1" ]; then
  python slurm/log_job_energy_mlflow.py \
    --summary-json "${GPU_SUMMARY_JSON}" \
    --tracking-uri "${MLFLOW_TRACKING_URI}" \
    --experiment "${MLFLOW_JOB_ENERGY_EXPERIMENT}" \
    --run-name "job-energy-${JOB_ID}" || true
fi

python slurm/export_job_metrics_prom.py \
  --summary-json "${GPU_SUMMARY_JSON}" \
  --output-dir "${PROM_TEXTFILE_DIR}" \
  --aggregate-dir "${GPU_METRICS_DIR}"

if [ -f "${PHASE_SUMMARY_JSON}" ]; then
  python slurm/export_job_phase_metrics_prom.py \
    --summary-json "${PHASE_SUMMARY_JSON}" \
    --output-dir "${PROM_TEXTFILE_DIR}" || true
fi

exit "${TRAIN_EXIT}"
