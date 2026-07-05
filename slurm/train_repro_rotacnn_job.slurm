#!/bin/bash
#SBATCH --job-name=rotacnn-repro-energy
#SBATCH --partition=debug
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00
#SBATCH --chdir=/workspace
#SBATCH --output=/workspace/logs/%x-%j.out
#SBATCH --error=/workspace/logs/%x-%j.err

set -euo pipefail

cd "${SLURM_SUBMIT_DIR}"

export MLFLOW_TRACKING_URI="${MLFLOW_TRACKING_URI:-http://mlflow:5000}"
export GPU_SAMPLE_INTERVAL_SEC="${GPU_SAMPLE_INTERVAL_SEC:-0.5}"
export ELECTRICITY_PRICE_EUR_PER_KWH="${ELECTRICITY_PRICE_EUR_PER_KWH:-0.30}"
export GRID_CO2_KG_PER_KWH="${GRID_CO2_KG_PER_KWH:-0.4}"
export PUE_FACTOR="${PUE_FACTOR:-1.0}"
export GPU_METRICS_DIR="${GPU_METRICS_DIR:-/workspace/energy_metrics}"
export PROM_TEXTFILE_DIR="${PROM_TEXTFILE_DIR:-${GPU_METRICS_DIR}/node_exporter}"
export MLFLOW_LOG_JOB_ENERGY="${MLFLOW_LOG_JOB_ENERGY:-1}"
export MLFLOW_JOB_ENERGY_EXPERIMENT="${MLFLOW_JOB_ENERGY_EXPERIMENT:-ml-energy-poc}"

export ROTA_REPO_ROOT="${ROTA_REPO_ROOT:-/workspace/rotationally-invariant-cnns}"
export REPRO_SCRIPT="${REPRO_SCRIPT:-/workspace/slurm/train_repro_phase_tracked.py}"
export REPRO_DATASET="${REPRO_DATASET:-carpk}"
export REPRO_MODEL="${REPRO_MODEL:-yolov8}"
# format: key=value;key2=value2
export REPRO_OVERRIDES="${REPRO_OVERRIDES:-}"
export ENERGY_ADAPTIVE_ENABLED="${ENERGY_ADAPTIVE_ENABLED:-0}"
export ENERGY_ADAPTIVE_MIN_EPOCHS="${ENERGY_ADAPTIVE_MIN_EPOCHS:-20}"
export ENERGY_ADAPTIVE_PATIENCE="${ENERGY_ADAPTIVE_PATIENCE:-3}"
export ENERGY_ADAPTIVE_SMOOTHING_WINDOW="${ENERGY_ADAPTIVE_SMOOTHING_WINDOW:-3}"
export ENERGY_ADAPTIVE_MIN_DELTA_MAP50="${ENERGY_ADAPTIVE_MIN_DELTA_MAP50:-0.001}"
export ENERGY_ADAPTIVE_MIN_MAPE_MAP50_PER_WH="${ENERGY_ADAPTIVE_MIN_MAPE_MAP50_PER_WH:-0.0001}"

mkdir -p "${GPU_METRICS_DIR}" "${PROM_TEXTFILE_DIR}" /workspace/logs

JOB_ID="${SLURM_JOB_ID:-local}"
GPU_CSV="${GPU_METRICS_DIR}/gpu_metrics_job_${JOB_ID}.csv"
GPU_SUMMARY_JSON="${GPU_METRICS_DIR}/gpu_summary_job_${JOB_ID}.json"
GPU_SUMMARY_PHASES_JSON="${GPU_METRICS_DIR}/gpu_summary_job_${JOB_ID}_phases.json"
TIMELINE_JSONL="${GPU_METRICS_DIR}/phase_timeline_job_${JOB_ID}.jsonl"
EPOCH_TIMELINE_JSONL="${GPU_METRICS_DIR}/epoch_timeline_job_${JOB_ID}.jsonl"
EPOCH_SUMMARY_JSON="${GPU_METRICS_DIR}/epoch_summary_job_${JOB_ID}.json"
PHASE_CC_DIR="${GPU_METRICS_DIR}/codecarbon_job_${JOB_ID}"
JOB_CC_JSON="${GPU_METRICS_DIR}/codecarbon_job_total_job_${JOB_ID}.json"
MLFLOW_RUN_ID_FILE="${GPU_METRICS_DIR}/mlflow_run_id_job_${JOB_ID}.txt"
PHASE_PROM="${PROM_TEXTFILE_DIR}/job_${JOB_ID}_phases.prom"
EPOCH_PROM="${PROM_TEXTFILE_DIR}/job_${JOB_ID}_epochs.prom"

export MLFLOW_EXPERIMENT_NAME="${MLFLOW_JOB_ENERGY_EXPERIMENT}"
export MLFLOW_RUN_NAME="job-energy-${JOB_ID}"
export MLFLOW_RUN_ID_FILE

bash slurm/gpu_monitor.sh "${GPU_CSV}" "${GPU_SAMPLE_INTERVAL_SEC}" &
MONITOR_PID=$!

set +e
OVR_ARR=()
if [ -n "${REPRO_OVERRIDES}" ]; then
  IFS=';' read -r -a OVR_RAW <<< "${REPRO_OVERRIDES}"
  for kv in "${OVR_RAW[@]}"; do
    [ -n "$kv" ] && OVR_ARR+=(--override "$kv")
  done
fi

ADAPTIVE_FLAG=()
if [ "${ENERGY_ADAPTIVE_ENABLED}" = "1" ] || [ "${ENERGY_ADAPTIVE_ENABLED,,}" = "true" ] || [ "${ENERGY_ADAPTIVE_ENABLED,,}" = "yes" ]; then
  ADAPTIVE_FLAG+=(--adaptive-enabled)
fi

python "${REPRO_SCRIPT}" \
  --repo-root "${ROTA_REPO_ROOT}" \
  --dataset "${REPRO_DATASET}" \
  --model "${REPRO_MODEL}" \
  --timeline-path "${TIMELINE_JSONL}" \
  --phase-metrics-dir "${PHASE_CC_DIR}" \
  --job-metrics-path "${JOB_CC_JSON}" \
  --gpu-csv "${GPU_CSV}" \
  --epoch-timeline-path "${EPOCH_TIMELINE_JSONL}" \
  --epoch-summary-path "${EPOCH_SUMMARY_JSON}" \
  --price-eur-kwh "${ELECTRICITY_PRICE_EUR_PER_KWH}" \
  --co2-kg-kwh "${GRID_CO2_KG_PER_KWH}" \
  --pue-factor "${PUE_FACTOR}" \
  --adaptive-min-epochs "${ENERGY_ADAPTIVE_MIN_EPOCHS}" \
  --adaptive-patience "${ENERGY_ADAPTIVE_PATIENCE}" \
  --adaptive-smoothing-window "${ENERGY_ADAPTIVE_SMOOTHING_WINDOW}" \
  --adaptive-min-delta-map50 "${ENERGY_ADAPTIVE_MIN_DELTA_MAP50}" \
  --adaptive-min-mape-map50-per-wh "${ENERGY_ADAPTIVE_MIN_MAPE_MAP50_PER_WH}" \
  "${ADAPTIVE_FLAG[@]}" \
  "${OVR_ARR[@]}"
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
  --output-json "${GPU_SUMMARY_PHASES_JSON}" \
  --job-codecarbon-json "${JOB_CC_JSON}" \
  --price-eur-kwh "${ELECTRICITY_PRICE_EUR_PER_KWH}" \
  --co2-kg-kwh "${GRID_CO2_KG_PER_KWH}" \
  --pue "${PUE_FACTOR}"

python slurm/summarize_gpu_metrics_epochs.py \
  --epoch-timeline-jsonl "${EPOCH_TIMELINE_JSONL}" \
  --output-json "${EPOCH_SUMMARY_JSON}" \
  --job-id "${JOB_ID}" \
  --price-eur-kwh "${ELECTRICITY_PRICE_EUR_PER_KWH}" \
  --co2-kg-kwh "${GRID_CO2_KG_PER_KWH}" \
  "${ADAPTIVE_FLAG[@]}"

if [ "${MLFLOW_LOG_JOB_ENERGY}" = "1" ] || [ "${MLFLOW_LOG_JOB_ENERGY,,}" = "true" ] || [ "${MLFLOW_LOG_JOB_ENERGY,,}" = "yes" ]; then
  RUN_ID=""
  if [ -f "${MLFLOW_RUN_ID_FILE}" ]; then
    RUN_ID="$(cat "${MLFLOW_RUN_ID_FILE}" 2>/dev/null || true)"
  fi

  python slurm/log_job_energy_mlflow.py \
    --summary-json "${GPU_SUMMARY_PHASES_JSON}" \
    --tracking-uri "${MLFLOW_TRACKING_URI}" \
    --experiment "${MLFLOW_JOB_ENERGY_EXPERIMENT}" \
    --run-name "job-energy-${JOB_ID}" \
    ${RUN_ID:+--run-id "${RUN_ID}"} || true
fi

python slurm/export_job_metrics_prom.py \
  --summary-json "${GPU_SUMMARY_PHASES_JSON}" \
  --output-dir "${PROM_TEXTFILE_DIR}" \
  --aggregate-dir "${GPU_METRICS_DIR}"

python slurm/export_job_phase_metrics_prom.py \
  --summary-json "${GPU_SUMMARY_PHASES_JSON}" \
  --output-prom "${PHASE_PROM}"

python slurm/export_job_epoch_metrics_prom.py \
  --summary-json "${EPOCH_SUMMARY_JSON}" \
  --output-prom "${EPOCH_PROM}"

echo "Done: ${GPU_SUMMARY_PHASES_JSON}"
exit "${TRAIN_EXIT}"
