#!/bin/bash
#SBATCH --job-name=controller-benchmark
#SBATCH --partition=debug
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --chdir=/workspace
#SBATCH --output=/workspace/logs/controller-benchmark-%j.out
#SBATCH --error=/workspace/logs/controller-benchmark-%j.err

set -euo pipefail

BENCHMARK_WORKSPACE="${BENCHMARK_WORKSPACE:-/workspace}"
PERSISTENT_PYTHON_DEPS="${PERSISTENT_PYTHON_DEPS:-/workspace-cache/controller-python-deps}"
export PYTHONPATH="${BENCHMARK_WORKSPACE}:${PERSISTENT_PYTHON_DEPS}:${PYTHONPATH:-}"
export MLFLOW_TRACKING_URI="${MLFLOW_TRACKING_URI:-http://mlflow:5000}"
export RUNTIME_IMAGE_ID="${RUNTIME_IMAGE_ID:-unknown}"
export ELECTRICITY_PRICE_EUR_PER_KWH="${ELECTRICITY_PRICE_EUR_PER_KWH:-0.30}"
export GRID_CO2_KG_PER_KWH="${GRID_CO2_KG_PER_KWH:-0.40}"
export PUE_FACTOR="${PUE_FACTOR:-1.0}"
ENERGY_DIR="${BENCHMARK_METRICS_DIR:-/workspace/energy_metrics}"
PROM_DIR="${ENERGY_DIR}/node_exporter"
JOB_ID="${SLURM_JOB_ID}"
mkdir -p "${ENERGY_DIR}" "${PROM_DIR}" /workspace/logs

cd "${BENCHMARK_WORKSPACE}"

case "${BENCHMARK_RUNNER}" in
  vedai_yolo)
    python -m controller_benchmark.runners.vedai_detection \
      --run-spec-base64 "${RUN_SPEC_BASE64}" \
      --output-dir "${ENERGY_DIR}"
    ;;
  carpk_task_generalisation)
    python -m controller_benchmark.runners.carpk_task_generalisation \
      --run-spec-base64 "${RUN_SPEC_BASE64}" \
      --output-dir "${ENERGY_DIR}"
    ;;
  visdrone_yolo)
    python -m controller_benchmark.runners.visdrone_detection \
      --run-spec-base64 "${RUN_SPEC_BASE64}" \
      --output-dir "${ENERGY_DIR}"
    ;;
  standard_image_classification)
    python -m controller_benchmark.runners.standard_image_classification \
      --run-spec-base64 "${RUN_SPEC_BASE64}" \
      --output-dir "${ENERGY_DIR}"
    ;;
  standard_semantic_segmentation)
    python -m controller_benchmark.runners.standard_semantic_segmentation \
      --run-spec-base64 "${RUN_SPEC_BASE64}" \
      --output-dir "${ENERGY_DIR}"
    ;;
  cross_domain_task)
    python -m controller_benchmark.runners.cross_domain_tasks \
      --run-spec-base64 "${RUN_SPEC_BASE64}" \
      --output-dir "${ENERGY_DIR}"
    ;;
  *)
    echo "Unsupported generic benchmark runner: ${BENCHMARK_RUNNER}" >&2
    exit 2
    ;;
esac

python "${BENCHMARK_WORKSPACE}/slurm/export_job_metrics_prom.py" \
  --summary-json "${ENERGY_DIR}/gpu_summary_job_${JOB_ID}.json" \
  --output-dir "${PROM_DIR}" \
  --aggregate-dir "${ENERGY_DIR}"
python "${BENCHMARK_WORKSPACE}/slurm/export_job_epoch_metrics_prom.py" \
  --summary-json "${ENERGY_DIR}/epoch_summary_job_${JOB_ID}.json" \
  --output-prom "${PROM_DIR}/job_${JOB_ID}_epochs.prom"
python "${BENCHMARK_WORKSPACE}/slurm/export_job_phase_metrics_prom.py" \
  --summary-json "${ENERGY_DIR}/gpu_summary_job_${JOB_ID}.json" \
  --output-prom "${PROM_DIR}/job_${JOB_ID}_phases.prom"

echo "Generic benchmark job ${JOB_ID} completed."
