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
export EARLY_STOP_CONTROLLER_MODE="${EARLY_STOP_CONTROLLER_MODE:-none}"
export COMPARISON_STRATEGY="${COMPARISON_STRATEGY:-unspecified}"
export ENERGY_ADAPTIVE_MONITOR_METRIC="${ENERGY_ADAPTIVE_MONITOR_METRIC:-map50}"
export ENERGY_ADAPTIVE_MIN_EPOCHS="${ENERGY_ADAPTIVE_MIN_EPOCHS:-20}"
export ENERGY_ADAPTIVE_PATIENCE="${ENERGY_ADAPTIVE_PATIENCE:-3}"
export ENERGY_ADAPTIVE_SMOOTHING_WINDOW="${ENERGY_ADAPTIVE_SMOOTHING_WINDOW:-3}"
export ENERGY_ADAPTIVE_MIN_DELTA_MAP50="${ENERGY_ADAPTIVE_MIN_DELTA_MAP50:-0.001}"
export ENERGY_ADAPTIVE_MIN_MAPE_MAP50_PER_WH="${ENERGY_ADAPTIVE_MIN_MAPE_MAP50_PER_WH:-0.0001}"
export UNCERTAINTY_TARGET_EPOCH="${UNCERTAINTY_TARGET_EPOCH:-100}"
export UNCERTAINTY_EPSILON="${UNCERTAINTY_EPSILON:-0.01}"
export UNCERTAINTY_ALPHA="${UNCERTAINTY_ALPHA:-0.05}"
export UNCERTAINTY_BOOTSTRAP_SAMPLES="${UNCERTAINTY_BOOTSTRAP_SAMPLES:-24}"
export UNCERTAINTY_MIN_FIT_POINTS="${UNCERTAINTY_MIN_FIT_POINTS:-8}"
export STANDARD_EARLY_STOP_MIN_DELTA="${STANDARD_EARLY_STOP_MIN_DELTA:-0.0005}"
export UNCERTAINTY_MIN_QUALITY="${UNCERTAINTY_MIN_QUALITY:-0.55}"
export UNCERTAINTY_REQUIRE_LOW_MAPE="${UNCERTAINTY_REQUIRE_LOW_MAPE:-true}"
export CONFORMAL_CALIBRATION_PATH="${CONFORMAL_CALIBRATION_PATH:-/workspace/slurm/conformal_calibration.json}"
export CONFORMAL_REGRET_TOLERANCE="${CONFORMAL_REGRET_TOLERANCE:-0.015}"
export CONFORMAL_FUTURE_EFFICIENCY_THRESHOLD="${CONFORMAL_FUTURE_EFFICIENCY_THRESHOLD:-0.05}"
export CONFORMAL_MAX_INTERVAL_WIDTH="${CONFORMAL_MAX_INTERVAL_WIDTH:-0.08}"
export CONFORMAL_ENERGY_WINDOW="${CONFORMAL_ENERGY_WINDOW:-5}"
export CONFORMAL_REQUIRE_CALIBRATION="${CONFORMAL_REQUIRE_CALIBRATION:-true}"
export CONTROLLER_EVALUATION_INTERVAL="${CONTROLLER_EVALUATION_INTERVAL:-1}"
export HYBRID_QUALITY_TARGET="${HYBRID_QUALITY_TARGET:-0.68}"
export HYBRID_PLATEAU_WINDOW="${HYBRID_PLATEAU_WINDOW:-12}"
export HYBRID_PLATEAU_SLOPE_THRESHOLD="${HYBRID_PLATEAU_SLOPE_THRESHOLD:-0.0015}"
export HYBRID_THRESHOLD_GROWTH="${HYBRID_THRESHOLD_GROWTH:-0.25}"
export HYBRID_CANDIDATE_DECAY="${HYBRID_CANDIDATE_DECAY:-0.5}"
export HYBRID_LATE_EPOCH_FRACTION="${HYBRID_LATE_EPOCH_FRACTION:-0.90}"
export HYBRID_LATE_PATIENCE="${HYBRID_LATE_PATIENCE:-2}"
export HYBRID_MAX_EPOCH_BUDGET="${HYBRID_MAX_EPOCH_BUDGET:-95}"
export HYBRID_MAX_NET_ENERGY_WH="${HYBRID_MAX_NET_ENERGY_WH:-0}"
export HYBRID_REGRET_WEIGHT="${HYBRID_REGRET_WEIGHT:-1.0}"
export HYBRID_ENERGY_WEIGHT="${HYBRID_ENERGY_WEIGHT:-0.05}"
export ENERGY_GUARD_TARGET_SAVING_FRACTION="${ENERGY_GUARD_TARGET_SAVING_FRACTION:-0.20}"
export ENERGY_GUARD_SAFETY_MARGIN_FRACTION="${ENERGY_GUARD_SAFETY_MARGIN_FRACTION:-0.01}"
export ENERGY_GUARD_POST_TRAINING_RESERVE_WH="${ENERGY_GUARD_POST_TRAINING_RESERVE_WH:-0.90}"
export ENERGY_GUARD_WINDOW="${ENERGY_GUARD_WINDOW:-5}"
export ENERGY_GUARD_FALLBACK_EPOCH_FRACTION="${ENERGY_GUARD_FALLBACK_EPOCH_FRACTION:-0.77}"
export ENERGY_GUARD_STRICT="${ENERGY_GUARD_STRICT:-true}"
export CONTROLLER_PLUGIN="${CONTROLLER_PLUGIN:-}"
export CONTROLLER_PARAMETERS_JSON="${CONTROLLER_PARAMETERS_JSON:-{}}"
export CONTROLLER_PARAMETERS_BASE64="${CONTROLLER_PARAMETERS_BASE64:-}"
export CONTROLLER_ID="${CONTROLLER_ID:-unspecified}"
export BENCHMARK_VERSION="${BENCHMARK_VERSION:-unversioned}"
export BENCHMARK_RUN_ID="${BENCHMARK_RUN_ID:-none}"
export BENCHMARK_CASE_ID="${BENCHMARK_CASE_ID:-none}"
export BENCHMARK_STAGE="${BENCHMARK_STAGE:-none}"
export BENCHMARK_TASK_TYPE="${BENCHMARK_TASK_TYPE:-object_detection}"
export BENCHMARK_QUALITY_METRIC="${BENCHMARK_QUALITY_METRIC:-map50_95}"
export EXPERIMENT_SCENARIO="${EXPERIMENT_SCENARIO:-unspecified}"
export TRAINING_SEED="${TRAINING_SEED:-0}"
export SPLIT_SEED="${SPLIT_SEED:-42}"
export CACHE_POLICY="${CACHE_POLICY:-ram}"
export GPU_IDLE_SAMPLE_SECONDS="${GPU_IDLE_SAMPLE_SECONDS:-10}"
export RUNTIME_IMAGE_ID="${RUNTIME_IMAGE_ID:-unknown}"

CONTROLLER_PARAMETERS_ARGS=(--controller-parameters-json "${CONTROLLER_PARAMETERS_JSON}")
if [[ -n "${CONTROLLER_PARAMETERS_BASE64}" ]]; then
  CONTROLLER_PARAMETERS_ARGS=(--controller-parameters-base64 "${CONTROLLER_PARAMETERS_BASE64}")
fi

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
RUN_METADATA_JSON="${GPU_METRICS_DIR}/run_metadata_job_${JOB_ID}.json"
export RUN_METADATA_JSON

export MLFLOW_EXPERIMENT_NAME="${MLFLOW_JOB_ENERGY_EXPERIMENT}"
export MLFLOW_RUN_NAME="job-energy-${JOB_ID}"
export MLFLOW_RUN_ID_FILE

IDLE_POWER_W="$(python slurm/capture_run_metadata.py \
  --output "${RUN_METADATA_JSON}" \
  --idle-sample-seconds "${GPU_IDLE_SAMPLE_SECONDS}" \
  --print-idle-power)"
export GPU_IDLE_POWER_W="${IDLE_POWER_W:-0}"
echo "Measured idle GPU power: ${GPU_IDLE_POWER_W} W"

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
  --idle-power-w "${GPU_IDLE_POWER_W}" \
  --controller-mode "${EARLY_STOP_CONTROLLER_MODE}" \
  --comparison-strategy "${COMPARISON_STRATEGY}" \
  --scenario "${EXPERIMENT_SCENARIO}" \
  --training-seed "${TRAINING_SEED}" \
  --split-seed "${SPLIT_SEED}" \
  --cache-policy "${CACHE_POLICY}" \
  --adaptive-monitor-metric "${ENERGY_ADAPTIVE_MONITOR_METRIC}" \
  --adaptive-min-epochs "${ENERGY_ADAPTIVE_MIN_EPOCHS}" \
  --adaptive-patience "${ENERGY_ADAPTIVE_PATIENCE}" \
  --adaptive-smoothing-window "${ENERGY_ADAPTIVE_SMOOTHING_WINDOW}" \
  --adaptive-min-delta-map50 "${ENERGY_ADAPTIVE_MIN_DELTA_MAP50}" \
  --adaptive-min-mape-map50-per-wh "${ENERGY_ADAPTIVE_MIN_MAPE_MAP50_PER_WH}" \
  --uncertainty-target-epoch "${UNCERTAINTY_TARGET_EPOCH}" \
  --uncertainty-epsilon "${UNCERTAINTY_EPSILON}" \
  --uncertainty-alpha "${UNCERTAINTY_ALPHA}" \
  --uncertainty-bootstrap-samples "${UNCERTAINTY_BOOTSTRAP_SAMPLES}" \
  --uncertainty-min-fit-points "${UNCERTAINTY_MIN_FIT_POINTS}" \
  --standard-min-delta "${STANDARD_EARLY_STOP_MIN_DELTA}" \
  --uncertainty-min-quality "${UNCERTAINTY_MIN_QUALITY}" \
  --uncertainty-require-low-mape "${UNCERTAINTY_REQUIRE_LOW_MAPE}" \
  --conformal-calibration-path "${CONFORMAL_CALIBRATION_PATH}" \
  --conformal-regret-tolerance "${CONFORMAL_REGRET_TOLERANCE}" \
  --conformal-future-efficiency-threshold "${CONFORMAL_FUTURE_EFFICIENCY_THRESHOLD}" \
  --conformal-max-interval-width "${CONFORMAL_MAX_INTERVAL_WIDTH}" \
  --conformal-energy-window "${CONFORMAL_ENERGY_WINDOW}" \
  --conformal-require-calibration "${CONFORMAL_REQUIRE_CALIBRATION}" \
  --controller-evaluation-interval "${CONTROLLER_EVALUATION_INTERVAL}" \
  --hybrid-quality-target "${HYBRID_QUALITY_TARGET}" \
  --hybrid-plateau-window "${HYBRID_PLATEAU_WINDOW}" \
  --hybrid-plateau-slope-threshold "${HYBRID_PLATEAU_SLOPE_THRESHOLD}" \
  --hybrid-threshold-growth "${HYBRID_THRESHOLD_GROWTH}" \
  --hybrid-candidate-decay "${HYBRID_CANDIDATE_DECAY}" \
  --hybrid-late-epoch-fraction "${HYBRID_LATE_EPOCH_FRACTION}" \
  --hybrid-late-patience "${HYBRID_LATE_PATIENCE}" \
  --hybrid-max-epoch-budget "${HYBRID_MAX_EPOCH_BUDGET}" \
  --hybrid-max-net-energy-wh "${HYBRID_MAX_NET_ENERGY_WH}" \
  --hybrid-regret-weight "${HYBRID_REGRET_WEIGHT}" \
  --hybrid-energy-weight "${HYBRID_ENERGY_WEIGHT}" \
  --energy-guard-target-saving-fraction "${ENERGY_GUARD_TARGET_SAVING_FRACTION}" \
  --energy-guard-safety-margin-fraction "${ENERGY_GUARD_SAFETY_MARGIN_FRACTION}" \
  --energy-guard-post-training-reserve-wh "${ENERGY_GUARD_POST_TRAINING_RESERVE_WH}" \
  --energy-guard-window "${ENERGY_GUARD_WINDOW}" \
  --energy-guard-fallback-epoch-fraction "${ENERGY_GUARD_FALLBACK_EPOCH_FRACTION}" \
  --energy-guard-strict "${ENERGY_GUARD_STRICT}" \
  --controller-plugin "${CONTROLLER_PLUGIN}" \
  "${CONTROLLER_PARAMETERS_ARGS[@]}" \
  --controller-id "${CONTROLLER_ID}" \
  --benchmark-version "${BENCHMARK_VERSION}" \
  --benchmark-run-id "${BENCHMARK_RUN_ID}" \
  --benchmark-case-id "${BENCHMARK_CASE_ID}" \
  --benchmark-stage "${BENCHMARK_STAGE}" \
  --task-type "${BENCHMARK_TASK_TYPE}" \
  --quality-metric "${BENCHMARK_QUALITY_METRIC}" \
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
  --pue "${PUE_FACTOR}" \
  --idle-power-w "${GPU_IDLE_POWER_W}" \
  --metadata-json "${RUN_METADATA_JSON}"

python slurm/summarize_gpu_metrics_phases.py \
  --gpu-csv "${GPU_CSV}" \
  --timeline-jsonl "${TIMELINE_JSONL}" \
  --base-summary-json "${GPU_SUMMARY_JSON}" \
  --output-json "${GPU_SUMMARY_PHASES_JSON}" \
  --job-codecarbon-json "${JOB_CC_JSON}" \
  --price-eur-kwh "${ELECTRICITY_PRICE_EUR_PER_KWH}" \
  --co2-kg-kwh "${GRID_CO2_KG_PER_KWH}" \
  --pue "${PUE_FACTOR}" \
  --idle-power-w "${GPU_IDLE_POWER_W}"

# The orchestrator and all generic runners use the unsuffixed summary as the
# canonical, fully enriched artifact. Preserve the legacy phase-suffixed copy,
# then publish the same complete document at the canonical path.
cp "${GPU_SUMMARY_PHASES_JSON}" "${GPU_SUMMARY_JSON}"

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

python slurm/export_study_metrics_prom.py \
  --metrics-dir "${GPU_METRICS_DIR}" \
  --output-prom "${PROM_TEXTFILE_DIR}/study.prom"

echo "Done: ${GPU_SUMMARY_PHASES_JSON}"
exit "${TRAIN_EXIT}"
