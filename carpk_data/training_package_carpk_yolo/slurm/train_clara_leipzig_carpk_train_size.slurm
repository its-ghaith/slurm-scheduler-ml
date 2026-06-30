#!/bin/bash
#SBATCH -J recv-carpk-yolo-augment
#SBATCH -p clara
#SBATCH --ntasks=1
#SBATCH --gres=gpu:v100:1
#SBATCH --mem-per-gpu=32G
#SBATCH --cpus-per-task=14
#SBATCH --time=05:00:00
#SBATCH -o /work/tz51aquc-reCV/output/logs/%x-%j-%u.log # %x=jobname, %j=slurm id, %u=user
#SBATCH -e /work/tz51aquc-reCV/output/logs/%x-%j-%u.log


# Modules
module load Python/3.12.3-GCCcore-13.3.0
module load CUDA/12.6.0

source $HOME/rotationally-invariant-cnns/.venv/bin/activate

# HPC settings
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
export PYTHONUNBUFFERED=1
export HYDRA_FULL_ERROR=1

# set paths
PROJECT="$HOME/rotationally-invariant-cnns"
OUTDIR="/work/tz51aquc-reCV/output"
RUNDIR="$OUTDIR/runs/${SLURM_JOB_NAME}-${SLURM_JOB_ID}-${SLURM_USER}"
export RUNDIR
mkdir -p "$RUNDIR"

# use fast but temporary scratch directory for data
SCRATCH="/lscratch"
if [ ! -d "$SCRATCH" ]; then
    SCRATCH="$HOME/tmp"
fi
DATADIR="${SCRATCH}/${SLURM_JOB_ID}/carpk"
mkdir -p $DATADIR
echo "Using scratch: $DATADIR"

trap 'echo "==> Cleaning scratch: $SCRATCH"; rm -rf "$DATADIR"' EXIT
rsync -a /work/tz51aquc-reCV/data/carpk/ "$DATADIR/"

# Local MLflow store + YOLO runs dir + codecarbon
export CODECARBON_OUTPUT_DIR="${RUNDIR}/energy_logs"
export MLFLOW_TRACKING_URI="${OUTDIR}/mlruns" # for local mlflow
yolo settings runs_dir="${RUNDIR}" >/dev/null 2>&1 || true

# Run training/eval (Hydra/multirun example)
cd "$PROJECT"

uv sync
srun uv run python src/effcv/run_experiment.py \
  --multirun \
  +model=yolov8 ++model.version=yolov8s.pt,yolov8m.pt,yolov8l.pt \
  +dataset=carpk ++experiment=carpk_leipzig_yolo_model_size \
  ++dataset.train_size=500 \
  ++model.batch_size=6 \
  ++dataset.path="$DATADIR" \
  ++dataset.data_yaml="$DATADIR"/data.yaml \
  ++model.epochs=100 ++model.patience=6 ++dataset.img_size=640 \

echo "Job ${SLURM_JOB_ID} finished; results in $RUNDIR"
