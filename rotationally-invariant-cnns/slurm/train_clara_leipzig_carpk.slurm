#!/bin/bash
#SBATCH -J recv-carpk-yolo-performancetests
#SBATCH -p clara
#SBATCH --ntasks=1
#SBATCH --gres=gpu:v100:1
#SBATCH --mem-per-gpu=32G
#SBATCH --cpus-per-task=14
#SBATCH --time=16:00:00
#SBATCH -o /work/tz51aquc-reCV/output/logs/%x-%j-%u.log
#SBATCH -e /work/tz51aquc-reCV/output/logs/%x-%j-%u.log

module load Python/3.12.3-GCCcore-13.3.0
module load CUDA/12.6.0

source "$HOME/rotationally-invariant-cnns/.venv/bin/activate"

export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export MKL_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export PYTHONUNBUFFERED=1
export HYDRA_FULL_ERROR=1

PROJECT="$HOME/rotationally-invariant-cnns"
OUTDIR="/work/tz51aquc-reCV/output"
RUNDIR="$OUTDIR/runs/${SLURM_JOB_NAME}-${SLURM_JOB_ID}-${SLURM_USER}"
export RUNDIR
mkdir -p "$RUNDIR"

SCRATCH="/lscratch"
if [ ! -d "$SCRATCH" ]; then
    SCRATCH="$HOME/tmp"
fi
DATADIR="${SCRATCH}/${SLURM_JOB_ID}/carpk"
mkdir -p "$DATADIR"
echo "Using scratch: $DATADIR"

trap 'echo "==> Cleaning scratch: $SCRATCH"; rm -rf "$DATADIR"' EXIT
rsync -a /work/tz51aquc-reCV/data/carpk/ "$DATADIR/"

export CODECARBON_OUTPUT_DIR="$RUNDIR/energy_logs"
export MLFLOW_TRACKING_URI="$OUTDIR/mlruns"
yolo settings runs_dir="$RUNDIR" >/dev/null 2>&1 || true

cd "$PROJECT"
uv sync

srun uv run python src/countcv/effcv/run_experiment.py --multirun \
  +model=yolov8 ++model.version=yolov8n.pt,yolov8s.pt +dataset=carpk ++experiment=carpk_hpc_modelsize \
  ++dataset.img_size=[1280,720] \
  ++remote_mlflow=False \
  ++model.epochs=100 \
  ++model.patience=8 \
  ++dataset.train_size=500 \
  ++model.batch_size=8 \
  ++dataset.path="$DATADIR" \
  ++dataset.data_yaml="$DATADIR/data.yaml"

echo "Job ${SLURM_JOB_ID} finished; results in $RUNDIR"
