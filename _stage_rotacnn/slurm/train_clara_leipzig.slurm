#!/bin/bash
#SBATCH -J recv-test-train
#SBATCH -p clara
#SBATCH --ntasks=1
#SBATCH --gres=gpu:v100:1
#SBATCH --cpus-per-task=14
#SBATCH --time=00:30:00
#SBATCH -o /work/tz51aquc-reCV/output/logs/%x-%j-%u.log # %x=jobname, %j=slurm id, %u=user
#SBATCH -e /work/tz51aquc-reCV/output/logs/%x-%j-%u.log


# Modules
module load Python/3.12.3-GCCcore-13.3.0
module load CUDA/12.6.0

source $HOME/rotationally-invariant-cnns/.venv/bin/activate

# HPC settings (check if this is automatically used in our code?)
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
DATADIR="${SCRATCH}/${SLURM_JOB_ID}/uc_cells"
mkdir -p $DATADIR
echo "Using scratch: $DATADIR"

trap 'echo "==> Cleaning scratch: $SCRATCH"; rm -rf "$DATADIR"' EXIT
rsync -a /work/tz51aquc-reCV/data/uc_cells/ "$DATADIR/"

# Local MLflow store + YOLO runs dir + codecarbon
export CODECARBON_OUTPUT_DIR="${RUNDIR}/energy_logs"
export MLFLOW_TRACKING_URI="${OUTDIR}/mlruns" # for local mlflow
yolo settings runs_dir="${RUNDIR}" >/dev/null 2>&1 || true

# Run training/eval (Hydra/multirun example)
cd "$PROJECT"
srun uv run python src/countcv/effcv/run_experiment.py \
  --multirun \
  +model=yolov8 +dataset=foci ++experiment=test_hpc \
  ++model.batch_size=4 ++dataset.train_size=50 \
  ++dataset.path="$DATADIR" \
  ++dataset.data_yaml="$DATADIR"/data.yaml \
  ++remote_mlflow=False \
  ++model.epochs=2

echo "Job ${SLURM_JOB_ID} finished; results in $RUNDIR"
