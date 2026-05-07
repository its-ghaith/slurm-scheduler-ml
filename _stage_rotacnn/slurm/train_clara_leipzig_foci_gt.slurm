#!/bin/bash
#SBATCH -J recv-foci-yolo-augment
#SBATCH -p clara
#SBATCH --ntasks=1
#SBATCH --gres=gpu:v100:1
#SBATCH --mem-per-gpu=32G
#SBATCH --cpus-per-task=10
#SBATCH --time=04:00:00
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

DATASET_FOLDER="uc_cells"
DATASET="foci"
DATADIR="${SCRATCH}/${SLURM_JOB_ID}/${DATASET_FOLDER}"
mkdir -p $DATADIR
echo "Using scratch: $DATADIR"

trap 'echo "==> Cleaning scratch: $SCRATCH"; rm -rf "$DATADIR"' EXIT
rsync -a /work/tz51aquc-reCV/data/${DATASET_FOLDER}/raw "$DATADIR/"
cp /work/tz51aquc-reCV/data/${DATASET_FOLDER}/data.yaml "$DATADIR/"

# Local MLflow store + YOLO runs dir + codecarbon
export CODECARBON_OUTPUT_DIR="${RUNDIR}/energy_logs"
export MLFLOW_TRACKING_URI="${OUTDIR}/mlruns" # for local mlflow
yolo settings runs_dir="${RUNDIR}" >/dev/null 2>&1 || true

# Run training/eval (Hydra/multirun example)
cd "$PROJECT"

srun uv run python src/countcv/effcv/run_experiment.py \
  --multirun \
  +dataset="$DATASET" ++experiment=foci_leipzig_yolo_train_size \
  +model=yolov11 \
  ++dataset.train_size=50,100,150,159 ++dataset.img_size=660 \
  ++dataset.path="$DATADIR" \
  ++dataset.data_yaml="$DATADIR"/data.yaml \
  ++model.epochs=100 ++model.patience=20 \
  ++model.batch_size=8 ++model.flipud=0.3 ++model.fliplr=0.3 \
  ++model.hsv_s=0.4 ++model.translate=0.0 ++model.degrees=15

srun uv run python src/countcv/effcv/run_experiment.py \
  --multirun \
  +model=cnn_base ++model.version=fcrn-base,SAUnet +dataset=foci ++experiment=tagderforschung \
  ++dataset.train_size=false \
  ++dataset.img_size=220 \
  ++remote_mlflow=False \
  ++model.epochs=300 \
  ++model.sigma=2 \
  ++model.alpha=100 \
  ++model.tilesize=224 \
  ++model.lambda_count=0.001 \
  ++model.batch_size=32 \
  ++dataset.path="$DATADIR" \
  ++dataset.data_yaml="$DATADIR"/data.yaml 


echo "Job ${SLURM_JOB_ID} finished; results in $RUNDIR"
