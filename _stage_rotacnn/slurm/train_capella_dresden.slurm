#!/bin/bash
#SBATCH -J recv-ultralytics-train-size
#SBATCH --partition=capella
#SBATCH --cpus-per-task=14 # maximum cpus allowed
#SBATCH --gpus=1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=06:00:00
#SBATCH -o /data/horse/ws/imvo807h-re_cv/output/logs/%x-%j-%u.log # %x=jobname, %j=slurm id, %u=user
#SBATCH -e /data/horse/ws/imvo807h-re_cv/output/logs/%x-%j-%u.log

SHARED_HORSE=/data/horse/ws/imvo807h-re_cv
# Modules
module load release/24.10  GCCcore/13.3.0 Python/3.12.3
module load CUDA/12.8.0

source $SHARED_HORSE/.venv/bin/activate

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
export PYTHONUNBUFFERED=1
export HYDRA_FULL_ERROR=1

# set paths
PROJECT="$HOME/rotationally-invariant-cnns"
OUTDIR="$SHARED_HORSE/output"
RUNDIR="$OUTDIR/runs/${SLURM_JOB_NAME}-${SLURM_JOB_ID}-${SLURM_USER}"
export RUNDIR

DATADIR="$SHARED_HORSE/data/carpk"

# Local MLflow store + YOLO runs dir + codecarbon
export CODECARBON_OUTPUT_DIR="${RUNDIR}/energy_logs"
export MLFLOW_TRACKING_URI="${OUTDIR}/mlruns" # for local mlflow
yolo settings runs_dir="${RUNDIR}" >/dev/null 2>&1 || true

# Run training/eval (Hydra/multirun example)
cd "$PROJECT"
export PYTHONPATH="$PROJECT:$PYTHONPATH"

# check python path and project files
python -c "import sys,inspect,countcv; print('venv:', sys.executable); print('countcv @', inspect.getfile(countcv))"

uv sync --active

srun uv run --active python src/countcv/effcv/run_experiment.py \
  --multirun \
  +model=yolov8 +dataset=carpk ++experiment=carpk_dresden_yolo_model_size \
  ++dataset.train_size=500 \
  ++model.batch_size=6 \
  ++dataset.path="$DATADIR" \
  ++dataset.data_yaml="$DATADIR"/data.yaml \
  ++model.epochs=100 ++model.patience=6 ++dataset.img_size=640 \

# srun uv run python src/countcv/effcv/run_experiment.py \
#   --multirun \
#   +model=cnn_base model.version=fcrn-base,SAUnet +dataset=synthcells ++experiment=train_size \
#   ++dataset.train_size=50,100,128 \
#   ++dataset.path="$DATADIR" \
#   ++model.epochs=100

DATADIR="$SHARED_HORSE/data/uc_cells"
srun uv run --active python src/countcv/effcv/run_experiment.py \
  --multirun \
  +dataset="foci" ++experiment=foci_dresden_yolo_model_size \
  +model=yolov8 ++model.version=yolov8s.pt,yolov8m.pt,yolov8l.pt \
  ++dataset.train_size=159 ++dataset.img_size=660 \
  ++dataset.path="$DATADIR" \
  ++dataset.data_yaml="$DATADIR"/data.yaml \
  ++model.epochs=100 ++model.patience=20 \
  ++model.batch_size=8 ++model.flipud=0.3 ++model.fliplr=0.3 \
  ++model.hsv_s=0.4 ++model.translate=0.0 ++model.degrees=15

DATADIR="$SHARED_HORSE/data/synthetic_cells"
srun uv run --active python src/countcv/effcv/run_experiment.py \
  --multirun \
  +dataset=synthcells ++experiment=model_size \
  ++num_worker=14 +model=yolov8 ++model.version=yolov8s.pt,yolov8m.pt,yolov8l.pt ++experiment=synth_dresden_train_size \
  ++dataset.train_size=128 \
  ++dataset.path="$DATADIR" \
  ++dataset.data_yaml="$DATADIR"/data.yaml \
    ++model.epochs=100 ++model.patience=8 \
  ++model.batch_size=8 \
  ++model.degrees=20 ++model.hsv_s=0.7 \
  ++model.mosaic=0.5 ++dataset.img_size=256

echo "Job ${SLURM_JOB_ID} finished; results in $RUNDIR"


