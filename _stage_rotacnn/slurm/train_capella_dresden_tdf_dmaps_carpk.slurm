#!/bin/bash
#SBATCH -J recv-tagderforschung
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
python -c "import sys,inspect,effcv; print('venv:', sys.executable); print('effcv @', inspect.getfile(effcv))"


srun uv run python src/countcv/effcv/run_experiment.py \
  --multirun \
  +model=cnn_base ++model.version=fcrn-base,SAUnet +dataset=carpk ++experiment=tagderforschung\
  ++dataset.img_size=[512,256]\
  ++remote_mlflow=False \
  ++model.epochs=300 \
  ++model.sigma=2 \
  ++model.alpha=100 \
  ++model.tilesize=null\
  ++model.lambda_count=0.001 \
  ++model.infer_normalisation=False \
  ++dataset.train_size=500 \
  ++model.batch_size=32\
  ++dataset.path="$DATADIR" \
  ++dataset.data_yaml="$DATADIR"/data.yaml 

echo "Job ${SLURM_JOB_ID} finished; results in $RUNDIR"


