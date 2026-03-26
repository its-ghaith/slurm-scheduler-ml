# Lokale HPC-Simulation mit Docker + SLURM

Damit simulierst du einen kleinen HPC-Cluster lokal und nutzt spaeter auf echtem HPC fast denselben Workflow.

## 1) Cluster lokal starten

```powershell
docker compose -f docker-compose.slurm.yml up -d --build
```

Pruefen:

```powershell
docker exec -it slurmctld sinfo
docker exec -it slurmctld squeue
```

## 2) Trainingsjob submitten (lokal)

Einzeljob:

```powershell
docker exec -it slurmctld sbatch /workspace/slurm/train_mlflow_local.slurm
docker exec -it slurmctld squeue
```

Beliebiges Trainingskommando ohne Datei-Aenderung im Job-Wrapping:

```powershell
docker exec -it slurmctld bash -lc "TRAIN_CMD='python train.py' sbatch /workspace/slurm/train_mlflow_local.slurm"
```

Batch-Size vorgeben:

```powershell
docker exec -it slurmctld bash -lc "BATCH_SIZE=64 sbatch /workspace/slurm/train_mlflow_local.slurm"
```

## 3) Logs und MLflow

Logs:

```powershell
docker exec -it slurmctld bash -lc "ls -lah /workspace/logs && tail -n 100 /workspace/logs/*.out"
```

MLflow UI lokal:

- [http://localhost:5000](http://localhost:5000)

Pro Job werden erzeugt:

- `/workspace/energy_metrics/gpu_metrics_job_<JOBID>.csv`
- `/workspace/energy_metrics/gpu_summary_job_<JOBID>.json`

Optionales Logging der Job-Energie nach MLflow:

- Aktiv: `MLFLOW_LOG_JOB_ENERGY=1` (Default)
- Experiment: `slurm-job-energy` (anpassbar via `MLFLOW_JOB_ENERGY_EXPERIMENT`)

## 4) Auf echtes HPC migrieren

1. Nutze `slurm/train_mlflow_job.slurm` (statt `train_mlflow_local.slurm`).
2. Passe `#SBATCH` Werte an (`partition`, `gres`, `time`, `mem`, `cpus-per-task`).
3. Setze `MLFLOW_TRACKING_URI` auf den echten MLflow-Service im Cluster.
