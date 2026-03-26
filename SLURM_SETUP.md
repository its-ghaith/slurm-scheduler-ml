# SLURM Setup fuer dieses Projekt

Diese Variante ersetzt `docker-compose` fuer das Training auf einem HPC-Cluster mit SLURM.

## 1) Einmalig vorbereiten

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Hinweis: Falls dein Cluster mit `module` arbeitet, lade zuerst passende Module (Python/CUDA).

## 2) Optional: MLflow als SLURM Job starten

```bash
sbatch slurm/mlflow_server.slurm
squeue -u $USER
```

Den Hostnamen des MLflow-Jobs siehst du in `logs/mlflow-<jobid>.out`.

Optionaler Tunnel vom lokalen Rechner:

```bash
ssh -L 5000:<mlflow-hostname>:5000 <user>@<cluster-login-host>
```

Dann ist MLflow lokal unter `http://127.0.0.1:5000` erreichbar.

## 3) Training starten

Einzeljob:

```bash
export MLFLOW_TRACKING_URI=http://<mlflow-hostname>:5000
sbatch slurm/train_mlflow_job.slurm
```

Beliebiges Trainingskommando ohne Code-Aenderung in der Train-Datei:

```bash
export MLFLOW_TRACKING_URI=http://<mlflow-hostname>:5000
export TRAIN_CMD="python train.py"
sbatch slurm/train_mlflow_job.slurm
```

Batch-Sweep (16/32/64/128):

```bash
export MLFLOW_TRACKING_URI=http://<mlflow-hostname>:5000
bash slurm/submit_batch_sweep.sh
```

## 4) Monitoring

- Jobstatus: `squeue -u $USER`
- Logs: `tail -f logs/ml-energy-<jobid>.out`
- MLflow UI: `http://<mlflow-hostname>:5000` (oder via SSH-Tunnel lokal)
- Job-Energie-CSV: `energy_metrics/gpu_metrics_job_<jobid>.csv`
- Job-Energie-JSON: `energy_metrics/gpu_summary_job_<jobid>.json`

## 5) Cluster-Anpassungen

Passe in den SLURM-Dateien bei Bedarf an:

- `#SBATCH --partition=...`
- `#SBATCH --gres=gpu:...`
- `#SBATCH --time=...`
- `module load ...`
