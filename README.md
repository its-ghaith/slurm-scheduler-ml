# Hybrid MLOps Architecture: Kubernetes + External Slurm

Diese Umgebung setzt das empfohlene Hybrid-Modell um:

- Kubernetes als zentrale MLOps-Plattform (MLflow, Prometheus, Grafana, Connector)
- Externes Slurm/HPC-Cluster fuer verteiltes GPU-Training und Batch-Jobs
- Job-Dispatch von Kubernetes nach Slurm ueber SSH (`sbatch`)

## CARPK YOLO Pipeline (phasenbasiert)

Die Trainingslogik ist jetzt in separaten Phasen organisiert:

- `preprocessing`
- `hyperparameter_tuning`
- `training` (distributed ueber mehrere GPUs)
- `evaluation`
- `counting_inference` (Autozaehlung pro Bild)

Entrypoints:
- [run_carpk_yolo_pipeline.py](D:/OLD/Sync/Projekte/Masterarbeit/Project/PoC4/run_carpk_yolo_pipeline.py)
- [train_distributed_with_energy_tracking_mlflow.py](D:/OLD/Sync/Projekte/Masterarbeit/Project/PoC4/train_distributed_with_energy_tracking_mlflow.py)

Phasen-Implementierung:
- [phase_tracker.py](D:/OLD/Sync/Projekte/Masterarbeit/Project/PoC4/carpk_yolo_pipeline/phase_tracker.py)
- [phases.py](D:/OLD/Sync/Projekte/Masterarbeit/Project/PoC4/carpk_yolo_pipeline/phases.py)

Erwartete CARPK-Datenstruktur:

```text
data/carpk/
  images/
    train/
    val/
    test/
  labels/
    train/
    val/
```

## Architektur

Kubernetes Namespace `mlops-energy`:
- `mlflow`
- `prometheus`
- `grafana`
- `slurm-connector` (SSH-Job-Connector)

Manifest:
- `rancherConfigs/hybrid-stack.yaml`

Ablauf:
1. Pipeline/Operator in Kubernetes triggert Submit.
2. `slurm-connector` verbindet sich per SSH zum externen Slurm-Login-Node.
3. Der Connector fuehrt remote `sbatch slurm/train_distributed_mlflow_job.slurm` aus.
4. Training laeuft auf externem Slurm/GPU-Cluster.
5. Metrics/Artifacts werden nach MLflow (Kubernetes) geschrieben.
6. Deployment/Serving/Monitoring verbleibt in Kubernetes.

## Voraussetzungen

- Externes Slurm-Cluster mit Login-Node
- SSH-Zugang (Private Key + `known_hosts` Eintrag)
- Auf dem Slurm-Cluster liegt das Projekt unter `REMOTE_PROJECT_DIR`
- Das Script `REMOTE_SLURM_SCRIPT` ist dort vorhanden
- Slurm-Knoten koennen `REMOTE_MLFLOW_TRACKING_URI` erreichen

## Konfiguration

1. `.env` erstellen:

```powershell
Copy-Item .env.example .env
```

2. Folgende Werte setzen:

- `CR_PAT`
- `SLURM_SSH_HOST`
- `SLURM_SSH_PORT`
- `SLURM_SSH_USER`
- `SLURM_SSH_PRIVATE_KEY`
- `SLURM_SSH_KNOWN_HOSTS`
- `REMOTE_PROJECT_DIR`
- `REMOTE_SLURM_SCRIPT`
- `REMOTE_MLFLOW_TRACKING_URI`

## Profile umschalten (Dev/Prod)

- Dev (Fake Slurm in Kubernetes):
```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\setup_rancher.ps1 -EnvFile .env.dev -Action bootstrap-dev
```

- Prod (externes echtes Slurm):
```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\setup_rancher.ps1 -EnvFile .env.prod -Action bootstrap
```

- Submit mit Profil:
```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\setup_rancher.ps1 -EnvFile .env.dev -Action submit
powershell -NoProfile -ExecutionPolicy Bypass -File .\setup_rancher.ps1 -EnvFile .env.prod -Action submit
```

## Betrieb

### Bootstrap

```powershell
./setup_rancher.ps1 -Action bootstrap
```

Macht:
- Namespace + Secrets
- Apply von `rancherConfigs/hybrid-stack.yaml`
- Connector-Config Patch
- Rollout und Health-Checks
- SSH-Test gegen externes Slurm (`sinfo -N -l`)

### Job-Submit

```powershell
./setup_rancher.ps1 -Action submit
```

Optional:

```powershell
./setup_rancher.ps1 -Action submit -BatchSize 32
```

### Distributed GPU-Submit (DDP ueber Slurm)

Neues Job-Template:
- `slurm/train_distributed_mlflow_job.slurm`
- `train_distributed_with_energy_tracking_mlflow.py`

Direkt ueber den Connector:

```powershell
kubectl --kubeconfig rancherConfigs/main.yaml -n mlops-energy exec deploy/slurm-connector -- `
  bash -lc "REMOTE_SLURM_SCRIPT='slurm/train_distributed_mlflow_job.slurm' BATCH_SIZE=64 EPOCHS=3 /opt/slurm-connector/submit_remote_slurm.sh"
```

Wichtige Env-Optionen fuer CARPK+YOLO:
- `CARPK_DATASET_DIR` (default: `${SLURM_SUBMIT_DIR}/data/carpk`)
- `CARPK_INFER_DIR` (default: `${CARPK_DATASET_DIR}/images/test`)
- `YOLO_MODEL` (default: `yolov8n.pt`)
- `RUN_TUNING` (`1` oder `0`)
- `TRAIN_DEVICE` (z. B. `0,1` fuer Multi-GPU)

Hinweis:
- Mit `fake-slurm-dev` testest du den End-to-End-Connector-Pfad.
- Den echten Slurm-Performance-Vorteil (Scheduling/Multi-GPU) siehst du nur mit externem Slurm-Cluster und >=2 GPUs.

### Test

```powershell
./setup_rancher.ps1 -Action test
```

## Port-Forward

```powershell
./setup_rancher.ps1 -Action portforward
```

## Wichtiger Unterschied zum alten Stack

Der alte Stack `rancherConfigs/slurm-stack.yaml` startet `slurmctld/slurmd` in Kubernetes.
Der neue Hybrid-Stack nutzt kein In-Cluster-Slurm, sondern einen externen Slurm-Cluster ueber SSH.
