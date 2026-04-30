# Rancher Branch: SLURM + MLflow + Prometheus + Grafana (Kubernetes)

Dieser Branch (`Rancher-sulrm`) enthaelt nur die Rancher/Kubernetes-Umgebung.
Ziel: reproduzierbarer End-to-End-Workflow fuer GPU-Training ueber SLURM mit Sichtbarkeit in MLflow und Grafana.

## Zielbild

Pro SLURM-Job soll es geben:
- genau **einen** MLflow-Run (`job-energy-<JOBID>`) mit Trainings- und Energie-Metriken
- Prometheus-Metriken fuer Dashboard/KPIs in Grafana

## Architektur

Komponenten im Namespace `mlops-energy`:
- `slurmctld` (Controller)
- `slurmd` (Worker mit GPU)
- `slurmd-node-exporter` (Textfile Collector Sidecar)
- `mlflow`
- `prometheus`
- `grafana`

Manifest:
- `rancherConfigs/slurm-stack.yaml`

Visualisierung:

![Workflow](/D:/OLD/Sync/Projekte/Masterarbeit/Project/PoC4/docs_assets/workflow.png)

![Datenfluss](/D:/OLD/Sync/Projekte/Masterarbeit/Project/PoC4/docs_assets/dataflow.png)

## Datenfluss

1. `sbatch` wird in `slurmctld` gestartet.
2. `slurmd` fuehrt Training auf GPU aus.
3. GPU-CSV + Summary-JSON werden in `/workspace/energy_metrics` erzeugt.
4. `export_job_metrics_prom.py` schreibt `job_<id>.prom` und `aggregate.prom`.
5. `slurmd-node-exporter` liest Textfiles, Prometheus scraped, Grafana visualisiert.
6. MLflow erhaelt Trainings- und Energie-Metriken im selben Run `job-energy-<JOBID>`.

## Relevante Dateien

- `setup_rancher.ps1`
- `.env.example`
- `rancherConfigs/slurm-stack.yaml`
- `slurm/train_mlflow_local.slurm`
- `slurm/log_job_energy_mlflow.py`
- `train_with_energy_tracking_mlflow.py`
- `slurm/export_job_metrics_prom.py`
- `grafana/dashboards/slurm-energy-overview.json`
- `prometheus/prometheus.yml`

## Konfiguration

### 1) `.env` erstellen

```powershell
Copy-Item .env.example .env
```

`.env` wird nicht committed (`.gitignore`).

### 2) Kubeconfig lokal bereitstellen

Pfad:

`rancherConfigs/main.yaml`

Wenn die Datei fehlt, schlagen `bootstrap`, `submit` und `test` sofort fehl.

### 3) Pflicht-/Empfehlungswerte

```env
CR_PAT=ghp_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
GHCR_USER=<your-github-username>
GHCR_SERVER=ghcr.io
GHCR_IMAGE=ghcr.io/<your-github-username>/slurm-scheduler-ml/mlops-slurm-runtime:latest

ACTION=bootstrap
KUBECONFIG_PATH=rancherConfigs/main.yaml
K8S_NAMESPACE=<your-k8s-namespace>
K8S_MANIFEST=rancherConfigs/slurm-stack.yaml

TRAIN_CMD=python train_with_energy_tracking_mlflow.py
BATCH_SIZE=64
MLFLOW_TRACKING_URI=http://mlflow:5000
GPU_SAMPLE_INTERVAL_SEC=1
ELECTRICITY_PRICE_EUR_PER_KWH=0.30
GRID_CO2_KG_PER_KWH=0.40
PUE_FACTOR=1.0
MLFLOW_LOG_JOB_ENERGY=1
MLFLOW_JOB_ENERGY_EXPERIMENT=ml-energy-poc
```

## Betrieb

### A) Bootstrap

```powershell
./setup_rancher.ps1 -Action bootstrap
```

Macht:
- Namespace check
- GHCR Secret (`ghcr-pull-secret`) mit `CR_PAT`
- `kubectl apply` auf Manifest
- Rollout restart/wait fuer `mlflow`, `slurmctld`, `slurmd`
- Basis-Checks

### B) Training-Job submit

```powershell
./setup_rancher.ps1 -Action submit
```

Optional:

```powershell
./setup_rancher.ps1 -Action submit -TrainCmd "python train.py"
```

### C) Schnelltest

```powershell
./setup_rancher.ps1 -Action test
```

## Webzugriff

Port-Forward manuell (separate Terminals):

```powershell
kubectl --kubeconfig rancherConfigs/main.yaml -n mlops-energy port-forward svc/grafana 3000:3000
kubectl --kubeconfig rancherConfigs/main.yaml -n mlops-energy port-forward svc/mlflow 5000:5000
kubectl --kubeconfig rancherConfigs/main.yaml -n mlops-energy port-forward svc/prometheus 9090:9090
```

UIs:
- Grafana: [http://localhost:3000](http://localhost:3000)
- MLflow: [http://localhost:5000](http://localhost:5000)
- Prometheus: [http://localhost:9090](http://localhost:9090)

## End-to-End Verifikation

1. Pods gesund:
```powershell
kubectl --kubeconfig rancherConfigs/main.yaml -n mlops-energy get pods -o wide
```

2. SLURM gesund:
```powershell
kubectl --kubeconfig rancherConfigs/main.yaml -n mlops-energy exec deploy/slurmctld -- bash -lc "scontrol ping && sinfo -N -l"
```

3. Job submit + warten:
```powershell
./setup_rancher.ps1 -Action submit
kubectl --kubeconfig rancherConfigs/main.yaml -n mlops-energy exec deploy/slurmctld -- squeue
```

4. Energieartefakte vorhanden:
```powershell
kubectl --kubeconfig rancherConfigs/main.yaml -n mlops-energy exec deploy/slurmd -c slurmd -- bash -lc "ls -lah /workspace/energy_metrics && ls -lah /workspace/energy_metrics/node_exporter"
```

5. Prometheus Query liefert Werte:
```powershell
kubectl --kubeconfig rancherConfigs/main.yaml -n mlops-energy exec deploy/prometheus -- sh -lc "wget -qO- 'http://localhost:9090/api/v1/query?query=slurm_jobs_total'"
kubectl --kubeconfig rancherConfigs/main.yaml -n mlops-energy exec deploy/prometheus -- sh -lc "wget -qO- 'http://localhost:9090/api/v1/query?query=slurm_job_training_energy_kwh'"
```

6. MLflow: Experiment `ml-energy-poc` zeigt Run `job-energy-<JOBID>` mit:
- Training: `loss`, `duration_seconds`
- Energie: `training_energy_kwh`, `estimated_electricity_cost_eur`, `estimated_co2_kg`, etc.

## Troubleshooting

### MLflow "Die Website ist nicht erreichbar"
- Ursache meist lokaler Port-Forward.
- Neu starten:
```powershell
Get-Process kubectl -ErrorAction SilentlyContinue | Stop-Process -Force
kubectl --kubeconfig rancherConfigs/main.yaml -n mlops-energy port-forward svc/mlflow 5000:5000
```
- Check:
```powershell
Invoke-WebRequest http://localhost:5000 -UseBasicParsing
```

### Grafana "No data"
- Prometheus Query direkt pruefen (`slurm_jobs_total`, `slurm_job_training_energy_kwh`).
- Zeitfenster in Grafana vergroessern (`Last 30 days`).
- `job_*.prom` und `aggregate.prom` im `slurmd` Pod pruefen.

### CR_PAT Fehler bei Bootstrap
- `.env` muss `CR_PAT` enthalten oder `CR_PAT` in Shell setzen.
- Secret bei Bedarf neu erzeugen (`bootstrap` macht das automatisch).

### Zwei Experimente / zwei Runs pro Job
- Zielzustand ist **ein Experiment** `ml-energy-poc` und **ein Run** `job-energy-<JOBID>` pro Job.
- Altes Experiment `slurm-job-energy` wurde auf `deleted` gesetzt.

## Wichtiger Hinweis zum Runtime-Image

Wenn Pods ein aelteres Runtime-Image nutzen, koennen Skript-Aenderungen lokal nicht sofort im Cluster aktiv sein.
Fuer dauerhafte Wirkung:
1. Image neu bauen/pushen.
2. `rancherConfigs/slurm-stack.yaml` mit neuem Tag aktualisieren.
3. `./setup_rancher.ps1 -Action rollout` oder `bootstrap`.

## Branch-Policy

- Dieser Branch ist strikt Rancher/Kubernetes.
- Local-Docker-Setup liegt in `local-sulrm`.
- `.env` nie committen, nur `.env.example`.
- `rancherConfigs/main.yaml` bleibt lokal/ignoriert.
